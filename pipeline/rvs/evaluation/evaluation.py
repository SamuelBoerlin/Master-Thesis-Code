import logging
import traceback
from dataclasses import dataclass, field, replace
from logging import Formatter, Logger
from pathlib import Path
from typing import List, Optional, Set, Type

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.utils.rich_utils import CONSOLE
from torch.multiprocessing import Process

from rvs.evaluation.embedder import Embedder, EmbedderConfig
from rvs.evaluation.evaluation_method import evaluate_results
from rvs.evaluation.index import load_index
from rvs.evaluation.lvis import LVISDataset
from rvs.evaluation.pipeline import PipelineEvaluationInstance
from rvs.evaluation.process import ProcessResult, start_process, stop_process
from rvs.evaluation.worker import pipeline_worker_func
from rvs.pipeline.pipeline import Pipeline, PipelineConfig
from rvs.utils.console import file_link


@dataclass
class EvaluationConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Evaluation)

    pipeline: PipelineConfig = field(default_factory=lambda: PipelineConfig)
    """Configuration of the pipeline to use for the evaluation"""

    lvis_categories: Optional[Set[str]] = None
    """List of LVIS categories used in the evaluation (unconfigured = all)"""

    lvis_uids: Optional[Set[str]] = None
    """List of LVIS uids used in the evaluation (unconfigured = all)"""

    lvis_download_processes: int = 8
    """Number of processes to use for downloading the 3D model files"""

    lvis_per_category_limit: Optional[int] = None
    """Per category limit, useful for testing"""

    output_dir: Path = Path("outputs")
    """Relative or absolute output directory to save all output data"""

    timestamp: str = "{timestamp}"
    """Evaluation/experiment timestamp."""

    stage_by_stage: bool = False
    """Whether to process the objects stage-by-stage (i.e. first SAMPLE_VIEWS for all objects, then RENDER_VIEWS, etc.) instead of all stages object-by-object"""

    from_stage: Optional[Pipeline.Stage] = None
    """If configured the pipeline is only run from this stage and no earlier"""

    to_stage: Optional[Pipeline.Stage] = None
    """If configured the pipeline is only run up to this stage and no further"""

    pipeline_run_limit: Optional[int] = None
    """Maximum number of pipeline runs before it aborts"""

    skip_finished: bool = False
    """Whether to skip re-running pipelines if final output is already available"""

    # FIXME This (or any other further "sub-configs") breaks tyro for some strange reason...
    # embedder: EmbedderConfig = field(defaulut_factory=lambda: EmbedderConfig)
    # """Configuration of the CLIP embedder used for the precision/recall/accuracy evaluation"""

    eval_only: bool = False
    """Run evaluation part only"""


class Evaluation:
    config: EvaluationConfig

    lvis: LVISDataset
    """Objaverse 1.0 LVIS dataset"""

    lvis_cache_dir: Path
    """Directory for caching lvis dataset"""

    intermediate_dir: Path
    """Output directory for intermediate results"""

    results_dir: Path
    """Output directory for final results"""

    @property
    def log_file_path(self) -> Path:
        return self.config.output_dir / "logs.log"

    logger: Logger
    logger_format: Formatter

    embedder: Embedder

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def init(self) -> None:
        self.logger = Logger(__name__)
        self.logger_format = Formatter(fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        self.intermediate_dir = self.config.output_dir / "intermediate"
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = self.config.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        CONSOLE.log("Setting up embedder...")
        self.embedder = EmbedderConfig().setup()

        self.lvis_cache_dir = self.config.output_dir / "lvis"
        self.lvis_cache_dir.mkdir(parents=True, exist_ok=True)

        self.lvis = LVISDataset(
            self.config.lvis_categories,
            self.config.lvis_uids,
            self.config.lvis_download_processes,
            self.config.lvis_per_category_limit,
        )
        if self.lvis.load_cache(self.lvis_cache_dir) is None:
            self.lvis.load()
            self.lvis.save_cache(self.lvis_cache_dir)

    def run(self) -> bool:
        logger_file_handler = logging.FileHandler(self.log_file_path)
        logger_file_handler.setFormatter(self.logger_format)
        self.logger.addHandler(logger_file_handler)

        instance = PipelineEvaluationInstance(self.__configure_pipeline(self.config.pipeline), self.intermediate_dir)

        aborted = False

        if not self.config.eval_only:
            try:
                if self.config.stage_by_stage:
                    aborted = not self.__run_stage_by_stage(instance)
                else:
                    aborted = not self.__run_object_by_object(instance)
            finally:
                self.logger.removeHandler(logger_file_handler)
                logger_file_handler.close()

        if aborted:
            return False

        evaluate_results(self.lvis, self.embedder, instance, self.results_dir)

        return True

    def __run_stage_by_stage(self, instance: PipelineEvaluationInstance) -> bool:
        stages = Pipeline.Stage.between(self.config.from_stage, self.config.to_stage, default=Pipeline.Stage.all())

        num_runs = 0

        for stage in stages:
            CONSOLE.log(f"Processing stage {str(stage)}...")

            for category in self.lvis.dataset.keys():
                CONSOLE.log(f"Processing category {category}...")

                for uid in self.lvis.dataset[category]:
                    if self.config.pipeline_run_limit is not None and num_runs >= self.config.pipeline_run_limit:
                        return False

                    file = Path(self.lvis.uid_to_file[uid])

                    CONSOLE.log(f"Processing uid {uid} ({file_link(file)})...")

                    if self.__run_pipeline(instance, file, [stage]):
                        num_runs += 1

        return True

    def __run_object_by_object(self, instance: PipelineEvaluationInstance) -> bool:
        stages = Pipeline.Stage.between(self.config.from_stage, self.config.to_stage, default=Pipeline.Stage.all())

        num_runs = 0

        for category in self.lvis.dataset.keys():
            CONSOLE.log(f"Processing category {category}...")

            for uid in self.lvis.dataset[category]:
                if self.config.pipeline_run_limit is not None and num_runs >= self.config.pipeline_run_limit:
                    return False

                file = Path(self.lvis.uid_to_file[uid])

                CONSOLE.log(f"Processing uid {uid} ({file_link(file)})...")

                if self.__run_pipeline(instance, file, stages):
                    num_runs += 1

        return True

    def __run_pipeline(
        self,
        instance: PipelineEvaluationInstance,
        file: Path,
        stages: Optional[List[Pipeline.Stage]],
        handle_errors: bool = True,
    ) -> bool:
        pipeline_str = f'"{self.__safe_resolve(file)}" -> [{", ".join([stage.name for stage in stages])}] -> "{self.__safe_resolve(instance.get_pipeline_dir(file))}" -> "{self.__safe_resolve(instance.get_results_dir(file))}"'

        skip = False

        if self.config.skip_finished:
            try:
                index_file = instance.get_index_file(file)
                if index_file.exists():
                    load_index(index_file, validate=True)
                    skip = True  # If index exists and is valid then pipeline has finished
            except Exception:
                self.logger.warning("Index of pipeline %s is invalid:\n%s", pipeline_str, traceback.format_exc())

        if skip:
            self.logger.info("Skipping pipeline %s", pipeline_str)
            return False

        with ProcessResult() as result:
            self.logger.info("Starting pipeline %s", pipeline_str)

            process: Process = None
            try:
                process = start_process(
                    target=pipeline_worker_func,
                    args=(
                        instance,
                        file,
                        stages,
                        result,
                    ),
                )
                process.join()
            except Exception as ex:
                if handle_errors:
                    result.success = False
                    result.msg = traceback.format_exc()
                else:
                    raise ex
            finally:
                try:
                    stop_process(process)
                    process.join()
                    process.close()
                except Exception:
                    pass

                if result.success:
                    self.logger.info("Finished pipeline %s", pipeline_str)
                elif result.msg is not None:
                    self.logger.error("Failed pipeline %s due to exception:\n%s", pipeline_str, result.msg)
                else:
                    self.logger.error("Failed pipeline %s due to unknown reason", pipeline_str)

            return result.success

    def __safe_resolve(self, file: Path) -> str:
        try:
            return str(file.resolve())
        except Exception:
            return str(file)

    def __configure_pipeline(self, config: PipelineConfig) -> PipelineConfig:
        config = replace(self.config.pipeline)
        config.timestamp = self.config.timestamp
        config.set_timestamp()
        return config
