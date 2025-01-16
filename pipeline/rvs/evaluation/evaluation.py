import traceback
from dataclasses import dataclass, field, replace
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import List, Optional, Set, Tuple, Type, Union

import yaml
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
from rvs.pipeline.pipeline import Pipeline, PipelineConfig, PipelineStage
from rvs.utils.console import file_link
from rvs.utils.logging import create_logger


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

    from_stage: Optional[PipelineStage] = None
    """If configured the pipeline is only run from this stage and no earlier"""

    to_stage: Optional[PipelineStage] = None
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


@dataclass
class EvaluationRun:
    instance: PipelineEvaluationInstance
    progress_logger: Logger


@dataclass
class PipelineRun:
    parent: EvaluationRun
    file: Path
    stages: Optional[List[PipelineStage]]


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

    runs_dir: Path
    """Output directory for eval configs and logs"""

    embedder: Embedder

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def init(self) -> None:
        self.intermediate_dir = self.config.output_dir / "intermediate"
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = self.config.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.runs_dir = self.config.output_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

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
        run_dir = self.__create_run_dir()

        self.__save_eval_config([self.config.output_dir / "config.yaml", run_dir / "config.yaml"])

        with create_logger(
            __name__, files=[self.config.output_dir / "progress.log", run_dir / "progress.log"]
        ) as progress_logger_handle:
            run = EvaluationRun(
                instance=PipelineEvaluationInstance(
                    self.__configure_pipeline(self.config.pipeline), self.intermediate_dir
                ),
                progress_logger=progress_logger_handle.logger,
            )

            aborted = False

            if not self.config.eval_only:
                if self.config.stage_by_stage:
                    aborted = not self.__run_stage_by_stage(run)
                else:
                    aborted = not self.__run_object_by_object(run)

            if aborted:
                return False

            evaluate_results(self.lvis, self.embedder, run.instance, self.results_dir)

            return True

    def __create_run_dir(self) -> Path:
        base_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = self.runs_dir / base_name
        if run_dir.exists():
            i = 2
            while True:
                run_dir = self.runs_dir / (base_name + "_" + str(i))
                if not run_dir.exists():
                    break
                i += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def __save_eval_config(self, files: Union[Path, List[Path]]) -> None:
        if isinstance(files, Path):
            files = [files]
        cfg = yaml.dump(self.config)
        for file in files:
            file.parent.mkdir(parents=True, exist_ok=True)
            CONSOLE.log(f"Saving evaluation config to: {file_link(file)}")
            file.write_text(cfg, "utf8")

    def __run_stage_by_stage(self, run: EvaluationRun) -> bool:
        stages = PipelineStage.between(self.config.from_stage, self.config.to_stage, default=PipelineStage.all())

        num_successful_runs = 0

        num_runs = 0
        total_runs = 0
        for stage in stages:
            for category in self.lvis.dataset.keys():
                total_runs += len(self.lvis.dataset[category])

        for stage in stages:
            CONSOLE.log(f"Processing stage {str(stage)}...")

            for category in self.lvis.dataset.keys():
                CONSOLE.log(f"Processing category {category}...")

                for uid in self.lvis.dataset[category]:
                    if (
                        self.config.pipeline_run_limit is not None
                        and num_successful_runs >= self.config.pipeline_run_limit
                    ):
                        return False

                    file = Path(self.lvis.uid_to_file[uid])

                    CONSOLE.log(f"Processing uid {uid} ({file_link(file)})...")

                    if self.__run_pipeline(PipelineRun(parent=run, file=file, stages=[stage])):
                        num_successful_runs += 1
                    num_runs += 1

                    run.progress_logger.info(
                        f"Progress: {num_runs} / {total_runs} ({'{:.2f}'.format(float(num_runs) / total_runs * 100.0)}%)"
                    )

        return True

    def __run_object_by_object(self, run: EvaluationRun) -> bool:
        stages = PipelineStage.between(self.config.from_stage, self.config.to_stage, default=PipelineStage.all())

        num_successful_runs = 0

        num_runs = 0
        total_runs = 0
        for category in self.lvis.dataset.keys():
            total_runs += len(self.lvis.dataset[category])

        for category in self.lvis.dataset.keys():
            CONSOLE.log(f"Processing category {category}...")

            for uid in self.lvis.dataset[category]:
                if self.config.pipeline_run_limit is not None and num_successful_runs >= self.config.pipeline_run_limit:
                    return False

                file = Path(self.lvis.uid_to_file[uid])

                CONSOLE.log(f"Processing uid {uid} ({file_link(file)})...")

                if self.__run_pipeline(PipelineRun(parent=run, file=file, stages=stages)):
                    num_successful_runs += 1
                num_runs += 1

                run.progress_logger.info(
                    f"Progress: {num_runs} / {total_runs} ({'{:.2f}'.format(float(num_runs) / total_runs * 100.0)}%)"
                )

        return True

    def __run_pipeline(
        self,
        run: PipelineRun,
        handle_errors: bool = True,
    ) -> bool:
        stages_str = "..." if run.stages is None else ", ".join([stage.name for stage in run.stages])
        pipeline_str = f'"{self.__safe_resolve(run.file)}" -> [{stages_str}] -> "{self.__safe_resolve(run.parent.instance.get_pipeline_dir(run.file))}" -> "{self.__safe_resolve(run.parent.instance.get_results_dir(run.file))}"'

        skip = False

        if self.config.skip_finished:
            try:
                index_file = run.parent.instance.get_index_file(run.file)
                if index_file.exists():
                    load_index(index_file, validate=True)
                    skip = True  # If index exists and is valid then pipeline has finished
            except Exception:
                run.parent.progress_logger.warning(
                    "Index of pipeline %s is invalid:\n%s", pipeline_str, traceback.format_exc()
                )

        if skip:
            run.parent.progress_logger.info("Skipping pipeline %s", pipeline_str)
            return False

        with ProcessResult() as result:
            run.parent.progress_logger.info("Starting pipeline %s", pipeline_str)

            process: Process = None
            try:
                process = start_process(
                    target=pipeline_worker_func,
                    args=(
                        run.parent.instance,
                        run.file,
                        run.stages,
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
                    run.parent.progress_logger.info("Finished pipeline %s", pipeline_str)
                elif result.msg is not None:
                    run.parent.progress_logger.error(
                        "Failed pipeline %s due to exception:\n%s", pipeline_str, result.msg
                    )
                else:
                    run.parent.progress_logger.error("Failed pipeline %s due to unknown reason", pipeline_str)

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


@dataclass
class EvaluationResumeConfig:
    config: Path

    def load(self) -> Tuple[EvaluationConfig, Optional[Path]]:
        if self.config is None:
            raise ValueError("No config path specified")

        if not self.config.exists() or not self.config.is_file():
            raise ValueError(f'Config file "{self.config}" does not exist')

        config: EvaluationConfig = yaml.load(self.config.read_text(encoding="utf8"), Loader=yaml.Loader)

        working_dir = self.__find_working_dir(config, self.config.parent)

        return (config, working_dir)

    def __find_working_dir(self, config: EvaluationConfig, config_dir: Path) -> Optional[Path]:
        if config.output_dir.is_absolute():
            return None

        assert config_dir.is_dir()

        config_dir = config_dir.resolve()

        common_dir = EvaluationResumeConfig.__find_common_base_dir(config_dir, config.output_dir)

        if common_dir is None:
            raise Exception(
                f'Unable to determine output directory from previous output directory "{str(config.output_dir)}" and config directory "{str(config_dir)}"'
            )

        return common_dir.parent

    @staticmethod
    def __find_common_base_dir(full: Path, part: Path) -> Optional[Path]:
        full = full.resolve()

        while True:
            match = EvaluationResumeConfig.__match_dirs(full, part)
            if match is not None:
                return match

            next_full = full.parent
            if next_full == full:
                return None
            full = next_full

    @staticmethod
    def __match_dirs(full: Path, part: Path) -> Optional[Path]:
        while True:
            if part.is_absolute():
                raise ValueError(f'Partial path "{str(part)}" is absolute path')

            if part.name != full.name:
                return None

            next_part = part.parent
            if next_part == next_part.parent:
                return full
            part = next_part

            next_full = full.parent
            if next_full == full:
                return None
            full = next_full
