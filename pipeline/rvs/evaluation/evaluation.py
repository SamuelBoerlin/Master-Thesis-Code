import os
import signal
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import sleep, time
from typing import Dict, List, Optional, Set, Type

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.utils.rich_utils import CONSOLE
from objaverse import load_lvis_annotations, load_objects
from torch.multiprocessing import Process, set_start_method

from rvs.evaluation.pipeline import PipelineEvaluationInstance
from rvs.pipeline.pipeline import Pipeline, PipelineConfig
from rvs.utils.console import file_link


@dataclass
class EvaluationConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Evaluation)

    pipeline: PipelineConfig = field(default_factory=lambda: PipelineConfig)
    """Configuration of the pipeline to use for the evaluation"""

    lvis_categories: Set[str] = None
    """List of LVIS categories used in the evaluation (unconfigured = all)"""

    lvis_uids: Set[str] = None
    """List of LVIS uids used in the evaluation (unconfigured = all)"""

    lvis_download_processes = 8
    """Number of processes to use for downloading the 3D model files"""

    output_dir: Path = Path("outputs")
    """Relative or absolute output directory to save all output data"""

    timestamp: str = "{timestamp}"
    """Evaluation/experiment timestamp."""

    stage_by_stage: bool = False
    """Whether to process the objects stage-by-stage (i.e. first SAMPLE_VIEWS for all objects, then RENDER_VIEWS, etc.) instead of all stages object-by-object"""

    up_to_stage: Optional[Pipeline.Stage] = None
    """If configured the pipeline is only run up to this stage and no further"""


def run_pipeline_worker(
    instance: PipelineEvaluationInstance, file: Path, stages: Optional[List[Pipeline.Stage]] = None
) -> None:
    instance.run(file, stages)


class Evaluation:
    config: EvaluationConfig

    lvis_dataset: Dict[str, List[str]]
    """Mapping of LVIS category to list of objaverse 1.0 uids"""

    lvis_files: Dict[str, str]
    """Mapping of LVIS objaverse 1.0 uid to local file path"""

    intermediate_dir: Path
    """Output directory for intermediate results"""

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def init(self) -> None:
        self.intermediate_dir = self.config.output_dir / "intermediate"
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)

        CONSOLE.log("Loading LVIS dataset...")
        self.lvis_dataset = self.__load_lvis_dataset(self.config.lvis_categories, self.config.lvis_uids)

        CONSOLE.rule("Loading LVIS files...")
        self.lvis_files = {}
        for k in self.lvis_dataset.keys():
            CONSOLE.log(f"Category: {k}")
            category_files = load_objects(self.lvis_dataset[k], download_processes=self.config.lvis_download_processes)
            CONSOLE.log(f"Files: {len(category_files)}")
            self.lvis_files.update(category_files)
        CONSOLE.rule()

    def __load_lvis_dataset(self, categories: Optional[Set[str]], uids: Optional[Set[str]]) -> Dict[str, List[str]]:
        dataset = load_lvis_annotations()
        if categories is not None:
            for k in list(dataset.keys()):
                if k not in categories:
                    del dataset[k]
        if uids is not None:
            for k in list(dataset.keys()):
                filtered = [u for u in dataset[k] if u in uids]
                if len(filtered) > 0:
                    dataset[k] = filtered
                else:
                    del dataset[k]
        return dataset

    def run(self) -> None:
        if self.config.stage_by_stage:
            self.__run_stage_by_stage()
        else:
            self.__run_object_by_object()

    def __run_stage_by_stage(self) -> None:
        stages = self.config.up_to_stage.up_to() if self.config.up_to_stage is not None else None

        for stage in stages:
            CONSOLE.log(f"Processing stage {str(stage)}...")

            for category in self.lvis_dataset.keys():
                CONSOLE.log(f"Processing category {category}...")

                for uid in self.lvis_dataset[category]:
                    file = Path(self.lvis_files[uid])
                    CONSOLE.log(f"Processing uid {uid} ({file_link(file)})...")
                    self.__run_pipeline(file, [stage])

    def __run_object_by_object(self) -> None:
        stages = self.config.up_to_stage.up_to() if self.config.up_to_stage is not None else None

        for category in self.lvis_dataset.keys():
            CONSOLE.log(f"Processing category {category}...")

            for uid in self.lvis_dataset[category]:
                file = Path(self.lvis_files[uid])
                CONSOLE.log(f"Processing uid {uid} ({file_link(file)})...")
                self.__run_pipeline(file, stages)

    def __run_pipeline(self, file: Path, stages: Optional[List[Pipeline.Stage]] = None) -> None:
        set_start_method(
            "spawn", force=True
        )  # Required for CUDA: https://pytorch.org/docs/main/notes/multiprocessing.html
        process = Process(
            target=run_pipeline_worker,
            args=(
                PipelineEvaluationInstance(self.__configure_pipeline(self.config.pipeline), self.intermediate_dir),
                file,
                stages,
            ),
            daemon=True,
        )
        try:
            process.start()
            process.join()
        finally:
            stop_time = time()
            while process.is_alive():
                elapsed_time = time() - stop_time
                if elapsed_time > 1.0:
                    process.kill()
                elif elapsed_time > 0.1:
                    os.kill(process.pid, signal.SIGINT)
                sleep(0.01)

    def __configure_pipeline(self, config: PipelineConfig) -> PipelineConfig:
        config = replace(self.config.pipeline)
        config.timestamp = self.config.timestamp
        return config
