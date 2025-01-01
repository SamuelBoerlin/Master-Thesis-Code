import gc
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Set, Type

import torch
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.utils.rich_utils import CONSOLE
from objaverse import load_lvis_annotations, load_objects

from rvs.pipeline.pipeline import Pipeline, PipelineConfig
from rvs.scripts.rvs import _set_random_seed
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


class Evaluation:
    config: EvaluationConfig

    lvis_dataset: Dict[str, List[str]]
    """Mapping of LVIS category to list of objaverse 1.0 uids"""

    lvis_files: Dict[str, str]
    """Mapping of LVIS objaverse 1.0 uid to local file path"""

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def init(self) -> None:
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
        for category in self.lvis_dataset.keys():
            CONSOLE.log(f"Processing category {category}...")

            for uid in self.lvis_dataset[category]:
                file = Path(self.lvis_files[uid])
                CONSOLE.log(f"Processing uid {uid} ({file_link(file)})...")
                self.__process_file(file)

    def __process_file(self, file: Path) -> None:
        pipeline = self.__load_pipeline(self.config.pipeline, file)

        results = pipeline.run()

        # Clear memory for next run
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    def __load_pipeline(self, config: PipelineConfig, file: Path) -> Pipeline:
        # Clone config
        config = replace(config)

        config = self.__configure_pipeline(config, file)

        config.set_timestamp()

        _set_random_seed(config.machine.seed)

        pipeline: Pipeline = config.setup(local_rank=0, world_size=1)

        pipeline.init()

        config.print_to_terminal()
        config.save_config()

        return pipeline

    def __configure_pipeline(self, config: PipelineConfig, file: Path) -> PipelineConfig:
        config.output_dir = Path.joinpath(self.config.output_dir, file.name)
        config.experiment_name = "evaluation"
        config.model_file = file
        config.timestamp = self.config.timestamp
        return config
