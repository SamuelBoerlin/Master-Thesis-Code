import gc
import shutil
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import torch

from rvs.evaluation.index import save_index
from rvs.pipeline.pipeline import Pipeline, PipelineConfig
from rvs.scripts.rvs import _set_random_seed
from rvs.utils.nerfstudio import create_transforms_json, get_frame_name

INDEX_FILE_NAME = "index.json"


class PipelineEvaluationInstance:
    output_dir: Path

    config: PipelineConfig

    @property
    def results_dir(self) -> Path:
        return self.output_dir / "results"

    @property
    def pipeline_dir(self) -> Path:
        return self.output_dir / "pipeline"

    def __init__(self, config: PipelineConfig, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.config = config

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_dir.mkdir(parents=True, exist_ok=True)

    def run(self, file: Path, stages: Optional[List[Pipeline.Stage]] = None) -> None:
        pipeline = PipelineEvaluationInstance.load_pipeline(self.config, self.pipeline_dir, file, stages)

        results = pipeline.run()

        PipelineEvaluationInstance.save_results(results, self.results_dir, file)

        # Clear memory for next run
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def save_results(results: Pipeline.State, output_dir: Path, file: Path) -> None:
        output_dir = output_dir / file.name
        output_dir.mkdir(parents=True, exist_ok=True)

        if results.selected_views is not None:
            # Save selected views and index once pipeline is fully done

            images: List[Path] = []
            transforms: List[Path] = []

            for view in results.selected_views:
                if view.path is None:
                    raise ValueError(f"View {view.index + 1} is missing path (hasn't been saved to a file?)")

                frame_name = get_frame_name(view)

                image_path = output_dir / frame_name

                shutil.copyfile(view.path, image_path)

                images.append(image_path)

                transforms_json_path = output_dir / (frame_name + ".transforms.json")
                transforms_json = create_transforms_json(
                    [view],
                    focal_length_x=results.pipeline.renderer.config.focal_length_x,
                    focal_length_y=results.pipeline.renderer.config.focal_length_y,
                    width=results.pipeline.renderer.config.width,
                    height=results.pipeline.renderer.config.height,
                    frame_dir=Path("."),
                    frame_name=frame_name,
                )

                with transforms_json_path.open("w") as f:
                    f.write(transforms_json)

                transforms.append(transforms_json_path)

            save_index(output_dir / INDEX_FILE_NAME, images, transforms)

    def get_index_file(self, file: Path) -> Path:
        return self.results_dir / file.name / INDEX_FILE_NAME

    def create_pipeline_config(self, file: Path, stages: Optional[List[Pipeline.Stage]] = None) -> PipelineConfig:
        return PipelineEvaluationInstance.configure_pipeline(self.config, self.pipeline_dir, file, stages)

    @staticmethod
    def configure_pipeline(
        config: PipelineConfig, output_dir: Path, file: Path, stages: Optional[List[Pipeline.Stage]]
    ) -> PipelineConfig:
        config = replace(config)
        config = PipelineEvaluationInstance.__configure_pipeline_run_settings(config, output_dir, file, stages)

        config.set_timestamp()

        _set_random_seed(config.machine.seed)

        return config

    @staticmethod
    def load_pipeline(
        config: PipelineConfig, output_dir: Path, file: Path, stages: Optional[List[Pipeline.Stage]]
    ) -> Pipeline:
        pipeline: Pipeline = PipelineEvaluationInstance.configure_pipeline(config, output_dir, file, stages).setup(
            local_rank=0, world_size=1
        )

        pipeline.init()

        config.print_to_terminal()
        config.save_config()

        return pipeline

    @staticmethod
    def __configure_pipeline_run_settings(
        config: PipelineConfig, output_dir: Path, file: Path, stages: Optional[List[Pipeline.Stage]]
    ) -> PipelineConfig:
        config.output_dir = output_dir / file.name
        config.experiment_name = "evaluation"
        config.model_file = file
        config.stages = stages
        return config
