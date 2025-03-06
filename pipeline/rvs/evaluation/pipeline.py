import gc
import hashlib
import shutil
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Set

import torch

from rvs.evaluation.index import save_index
from rvs.pipeline.io import PipelineIO
from rvs.pipeline.pipeline import Pipeline, PipelineConfig
from rvs.pipeline.stage import PipelineStage
from rvs.scripts.rvs import _set_random_seed
from rvs.utils.hash import hash_file_sha1
from rvs.utils.nerfstudio import create_transforms_json, get_frame_name

INDEX_FILE_NAME = "index.json"


class PipelineEvaluationInstance:
    from rvs.pipeline.state import PipelineState

    output_dir: Path

    config: PipelineConfig

    @property
    def results_dir(self) -> Path:
        return self.output_dir / "results"

    @property
    def pipeline_dir(self) -> Path:
        return self.output_dir / "pipeline"

    input_pipelines: Optional[List["PipelineEvaluationInstance"]]

    stages: Optional[List[PipelineStage]]

    def __init__(
        self,
        config: PipelineConfig,
        output_dir: Path,
        stages: Optional[List[PipelineStage]] = None,
        input_pipelines: Optional[List["PipelineEvaluationInstance"]] = None,
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.stages = stages
        self.input_pipelines = input_pipelines

    def init(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_dir.mkdir(parents=True, exist_ok=True)

    def run(self, file: Path, stages_filter: Optional[Set[PipelineStage]] = None) -> None:
        io = self.create_pipeline_io(file)

        run_stages = self.get_pipeline_stages(stages_filter)

        pipeline = PipelineEvaluationInstance.create_pipeline(
            self.config,
            self.pipeline_dir,
            io.input_dirs,
            file,
            run_stages,
        )

        assert pipeline.config.stages == run_stages

        assert pipeline.io and pipeline.io.input_dirs == io.input_dirs

        assert pipeline.config.output_dir == self.get_pipeline_dir(file)

        results = pipeline.run()

        index_file = PipelineEvaluationInstance.save_results(results, self.results_dir, file)
        if index_file is not None:
            assert index_file == self.get_index_file(file)
            assert index_file.parent == self.get_results_dir(file)

        # Clear memory for next run
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def save_results(results: PipelineState, output_dir: Path, file: Path) -> Optional[Path]:
        output_dir = output_dir / file.name
        output_dir.mkdir(parents=True, exist_ok=True)

        if results.selected_views is not None:
            # Save selected views and index once pipeline is fully done

            images: List[Path] = []
            transforms: List[Path] = []

            for view in results.selected_views:
                view_path = view.resolve_path(results.pipeline.io)
                if view_path is None:
                    raise ValueError(f"View {view.index + 1} is missing path (hasn't been saved to a file?)")

                frame_name = get_frame_name(view)

                image_path = output_dir / frame_name

                shutil.copyfile(view_path, image_path)

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

            index_file = output_dir / INDEX_FILE_NAME

            save_index(index_file, images, transforms)

            return index_file

        return None

    def get_pipeline_stages(self, stages_filter: Optional[Set[PipelineStage]]) -> Optional[List[PipelineStage]]:
        if self.stages is None:
            return None

        if stages_filter is None:
            return list(self.stages)

        for stage in stages_filter:
            if stage not in self.stages:
                raise ValueError(
                    f"Invalid filter stage {stage}, expected {', '.join([stage.name for stage in self.stages])}"
                )

        filtered_stages = [stage for stage in self.stages if stage in stages_filter]

        if len(filtered_stages) == 0:
            raise ValueError("No stages specified")

        return filtered_stages

    def get_pipeline_dir(self, file: Path) -> Path:
        return self.pipeline_dir / file.name

    def get_results_dir(self, file: Path) -> Path:
        return self.results_dir / file.name

    def get_index_file(self, file: Path) -> Path:
        return self.get_results_dir(file) / INDEX_FILE_NAME

    def create_pipeline_config(self, file: Path, stages_filter: Optional[Set[PipelineStage]] = None) -> PipelineConfig:
        return PipelineEvaluationInstance.configure_pipeline(
            self.config,
            self.pipeline_dir,
            file,
            self.get_pipeline_stages(stages_filter),
        )

    def create_pipeline_io(self, file: Path, stages_filter: Optional[Set[PipelineStage]] = None) -> PipelineIO:
        input_dirs = None

        if self.input_pipelines is not None and len(self.input_pipelines) > 0:
            input_dirs = []

            for input_pipeline in self.input_pipelines:
                input_pipeline_config = input_pipeline.create_pipeline_config(file)
                input_dirs.append(input_pipeline_config.get_base_dir())

        return self.create_pipeline_config(file, stages_filter).create_io(input_dirs)

    @staticmethod
    def configure_pipeline(
        config: PipelineConfig,
        output_dir: Path,
        file: Path,
        stages: Optional[List[PipelineStage]],
        derived_seed: bool = True,
    ) -> PipelineConfig:
        config = replace(config)
        config = PipelineEvaluationInstance.__configure_pipeline_run_settings(
            config, output_dir, file, stages, derived_seed
        )

        config.set_timestamp()

        _set_random_seed(config.machine.seed)

        return config

    @staticmethod
    def create_pipeline(
        config: PipelineConfig,
        output_dir: Path,
        input_dirs: Optional[List[Path]],
        file: Path,
        stages: Optional[List[PipelineStage]],
        derived_seed: bool = True,
    ) -> Pipeline:
        pipeline: Pipeline = PipelineEvaluationInstance.configure_pipeline(
            config, output_dir, file, stages, derived_seed=derived_seed
        ).setup(local_rank=0, world_size=1)

        pipeline.init(input_dirs=input_dirs)

        pipeline.config.print_to_terminal()
        pipeline.config.save_config()

        return pipeline

    @staticmethod
    def __configure_pipeline_run_settings(
        config: PipelineConfig,
        output_dir: Path,
        file: Path,
        stages: Optional[List[PipelineStage]],
        derived_seed: bool,
    ) -> PipelineConfig:
        config.output_dir = output_dir / file.name
        config.model_file = file
        config.stages = None if stages is None else list(stages)

        if derived_seed and file.exists() and file.is_file():
            digest = hashlib.sha1(str(config.machine.seed).encode())
            digest.update(hash_file_sha1(file).encode())

            new_seed = int(digest.hexdigest(), 16)
            if new_seed < 0:
                new_seed = -new_seed
            new_seed %= 2**32 - 1

            machine = replace(config.machine, seed=new_seed)
            config = replace(config, machine=machine)

        return config
