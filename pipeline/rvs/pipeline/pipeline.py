import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Type

from nerfstudio.configs.experiment_config import ExperimentConfig
from nerfstudio.utils.rich_utils import CONSOLE
from PIL import Image as im

from rvs.pipeline.clustering import Clustering, ClusteringConfig
from rvs.pipeline.field import Field, FieldConfig
from rvs.pipeline.renderer import Renderer, RendererConfig
from rvs.pipeline.sampler import PositionSampler, PositionSamplerConfig
from rvs.pipeline.selection import ViewSelection, ViewSelectionConfig
from rvs.pipeline.views import View, Views, ViewsConfig
from rvs.utils.debug import render_sample_clusters, render_sample_positions
from rvs.utils.nerfstudio import save_transforms_frame, save_transforms_json


@dataclass
class PipelineConfig(ExperimentConfig):
    _target: Type = dataclasses.field(default_factory=lambda: Pipeline)

    views: ViewsConfig = dataclasses.field(default_factory=lambda: ViewsConfig)
    """Configuration for training views selection."""

    renderer: RendererConfig = dataclasses.field(default_factory=lambda: RendererConfig)
    """Configuration for training views renderer."""

    field: FieldConfig = dataclasses.field(default_factory=lambda: FieldConfig)
    """Configuration for radiance field."""

    sampler: PositionSamplerConfig = dataclasses.field(default_factory=lambda: PositionSamplerConfig)
    """Configuration for position sampler."""

    clustering: ClusteringConfig = dataclasses.field(default_factory=lambda: ClusteringConfig)
    """Configuration for embeddings clustering."""

    selection: ViewSelectionConfig = dataclasses.field(default_factory=lambda: ViewSelectionConfig)
    """Configuration for final view selection algorithm."""

    model_file: Path = None
    """Path to .glb 3D model file"""

    #################
    # Debug options #
    #################
    skip_rendering: bool = False
    """Whether to skip rendering views"""

    def setup(self, **kwargs) -> Any:
        self.propagate_experiment_settings()
        return super().setup(**kwargs)

    def propagate_experiment_settings(self):
        self.set_experiment_name()
        self.field.trainer.experiment_name = self.experiment_name
        self.field.trainer.timestamp = self.timestamp
        self.field.trainer.machine.seed = self.machine.seed
        # TODO: There are some others like method_name, project_name and possibly more that should be propagated

    def print_to_terminal(self) -> None:
        CONSOLE.rule("Config")
        CONSOLE.print(self)
        CONSOLE.rule("")


class Pipeline:
    config: PipelineConfig
    views: Views
    renderer: Renderer
    field: Field
    sampler: PositionSampler
    clustering: Clustering
    selection: ViewSelection

    __renderer_output_dir: Path
    __field_output_dir: Path

    def __init__(self, config: PipelineConfig, **kwargs) -> None:
        self.config = config
        self.kwargs = kwargs

    def init(self) -> None:
        output_dir = self.config.get_base_dir()  # FIXME: This has "unnamed" as experiment name...

        self.__renderer_output_dir = output_dir / "renderer"
        self.__renderer_output_dir.mkdir(parents=True, exist_ok=True)

        self.views = self.config.views.setup()

        self.renderer = self.config.renderer.setup()

        self.__field_output_dir = output_dir / "field"
        self.__field_output_dir.mkdir(parents=True, exist_ok=True)

        self.field = self.config.field.setup()

        self.sampler = self.config.sampler.setup()

        self.clustering = self.config.clustering.setup()

        self.selection = self.config.selection.setup()

    def run(self) -> None:
        pipeline_state = Pipeline.State(self)

        CONSOLE.log("Generating views...")
        pipeline_state.training_views = self.views.generate()
        transforms_path = self.__save_transforms(pipeline_state.training_views)

        if not self.config.skip_rendering:
            CONSOLE.log("Rendering views...")
            self.renderer.render(self.config.model_file, pipeline_state.training_views, self.__save_view)

        CONSOLE.log("Sampling positions...")
        sample_positions = self.sampler.sample(self.config.model_file)

        # TODO: Debug
        render_sample_positions(
            self.config.model_file,
            pipeline_state.training_views[12],
            sample_positions,
            lambda sample_view, image: save_transforms_frame(
                self.__renderer_output_dir, sample_view, image, frame_name="sample_positions.png"
            ),
        )

        CONSOLE.log("Training radiance field...")
        self.field.init(pipeline_state, transforms_path, self.__field_output_dir, **self.kwargs)
        self.field.train()

        CONSOLE.log("Sampling embeddings...")
        sample_embeddings = self.field.sample(sample_positions)

        CONSOLE.log("Clustering embeddings...")
        sample_clusters = self.clustering.cluster(sample_embeddings)

        # TODO: Debug
        render_sample_clusters(
            self.config.model_file,
            pipeline_state.training_views[12],
            sample_positions,
            sample_embeddings,
            sample_clusters,
            lambda sample_view, image: save_transforms_frame(
                self.__renderer_output_dir, sample_view, image, frame_name="sample_clusters.png"
            ),
            hard_assignments=True,
        )

        CONSOLE.log("Selecting views...")
        selected_views = self.selection.select(sample_clusters, pipeline_state)

        CONSOLE.log("Selected views:")
        for i, view in enumerate(selected_views):
            CONSOLE.log(view.index + 1)

        # TODO: Debug
        for i, view in enumerate(selected_views):
            render_sample_clusters(
                self.config.model_file,
                view,
                sample_positions,
                sample_embeddings,
                sample_clusters,
                lambda sample_view, image: save_transforms_frame(
                    self.__renderer_output_dir,
                    sample_view,
                    image,
                    frame_name=f"selected_{str(i + 1).zfill(5)}.png",
                ),
                hard_assignments=True,
            )

        CONSOLE.log("Done...")

    def __save_transforms(self, views: List[View]) -> Path:
        path = save_transforms_json(
            self.__renderer_output_dir,
            views,
            self.config.renderer.focal_length_x,
            self.config.renderer.focal_length_y,
            self.config.renderer.width,
            self.config.renderer.height,
        )
        CONSOLE.log(f"Saved views transforms to {str(path)}")
        return path

    def __save_view(self, view: View, image: im.Image) -> Path:
        path = save_transforms_frame(self.__renderer_output_dir, view, image)
        CONSOLE.log(f"Saved view {view.index} to {str(path)}")
        return path

    class State:
        pipeline: "Pipeline"

        training_views: List[View]

        def __init__(self, pipeline: "Pipeline") -> None:
            self.pipeline = pipeline
