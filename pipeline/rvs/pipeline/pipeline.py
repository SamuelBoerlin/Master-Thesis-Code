import dataclasses
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Set, Type

from nerfstudio.configs.experiment_config import ExperimentConfig
from nerfstudio.utils.rich_utils import CONSOLE
from numpy.typing import NDArray
from PIL import Image as im

from rvs.pipeline.clustering import Clustering, ClusteringConfig
from rvs.pipeline.field import Field, FieldConfig
from rvs.pipeline.renderer import Renderer, RendererConfig
from rvs.pipeline.sampler import PositionSampler, PositionSamplerConfig
from rvs.pipeline.selection import ViewSelection, ViewSelectionConfig
from rvs.pipeline.views import View, Views, ViewsConfig
from rvs.utils.console import file_link
from rvs.utils.debug import render_sample_clusters, render_sample_positions
from rvs.utils.nerfstudio import get_frame_name, save_transforms_frame, save_transforms_json


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
    """Path to .glb 3D model file."""

    stages: Optional[Set["Pipeline.Stage"]] = None
    """Which stages of the pipeline should be run. If empty all stages are run."""

    #################
    # Debug options #
    #################
    render_sample_positions_of_views: Optional[List[int]] = None
    """Views to render the sample positions of"""

    render_sample_clusters_of_views: Optional[List[int]] = None
    """Views to render the sample clusters of"""

    render_sample_clusters_hard_assignment: bool = True
    """Whether cluster assignments should be rendered as hard assignments"""

    render_selected_views: bool = False
    """Whether to render selected views"""

    def setup(self, **kwargs) -> Any:
        self.propagate_experiment_settings()
        return super().setup(**kwargs)

    def propagate_experiment_settings(self):
        self.set_experiment_name()
        self.field.trainer.experiment_name = self.experiment_name
        self.field.trainer.timestamp = self.timestamp
        self.field.trainer.machine.seed = self.machine.seed
        # TODO: There are some others like method_name, project_name and possibly more that should be propagated

    def set_experiment_name(self) -> None:
        if self.experiment_name is None:
            datapath = self.model_file if self.model_file is not None else self.pipeline.datamanager.data
            if datapath is not None:
                datapath = datapath.parent if datapath.is_file() else datapath
                self.experiment_name = str(datapath.stem)
            else:
                self.experiment_name = "unnamed"

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
        output_dir = self.config.get_base_dir()

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

    def run(self) -> "Pipeline.State":
        pipeline_state = Pipeline.State(self)

        if self.should_run_stage(Pipeline.Stage.SAMPLE_VIEWS):
            CONSOLE.log("Generating views...")
            pipeline_state.training_views = self.views.generate()
            transforms_path = self.__save_transforms(pipeline_state.training_views)

        if self.should_run_stage(Pipeline.Stage.RENDER_VIEWS):
            CONSOLE.log("Rendering views...")
            self.renderer.render(self.config.model_file, pipeline_state.training_views, self.__save_view)

        for view in pipeline_state.training_views:
            self.__set_view_path(view)

        if self.should_run_stage(Pipeline.Stage.SAMPLE_POSITIONS):
            CONSOLE.log("Sampling positions...")
            pipeline_state.sample_positions = self.sampler.sample(self.config.model_file)

            if self.config.render_sample_positions_of_views is not None:
                for i in self.config.render_sample_positions_of_views:
                    render_sample_positions(
                        self.config.model_file,
                        pipeline_state.training_views[i],
                        pipeline_state.sample_positions,
                        lambda sample_view, image: save_transforms_frame(
                            self.__renderer_output_dir,
                            sample_view,
                            image,
                            frame_name=get_frame_name(
                                pipeline_state.training_views[i], frame_name="sample_positions_{}.png"
                            ),
                        ),
                    )

        if self.should_run_stage(Pipeline.Stage.TRAIN_FIELD):
            CONSOLE.log("Training radiance field...")
            self.field.init(pipeline_state, transforms_path, self.__field_output_dir, **self.kwargs)
            self.field.train()

        if self.should_run_stage(Pipeline.Stage.SAMPLE_EMBEDDINGS):
            CONSOLE.log("Sampling embeddings...")
            pipeline_state.sample_embeddings = self.field.sample(pipeline_state.sample_positions)

        if self.should_run_stage(Pipeline.Stage.CLUSTER_EMBEDDINGS):
            CONSOLE.log("Clustering embeddings...")
            pipeline_state.sample_clusters = self.clustering.cluster(pipeline_state.sample_embeddings)

            if self.config.render_sample_clusters_of_views is not None:
                for i in self.config.render_sample_clusters_of_views:
                    render_sample_clusters(
                        self.config.model_file,
                        pipeline_state.training_views[i],
                        pipeline_state.sample_positions,
                        pipeline_state.sample_embeddings,
                        pipeline_state.sample_clusters,
                        lambda sample_view, image: save_transforms_frame(
                            self.__renderer_output_dir,
                            sample_view,
                            image,
                            frame_name=get_frame_name(
                                pipeline_state.training_views[i], frame_name="sample_clusters_{}.png"
                            ),
                        ),
                        hard_assignments=self.config.render_sample_clusters_hard_assignment,
                    )

        if self.should_run_stage(Pipeline.Stage.SELECT_VIEWS):
            CONSOLE.log("Selecting views...")
            pipeline_state.selected_views = self.selection.select(pipeline_state.sample_clusters, pipeline_state)

            if self.config.render_selected_views:
                for i, view in enumerate(pipeline_state.selected_views):
                    render_sample_clusters(
                        self.config.model_file,
                        view,
                        pipeline_state.sample_positions,
                        pipeline_state.sample_embeddings,
                        pipeline_state.sample_clusters,
                        lambda sample_view, image: save_transforms_frame(
                            self.__renderer_output_dir,
                            sample_view,
                            image,
                            frame_name=get_frame_name(None, frame_index=i, frame_name="selected_{}.png"),
                        ),
                        hard_assignments=self.config.render_sample_clusters_hard_assignment,
                    )

            CONSOLE.log("Selected views:")
            for i, view in enumerate(pipeline_state.selected_views):
                CONSOLE.log(f"{view.index + 1} ({file_link(view.path) if view.path is not None else 'N/A'})")

        CONSOLE.log("Done...")

        return pipeline_state

    def should_run_stage(self, stage: "Pipeline.Stage") -> bool:
        return self.config.stages is None or stage in self.config.stages

    def __save_transforms(self, views: List[View]) -> Path:
        path = save_transforms_json(
            self.__renderer_output_dir,
            views,
            self.config.renderer.focal_length_x,
            self.config.renderer.focal_length_y,
            self.config.renderer.width,
            self.config.renderer.height,
        )
        CONSOLE.log(f"Saved views transforms to {file_link(path)}")
        return path

    def __save_view(self, view: View, image: im.Image) -> Path:
        path = save_transforms_frame(self.__renderer_output_dir, view, image, set_path=True)
        CONSOLE.log(f"Saved view {view.index} to {file_link(path)}")
        return path

    def __set_view_path(self, view: View) -> None:
        """Sets the view's path if path is not yet set and if a corresponding image file already exists"""
        if view.path is None:
            path = self.__renderer_output_dir / "images" / get_frame_name(view)
            if path.exists():
                view.path = path

    class State:
        pipeline: "Pipeline"
        training_views: Optional[List[View]] = None
        sample_positions: Optional[NDArray] = None
        sample_embeddings: Optional[NDArray] = None
        sample_clusters: Optional[NDArray] = None
        selected_views: Optional[List[View]] = None

        def __init__(self, pipeline: "Pipeline") -> None:
            self.pipeline = pipeline

    class Stage(Enum):
        SAMPLE_VIEWS = 1
        RENDER_VIEWS = 2
        SAMPLE_POSITIONS = 3
        TRAIN_FIELD = 4
        SAMPLE_EMBEDDINGS = 5
        CLUSTER_EMBEDDINGS = 6
        SELECT_VIEWS = 7

        def up_to(self) -> List["Pipeline.Stage"]:
            return [stage for stage in Pipeline.Stage if stage.value <= self.value]
