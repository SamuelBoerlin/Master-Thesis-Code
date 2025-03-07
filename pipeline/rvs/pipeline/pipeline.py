import dataclasses
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import numpy as np
from nerfstudio.configs.experiment_config import ExperimentConfig
from nerfstudio.utils.rich_utils import CONSOLE
from numpy.typing import NDArray
from PIL import Image as im

from rvs.pipeline.clustering import Clustering, ClusteringConfig
from rvs.pipeline.embedding import ClipEmbeddingConfig, EmbeddingConfig
from rvs.pipeline.field import Field, FieldConfig
from rvs.pipeline.io import PipelineIO
from rvs.pipeline.renderer import Normalization, Renderer, RendererConfig, RenderOutput
from rvs.pipeline.sampler import PositionSampler, PositionSamplerConfig
from rvs.pipeline.selection import ViewSelection, ViewSelectionConfig
from rvs.pipeline.stage import PipelineStage, RequirePipelineStage
from rvs.pipeline.state import PipelineState
from rvs.pipeline.transform import Transform, TransformConfig
from rvs.pipeline.views import View, Views, ViewsConfig
from rvs.utils.console import file_link
from rvs.utils.debug import render_sample_clusters, render_sample_positions
from rvs.utils.nerfstudio import (
    ThreadedImageSaver,
    get_frame_name,
    get_transforms_frame_path,
    load_transforms_json,
    save_transforms_frame,
    save_transforms_json,
)


@dataclass
class PipelineConfig(ExperimentConfig):
    _target: Type = dataclasses.field(default_factory=lambda: Pipeline)

    views: ViewsConfig = dataclasses.field(default_factory=lambda: ViewsConfig)
    """Configuration for training views selection."""

    renderer: RendererConfig = dataclasses.field(default_factory=lambda: RendererConfig)
    """Configuration for training views renderer."""

    embeddings: Tuple[EmbeddingConfig, ...] = dataclasses.field(default_factory=lambda: (ClipEmbeddingConfig(),))
    """Configurations for embeddings to be used."""

    field: FieldConfig = dataclasses.field(default_factory=lambda: FieldConfig)
    """Configuration for radiance field."""

    sampler: PositionSamplerConfig = dataclasses.field(default_factory=lambda: PositionSamplerConfig)
    """Configuration for position sampler."""

    transform: TransformConfig = dataclasses.field(default_factory=lambda: TransformConfig)
    """Configuration for embeddings transform"""

    clustering: ClusteringConfig = dataclasses.field(default_factory=lambda: ClusteringConfig)
    """Configuration for embeddings clustering."""

    selection: ViewSelectionConfig = dataclasses.field(default_factory=lambda: ViewSelectionConfig)
    """Configuration for final view selection algorithm."""

    model_file: Path = None
    """Path to .glb 3D model file."""

    stages: Optional[Set[PipelineStage]] = None
    """Which stages of the pipeline should be run. If empty all stages are run."""

    #################
    # Debug options #
    #################
    render_sample_positions_of_views: Optional[List[int]] = None
    """Views to render the sample positions of"""

    render_sample_clusters_of_views: Optional[List[int]] = None
    """Views to render the sample clusters of"""

    render_sample_clusters_of_selected_views: bool = False
    """Whether to render the sample clusters from the selected views"""

    render_sample_clusters_hard_assignment: bool = True
    """Whether cluster assignments should be rendered as hard assignments"""

    render_sample_as_plot: bool = False
    """Whether to render the samples inside a plot"""

    def setup(self, **kwargs) -> Any:
        self.propagate_experiment_settings()

        if len(self.embeddings) == 0:
            raise ValueError("At least 1 embeddings config required")

        embeddings_types = set()
        for embedding_config in self.embeddings:
            if embedding_config.type in embeddings_types:
                raise ValueError(f"Duplicate embeddings type {embedding_config.type}")
            embeddings_types.add(embedding_config.type)

        return super().setup(**kwargs)

    def propagate_experiment_settings(self):
        self.set_experiment_name()
        self.field.trainer.experiment_name = self.experiment_name
        # self.field.trainer.method_name = self.method_name
        self.field.trainer.project_name = self.project_name
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

    def create_io(self, input_dirs: Optional[List[Path]]) -> PipelineIO:
        return PipelineIO(self.get_base_dir(), self.stages, input_dirs=input_dirs)


class Pipeline:
    config: PipelineConfig
    views: Views
    renderer: Renderer
    field: Field
    sampler: PositionSampler
    transform: Transform
    clustering: Clustering
    selection: ViewSelection

    __io: PipelineIO

    @property
    def io(self) -> Optional[PipelineIO]:
        return self.__io

    __renderer_output_dir: Path
    __sampler_output_dir: Path
    __field_output_dir: Path
    __embedding_output_dir: Path
    __transform_output_dir: Path
    __clustering_output_dir: Path
    __selection_output_dir: Path

    __stage_dependencies: Dict[PipelineStage, Set[PipelineStage]]

    @property
    def output_dir(self) -> Path:
        return self.config.get_base_dir()

    def __init__(self, config: PipelineConfig, **kwargs) -> None:
        self.config = replace(config)
        self.kwargs = kwargs

    def init(self, input_dirs: Optional[List[Path]] = None) -> None:
        self.__io = self.config.create_io(input_dirs=input_dirs)

        self.__renderer_output_dir = Path("renderer")
        self.__io.mk_output_path(self.__renderer_output_dir)

        self.views = self.config.views.setup()

        self.renderer = self.config.renderer.setup()

        self.__field_output_dir = Path("field")
        self.__io.mk_output_path(self.__field_output_dir)

        self.field = self.config.field.setup()

        self.__sampler_output_dir = Path("sampler")
        self.__io.mk_output_path(self.__sampler_output_dir)

        self.sampler = self.config.sampler.setup()

        self.__embedding_output_dir = Path("embedding")
        self.__io.mk_output_path(self.__embedding_output_dir)

        self.__transform_output_dir = Path("transform")
        self.__io.mk_output_path(self.__transform_output_dir)

        self.transform = self.config.transform.setup()

        self.__clustering_output_dir = Path("clustering")
        self.__io.mk_output_path(self.__clustering_output_dir)

        self.clustering = self.config.clustering.setup()

        self.__selection_output_dir = Path("selection")
        self.__io.mk_output_path(self.__selection_output_dir)

        self.selection = self.config.selection.setup()

        self.__stage_dependencies = {
            stage: set(dependencies) for stage, dependencies in PipelineStage.DEPENDENCIES.items()
        }

        for stage in PipelineStage:
            assert stage in self.__stage_dependencies

        stages_components_dict: Dict[PipelineStage, List[Any]] = {
            PipelineStage.SAMPLE_VIEWS: [self.views],
            PipelineStage.RENDER_VIEWS: [self.renderer],
            PipelineStage.TRAIN_FIELD: [self.field],
            PipelineStage.SAMPLE_POSITIONS: [self.sampler],
            PipelineStage.SAMPLE_EMBEDDINGS: [self.field],
            PipelineStage.TRANSFORM_EMBEDDINGS: [self.transform],
            PipelineStage.CLUSTER_EMBEDDINGS: [self.clustering],
            PipelineStage.SELECT_VIEWS: [self.selection],
            PipelineStage.OUTPUT: [],
        }

        for stage in PipelineStage:
            assert stage in stages_components_dict

        for stage, components in stages_components_dict.items():
            for component in components:
                if isinstance(component, RequirePipelineStage):
                    requirements: RequirePipelineStage = component
                    self.__stage_dependencies[stage].update(requirements.required_stages)

        if self.config.render_sample_positions_of_views:
            self.__stage_dependencies[PipelineStage.SAMPLE_POSITIONS].update(
                {PipelineStage.SAMPLE_VIEWS, PipelineStage.RENDER_VIEWS}
            )

        if self.config.render_sample_clusters_of_views:
            self.__stage_dependencies[PipelineStage.CLUSTER_EMBEDDINGS].update(
                {PipelineStage.SAMPLE_VIEWS, PipelineStage.RENDER_VIEWS, PipelineStage.SAMPLE_POSITIONS}
            )

        if self.config.render_sample_clusters_of_selected_views:
            self.__stage_dependencies[PipelineStage.SELECT_VIEWS].update(
                {
                    PipelineStage.RENDER_VIEWS,
                    PipelineStage.SAMPLE_POSITIONS,
                    PipelineStage.SAMPLE_EMBEDDINGS,
                }
            )

    def run(self) -> "PipelineState":
        pipeline_state = PipelineState(self)

        CONSOLE.log(
            f"Loading stages: [{', '.join([stage for stage in PipelineStage if not self.should_run_stage(stage) and self.should_load_stage(stage)])}]"
        )
        CONSOLE.log(
            f"Running stages: [{', '.join([stage for stage in PipelineStage if self.should_run_stage(stage)])}]"
        )

        pipeline_state.scratch_output_dir = None

        if self.should_run_stage(PipelineStage.SAMPLE_VIEWS):
            CONSOLE.log("Generating view transforms...")

            pipeline_state.scratch_output_dir = self.__io.mk_output_path(self.__renderer_output_dir / "scratch")
            pipeline_state.scratch_output_dir.mkdir(parents=True, exist_ok=True)

            views = self.views.generate(pipeline_state)
            pipeline_state.training_views = [View.copy(view) for view in views]
            for view in pipeline_state.training_views:
                view.transform.setflags(write=False)

            transforms_path = self.__save_transforms(pipeline_state)
        elif self.should_load_stage(PipelineStage.SAMPLE_VIEWS):
            CONSOLE.log("Loading view transforms...")
            transforms_path = self.__load_transforms(pipeline_state)

        pipeline_state.scratch_output_dir = None

        if self.should_run_stage(PipelineStage.RENDER_VIEWS):
            CONSOLE.log("Rendering view images...")

            pipeline_state.scratch_output_dir = self.__io.mk_output_path(self.__renderer_output_dir / "scratch")
            pipeline_state.scratch_output_dir.mkdir(parents=True, exist_ok=True)

            with ThreadedImageSaver(
                self.__io.get_output_path(self.__renderer_output_dir), callback=self.__on_saved_view
            ) as saver:

                def get_save_path(view: View) -> Path:
                    return get_transforms_frame_path(saver.output_dir, view)

                def callback(view: View, image: Optional[im.Image]):
                    if image is not None:
                        saver.save(view, image)
                    else:
                        path = get_save_path(view)
                        if not path.exists():
                            raise Exception(f"Render was not directly saved to {str(path)}")

                render_output = RenderOutput(path=get_save_path, callback=callback)

                normalization = self.renderer.render(
                    self.config.model_file, pipeline_state.training_views, render_output, pipeline_state
                )
                pipeline_state.model_normalization = Normalization(
                    normalization.scale.copy(), normalization.offset.copy()
                )
                pipeline_state.model_normalization.scale.setflags(write=False)
                pipeline_state.model_normalization.offset.setflags(write=False)

            self.__save_model_normalization(pipeline_state)

            for view in pipeline_state.training_views:
                self.__set_view_path(view)
        elif self.should_load_stage(PipelineStage.RENDER_VIEWS):
            CONSOLE.log("Loading normalization...")
            self.__load_model_normalization(pipeline_state)

            CONSOLE.log("Loading view images...")
            self.__load_view_paths(pipeline_state)

        pipeline_state.scratch_output_dir = None

        if self.should_run_stage(PipelineStage.TRAIN_FIELD):
            CONSOLE.log("Training radiance field...")

            pipeline_state.scratch_output_dir = self.__io.mk_output_path(self.__field_output_dir / "scratch")
            pipeline_state.scratch_output_dir.mkdir(parents=True, exist_ok=True)

            self.field.init(
                pipeline_state,
                transforms_path,
                self.__io.get_output_path(self.__field_output_dir),
                **self.kwargs,
            )

            self.field.train()
        elif self.should_load_stage(PipelineStage.TRAIN_FIELD):
            CONSOLE.log("Loading radiance field...")

            def has_checkpoint(path: Path) -> bool:
                if path.exists():
                    for _ in path.rglob("nerfstudio_models/step-*.ckpt"):
                        return True
                return False

            self.field.init(
                pipeline_state,
                transforms_path,
                self.__io.get_input_path(self.__field_output_dir, condition=has_checkpoint),
                load_from_checkpoint=True,
                **self.kwargs,
            )

        pipeline_state.scratch_output_dir = None

        if self.should_run_stage(PipelineStage.SAMPLE_POSITIONS):
            CONSOLE.log("Sampling positions...")

            pipeline_state.scratch_output_dir = self.__io.mk_output_path(self.__sampler_output_dir / "scratch")
            pipeline_state.scratch_output_dir.mkdir(parents=True, exist_ok=True)

            pipeline_state.sample_positions = self.sampler.sample(
                self.config.model_file, pipeline_state.model_normalization, pipeline_state
            ).copy()
            pipeline_state.sample_positions.setflags(write=False)

            self.__save_sample_positions(pipeline_state)

            if self.config.render_sample_positions_of_views is not None:
                for i, view_idx in enumerate(self.config.render_sample_positions_of_views):
                    render_sample_positions(
                        self.config.model_file,
                        pipeline_state.training_views[view_idx],
                        pipeline_state.model_normalization,
                        pipeline_state.sample_positions,
                        lambda sample_view, image: save_transforms_frame(
                            self.__io.get_output_path(self.__sampler_output_dir),
                            sample_view,
                            image,
                            frame_dir="visualization/",
                            frame_name=get_frame_name(
                                pipeline_state.training_views[view_idx], frame_name=f"positions_{i}_{{}}.png"
                            ),
                        ),
                        render_as_plot=self.config.render_sample_as_plot,
                    )
        elif self.should_load_stage(PipelineStage.SAMPLE_POSITIONS):
            CONSOLE.log("Loading positions...")
            self.__load_sample_positions(pipeline_state)

        pipeline_state.scratch_output_dir = None

        if self.should_run_stage(PipelineStage.SAMPLE_EMBEDDINGS):
            CONSOLE.log("Sampling embeddings...")

            pipeline_state.scratch_output_dir = self.__io.mk_output_path(self.__embedding_output_dir / "scratch")
            pipeline_state.scratch_output_dir.mkdir(parents=True, exist_ok=True)

            pipeline_state.sample_embeddings = None
            pipeline_state.sample_embeddings_type = None
            pipeline_state.sample_embeddings_dict = dict()

            for i, embedding_config in enumerate(self.config.embeddings):
                embeddings = self.field.sample(embedding_config, pipeline_state.sample_positions).copy()
                embeddings.setflags(write=False)

                if i == 0:
                    pipeline_state.sample_embeddings = embeddings
                    pipeline_state.sample_embeddings_type = embedding_config.type

                assert embedding_config.type not in pipeline_state.sample_embeddings_dict.keys()

                pipeline_state.sample_embeddings_dict[embedding_config.type] = embeddings

            assert pipeline_state.sample_embeddings is not None
            assert pipeline_state.sample_embeddings_type is not None
            assert pipeline_state.sample_embeddings_dict is not None
            assert len(pipeline_state.sample_embeddings_dict) == len(self.config.embeddings)

            self.__save_sample_embeddings(pipeline_state)
        elif self.should_load_stage(PipelineStage.SAMPLE_EMBEDDINGS):
            CONSOLE.log("Loading embeddings...")
            self.__load_sample_embeddings(pipeline_state)

        pipeline_state.scratch_output_dir = None

        if self.should_run_stage(PipelineStage.TRANSFORM_EMBEDDINGS):
            CONSOLE.log("Transforming embeddings...")

            pipeline_state.scratch_output_dir = self.__io.mk_output_path(self.__transform_output_dir / "scratch")
            pipeline_state.scratch_output_dir.mkdir(parents=True, exist_ok=True)

            pipeline_state.transform_parameters = None
            pipeline_state.transform_parameters_dict = dict()

            pipeline_state.transformed_embeddings = None
            pipeline_state.transformed_embeddings_type = None
            pipeline_state.transformed_embeddings_dict = dict()

            for i, embedding_config in enumerate(self.config.embeddings):
                parameters = self.transform.create(
                    pipeline_state.sample_embeddings_dict[embedding_config.type], pipeline_state
                )
                pipeline_state.transform_parameters_dict[embedding_config.type] = {
                    key: value.copy() for key, value in parameters.items()
                }
                for value in pipeline_state.transform_parameters_dict[embedding_config.type].values():
                    value.setflags(write=False)

                transformed_embeddings = self.transform.apply(
                    pipeline_state.sample_embeddings_dict[embedding_config.type],
                    pipeline_state.transform_parameters_dict[embedding_config.type],
                ).copy()
                transformed_embeddings.setflags(write=False)

                if i == 0:
                    pipeline_state.transform_parameters = pipeline_state.transform_parameters_dict[
                        embedding_config.type
                    ]
                    pipeline_state.transformed_embeddings = transformed_embeddings
                    pipeline_state.transformed_embeddings_type = embedding_config.type

                assert embedding_config.type not in pipeline_state.transformed_embeddings_dict.keys()

                pipeline_state.transformed_embeddings_dict[embedding_config.type] = transformed_embeddings

            assert pipeline_state.transform_parameters is not None
            assert pipeline_state.transform_parameters_dict is not None
            assert len(pipeline_state.transform_parameters_dict) == len(self.config.embeddings)

            assert pipeline_state.transformed_embeddings is not None
            assert pipeline_state.transformed_embeddings_type is not None
            assert pipeline_state.transformed_embeddings_dict is not None
            assert len(pipeline_state.transformed_embeddings_dict) == len(self.config.embeddings)

            assert pipeline_state.transformed_embeddings_type == pipeline_state.sample_embeddings_type

            self.__save_embeddings_transform(pipeline_state)
        elif self.should_load_stage(PipelineStage.TRANSFORM_EMBEDDINGS):
            CONSOLE.log("Loading embeddings transform...")
            self.__load_embeddings_transform(pipeline_state)

        pipeline_state.scratch_output_dir = None

        if self.should_run_stage(PipelineStage.CLUSTER_EMBEDDINGS):
            CONSOLE.log("Clustering embeddings...")

            pipeline_state.scratch_output_dir = self.__io.mk_output_path(self.__clustering_output_dir / "scratch")
            pipeline_state.scratch_output_dir.mkdir(parents=True, exist_ok=True)

            parameters, indices = self.clustering.cluster(pipeline_state.transformed_embeddings, pipeline_state)

            pipeline_state.cluster_parameters = {key: value.copy() for key, value in parameters.items()}
            for value in pipeline_state.cluster_parameters.values():
                value.setflags(write=False)
            pipeline_state.cluster_indices = indices.astype(np.intp, copy=True)
            pipeline_state.cluster_indices.setflags(write=False)

            self.__save_clusters(pipeline_state)

            if self.config.render_sample_clusters_of_views is not None:
                for i, view_idx in enumerate(self.config.render_sample_clusters_of_views):
                    render_sample_clusters(
                        self.config.model_file,
                        pipeline_state.training_views[view_idx],
                        pipeline_state.model_normalization,
                        pipeline_state.sample_positions,
                        pipeline_state.transformed_embeddings,
                        self.clustering.get_number_of_clusters(pipeline_state.cluster_parameters),
                        lambda xs: self.clustering.hard_classifier(xs, pipeline_state.cluster_parameters),
                        lambda xs: self.clustering.soft_classifier(xs, pipeline_state.cluster_parameters),
                        lambda sample_view, image: save_transforms_frame(
                            self.__io.get_output_path(self.__clustering_output_dir),
                            sample_view,
                            image,
                            frame_dir="visualization/",
                            frame_name=get_frame_name(
                                pipeline_state.training_views[view_idx], frame_name=f"clusters_{i}_{{}}.png"
                            ),
                        ),
                        hard_assignments=self.config.render_sample_clusters_hard_assignment,
                        render_as_plot=self.config.render_sample_as_plot,
                    )
        elif self.should_load_stage(PipelineStage.CLUSTER_EMBEDDINGS):
            CONSOLE.log("Loading clusters...")
            self.__load_clusters(pipeline_state)

        pipeline_state.scratch_output_dir = None

        if self.should_run_stage(PipelineStage.SELECT_VIEWS):
            CONSOLE.log("Selecting views...")

            pipeline_state.scratch_output_dir = self.__io.mk_output_path(self.__selection_output_dir / "scratch")
            pipeline_state.scratch_output_dir.mkdir(parents=True, exist_ok=True)

            views = self.selection.select(
                self.clustering.get_number_of_clusters(pipeline_state.cluster_parameters),
                lambda xs: self.clustering.hard_classifier(xs, pipeline_state.cluster_parameters),
                lambda xs: self.clustering.soft_classifier(xs, pipeline_state.cluster_parameters),
                pipeline_state,
            )
            pipeline_state.selected_views = [View.copy(view) for view in views]
            for view in pipeline_state.selected_views:
                view.transform.setflags(write=False)

            self.__save_selected_views(pipeline_state)

            if self.config.render_sample_clusters_of_selected_views:
                for i, view in enumerate(pipeline_state.selected_views):
                    render_sample_clusters(
                        self.config.model_file,
                        view,
                        pipeline_state.model_normalization,
                        pipeline_state.sample_positions,
                        pipeline_state.transformed_embeddings,
                        self.clustering.get_number_of_clusters(pipeline_state.cluster_parameters),
                        lambda xs: self.clustering.hard_classifier(xs, pipeline_state.cluster_parameters),
                        lambda xs: self.clustering.soft_classifier(xs, pipeline_state.cluster_parameters),
                        lambda sample_view, image: save_transforms_frame(
                            self.__io.get_output_path(self.__selection_output_dir),
                            sample_view,
                            image,
                            frame_dir="visualization/",
                            frame_name=get_frame_name(view, frame_name=f"clusters_{i}_{{}}.png"),
                        ),
                        hard_assignments=self.config.render_sample_clusters_hard_assignment,
                        render_as_plot=self.config.render_sample_as_plot,
                    )

            CONSOLE.log("Selected views:")
            for i, view in enumerate(pipeline_state.selected_views):
                resolved_path = view.resolve_path(self.__io)
                CONSOLE.log(f"{view.index + 1} ({file_link(resolved_path) if resolved_path is not None else 'N/A'})")
        elif self.should_load_stage(PipelineStage.SELECT_VIEWS):
            CONSOLE.log("Loading selected views...")
            self.__load_selected_views(pipeline_state)

        pipeline_state.scratch_output_dir = None

        CONSOLE.log("Done...")

        return pipeline_state

    def should_run_stage(self, stage: PipelineStage) -> bool:
        return self.config.stages is None or stage in self.config.stages

    def should_load_stage(self, stage: PipelineStage) -> bool:
        if self.config.stages is None or stage.required_by(self.config.stages, transitive=False):
            return True

        for descendant_stage in self.config.stages:
            if descendant_stage.depends_on(stage, transitive=True):
                if stage in self.__stage_dependencies[descendant_stage]:
                    return True

        return False

    def __save_transforms(self, state: "PipelineState") -> Path:
        path = save_transforms_json(
            self.__io.get_output_path(self.__renderer_output_dir),
            state.training_views,
            self.config.renderer.focal_length_x,
            self.config.renderer.focal_length_y,
            self.config.renderer.width,
            self.config.renderer.height,
        )
        CONSOLE.log(f"Saved views transforms to {file_link(path)}")
        return path

    def __load_transforms(self, state: "PipelineState") -> Path:
        try:
            path, views, fl_x, fl_y, w, h = self.__io.load_input(
                self.__renderer_output_dir, lambda path: load_transforms_json(path)
            )
            # TODO Check if fl_x etc. matches with renderer config
            state.training_views = views
            for view in state.training_views:
                view.transform.setflags(write=False)
            CONSOLE.log(f"Loaded views transforms from {file_link(path)}")
            return path
        except Exception as ex:
            CONSOLE.log("Failed loading views transforms:")
            raise ex

    def __on_saved_view(self, view: View, image: im.Image, path: Path) -> None:
        CONSOLE.log(f"Saved view {view.index} to {file_link(path)}")
        return path

    def __load_view_paths(self, state: "PipelineState") -> None:
        if state.training_views is not None:
            try:
                for view in state.training_views:
                    self.__set_view_path(view)
            except Exception as ex:
                CONSOLE.log("Failed loading view images:")
                raise ex

    def __set_view_path(self, view: View) -> None:
        """Sets the view's path if path is not yet set and if a corresponding image file already exists"""
        if view.path is None:
            view.path = self.__renderer_output_dir / "images" / get_frame_name(view)
            resolved_path = view.resolve_path(self.__io)
            if resolved_path is None:
                raise FileNotFoundError(f"View image {view.path} not found")

    def __save_model_normalization(self, state: "PipelineState") -> Path:
        path = self.__io.get_output_path(self.__renderer_output_dir / "normalization.json")
        with path.open("w") as f:
            json.dump(
                {
                    "scale": state.model_normalization.scale.tolist(),
                    "offset": state.model_normalization.offset.tolist(),
                },
                f,
            )
        CONSOLE.log(f"Saved normalization to {file_link(path)}")
        return path

    def __load_model_normalization(self, state: "PipelineState") -> Path:
        path = self.__io.get_input_path(self.__renderer_output_dir / "normalization.json")
        try:
            with path.open("r") as f:
                obj = json.load(f)
                state.model_normalization = Normalization(
                    scale=np.array(obj["scale"]),
                    offset=np.array(obj["offset"]),
                )
                state.model_normalization.scale.setflags(write=False)
                state.model_normalization.offset.setflags(write=False)
        except Exception as ex:
            CONSOLE.log("Failed loading normalization:")
            raise ex

    def __save_sample_positions(self, state: "PipelineState") -> Path:
        path = self.__io.get_output_path(self.__sampler_output_dir / "positions.json")
        with path.open("w") as f:
            json.dump(state.sample_positions.tolist(), f)
        CONSOLE.log(f"Saved positions to {file_link(path)}")
        return path

    def __load_sample_positions(self, state: "PipelineState") -> Path:
        path = self.__io.get_input_path(self.__sampler_output_dir / "positions.json")
        try:
            with path.open("r") as f:
                state.sample_positions = np.array(json.load(f))
                state.sample_positions.setflags(write=False)
            CONSOLE.log(f"Loaded positions from {file_link(path)}")
            return path
        except Exception as ex:
            CONSOLE.log("Failed loading positions:")
            raise ex

    def __save_sample_embeddings(self, state: "PipelineState") -> Path:
        path = self.__io.get_output_path(self.__embedding_output_dir / "embeddings.json")
        with path.open("w") as f:
            embeddings_list_dict = {
                type: embeddings.tolist() for type, embeddings in state.sample_embeddings_dict.items()
            }
            obj = {
                "type": state.sample_embeddings_type,
                "dict": embeddings_list_dict,
            }
            json.dump(obj, f)
        CONSOLE.log(f"Saved embeddings to {file_link(path)}")
        return path

    def __load_sample_embeddings(self, state: "PipelineState") -> Path:
        path = self.__io.get_input_path(self.__embedding_output_dir / "embeddings.json")
        try:
            with path.open("r") as f:
                obj = json.load(f)
                embeddings_list_dict: Dict[str, NDArray] = obj["dict"]
                state.sample_embeddings_dict = {
                    type: np.array(embeddings) for type, embeddings in embeddings_list_dict.items()
                }
                for embeddings in state.sample_embeddings_dict.values():
                    embeddings.setflags(write=False)
                state.sample_embeddings_type = obj["type"]
                state.sample_embeddings = state.sample_embeddings_dict[state.sample_embeddings_type]
                state.sample_embeddings.setflags(write=False)
            CONSOLE.log(f"Loaded embeddings from {file_link(path)}")
            return path
        except Exception as ex:
            CONSOLE.log("Failed loading embeddings:")
            raise ex

    def __save_embeddings_transform(self, state: "PipelineState") -> Path:
        self.__save_transformed_embeddings(state)

        path = self.__io.get_output_path(self.__transform_output_dir / "transform.json")
        with path.open("w") as f:
            json.dump(
                {
                    embedding_type: {
                        "parameters": {key: value.tolist() for key, value in parameters.items()},
                    }
                    for embedding_type, parameters in state.transform_parameters_dict.items()
                },
                f,
            )
        CONSOLE.log(f"Saved transform to {file_link(path)}")
        return path

    def __load_embeddings_transform(self, state: "PipelineState") -> Path:
        self.__load_transformed_embeddings(state)

        path = self.__io.get_input_path(self.__transform_output_dir / "transform.json")
        try:
            with path.open("r") as f:
                obj: Dict[str, Dict[str, Dict[str, Any]]] = json.load(f)
                state.transform_parameters_dict = {
                    embedding_type: {key: np.array(value) for key, value in embedding_obj["parameters"].items()}
                    for embedding_type, embedding_obj in obj.items()
                }
                for parameters_dict in state.transform_parameters_dict.values():
                    for value in parameters_dict.values():
                        value.setflags(write=False)
                state.transform_parameters = state.transform_parameters_dict[state.transformed_embeddings_type]
            CONSOLE.log(f"Loaded transform from {file_link(path)}")
            return path
        except Exception as ex:
            CONSOLE.log("Failed loading transform:")
            raise ex

    def __save_transformed_embeddings(self, state: "PipelineState") -> Path:
        path = self.__io.get_output_path(self.__transform_output_dir / "embeddings.json")
        with path.open("w") as f:
            embeddings_list_dict = {
                type: embeddings.tolist() for type, embeddings in state.transformed_embeddings_dict.items()
            }
            obj = {
                "type": state.transformed_embeddings_type,
                "dict": embeddings_list_dict,
            }
            json.dump(obj, f)
        CONSOLE.log(f"Saved transformed embeddings to {file_link(path)}")
        return path

    def __load_transformed_embeddings(self, state: "PipelineState") -> Path:
        path = self.__io.get_input_path(self.__transform_output_dir / "embeddings.json")
        try:
            with path.open("r") as f:
                obj = json.load(f)
                embeddings_list_dict: Dict[str, NDArray] = obj["dict"]
                state.transformed_embeddings_dict = {
                    type: np.array(embeddings) for type, embeddings in embeddings_list_dict.items()
                }
                for embeddings in state.transformed_embeddings_dict.values():
                    embeddings.setflags(write=False)
                state.transformed_embeddings_type = obj["type"]
                state.transformed_embeddings = state.transformed_embeddings_dict[state.transformed_embeddings_type]
                state.transformed_embeddings.setflags(write=False)
            CONSOLE.log(f"Loaded transformed embeddings from {file_link(path)}")
            return path
        except Exception as ex:
            CONSOLE.log("Failed loading transformed embeddings:")
            raise ex

    def __save_clusters(self, state: "PipelineState") -> Path:
        path = self.__io.get_output_path(self.__clustering_output_dir / "clusters.json")
        with path.open("w") as f:
            json.dump(
                {
                    "parameters": {key: value.tolist() for key, value in state.cluster_parameters.items()},
                    "indices": state.cluster_indices.tolist(),
                },
                f,
            )
        CONSOLE.log(f"Saved clusters to {file_link(path)}")
        return path

    def __load_clusters(self, state: "PipelineState") -> Path:
        path = self.__io.get_input_path(self.__clustering_output_dir / "clusters.json")
        try:
            with path.open("r") as f:
                json_obj = json.load(f)
                parameters_dict: Dict[str, List] = json_obj["parameters"]
                state.cluster_parameters = {key: np.array(value) for key, value in parameters_dict.items()}
                for value in state.cluster_parameters.values():
                    value.setflags(write=False)
                state.cluster_indices = np.array(json_obj["indices"], dtype=np.intp)
                state.cluster_indices.setflags(write=False)
            CONSOLE.log(f"Loaded clusters from {file_link(path)}")
            return path
        except Exception as ex:
            CONSOLE.log("Failed loading clusters:")
            raise ex

    def __save_selected_views(self, state: "PipelineState") -> Path:
        views_json_arr: List[str] = []

        for view in state.selected_views:
            if view.path is None:
                raise ValueError(f"View {view.index + 1} is missing path (hasn't been saved to a file?)")

            if view.resolve_path(self.__io) is None:
                raise ValueError(f"View {view.index + 1} path {view.path} does not exist")

            views_json_arr.append(
                {
                    "index": view.index,
                    "transform": view.transform.tolist(),
                    "path": str(view.path),
                }
            )

        path = self.__io.get_output_path(self.__selection_output_dir / "views.json")
        with path.open("w") as f:
            json.dump(views_json_arr, f)

        CONSOLE.log(f"Saved selected views to {file_link(path)}")

        return path

    def __load_selected_views(self, state: "PipelineState") -> Path:
        path = self.__io.get_input_path(self.__selection_output_dir / "views.json")
        try:
            state.selected_views = []
            with path.open("r") as f:
                json_obj = json.load(f)
                for view_obj in json_obj:
                    view = View(
                        view_obj["index"],
                        np.array(view_obj["transform"]),
                        Path(view_obj["path"]),
                    )
                    view.transform.setflags(write=False)
                    resolved_path = view.resolve_path(self.__io)
                    if resolved_path is None:
                        raise ValueError(f"View {view.index + 1} path {view.path} does not exist")
                    state.selected_views.append(view)
            CONSOLE.log(f"Loaded selected views from {file_link(path)}")
            return path
        except Exception as ex:
            CONSOLE.log("Failed loading selected views:")
            raise ex
