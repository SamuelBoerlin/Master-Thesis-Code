import os

# TODO: This should probably be elsewhere
# Required for headless rendering with pyrenderer and trimesh
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYGLET_HEADLESS"] = "1"

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.cm
import matplotlib.colors
import numpy as np
import torch
import tyro
import umap
from lerf.encoders.openclip_encoder import OpenCLIPNetwork
from lerf.lerf import LERFModel
from lerf.lerf_pipeline import LERFPipeline
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.figure import Figure
from matplotlib.patches import Circle, ConnectionPatch, Patch, Rectangle
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.data.datamanagers.base_datamanager import DataManager, VanillaDataManager
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.fields.base_field import Field
from nerfstudio.utils.rich_utils import CONSOLE
from numpy.typing import NDArray
from PIL import Image, Image as im
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import Tensor

from rvs.evaluation.debug import DebugContext, Uid
from rvs.evaluation.evaluation import Evaluation, EvaluationConfig
from rvs.evaluation.evaluation_method import load_result
from rvs.evaluation.lvis import Category
from rvs.pipeline.embedding import DefaultEmbeddingTypes
from rvs.pipeline.stage import PipelineStage
from rvs.pipeline.views import View
from rvs.utils.cache import get_evaluation_prompt_embedding_cache_key, get_pipeline_render_embedding_cache_key
from rvs.utils.config import find_config_working_dir, load_config
from rvs.utils.debug import render_sample, render_sample_positions, render_sample_positions_and_colors
from rvs.utils.nerfstudio import transform_to_ns_field_space
from rvs.utils.plot import fit_suptitle, image_grid_plot, place_legend_outside, save_figure
from rvs.utils.random import derive_rng


@dataclass
class Command:
    config: Path = tyro.MISSING
    """Path to config file to resume"""

    uid: Optional[Uid] = None
    """Object uid"""

    category: Optional[Category] = None
    """Object category"""

    output_dir: Path = tyro.MISSING
    """Output directory"""

    def run(self) -> None:
        if not self.config.exists() or not self.config.is_file():
            raise ValueError(f'Config file "{self.config}" does not exist')

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.output_dir.resolve()  # Resolve before changing working dir

        config = load_config(
            self.config,
            EvaluationConfig,
            on_default_applied=lambda fpath, obj, fname, value: CONSOLE.log(
                f"[bold yellow]WARNING: Applied default value to missing field {fpath}: {value}"
            ),
        )

        working_dir = find_config_working_dir(self.config, config.output_dir)

        if working_dir is not None:
            os.chdir(working_dir)

        eval: Evaluation = config.setup(overrides=self.__apply_config_overrides, debug_hook=self._run)
        eval.init()
        eval.run()

    def __apply_config_overrides(self, config: EvaluationConfig) -> EvaluationConfig:
        runtime = replace(config.runtime)
        runtime.override_existing = False
        runtime.results_only = False
        runtime.partial_results = False
        runtime.threads = 1
        runtime.stage_by_stage = False
        runtime.skip_embedder = False
        config = replace(config, runtime=runtime)
        config.embedder_image_cache_required = False
        config.embedder_text_cache_required = False
        if self.uid is not None:
            config.lvis_uids = {self.uid}
            config.lvis_uids_file = None
        if self.category is not None:
            config.lvis_categories = {self.category}
            config.lvis_categories_file = None
        return config

    def _run(self, ctx: DebugContext) -> None:
        pass


@dataclass
class SampleTextEmbeddingSimilarity(Command):
    views: List[int] = tyro.MISSING

    prompts: List[str] = tyro.MISSING

    flat_color: Optional[List[float]] = field(default_factory=lambda: [0.4, 0.4, 0.4, 1.0])

    def _run(self, ctx: DebugContext) -> None:
        if ctx.stage == PipelineStage.SAMPLE_EMBEDDINGS:
            if ctx.state.sample_embeddings_type != DefaultEmbeddingTypes.CLIP:
                raise ValueError(f"Invalid embedding type {ctx.state.sample_embeddings_type}")

            flat_model_color = np.array(self.flat_color) if self.flat_color is not None else None

            lerf_pipeline: LERFPipeline = ctx.state.pipeline.field.trainer.pipeline
            lerf_clip_encoder: OpenCLIPNetwork = lerf_pipeline.image_encoder

            similarities: Dict[str, NDArray] = dict()
            min_similarity: float = 1.0
            max_similarity: float = -1.0

            for prompt in self.prompts:
                CONSOLE.log(f"Embedding prompt={prompt}")

                embedding = lerf_clip_encoder.model.encode_text(lerf_clip_encoder.tokenizer(prompt).to("cuda")).detach()
                embedding /= embedding.norm(dim=-1, keepdim=True)

                np_embedding = embedding.squeeze().cpu().numpy()

                sample_similarities = np.zeros((ctx.state.sample_embeddings.shape[0]))
                for i in range(ctx.state.sample_embeddings.shape[0]):
                    sample_similarities[i] = np.dot(ctx.state.sample_embeddings[i], np_embedding)

                    min_similarity = min(min_similarity, sample_similarities[i])
                    max_similarity = max(max_similarity, sample_similarities[i])

                similarities[prompt] = sample_similarities

            blank_image = Image.fromarray(
                np.zeros(
                    (ctx.state.pipeline.renderer.config.width, ctx.state.pipeline.renderer.config.height, 4),
                    dtype=np.uint8,
                )
            )

            renders: List[im.Image] = []

            def callback(v: View, i: im.Image) -> None:
                nonlocal renders
                renders.append(i.copy())

            for view_idx in self.views:
                view: View = None

                for v in ctx.state.training_views:
                    if v.index == view_idx:
                        view = v

                if view is None:
                    raise ValueError(f"View with index {view_idx} not found")

                CONSOLE.log(f"Rendering view={view_idx}")

                render_sample(
                    ctx.state.pipeline.config.model_file,
                    view,
                    ctx.state.model_normalization,
                    callback=callback,
                    render_as_plot=False,
                )

            renders.append(blank_image)

            cmap = get_cmap("viridis")

            for prompt in self.prompts:
                sample_similarities = similarities[prompt]

                sample_colors = np.zeros((sample_similarities.shape[0], 3))
                for k in range(sample_similarities.shape[0]):
                    sample_colors[k] = cmap(
                        (sample_similarities[k] - min_similarity) / (max_similarity - min_similarity)
                    )[:3]

                for view_idx in self.views:
                    view: View = None

                    for v in ctx.state.training_views:
                        if v.index == view_idx:
                            view = v

                    if view is None:
                        raise ValueError(f"View with index {view_idx} not found")

                    CONSOLE.log(f"Rendering prompt={prompt} view={view_idx}")

                    render_sample_positions_and_colors(
                        ctx.state.pipeline.config.model_file,
                        view,
                        ctx.state.model_normalization,
                        ctx.state.sample_positions,
                        sample_colors=sample_colors,
                        callback=callback,
                        render_as_plot=False,
                        flat_model_color=flat_model_color,
                    )

                renders.append(blank_image)

            CONSOLE.log("Rendering grid plot")

            fig = plt.figure()

            def create_grid() -> None:
                axes = image_grid_plot(
                    fig,
                    renders,
                    columns=len(self.views) + 1,
                    row_labels=[None] + self.prompts,
                    col_labels=[f"View {str(v + 1)}" for v in self.views] + [None],
                    label_face_alpha=0.5,
                    border_color="black",
                    border_alpha=([0.5] * len(self.views) + [0.0]) * (len(self.prompts) + 1),
                )

                for i in range(len(self.prompts)):
                    cb_axes = axes[1 + i][len(self.views)]
                    fig.colorbar(
                        matplotlib.cm.ScalarMappable(
                            matplotlib.colors.Normalize(vmin=min_similarity, vmax=max_similarity), cmap=cmap
                        ),
                        cax=inset_axes(cb_axes, width="5%", height="100%", loc="center left"),
                        orientation="vertical",
                        ticks=np.linspace(min_similarity, max_similarity, 8, endpoint=True),
                    )

            fit_suptitle(
                fig,
                create_grid,
                suptitle=f"Similarity Between Sample Embeddings of 3D Model '{ctx.uid}'\nfrom Category '{ctx.category}' and Text Prompt Embeddings",
            )

            fig.set_facecolor((0, 0, 0, 0))

            fig.tight_layout()

            save_figure(fig, self.output_dir / "sample_text_embedding_similarity.png")


@dataclass
class ViewTextEmbeddingSimilarity(Command):
    prompt: str = tyro.MISSING

    highlight_selected_views: bool = True

    highlight_random_views: bool = True

    show_only_highlighted_views: bool = False

    relative_similarity_bar: bool = True

    sort_by_similarity: bool = True

    show_only_top_n: Optional[int] = None

    show_only_bottom_n: Optional[int] = None

    def _run(self, ctx: DebugContext) -> None:
        if ctx.stage == PipelineStage.OUTPUT:
            if (
                self.show_only_top_n is not None or self.show_only_bottom_n is not None
            ) and not self.sort_by_similarity:
                raise ValueError("Can only use show_only_top_n or show_only_bottom_n if sort_by_similarity=True")

            CONSOLE.log(f"Embedding prompt={self.prompt}")

            prompt_embedding = ctx.eval.embedder.embed_text_numpy(
                self.prompt, get_evaluation_prompt_embedding_cache_key(self.prompt)
            )

            CONSOLE.log("Loading view indices")

            selected_view_indices: Optional[List[int]] = None
            if self.highlight_selected_views:
                selected_view_indices = [view.index for view in ctx.state.selected_views]

            random_view_indices: Optional[List[int]] = None
            if self.highlight_random_views:
                all_random_view_indices: Dict[Uid, List[int]] = load_result(
                    ctx.eval.results_dir / "dumps" / "all_equiv_random_views_indices.pkl"
                )
                if ctx.uid in all_random_view_indices:
                    random_view_indices = all_random_view_indices[ctx.uid]

            num_cols = None

            view_indices_list: List[List[int]] = None

            if (
                self.show_only_highlighted_views
                and selected_view_indices is not None
                and random_view_indices is not None
            ):
                if len(selected_view_indices) >= len(random_view_indices):
                    view_indices_list = [selected_view_indices, random_view_indices]

                    if not self.sort_by_similarity:
                        num_cols = len(selected_view_indices)
                else:
                    view_indices_list = [random_view_indices, selected_view_indices]

                    if not self.sort_by_similarity:
                        num_cols = len(random_view_indices)
            else:
                view_indices_list = [[view.index for view in ctx.state.training_views]]

            assert view_indices_list is not None

            images: List[im.Image] = []
            images_files: List[Path] = []
            images_indices: List[int] = []
            images_embeddings: List[NDArray] = []
            images_similarity: List[float] = []

            for view_indices in view_indices_list:
                for view_idx in view_indices:
                    if view_idx not in images_indices:
                        CONSOLE.log(f"Loading view={view_idx}")

                        view: View = None

                        for v in ctx.state.training_views:
                            if v.index == view_idx:
                                view = v

                        if view is None:
                            raise ValueError(f"View with index {view_idx} not found")

                        file = view.resolve_path(ctx.state.pipeline.io)
                        images.append(im.open(file))
                        images_files.append(file)
                        images_indices.append(view_idx)

                        CONSOLE.log(f"Embedding view={view_idx}")

                        image_embedding = ctx.eval.embedder.embed_image_numpy(
                            file, get_pipeline_render_embedding_cache_key(ctx.state.pipeline.config.model_file, file)
                        )

                        images_embeddings.append(image_embedding)

                        similarity = np.dot(image_embedding, prompt_embedding)
                        images_similarity.append(similarity)

            if self.sort_by_similarity:
                num_cols = None

                sort_indices: List[int] = np.argsort(-np.array(images_similarity)).tolist()

                def sort(lst: List) -> List:
                    return [lst[i] for i in sort_indices]

                def truncate(lst: List) -> List:
                    top_n = self.show_only_top_n if self.show_only_top_n is not None else len(lst)
                    bottom_n = self.show_only_bottom_n if self.show_only_bottom_n is not None else len(lst)
                    return [lst[i] for i in range(len(lst)) if i < top_n or (len(lst) - 1 - i) < bottom_n]

                images = truncate(sort(images))
                images_files = truncate(sort(images_files))
                images_indices = truncate(sort(images_indices))
                images_embeddings = truncate(sort(images_embeddings))
                images_similarity = truncate(sort(images_similarity))

            min_similarity = min(images_similarity)
            max_similarity = max(images_similarity)

            CONSOLE.log("Creating plot")

            fig = plt.figure()

            axes = image_grid_plot(
                fig,
                images,
                columns=num_cols,
                label_face_alpha=0.5,
                border_color="black",
                border_alpha=0.5,
            )

            i = 0
            for r in range(len(axes)):
                for c in range(len(axes[0])):
                    if i >= len(images_indices):
                        break

                    ax = axes[r][c]

                    view_idx = images_indices[i]

                    similarity = images_similarity[i]

                    ax.annotate(
                        "{0:.2f}".format(similarity),
                        (0.925, 0.075),
                        horizontalalignment="right",
                        verticalalignment="bottom",
                        xycoords="axes fraction",
                        fontsize=24.0,
                        fontweight="bold",
                        bbox={
                            "facecolor": fig.get_facecolor(),
                            "alpha": 0.5,
                            "boxstyle": "square",
                            "edgecolor": "none",
                            "linewidth": 0,
                        },
                    )

                    # bar_ax: Axes = inset_axes(ax, width="5%", height="97%", loc="center left")
                    bar_ax: Axes = make_axes_locatable(ax).append_axes("left", size="5%", pad="2%")

                    bar_height: float
                    if self.relative_similarity_bar:
                        bar_height = 2.0 * (similarity - min_similarity) / (max_similarity - min_similarity)
                    else:
                        bar_height = 1.0 + similarity

                    bar_ax.bar([0], [bar_height], bottom=-1.0)
                    bar_ax.set_ylim(bottom=-1.0, top=1.0)
                    bar_ax.axis("off")

                    is_selected_view = view_idx in selected_view_indices if selected_view_indices is not None else False
                    is_random_view = view_idx in random_view_indices if random_view_indices is not None else False

                    if is_random_view:
                        ax.add_patch(
                            Rectangle(
                                (0.0, 0.0),
                                1.0,
                                1.0,
                                transform=ax.transAxes,
                                linewidth=40 if is_selected_view else 20,
                                edgecolor="C1",
                                facecolor="none",
                                alpha=1.0,
                                zorder=-2,
                            )
                        )

                    if is_selected_view:
                        ax.add_patch(
                            Rectangle(
                                (0.0, 0.0),
                                1.0,
                                1.0,
                                transform=ax.transAxes,
                                linewidth=20,
                                edgecolor="C2",
                                facecolor="none",
                                alpha=1.0,
                                zorder=-1,
                            )
                        )

                    i = i + 1

            legend_handles: List[Patch] = []

            if self.highlight_selected_views:
                legend_handles.append(
                    Patch(
                        facecolor="C2",
                        edgecolor="C2",
                        label="Selected View",
                    )
                )

            if self.highlight_random_views:
                legend_handles.append(
                    Patch(
                        facecolor="C1",
                        edgecolor="C1",
                        label="Random View",
                    )
                )

            if len(legend_handles) > 0:
                axes[0][-1].legend(
                    handles=legend_handles,
                    loc="upper left",
                    fontsize=24,
                )
                place_legend_outside(axes[0][-1])

            fig.set_facecolor((0, 0, 0, 0))

            fig.tight_layout()

            save_figure(fig, self.output_dir / "view_text_embedding_similarity.png")


@dataclass
class SelectedViewEmbeddingSimilarity(Command):
    selected_views: Optional[List[int]] = None

    views: List[int] = tyro.MISSING

    flat_color: Optional[List[float]] = field(default_factory=lambda: [0.4, 0.4, 0.4, 1.0])

    def _run(self, ctx: DebugContext) -> None:
        if ctx.stage == PipelineStage.SELECT_VIEWS:
            if self.flat_color is not None and len(self.flat_color) != 4:
                raise ValueError("len(self.flat_color) != 4")

            flat_model_color = np.array(self.flat_color) if self.flat_color is not None else None

            if ctx.state.sample_embeddings_type != DefaultEmbeddingTypes.CLIP:
                raise ValueError(f"Invalid embedding type {ctx.state.sample_embeddings_type}")

            lerf_pipeline: LERFPipeline = ctx.state.pipeline.field.trainer.pipeline
            lerf_datamanager: DataManager = lerf_pipeline.datamanager
            lerf_model: LERFModel = lerf_pipeline.model

            selected_views: List[View] = []
            selected_views_indices: List[int] = []

            similarities: Dict[int, NDArray] = dict()
            min_similarity: float = 1.0
            max_similarity: float = -1.0

            for j, selected_view in enumerate(ctx.state.selected_views):
                if self.selected_views is None or j in self.selected_views:
                    CONSOLE.log(f"Embedding selected view={selected_view.index}")

                    embedding: NDArray = None

                    for i in range(len(lerf_datamanager.train_dataset)):
                        if lerf_datamanager.train_dataset[i]["image_idx"] == selected_view.index:
                            image: Tensor = lerf_datamanager.train_dataset[i]["image"]
                            image = image.to(lerf_model.device)
                            image = image[:, :, :3].permute(2, 0, 1).unsqueeze(0)

                            with torch.no_grad():
                                embedding = (
                                    lerf_model.image_encoder.encode_image(image).detach().cpu().numpy().reshape((-1,))
                                )

                            embedding = embedding / np.linalg.norm(embedding)

                            break

                    if embedding is None:
                        raise ValueError(
                            f"Could not find training view for selected view with index {selected_view.index}"
                        )

                    sample_similarities = np.zeros((ctx.state.sample_embeddings.shape[0]))
                    for i in range(ctx.state.sample_embeddings.shape[0]):
                        sample_similarities[i] = np.dot(ctx.state.sample_embeddings[i], embedding)

                        min_similarity = min(min_similarity, sample_similarities[i])
                        max_similarity = max(max_similarity, sample_similarities[i])

                    similarities[selected_view.index] = sample_similarities

                    selected_views.append(selected_view)
                    selected_views_indices.append(j)

            assert len(selected_views) == len(selected_views_indices)

            blank_image = Image.fromarray(
                np.zeros(
                    (ctx.state.pipeline.renderer.config.width, ctx.state.pipeline.renderer.config.height, 4),
                    dtype=np.uint8,
                )
            )

            renders: List[im.Image] = []

            def callback(v: View, i: im.Image) -> None:
                nonlocal renders
                renders.append(i.copy())

            renders.append(blank_image)
            renders.append(blank_image)

            for view_idx in self.views:
                view: View = None

                for v in ctx.state.training_views:
                    if v.index == view_idx:
                        view = v

                if view is None:
                    raise ValueError(f"View with index {view_idx} not found")

                CONSOLE.log(f"Rendering view={view_idx}")

                render_sample(
                    ctx.state.pipeline.config.model_file,
                    view,
                    ctx.state.model_normalization,
                    callback=callback,
                    render_as_plot=False,
                )

            renders.append(blank_image)

            cmap = get_cmap("viridis")

            for j, selected_view in enumerate(selected_views):
                CONSOLE.log(f"Rendering selected view={selected_view.index}")

                render_sample(
                    ctx.state.pipeline.config.model_file,
                    selected_view,
                    ctx.state.model_normalization,
                    callback=callback,
                    render_as_plot=False,
                )

                sample_similarities = similarities[selected_view.index]

                sample_colors = np.zeros((sample_similarities.shape[0], 3))
                for k in range(sample_similarities.shape[0]):
                    sample_colors[k] = cmap(
                        (sample_similarities[k] - min_similarity) / (max_similarity - min_similarity)
                    )[:3]

                cluster_sample_positions = (
                    np.ones((ctx.state.sample_positions.shape[0], 3)) * 100000.0
                )  # Outside render range

                for i in range(ctx.state.sample_positions.shape[0]):
                    if ctx.state.cluster_indices[i] == selected_views_indices[j]:
                        cluster_sample_positions[i] = ctx.state.sample_positions[i]

                CONSOLE.log(f"Rendering cluster view={selected_view.index}")

                render_sample_positions_and_colors(
                    ctx.state.pipeline.config.model_file,
                    selected_view,
                    ctx.state.model_normalization,
                    cluster_sample_positions,
                    sample_colors=sample_colors,
                    callback=callback,
                    render_as_plot=False,
                    flat_model_color=flat_model_color,
                )

                for view_idx in self.views:
                    view: View = None

                    for v in ctx.state.training_views:
                        if v.index == view_idx:
                            view = v

                    if view is None:
                        raise ValueError(f"View with index {view_idx} not found")

                    CONSOLE.log(f"Rendering view={view_idx}")

                    render_sample_positions_and_colors(
                        ctx.state.pipeline.config.model_file,
                        view,
                        ctx.state.model_normalization,
                        ctx.state.sample_positions,
                        sample_colors=sample_colors,
                        callback=callback,
                        render_as_plot=False,
                        flat_model_color=flat_model_color,
                    )

                renders.append(blank_image)

            CONSOLE.log("Rendering grid plot")

            fig = plt.figure()

            def create_grid() -> None:
                axes = image_grid_plot(
                    fig,
                    renders,
                    columns=len(self.views) + 2 + 1,
                    row_labels=[None] + [f"Selected View {str(v.index + 1)}" for v in selected_views],
                    col_labels=["Selected View", "Cluster"] + [f"View {str(v + 1)}" for v in self.views] + [None],
                    col_label_offsets=[1, 1] + [0] * (len(self.views) + 1),
                    label_face_alpha=0.5,
                    border_color="black",
                    border_alpha=[0.0, 0.0]
                    + [0.5] * len(self.views)
                    + [0.0]
                    + ([0.5] * (len(self.views) + 2) + [0.0]) * len(selected_views),
                )

                for i in range(len(selected_views)):
                    cb_axes = axes[1 + i][len(self.views) + 2 + 1 - 1]
                    fig.colorbar(
                        matplotlib.cm.ScalarMappable(
                            matplotlib.colors.Normalize(vmin=min_similarity, vmax=max_similarity), cmap=cmap
                        ),
                        cax=inset_axes(cb_axes, width="5%", height="100%", loc="center left"),
                        orientation="vertical",
                        ticks=np.linspace(min_similarity, max_similarity, 8, endpoint=True),
                    )

            fit_suptitle(
                fig,
                create_grid,
                suptitle=f"Similarity Between Sample Embeddings of 3D Model '{ctx.uid}'\nfrom Category '{ctx.category}' and Selected Views Embeddings",
            )

            fig.set_facecolor((0, 0, 0, 0))

            fig.tight_layout()

            save_figure(fig, self.output_dir / "selected_view_embedding_similarity.png")


@dataclass
class FieldWeights(Command):
    views: List[int] = tyro.MISSING

    num_contractions: int = 3
    num_expansions: int = 3

    contraction_factor: float = 0.95
    expansion_factor: float = 1.0 / 0.95

    per_view_scale: bool = True

    use_middle_scale: bool = False

    lower_scale_percentile: float = 10.0
    upper_scale_percentile: float = 90.0

    jitter_samples: int = 100
    jitter_scale: float = 0.01

    flat_color: Optional[List[float]] = field(default_factory=lambda: [0.4, 0.4, 0.4, 1.0])

    def _run(self, ctx: DebugContext) -> None:
        if ctx.stage == PipelineStage.SAMPLE_POSITIONS:
            if self.flat_color is not None and len(self.flat_color) != 4:
                raise ValueError("len(self.flat_color) != 4")

            flat_model_color = np.array(self.flat_color) if self.flat_color is not None else None

            lerf_pipeline: LERFPipeline = ctx.state.pipeline.field.trainer.pipeline
            lerf_model: LERFModel = lerf_pipeline.model

            datamanager: VanillaDataManager = lerf_pipeline.datamanager

            base_positions = ctx.state.sample_positions

            positions: Dict[int, NDArray] = dict()
            weights: Dict[int, NDArray] = dict()

            for i in reversed(list(range(self.num_contractions))):
                contracted_positions = base_positions * (self.contraction_factor ** (i + 1))
                positions[-(i + 1)] = contracted_positions
                weights[-(i + 1)] = self.__sample_weights(
                    lerf_model.field, lerf_model.device, contracted_positions, datamanager.train_dataparser_outputs
                )

            positions[0] = base_positions
            weights[0] = self.__sample_weights(
                lerf_model.field, lerf_model.device, base_positions, datamanager.train_dataparser_outputs
            )

            for i in range(self.num_expansions):
                expanded_positions = base_positions * (self.expansion_factor ** (i + 1))
                positions[i + 1] = expanded_positions
                weights[i + 1] = self.__sample_weights(
                    lerf_model.field, lerf_model.device, expanded_positions, datamanager.train_dataparser_outputs
                )

            all_weights: NDArray
            if self.use_middle_scale:
                all_weights = weights[0]
            else:
                all_weights = np.concatenate([ds for ds in weights.values()])
            min_weight = np.nanpercentile(all_weights, self.lower_scale_percentile)
            max_weight = np.nanpercentile(all_weights, self.upper_scale_percentile)

            visible_indices: List[NDArray] = []

            def projection_callback(p: NDArray) -> None:
                nonlocal visible_indices
                visible_indices.append(np.nonzero(p[:, 3] >= 0.5)[0])

            for view_idx in self.views:
                view: View = None

                for v in ctx.state.training_views:
                    if v.index == view_idx:
                        view = v

                if view is None:
                    raise ValueError(f"View with index {view_idx} not found")

                def noop_render_callback(v: View, i: im.Image):
                    pass

                # Just to determine which positions are visible
                render_sample_positions(
                    ctx.state.pipeline.config.model_file,
                    view,
                    ctx.state.model_normalization,
                    ctx.state.sample_positions,
                    callback=noop_render_callback,
                    render_as_plot=False,
                    flat_model_color=np.zeros((4,)),
                    projection_callback=projection_callback,
                    render_sample_positions=False,
                )

            assert len(visible_indices) == len(self.views)

            renders: List[List[im.Image]] = []

            cmap = get_cmap("viridis")

            blank_image = Image.fromarray(
                np.zeros(
                    (ctx.state.pipeline.renderer.config.width, ctx.state.pipeline.renderer.config.height, 4),
                    dtype=np.uint8,
                )
            )

            local_min_weights: List[float] = []
            local_max_weights: List[float] = []

            for i, view_idx in enumerate(self.views):
                view: View = None

                for v in ctx.state.training_views:
                    if v.index == view_idx:
                        view = v

                if view is None:
                    raise ValueError(f"View with index {view_idx} not found")

                local_min_weight = min_weight
                local_max_weight = max_weight

                if self.per_view_scale:
                    if self.use_middle_scale:
                        local_min_weight = np.nanpercentile(weights[0][visible_indices[i]], self.lower_scale_percentile)
                        local_max_weight = np.nanpercentile(weights[0][visible_indices[i]], self.upper_scale_percentile)
                    else:
                        local_visible_weights = np.concatenate(
                            [weights[j][visible_indices[i]] for j in positions.keys()]
                        )
                        local_min_weight = np.nanpercentile(local_visible_weights, self.lower_scale_percentile)
                        local_max_weight = np.nanpercentile(local_visible_weights, self.upper_scale_percentile)

                local_min_weights.append(local_min_weight)
                local_max_weights.append(local_max_weight)

                for j in sorted(list(positions.keys())):
                    visible_positions = positions[j][visible_indices[i], :]
                    visible_weights = weights[j][visible_indices[i]]

                    assert visible_positions.shape[0] == visible_weights.shape[0]

                    sample_render: im.Image = None

                    def sample_render_callback(v: View, i: im.Image) -> None:
                        nonlocal sample_render
                        sample_render = i.copy()

                    sample_colors = np.zeros((visible_weights.shape[0], 3))
                    for k in range(visible_weights.shape[0]):
                        sample_colors[k] = cmap(
                            np.clip(
                                (visible_weights[k] - local_min_weight) / (local_max_weight - local_min_weight),
                                0.0,
                                1.0,
                            )
                        )[:3]

                    render_sample_positions_and_colors(
                        ctx.state.pipeline.config.model_file,
                        view,
                        ctx.state.model_normalization,
                        visible_positions,
                        sample_colors,
                        callback=sample_render_callback,
                        render_as_plot=False,
                        flat_model_color=np.zeros((4,)),
                        render_model=False,
                    )

                    assert sample_render is not None

                    model_render: im.Image = None

                    def model_render_callback(v: View, i: im.Image) -> None:
                        nonlocal model_render
                        model_render = i.copy()

                    render_sample(
                        ctx.state.pipeline.config.model_file,
                        view,
                        ctx.state.model_normalization,
                        callback=model_render_callback,
                        render_as_plot=False,
                        flat_model_color=flat_model_color,
                    )

                    assert model_render is not None

                    assert model_render.width == sample_render.width
                    assert model_render.height == sample_render.height

                    renders.append(im.alpha_composite(model_render, sample_render))

                    sample_render.close()
                    model_render.close()

                renders.append(blank_image)

            fig = plt.figure()

            axes = image_grid_plot(
                fig,
                renders,
                columns=self.num_contractions + 1 + self.num_expansions + 1,
                col_labels=[str(scale) if scale <= 0 else f"+{str(scale)}" for scale in sorted(list(positions.keys()))]
                + [None],
                row_labels=[f"View {str(v + 1)}" for v in self.views],
                label_face_alpha=0.5,
                border_color="black",
                border_alpha=([0.5] * (self.num_contractions + 1 + self.num_expansions + 1 - 1) + [0.0])
                * len(self.views),
            )

            for i in range(len(self.views)):
                cb_axes = axes[i][-1]
                cbar = fig.colorbar(
                    matplotlib.cm.ScalarMappable(
                        matplotlib.colors.Normalize(vmin=local_min_weights[i], vmax=local_max_weights[i]), cmap=cmap
                    ),
                    cax=inset_axes(cb_axes, width="5%", height="100%", loc="center left"),
                    orientation="vertical",
                    ticks=np.linspace(local_min_weights[i], local_max_weights[i], 8, endpoint=True),
                )
                cbar.ax.set_ylabel("$1 - e^{\sigma}$", rotation=-90, va="bottom")

            fig.set_facecolor((0, 0, 0, 0))

            fig.tight_layout()

            save_figure(fig, self.output_dir / "field_weights.png")

    def __sample_weights(
        self, field: Field, device: Any, positions: NDArray, dataparser_outputs: DataparserOutputs
    ) -> NDArray:
        with torch.no_grad():
            base_positions = torch.from_numpy(positions.copy()).to(device, dtype=torch.float32)

            base_positions = transform_to_ns_field_space(base_positions, dataparser_outputs)

            def jitter_positions() -> Tensor:
                random_offsets = (torch.rand(size=base_positions.shape, device=device) - 0.5) * 2.0
                return base_positions + random_offsets * self.jitter_scale * dataparser_outputs.dataparser_scale

            batched_positions = torch.stack([base_positions] + [jitter_positions() for _ in range(self.jitter_samples)])

            batched_densities = field.density_fn(batched_positions).detach()

            batched_weights = 1.0 - torch.exp(-batched_densities)
            batched_weights = torch.nan_to_num(batched_weights)

            avg_weights = batched_weights.mean(dim=0, keepdim=False)

            return avg_weights.squeeze().cpu().numpy()


@dataclass
class EmbeddingCorrespondence(Command):
    views: List[int] = tyro.MISSING

    num_lines: int = 30

    lines_seed: Optional[int] = 42

    flat_color: Optional[List[float]] = None  # field(default_factory=lambda: [0.4, 0.4, 0.4, 1.0])

    render_sample_positions: bool = False

    flip_y: bool = False

    circle_x: Optional[float] = None
    circle_y: Optional[float] = None
    circle_radius: Optional[float] = None

    _xlabel: Optional[str] = "D1"
    _ylabel: Optional[str] = "D2"

    def _create_datapoints(self, ctx: DebugContext) -> NDArray:
        pass

    def _save_fig(self, fig: Figure) -> None:
        pass

    def _get_title(self, ctx: DebugContext) -> Optional[str]:
        pass

    def _run(self, ctx: DebugContext) -> None:
        if ctx.stage == PipelineStage.SAMPLE_EMBEDDINGS:
            flat_model_color = np.array(self.flat_color) if self.flat_color is not None else None

            if not (
                (self.circle_x is None) == (self.circle_y is None)
                and (self.circle_y is None) == (self.circle_radius is None)
                and (self.circle_radius is None) == (self.circle_x is None)
            ):
                raise ValueError("circle_x, circle_y and circle_radius must be used together")

            datapoints = self._create_datapoints(ctx)

            renders: List[im.Image] = []
            projections: List[NDArray] = []

            def render_callback(v: View, i: im.Image) -> None:
                nonlocal renders
                renders.append(i.copy())

            def projection_callback(p: NDArray) -> None:
                nonlocal projections
                projections.append(p)

            for view_idx in self.views:
                view: View = None

                for v in ctx.state.training_views:
                    if v.index == view_idx:
                        view = v

                if view is None:
                    raise ValueError(f"View with index {view_idx} not found")

                CONSOLE.log(f"Rendering view={view_idx}")

                render_sample_positions(
                    ctx.state.pipeline.config.model_file,
                    view,
                    ctx.state.model_normalization,
                    ctx.state.sample_positions,
                    callback=render_callback,
                    render_as_plot=False,
                    flat_model_color=flat_model_color,
                    projection_callback=projection_callback,
                    render_sample_positions=False,
                )

            assert len(renders) == len(projections)

            CONSOLE.log("Creating plots")

            axes: List[List[Axes]]

            fig, axes = plt.subplots(nrows=len(renders), ncols=2)

            if isinstance(axes[0], Axes):
                axes = [axes]

            fig_scale = 2.0
            fig.set_size_inches(6.4 * fig_scale, 4.8 * fig_scale * len(self.views))

            title = self._get_title(ctx)
            if title is not None:
                fig.suptitle(
                    title,
                    bbox={
                        "facecolor": fig.get_facecolor(),
                        "alpha": 0.5,
                        "boxstyle": "square",
                        "edgecolor": "none",
                        "linewidth": 0,
                    },
                    fontsize=18,
                    verticalalignment="top",
                    y=1.0,
                )

            lines_seed = self.lines_seed
            if lines_seed is None:
                lines_seed = np.random.randint(0, 2**32 - 1)

            lines_rng = derive_rng(str(lines_seed).encode())

            for i in range(len(renders)):
                image = renders[i]
                image_projections = projections[i]

                visible_indices = np.nonzero(image_projections[:, 3] >= 0.5)[0]
                invisible_indices = np.nonzero(image_projections[:, 3] < 0.5)[0]

                visible_positions = image_projections[visible_indices, :2] * np.array([image.width, image.height])
                visible_datapoints = datapoints[visible_indices]
                invisible_datapoints = datapoints[invisible_indices]

                # axes[i][0].set_aspect("equal")

                visible_scatter = axes[i][0].scatter(visible_datapoints[:, 0], visible_datapoints[:, 1], s=10.0)
                visible_scatter_colors = visible_scatter.get_facecolors()

                axes[i][0].scatter(invisible_datapoints[:, 0], invisible_datapoints[:, 1], s=1.0, alpha=0.5)

                if self._xlabel is not None:
                    axes[i][0].set_xlabel(self._xlabel)
                if self._ylabel is not None:
                    axes[i][0].set_ylabel(self._ylabel)
                axes[i][0].legend(["Visible", "Obstructed"])

                if self.flip_y:
                    axes[i][0].set_ylim(axes[i][0].get_ylim()[::-1])

                axes[i][1].axis("off")
                axes[i][1].set_aspect("equal")

                axes[i][1].imshow(image)
                # axes[i][1].scatter(visible_positions[:, 0], visible_positions[:, 1], s=0.5)

                available_indices: NDArray
                if self.circle_radius is None:
                    available_indices = np.arange(0, visible_positions.shape[0])
                else:
                    assert self.circle_x is not None

                    assert self.circle_y is not None
                    center = np.array([self.circle_x, self.circle_y])

                    distances = np.linalg.norm(visible_datapoints - center, axis=1)

                    inside: NDArray = distances <= self.circle_radius

                    available_indices = inside.nonzero()[0]

                    axes[i][0].add_patch(
                        Circle(
                            (self.circle_x, self.circle_y),
                            self.circle_radius,
                            facecolor="None",
                            edgecolor="black",
                            linestyle="--",
                        )
                    )

                random_indices: NDArray[np.signedinteger] = lines_rng.choice(
                    available_indices,
                    size=min(self.num_lines, available_indices.shape[0]),
                    replace=False,
                )

                lines_a_x: List[float] = []
                lines_a_y: List[float] = []

                lines_b_x: List[float] = []
                lines_b_y: List[float] = []

                lines_color: List[int] = []

                for j in range(random_indices.shape[0]):
                    idx = random_indices[j]

                    a = (visible_positions[idx, 0], visible_positions[idx, 1])
                    b = (visible_datapoints[idx, 0], visible_datapoints[idx, 1])

                    color = None
                    if isinstance(visible_scatter_colors, List):
                        if idx < len(visible_scatter_colors):
                            color = visible_scatter_colors[idx]
                        else:
                            color = visible_scatter_colors[0]
                    else:
                        color = visible_scatter_colors

                    connection_patch = ConnectionPatch(
                        xyA=a,
                        xyB=b,
                        coordsA="data",
                        coordsB="data",
                        axesA=axes[i][1],
                        axesB=axes[i][0],
                        color=color,
                    )

                    axes[i][1].add_artist(connection_patch)

                    lines_a_x.append(a[0])
                    lines_a_y.append(a[1])

                    lines_b_x.append(b[0])
                    lines_b_y.append(b[1])

                    lines_color.append(color)

                for j in range(len(lines_a_x)):
                    a_x = lines_a_x[j]
                    a_y = lines_a_y[j]

                    b_x = lines_b_x[j]
                    b_y = lines_b_y[j]

                    color = lines_color[j]

                    axes[i][1].scatter(a_x, a_y, s=30.0, color=color, edgecolors="black")
                    axes[i][0].scatter(b_x, b_y, s=30.0, color=color, edgecolors="black")

            fig.set_facecolor((0, 0, 0, 0))

            # fig.tight_layout()

            self._save_fig(fig)


@dataclass
class EmbeddingPCACorrespondence(EmbeddingCorrespondence):
    def __post_init__(self):
        self._xlabel = "PC1"
        self._ylabel = "PC2"

    def _create_datapoints(self, ctx: DebugContext) -> NDArray:
        CONSOLE.log("Calculating PCA")

        pca = PCA(n_components=2)

        pca.fit(ctx.state.sample_embeddings)

        return pca.transform(ctx.state.sample_embeddings)

    def _get_title(self, ctx: DebugContext) -> Optional[str]:
        return f"PCA of Sample Embeddings of 3D Model '{ctx.uid}'\nfrom Category '{ctx.category}'"

    def _save_fig(self, fig: Figure) -> None:
        save_figure(fig, self.output_dir / "pca_correspondence.png")


@dataclass
class EmbeddingUMAPCorrespondence(EmbeddingCorrespondence):
    umap_seed: Optional[int] = 42

    umap_metric: str = "cosine"

    umap_n_neighbors: int = 15

    umap_min_dist: float = 0.1

    def _create_datapoints(self, ctx: DebugContext) -> NDArray:
        CONSOLE.log("Calculating UMAP")

        umap_seed = self.umap_seed
        if umap_seed is None:
            umap_seed = np.random.randint(0, 2**32 - 1)

        mapper = umap.UMAP(
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            metric=self.umap_metric,
            random_state=self.umap_seed,
        )

        mapper.fit(ctx.state.sample_embeddings)

        return mapper.embedding_

    def _get_title(self, ctx: DebugContext) -> Optional[str]:
        return f"UMAP of Sample Embeddings of 3D Model '{ctx.uid}'\nfrom Category '{ctx.category}'"

    def _save_fig(self, fig: Figure) -> None:
        save_figure(fig, self.output_dir / "umap_correspondence.png")


@dataclass
class EmbeddingTSNECorrespondence(EmbeddingCorrespondence):
    tsne_seed: Optional[int] = 42

    tsne_metric: str = "cosine"

    tsne_n_iter: int = 1000

    tsne_perplexity: float = 30.0

    def _create_datapoints(self, ctx: DebugContext) -> NDArray:
        CONSOLE.log("Calculating t-SNE")

        tsne_seed = self.tsne_seed
        if tsne_seed is None:
            tsne_seed = np.random.randint(0, 2**32 - 1)

        tsne = TSNE(
            random_state=self.tsne_seed,
            metric=self.tsne_metric,
            n_iter=self.tsne_n_iter,
            perplexity=self.tsne_perplexity,
        )

        embedding = tsne.fit_transform(ctx.state.sample_embeddings)

        return embedding

    def _get_title(self, ctx: DebugContext) -> Optional[str]:
        return f"t-SNE of Sample Embeddings of 3D Model '{ctx.uid}'\nfrom Category '{ctx.category}'"

    def _save_fig(self, fig: Figure) -> None:
        save_figure(fig, self.output_dir / "tsne_correspondence.png")


commands = {
    "sample_text_embedding_similarity": SampleTextEmbeddingSimilarity(),
    "view_text_embedding_similarity": ViewTextEmbeddingSimilarity(),
    "selected_view_embedding_similarity": SelectedViewEmbeddingSimilarity(),
    "pca_correspondence": EmbeddingPCACorrespondence(),
    "umap_correspondence": EmbeddingUMAPCorrespondence(),
    "tsne_correspondence": EmbeddingTSNECorrespondence(),
    "field_weights": FieldWeights(),
}

SubcommandTypeUnion = tyro.conf.SuppressFixed[
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(
            defaults=commands, descriptions={key: type(command).__doc__ for key, command in commands.items()}
        )
    ]
]


def main(cmd: Command):
    cmd.run()


def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            SubcommandTypeUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    entrypoint()
