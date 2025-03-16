import os

import torch
from sklearn.decomposition import PCA
from torch import Tensor

# TODO: This should probably be elsewhere
# Required for headless rendering with pyrenderer and trimesh
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYGLET_HEADLESS"] = "1"

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.cm
import matplotlib.colors
import numpy as np
import tyro
import umap
from lerf.encoders.openclip_encoder import OpenCLIPNetwork
from lerf.lerf import LERFModel
from lerf.lerf_pipeline import LERFPipeline
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.data.datamanagers.base_datamanager import DataManager
from nerfstudio.utils.rich_utils import CONSOLE
from numpy.typing import NDArray
from PIL import Image, Image as im
from sklearn.manifold import TSNE

from rvs.evaluation.debug import DebugContext
from rvs.evaluation.evaluation import Evaluation, EvaluationConfig
from rvs.pipeline.embedding import DefaultEmbeddingTypes
from rvs.pipeline.stage import PipelineStage
from rvs.pipeline.views import View
from rvs.utils.config import find_config_working_dir, load_config
from rvs.utils.debug import render_sample, render_sample_positions, render_sample_positions_and_colors
from rvs.utils.plot import fit_suptitle, image_grid_plot, save_figure
from rvs.utils.random import derive_rng


@dataclass
class Command:
    config: Path = tyro.MISSING
    """Path to config file to resume"""

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
        config.runtime.from_stage = PipelineStage.OUTPUT
        config.runtime.to_stage = PipelineStage.OUTPUT
        config.runtime.override_existing = False
        config.runtime.results_only = False
        config.runtime.partial_results = False
        config.runtime.threads = 1
        config.runtime.stage_by_stage = False
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
            assert len(selected_views) == len(similarities)

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
class EmbeddingCorrespondence(Command):
    views: List[int] = tyro.MISSING

    num_lines: int = 30

    lines_seed: Optional[int] = 42

    flat_color: Optional[List[float]] = None  # field(default_factory=lambda: [0.4, 0.4, 0.4, 1.0])

    render_sample_positions: bool = False

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

                axes[i][1].axis("off")
                axes[i][1].set_aspect("equal")

                axes[i][1].imshow(image)
                # axes[i][1].scatter(visible_positions[:, 0], visible_positions[:, 1], s=0.5)

                random_indices: NDArray[np.signedinteger] = lines_rng.choice(
                    np.arange(0, visible_positions.shape[0]),
                    size=min(self.num_lines, visible_positions.shape[0]),
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

        print(embedding.shape)

        return embedding

    def _get_title(self, ctx: DebugContext) -> Optional[str]:
        return f"t-SNE of Sample Embeddings of 3D Model '{ctx.uid}'\nfrom Category '{ctx.category}'"

    def _save_fig(self, fig: Figure) -> None:
        save_figure(fig, self.output_dir / "tsne_correspondence.png")


commands = {
    "sample_text_embedding_similarity": SampleTextEmbeddingSimilarity(),
    "selected_view_embedding_similarity": SelectedViewEmbeddingSimilarity(),
    "pca_correspondence": EmbeddingPCACorrespondence(),
    "umap_correspondence": EmbeddingUMAPCorrespondence(),
    "tsne_correspondence": EmbeddingTSNECorrespondence(),
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
