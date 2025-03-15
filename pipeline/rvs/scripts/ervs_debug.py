import os

import torch
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
from lerf.encoders.openclip_encoder import OpenCLIPNetwork
from lerf.lerf import LERFModel
from lerf.lerf_pipeline import LERFPipeline
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.data.datamanagers.base_datamanager import DataManager
from nerfstudio.utils.rich_utils import CONSOLE
from numpy.typing import NDArray
from PIL import Image, Image as im

from rvs.evaluation.debug import DebugContext
from rvs.evaluation.evaluation import Evaluation, EvaluationConfig
from rvs.pipeline.embedding import DefaultEmbeddingTypes
from rvs.pipeline.stage import PipelineStage
from rvs.pipeline.views import View
from rvs.utils.config import find_config_working_dir, load_config
from rvs.utils.debug import render_sample, render_sample_positions_and_colors
from rvs.utils.plot import fit_suptitle, image_grid_plot, save_figure


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


commands = {
    "sample_text_embedding_similarity": SampleTextEmbeddingSimilarity(),
    "selected_view_embedding_similarity": SelectedViewEmbeddingSimilarity(),
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
