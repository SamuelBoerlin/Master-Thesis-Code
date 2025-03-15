import os

# TODO: This should probably be elsewhere
# Required for headless rendering with pyrenderer and trimesh
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYGLET_HEADLESS"] = "1"

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import tyro
from lerf.encoders.openclip_encoder import OpenCLIPNetwork
from lerf.lerf_pipeline import LERFPipeline
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE
from numpy.typing import NDArray
from PIL import Image as im

from rvs.evaluation.debug import DebugContext
from rvs.evaluation.evaluation import Evaluation, EvaluationConfig
from rvs.pipeline.embedding import DefaultEmbeddingTypes
from rvs.pipeline.stage import PipelineStage
from rvs.pipeline.views import View
from rvs.utils.config import find_config_working_dir, load_config
from rvs.utils.debug import render_sample_positions_and_colors
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

    def _run(self, ctx: DebugContext) -> None:
        if ctx.stage == PipelineStage.SAMPLE_EMBEDDINGS:
            if ctx.state.sample_embeddings_type != DefaultEmbeddingTypes.CLIP:
                raise ValueError(f"Invalid embedding type {ctx.state.sample_embeddings_type}")

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

            renders: List[im.Image] = []
            labels: List[str] = []

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

                    image: im.Image = None

                    def callback(v: View, i: im.Image) -> None:
                        nonlocal image
                        image = i.copy()

                    CONSOLE.log(f"Rendering prompt={prompt} view={view_idx}")

                    render_sample_positions_and_colors(
                        ctx.state.pipeline.config.model_file,
                        view,
                        ctx.state.model_normalization,
                        ctx.state.sample_positions,
                        sample_colors=sample_colors,
                        callback=callback,
                        render_as_plot=False,
                    )

                    assert image is not None

                    renders.append(image)

                    labels.append(f"{prompt}")

            assert len(renders) == len(labels)

            CONSOLE.log("Rendering grid plot")

            fig = plt.figure()

            fit_suptitle(
                fig,
                lambda: image_grid_plot(
                    fig,
                    renders,
                    columns=len(self.views),
                    # labels=labels,
                    row_labels=self.prompts,
                    col_labels=[f"View {str(v + 1)}" for v in self.views],
                    label_face_alpha=0.5,
                    border_color="black",
                ),
                suptitle=f"Similarity Between Sample Embeddings of 3D Model '{ctx.uid}'\nfrom Category '{ctx.category}' and Text Prompt Embeddings",
            )

            fig.tight_layout()

            fig.set_facecolor((0, 0, 0, 0))

            save_figure(fig, self.output_dir / "sample_text_embedding_similarity.png")


commands = {
    "sample_text_embedding_similarity": SampleTextEmbeddingSimilarity(),
    "test": SampleTextEmbeddingSimilarity(),
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
