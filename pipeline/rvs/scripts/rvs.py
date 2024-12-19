import os

# TODO: This should probably be elsewhere
# Required for headless rendering with pyrenderer and trimesh
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYGLET_HEADLESS"] = "1"

import random

import numpy as np
import torch
import tyro

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from rvs.configs.pipeline_configs import AnnotatedBaseConfigUnion
from rvs.pipeline.pipeline import Pipeline, PipelineConfig


def _set_random_seed(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(config: PipelineConfig):
    config.set_timestamp()

    config.print_to_terminal()

    _set_random_seed(config.machine.seed)

    pipeline: Pipeline = config.setup(local_rank=0, world_size=1)

    pipeline.init()

    config.save_config()

    pipeline.run()


def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    entrypoint()
