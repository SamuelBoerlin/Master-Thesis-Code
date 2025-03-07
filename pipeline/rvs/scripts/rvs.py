import os
import sys
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.style import Style
from tyro._argparse_formatter import THEME

from rvs.configs.pipeline_configs import (
    MethodDummyConfig,
    list_available_components,
    pipeline_method_format,
    setup_pipeline_tyro_union,
)

# TODO: This should probably be elsewhere
# Required for headless rendering with pyrenderer and trimesh
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYGLET_HEADLESS"] = "1"

import random

import numpy as np
import torch
import tyro
from nerfstudio.configs.config_utils import convert_markup_to_ansi

from rvs.pipeline.pipeline import Pipeline, PipelineConfig


def _set_random_seed(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(config: Any):
    if isinstance(config, PipelineConfig):
        start_pipeline(config)
    elif isinstance(config, MethodDummyConfig):
        console = Console(theme=THEME.as_rich_theme(), stderr=True)
        console.print(
            Panel(
                Group(
                    f"Pipeline format: {pipeline_method_format()}\n\n{list_available_components()}",
                ),
                title="[bold]Pipeline[/bold]",
                title_align="left",
                border_style=Style(color="yellow"),
                expand=False,
            )
        )
    else:
        raise Exception("Invalid config type")


def start_pipeline(config: PipelineConfig) -> None:
    config.set_timestamp()

    _set_random_seed(config.machine.seed)

    pipeline: Pipeline = config.setup(local_rank=0, world_size=1)

    pipeline.init()

    config.print_to_terminal()
    config.save_config()

    pipeline.run()


def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            setup_pipeline_tyro_union(sys.argv),
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    entrypoint()
