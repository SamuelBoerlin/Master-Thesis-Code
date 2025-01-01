import os

# TODO: This should probably be elsewhere
# Required for headless rendering with pyrenderer and trimesh
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYGLET_HEADLESS"] = "1"

import tyro
from nerfstudio.configs.config_utils import convert_markup_to_ansi

from rvs.configs.evaluation_configs import AnnotatedBaseConfigUnion
from rvs.evaluation.evaluation import Evaluation, EvaluationConfig


def main(config: EvaluationConfig):
    eval: Evaluation = config.setup()

    eval.init()

    eval.run()


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
