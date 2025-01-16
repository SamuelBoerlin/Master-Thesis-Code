import os

# TODO: This should probably be elsewhere
# Required for headless rendering with pyrenderer and trimesh
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYGLET_HEADLESS"] = "1"
from typing import Any

import tyro
from nerfstudio.configs.config_utils import convert_markup_to_ansi

from rvs.configs.evaluation_configs import AnnotatedBaseConfigUnion
from rvs.evaluation.evaluation import Evaluation, EvaluationConfig, EvaluationResumeConfig


def main(config: Any):
    if isinstance(config, EvaluationConfig):
        run_evaluation(config)
    elif isinstance(config, EvaluationResumeConfig):
        resume_evaluation(config)
    else:
        raise Exception("Invalid config type")


def run_evaluation(config: EvaluationConfig) -> None:
    eval: Evaluation = config.setup()
    eval.init()
    eval.run()


def resume_evaluation(config: EvaluationResumeConfig) -> None:
    eval_config, working_dir = config.load()

    if working_dir is not None:
        os.chdir(working_dir)

    run_evaluation(eval_config)


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
