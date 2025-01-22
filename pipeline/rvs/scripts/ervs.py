import os

# TODO: This should probably be elsewhere
# Required for headless rendering with pyrenderer and trimesh
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYGLET_HEADLESS"] = "1"
import sys
from dataclasses import replace
from shlex import quote
from typing import Any, Callable, Optional

import tyro
from nerfstudio.configs.config_utils import convert_markup_to_ansi

from rvs.configs.evaluation_configs import AnnotatedBaseConfigUnion
from rvs.evaluation.evaluation import Evaluation, EvaluationConfig, EvaluationResumeConfig


def main(config: Any):
    if isinstance(config, EvaluationConfig):
        start_evaluation(config)
    elif isinstance(config, EvaluationResumeConfig):
        resume_evaluation(config)
    else:
        raise Exception("Invalid config type")


def start_evaluation(eval_config: EvaluationConfig) -> None:
    eval_config = replace(eval_config)

    if eval_config.runtime.metadata is None:
        eval_config.runtime.metadata = dict()

    eval_config.runtime.metadata["args"] = sys.argv
    eval_config.runtime.metadata["unix_shell_command"] = sys.argv[0] + " ".join(
        [arg if arg.startswith("--") else quote(arg) for arg in sys.argv[1:]]
    )

    run_evaluation(eval_config)


def resume_evaluation(config: EvaluationResumeConfig) -> None:
    eval_config, eval_config_overrides, working_dir = config.load()

    if working_dir is not None:
        os.chdir(working_dir)

    run_evaluation(eval_config, overrides=eval_config_overrides)


def run_evaluation(
    config: EvaluationConfig,
    overrides: Optional[Callable[[EvaluationConfig], EvaluationConfig]] = None,
) -> None:
    eval: Evaluation = config.setup(overrides=overrides)
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
