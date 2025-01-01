import os

# TODO: This should probably be elsewhere
# Required for headless rendering with pyrenderer and trimesh
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYGLET_HEADLESS"] = "1"

from rvs.configs.pipeline_configs import pipeline_configs
from rvs.evaluation.evaluation import Evaluation, EvaluationConfig


def main():
    config = EvaluationConfig(
        pipeline=pipeline_configs["default"],
        lvis_categories={"amplifier"},
        lvis_uids={"31a843bd24d740158a57a59200ba0ac8"},
    )

    eval: Evaluation = config.setup()

    eval.init()

    eval.run()


def entrypoint():
    main()


if __name__ == "__main__":
    entrypoint()
