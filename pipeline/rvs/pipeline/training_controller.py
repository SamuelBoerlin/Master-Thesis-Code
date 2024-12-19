from dataclasses import dataclass, field
from typing import List, Type

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.utils.rich_utils import CONSOLE
from rvs.pipeline.model import WrapperHooks, WrapperHooksConfig


@dataclass
class TrainingControllerConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: TrainingController)

    rgb_only_iterations: int = 100
    """Number of iterations to train RGB only before starting to train embeddings"""


class TrainingController:
    @dataclass
    class HooksConfig(WrapperHooksConfig):
        _target: Type = field(default_factory=lambda: TrainingController.Hooks)

        controller: "TrainingController" = field(default=None)

    class Hooks(WrapperHooks):
        config: "TrainingController.HooksConfig"

        def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
        ) -> List[TrainingCallback]:
            """Hook for nerfstudio.models.base_model.Model.get_training_callbacks"""
            callbacks = self.wrapper_model.get_training_callbacks(training_callback_attributes)
            callbacks.append(
                TrainingCallback(
                    [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    lambda step: self.config.controller.on_iteration_done(step),
                )
            )
            return callbacks

    config: TrainingControllerConfig
    hooks: WrapperHooksConfig

    def __init__(self, config: TrainingControllerConfig) -> None:
        self.config = config
        self.hooks = TrainingController.HooksConfig(controller=self)

    def on_iteration_done(self, step: int) -> None:
        if step > self.config.rgb_only_iterations:
            if step == self.config.rgb_only_iterations + 1:
                CONSOLE.log("Turning on embedding training")
            # TODO: Implement switch
