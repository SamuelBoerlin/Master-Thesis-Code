from abc import ABC
from dataclasses import dataclass, field
from typing import List, Type

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.models.base_model import Model
from nerfstudio.utils.rich_utils import CONSOLE
from rvs.pipeline.model import WrapperHooks, WrapperHooksConfig


@dataclass
class TrainingControllerConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: TrainingController)

    rgb_only_iterations: int = 0
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
                    [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    lambda step: self.config.controller.on_iteration_start(self.wrapper_model, step),
                )
            )
            callbacks.append(
                TrainingCallback(
                    [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    lambda step: self.config.controller.on_iteration_done(self.wrapper_model, step),
                )
            )
            return callbacks

    config: TrainingControllerConfig
    hooks: WrapperHooksConfig

    def __init__(self, config: TrainingControllerConfig) -> None:
        self.config = config
        self.hooks = TrainingController.HooksConfig(controller=self)

    def on_iteration_start(self, model: Model, step: int) -> None:
        self.__update_parameters(model, step)

    def on_iteration_done(self, model: Model, step: int) -> None:
        self.__update_parameters(model, step)

    def __update_parameters(self, model: Model, step: int):
        if step + 1 > self.config.rgb_only_iterations:
            if isinstance(model, RuntimeModelParameters):
                params: RuntimeModelParameters = model
                if not params.enable_embeddings:
                    CONSOLE.log("Enabling embeddings")
                    params.enable_embeddings = True
        else:
            if isinstance(model, RuntimeModelParameters):
                params: RuntimeModelParameters = model
                if params.enable_embeddings:
                    CONSOLE.log("Disabling embeddings")
                    params.enable_embeddings = False


class RuntimeModelParameters(ABC):
    __enable_embeddings: bool = True

    @property
    def enable_embeddings(self) -> bool:
        return self.__enable_embeddings

    @enable_embeddings.setter
    def enable_embeddings(self, value: bool) -> None:
        self.__enable_embeddings = value
