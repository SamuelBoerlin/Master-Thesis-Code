from dataclasses import dataclass, field
from typing import Any, Type

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.models.base_model import Model, ModelConfig
from torch import nn


@dataclass
class WrapperHooksConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: WrapperHooks)


@dataclass
class WrapperModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: WrapperModel)

    wrapper_model: ModelConfig = field(default_factory=lambda: ModelConfig)

    wrapper_hooks: WrapperHooksConfig = field(default_factory=lambda: WrapperHooksConfig())

    def setup(self, **kwargs) -> Any:
        return super().setup(
            wrapper_model=self.wrapper_model.setup(**kwargs), wrapper_hooks=self.wrapper_hooks.setup(**kwargs)
        )


class WrapperHooks:
    config: WrapperHooksConfig
    wrapper_model: Model

    def __init__(self, config: WrapperHooksConfig, **kwargs) -> None:
        self.config = config


class WrapperModel(Model):
    wrapper_model: Model
    wrapper_hooks: WrapperHooks

    def __init__(self, _: WrapperModelConfig, wrapper_model: Model, wrapper_hooks: WrapperHooks) -> None:
        self.wrapper_model = wrapper_model
        self.wrapper_hooks = wrapper_hooks
        self.wrapper_hooks.wrapper_model = wrapper_model

    def __getattribute__(self, name):
        if name in ("wrapper_model", "wrapper_hooks"):
            return super(nn.Module, self).__getattribute__(name)
        if self.wrapper_hooks is not None and name != "__dict__":
            try:
                return getattr(self.wrapper_hooks, name)
            except AttributeError:
                pass
        return self.wrapper_model.__getattribute__(name)

    def __setattr__(self, name, value) -> Any:
        if name in ("wrapper_model", "wrapper_hooks"):
            super(nn.Module, self).__setattr__(name, value)
        else:
            self.wrapper_model.__setattr__(name, value)

    def __dir__(self):
        return dir(self.wrapper_model)

    def __call__(self, *args, **kwargs) -> Any:
        return self.wrapper_model(*args, **kwargs)

    def __getstate__(self):
        return self.wrapper_model.__getstate__()

    def __setstate__(self, state):
        return self.wrapper_model.__setstate__(state)
