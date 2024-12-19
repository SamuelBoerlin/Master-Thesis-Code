from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Type

import torch
from lerf.lerf import LERFModel
from numpy.typing import NDArray
from torch import Tensor

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.models.base_model import ModelConfig
from rvs.pipeline.model import WrapperModelConfig
from rvs.pipeline.training_controller import TrainingController, TrainingControllerConfig


@dataclass
class FieldConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Field)
    """target class to instantiate"""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    """Configuration of NERF trainer."""

    controller: TrainingControllerConfig = field(default_factory=TrainingControllerConfig)
    """Configuration of the NERF training controller that handles e.g. multiple training phases like rgb -> embeddings"""


class Field:
    config: FieldConfig
    trainer: Trainer
    controller: TrainingController

    def __init__(self, config: FieldConfig) -> None:
        self.config = config

    def init(self, data_path: Optional[Path], **kwargs) -> None:
        config = self.config
        if data_path:
            config = self.__replace_data(config, data_path)
        self.controller = self.config.controller.setup()
        config = self.__replace_model(
            config,
            WrapperModelConfig(wrapper_model=self.config.trainer.pipeline.model, wrapper_hooks=self.controller.hooks),
        )
        self.trainer = config.trainer.setup(**kwargs)
        self.trainer.config.save_config()
        self.trainer.setup()

    def __replace_data(self, config: FieldConfig, data: Path) -> FieldConfig:
        datamanager = replace(config.trainer.pipeline.datamanager, data=data)
        pipeline = replace(config.trainer.pipeline, datamanager=datamanager)
        trainer = replace(config.trainer, pipeline=pipeline)
        config = replace(config, trainer=trainer)
        return config

    def __replace_model(self, config: FieldConfig, model: ModelConfig) -> ModelConfig:
        pipeline = replace(config.trainer.pipeline, model=model)
        trainer = replace(config.trainer, pipeline=pipeline)
        config = replace(config, trainer=trainer)
        return config

    def train(self) -> None:
        # FIXME: LERF model caches clip embeddings under outputs/transforms.json/ which means they get mixed up
        self.trainer.train()
        self.trainer.shutdown()

    def sample(self, positions: NDArray) -> NDArray:
        # TODO: Proper typing
        lerf_model: LERFModel = self.trainer.pipeline.model

        # Convert positions np array to tensor
        positions: Tensor = torch.from_numpy(positions).to(lerf_model.device).reshape((1, -1, 3))

        positions = lerf_model.lerf_field.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0

        # Sample hashgrid values
        xs = [e(positions.view(-1, 3)) for e in lerf_model.lerf_field.clip_encs]
        x = torch.concat(xs, dim=-1)

        # TODO: Simply using 1.0 for the scale at the moment
        clip_scales = torch.ones((1, positions.shape[1], 1), device=positions.device)

        # Obtain embeddings for the given hashgrid values
        embeddings = lerf_model.lerf_field.clip_net(torch.cat([x, clip_scales.view(-1, 1)], dim=-1))

        # Normalize embeddings
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        # Convert embeddings tensor to np array
        embeddings = embeddings.view(-1, embeddings.shape[-1]).float()
        embeddings: NDArray = embeddings.cpu().data.numpy()

        return embeddings

    # TODO: Add method for sampling embeddings from field
