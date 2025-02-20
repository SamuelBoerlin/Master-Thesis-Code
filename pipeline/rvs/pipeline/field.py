from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Type

import torch
from lerf.lerf import LERFModel
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.models.base_model import ModelConfig
from numpy.typing import NDArray
from torch import Tensor

from rvs.pipeline.embedding import (
    ClipAtScaleEmbeddingConfig,
    ClipEmbeddingConfig,
    DefaultEmbeddingTypes,
    DinoEmbeddingConfig,
    EmbeddingConfig,
)
from rvs.pipeline.model import WrapperModelConfig
from rvs.pipeline.state import PipelineState
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

    def init(
        self,
        pipeline_state: PipelineState,
        data_path: Optional[Path],
        output_dir: Optional[Path],
        load_from_checkpoint: bool = False,
        **kwargs,
    ) -> None:
        config = self.config

        if data_path:
            config = self.__replace_data(config, data_path)

        if output_dir:
            cache_dir = Path.joinpath(output_dir, "cache", pipeline_state.pipeline.config.model_file.name)
            cache_dir.mkdir(parents=True, exist_ok=True)
            config = self.__replace_cache_dir(config, cache_dir)

            output_dir = Path.joinpath(output_dir, "nerf")
            output_dir.mkdir(parents=True, exist_ok=True)
            config = self.__replace_output_dir(config, output_dir)

        if load_from_checkpoint:
            config = self.__set_load_from_checkpoint(config)

        self.controller = config.controller.setup()

        config = self.__replace_model(
            config,
            WrapperModelConfig(wrapper_model=config.trainer.pipeline.model, wrapper_hooks=self.controller.hooks),
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

    def __replace_output_dir(self, config: FieldConfig, output_dir: Path) -> FieldConfig:
        trainer = replace(config.trainer, output_dir=output_dir)
        config = replace(config, trainer=trainer)
        return config

    def __set_load_from_checkpoint(self, config: FieldConfig) -> FieldConfig:
        trainer = replace(config.trainer, load_dir=config.trainer.get_checkpoint_dir())
        config = replace(config, trainer=trainer)
        return config

    def __replace_model(self, config: FieldConfig, model: ModelConfig) -> FieldConfig:
        pipeline = replace(config.trainer.pipeline, model=model)
        trainer = replace(config.trainer, pipeline=pipeline)
        config = replace(config, trainer=trainer)
        return config

    def __replace_cache_dir(self, config: FieldConfig, cache_dir: Path) -> FieldConfig:
        datamanager = replace(config.trainer.pipeline.datamanager, cache_dir=cache_dir)
        pipeline = replace(config.trainer.pipeline, datamanager=datamanager)
        trainer = replace(config.trainer, pipeline=pipeline)
        config = replace(config, trainer=trainer)
        return config

    def train(self) -> None:
        self.trainer.train()
        self.trainer.shutdown()

    def sample(self, config: EmbeddingConfig, positions: NDArray) -> NDArray:
        if config.type == DefaultEmbeddingTypes.CLIP:
            clip_scale = 1.0

            if isinstance(config, ClipAtScaleEmbeddingConfig):
                config_with_scale: ClipAtScaleEmbeddingConfig = config
                clip_scale = config_with_scale.scale
            elif not isinstance(config, ClipEmbeddingConfig):
                raise ValueError(f"Unknown CLIP config type {str(config)}")

            # TODO: Proper typing
            lerf_model: LERFModel = self.trainer.pipeline.model

            # Convert positions np array to tensor
            positions: Tensor = self.__transform_positions(lerf_model, positions)

            # Sample hashgrid values
            x = self.__sample_hashgrid_values(lerf_model, positions)

            # Obtain embeddings for the given hashgrid values
            clip_scales = clip_scale * torch.ones((1, positions.shape[1], 1), device=positions.device)
            embeddings = lerf_model.lerf_field.clip_net(torch.cat([x, clip_scales.view(-1, 1)], dim=-1))

            # Normalize embeddings
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            # Convert embeddings tensor to np array
            embeddings = embeddings.view(-1, embeddings.shape[-1]).float()
            embeddings: NDArray = embeddings.cpu().data.numpy()

            return embeddings
        elif config.type == DefaultEmbeddingTypes.DINO:
            if not isinstance(config, DinoEmbeddingConfig):
                raise ValueError(f"Unknown DINO config type {str(config)}")

            # TODO: Proper typing
            lerf_model: LERFModel = self.trainer.pipeline.model

            # Convert positions np array to tensor
            positions: Tensor = self.__transform_positions(lerf_model, positions)

            # Sample hashgrid values
            x = self.__sample_hashgrid_values(lerf_model, positions)

            # Obtain embeddings for the given hashgrid values
            embeddings = lerf_model.lerf_field.dino_net(x)

            # Convert embeddings tensor to np array
            embeddings = embeddings.view(-1, embeddings.shape[-1]).float()
            embeddings: NDArray = embeddings.cpu().data.numpy()

            return embeddings
        else:
            raise ValueError(f"Unknown embedding config type {str(config)}")

    def __transform_positions(self, lerf_model: LERFModel, positions: NDArray) -> Tensor:
        # Convert positions np array to tensor
        positions: Tensor = torch.from_numpy(positions).to(lerf_model.device).reshape((1, -1, 3))

        # Transform
        positions = lerf_model.lerf_field.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0

        return positions

    def __sample_hashgrid_values(self, lerf_model: LERFModel, positions: Tensor) -> Tensor:
        # Sample hashgrid values
        xs = [e(positions.view(-1, 3)) for e in lerf_model.lerf_field.clip_encs]
        x = torch.concat(xs, dim=-1)
        return x
