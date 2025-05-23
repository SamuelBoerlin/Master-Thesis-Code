from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Type

import numpy as np
import torch
from lerf.lerf import LERFModel
from lerf.lerf_pipeline import LERFPipeline
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.models.base_model import ModelConfig
from numpy.typing import NDArray
from torch import Tensor

from rvs.pipeline.embedding import (
    ClipAtRandomScaleEmbeddingConfig,
    ClipAtScaleEmbeddingConfig,
    ClipEmbeddingConfig,
    DefaultEmbeddingTypes,
    DinoEmbeddingConfig,
    EmbeddingConfig,
)
from rvs.pipeline.model import WrapperModelConfig
from rvs.pipeline.state import PipelineState
from rvs.pipeline.training_controller import TrainingController, TrainingControllerConfig
from rvs.pipeline.training_tracker import LocalWriterShimConfig, TrainingTrackerConfig
from rvs.utils.nerfstudio import transform_to_ns_field_space


@dataclass
class FieldConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Field)
    """target class to instantiate"""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    """Configuration of NERF trainer."""

    controller: TrainingControllerConfig = field(default_factory=TrainingControllerConfig)
    """Configuration of the NERF training controller that handles e.g. multiple training phases like rgb -> embeddings"""

    tracking: TrainingTrackerConfig = field(default_factory=TrainingTrackerConfig)
    """Configuration for NERF training tracking, i.e. saving training progress, losses, etc."""


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

            scratch_dir = Path.joinpath(output_dir, "scratch")
            scratch_dir.mkdir(parents=True, exist_ok=True)

            tracking_dir = scratch_dir / "tracking"
            tracking_dir.mkdir(parents=True, exist_ok=True)

            config = self.__configure_tracking(config, tracking_dir)

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

    def __configure_tracking(self, config: FieldConfig, tracking_dir: Path) -> FieldConfig:
        if config.tracking is not None:
            tracker = config.tracking
            if tracker.output_dir is None:
                tracker = replace(tracker, output_dir=tracking_dir)

            config = replace(config, tracking=tracker)

            logging = replace(
                config.trainer.logging,
                local_writer=LocalWriterShimConfig(
                    enable=True,
                    parent=config.trainer.logging.local_writer,
                    config=tracker,
                ),
            )
            trainer = replace(config.trainer, logging=logging)
            config = replace(config, trainer=trainer)
        return config

    def train(self) -> None:
        self.trainer.train()
        self.trainer.shutdown()

    def sample(self, config: EmbeddingConfig, positions: NDArray, pipeline_state: PipelineState) -> NDArray:
        if config.type == DefaultEmbeddingTypes.CLIP:
            assert isinstance(self.trainer.pipeline, LERFPipeline)
            lerf_pipeline: LERFPipeline = self.trainer.pipeline
            lerf_model: LERFModel = lerf_pipeline.model

            # Convert positions np array to tensor
            positions: Tensor = self.__transform_positions(lerf_pipeline, positions)

            # Sample hashgrid values
            x = self.__sample_hashgrid_values(lerf_model, positions)

            clip_scales = None
            if isinstance(config, ClipEmbeddingConfig):
                clip_scales = torch.ones((1, positions.shape[1], 1), device=positions.device)
            if isinstance(config, ClipAtScaleEmbeddingConfig):
                config_with_scale: ClipAtScaleEmbeddingConfig = config
                clip_scales = config_with_scale.scale * torch.ones((1, positions.shape[1], 1), device=positions.device)
            elif isinstance(config, ClipAtRandomScaleEmbeddingConfig):
                config_with_random_scale: ClipAtRandomScaleEmbeddingConfig = config
                random_values = (
                    np.random.default_rng(pipeline_state.pipeline.config.machine.seed).random(
                        (1, positions.shape[1], 1)
                    )
                    * (config_with_random_scale.max_scale - config_with_random_scale.min_scale)
                    + config_with_random_scale.min_scale
                )
                clip_scales = torch.from_numpy(random_values).to(positions.device)
            elif not isinstance(config, ClipEmbeddingConfig):
                raise ValueError(f"Unknown CLIP config type {str(config)}")
            assert clip_scales is not None

            # Obtain embeddings for the given hashgrid values
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

            assert isinstance(self.trainer.pipeline, LERFPipeline)
            lerf_pipeline: LERFPipeline = self.trainer.pipeline
            lerf_model: LERFModel = lerf_pipeline.model

            # Convert positions np array to tensor
            positions: Tensor = self.__transform_positions(lerf_pipeline, positions)

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

    def __transform_positions(self, lerf_pipeline: LERFPipeline, positions: NDArray) -> Tensor:
        lerf_model: LERFModel = lerf_pipeline.model
        datamanager: VanillaDataManager = lerf_pipeline.datamanager

        # Convert positions np array to tensor
        positions: Tensor = torch.from_numpy(positions.copy()).to(lerf_model.device).reshape((1, -1, 3))

        # Transform to NS field space
        positions = transform_to_ns_field_space(positions, datamanager.train_dataparser_outputs)

        # Transform to LERF field space
        positions = lerf_model.lerf_field.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0

        return positions

    def __sample_hashgrid_values(self, lerf_model: LERFModel, positions: Tensor) -> Tensor:
        # Sample hashgrid values
        xs = [e(positions.view(-1, 3)) for e in lerf_model.lerf_field.clip_encs]
        x = torch.concat(xs, dim=-1)
        return x
