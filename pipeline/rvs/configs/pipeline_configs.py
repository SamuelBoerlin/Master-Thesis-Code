from __future__ import annotations

import dataclasses
from typing import Dict

import tyro
from lerf.lerf_config import lerf_method, lerf_method_big, lerf_method_lite

from nerfstudio.engine.trainer import TrainerConfig
from rvs.pipeline.clustering import KMeansClusteringConfig
from rvs.pipeline.pipeline import FieldConfig, PipelineConfig
from rvs.pipeline.renderer import TrimeshRendererConfig
from rvs.pipeline.sampler import TrimeshPositionSamplerConfig
from rvs.pipeline.selection import BestTrainingViewSelectionConfig
from rvs.pipeline.views import SphereViewsConfig


def set_default_trainer_params(config: TrainerConfig) -> TrainerConfig:
    config = dataclasses.replace(config)
    # FIXME: For testing
    # config.max_num_iterations = 10
    # config.max_num_iterations = 200
    config.max_num_iterations = 2000
    config.pipeline.model.camera_optimizer.mode = "off"
    config.pipeline.model.background_color = "random"
    config.pipeline.model.disable_scene_contraction = True
    config.viewer.quit_on_train_completion = True
    return config


pipeline_configs: Dict[str, PipelineConfig] = {}
pipeline_descriptions = {
    "default": "Default model.",
    "default-lite": "Default big model.",
    "default-big": "Default lite model.",
}

pipeline_configs["default"] = PipelineConfig(
    method_name="default",
    views=SphereViewsConfig(),
    renderer=TrimeshRendererConfig(),
    field=FieldConfig(trainer=set_default_trainer_params(lerf_method.config)),
    sampler=TrimeshPositionSamplerConfig(),
    clustering=KMeansClusteringConfig(),
    selection=BestTrainingViewSelectionConfig(),
)

pipeline_configs["default-lite"] = PipelineConfig(
    method_name="default-lite",
    views=SphereViewsConfig(),
    renderer=TrimeshRendererConfig(),
    field=FieldConfig(trainer=set_default_trainer_params(lerf_method_lite.config)),
    sampler=TrimeshPositionSamplerConfig(),
    clustering=KMeansClusteringConfig(),
    selection=BestTrainingViewSelectionConfig(),
)

pipeline_configs["default-big"] = PipelineConfig(
    method_name="default-big",
    views=SphereViewsConfig(),
    renderer=TrimeshRendererConfig(),
    field=FieldConfig(trainer=set_default_trainer_params(lerf_method_big.config)),
    sampler=TrimeshPositionSamplerConfig(),
    clustering=KMeansClusteringConfig(),
    selection=BestTrainingViewSelectionConfig(),
)

all_methods, all_descriptions = pipeline_configs, pipeline_descriptions

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=all_methods, descriptions=all_descriptions)
    ]
]
