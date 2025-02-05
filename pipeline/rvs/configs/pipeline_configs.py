from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import Dict

import tyro
from lerf.lerf_config import lerf_method, lerf_method_big, lerf_method_lite
from nerfstudio.engine.trainer import TrainerConfig

from rvs.lerf.lerf_datamanager import CustomLERFDataManagerConfig
from rvs.lerf.lerf_model import CustomLERFModelConfig
from rvs.pipeline.clustering import ElbowKMeansClusteringConfig, KMeansClusteringConfig
from rvs.pipeline.pipeline import FieldConfig, PipelineConfig
from rvs.pipeline.renderer import TrimeshRendererConfig
from rvs.pipeline.sampler import TrimeshPositionSamplerConfig
from rvs.pipeline.selection import BestTrainingViewSelectionConfig
from rvs.pipeline.views import FermatSpiralViewsConfig, SphereViewsConfig
from rvs.utils.dataclasses import extend_dataclass_obj


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


def replace_lerf_datamanager(config: TrainerConfig) -> TrainerConfig:
    pipeline = replace(
        config.pipeline,
        datamanager=extend_dataclass_obj(config.pipeline.datamanager, CustomLERFDataManagerConfig),
    )
    config = replace(config, pipeline=pipeline)
    return config


def replace_lerf_model(config: TrainerConfig) -> TrainerConfig:
    pipeline = replace(
        config.pipeline,
        model=extend_dataclass_obj(config.pipeline.model, CustomLERFModelConfig),
    )
    config = replace(config, pipeline=pipeline)
    return config


def adapt_lerf_config(config: TrainerConfig) -> TrainerConfig:
    config = replace_lerf_datamanager(config)
    config = replace_lerf_model(config)
    config = set_default_trainer_params(config)
    return config


pipeline_configs: Dict[str, PipelineConfig] = {}
pipeline_descriptions = {
    "default": "Default model.",
    "default-lite": "Default big model.",
    "default-big": "Default lite model.",
    "elbow_kmeans": "Default model with elbow kmeans.",
    "fermat_views": "Default model with fermat spiral views.",
}

pipeline_configs["default"] = PipelineConfig(
    method_name="default",
    views=SphereViewsConfig(),
    renderer=TrimeshRendererConfig(),
    field=FieldConfig(trainer=adapt_lerf_config(lerf_method.config)),
    sampler=TrimeshPositionSamplerConfig(),
    clustering=KMeansClusteringConfig(),
    selection=BestTrainingViewSelectionConfig(),
)

pipeline_configs["default-lite"] = PipelineConfig(
    method_name="default-lite",
    views=SphereViewsConfig(),
    renderer=TrimeshRendererConfig(),
    field=FieldConfig(trainer=adapt_lerf_config(lerf_method_lite.config)),
    sampler=TrimeshPositionSamplerConfig(),
    clustering=KMeansClusteringConfig(),
    selection=BestTrainingViewSelectionConfig(),
)

pipeline_configs["default-big"] = PipelineConfig(
    method_name="default-big",
    views=SphereViewsConfig(),
    renderer=TrimeshRendererConfig(),
    field=FieldConfig(trainer=adapt_lerf_config(lerf_method_big.config)),
    sampler=TrimeshPositionSamplerConfig(),
    clustering=KMeansClusteringConfig(),
    selection=BestTrainingViewSelectionConfig(),
)

pipeline_configs["elbow_kmeans"] = PipelineConfig(
    method_name="elbow_kmeans",
    views=SphereViewsConfig(),
    renderer=TrimeshRendererConfig(),
    field=FieldConfig(trainer=adapt_lerf_config(lerf_method.config)),
    sampler=TrimeshPositionSamplerConfig(),
    clustering=ElbowKMeansClusteringConfig(),
    selection=BestTrainingViewSelectionConfig(),
)

pipeline_configs["fermat_views"] = PipelineConfig(
    method_name="fermat_views",
    views=FermatSpiralViewsConfig(),
    renderer=TrimeshRendererConfig(),
    field=FieldConfig(trainer=adapt_lerf_config(lerf_method.config)),
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
