from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import Dict, Tuple

import tyro
from lerf.lerf_config import lerf_method, lerf_method_big, lerf_method_lite
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.engine.trainer import TrainerConfig

from rvs.lerf.lerf_datamanager import CustomLERFDataManagerConfig
from rvs.lerf.lerf_model import CustomLERFModelConfig
from rvs.pipeline.clustering import (
    ClosestElbowKMeansClusteringConfig,
    FractionalElbowKMeansClusteringConfig,
    KMeansClusteringConfig,
)
from rvs.pipeline.embedding import ClipAtScaleEmbeddingConfig, DinoEmbeddingConfig
from rvs.pipeline.pipeline import FieldConfig, PipelineConfig, PipelineStage
from rvs.pipeline.renderer import BlenderRendererConfig, PyrenderRendererConfig, TrimeshRendererConfig
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


pipeline_components: Dict[PipelineStage, Dict[str, Tuple[str, InstantiateConfig]]] = {
    PipelineStage.SAMPLE_VIEWS: {
        "altaz_views": ("Altitude/Azimuth spherical views", SphereViewsConfig()),
        "fermat_views": ("Fermat spiral spherical views", FermatSpiralViewsConfig()),
    },
    PipelineStage.RENDER_VIEWS: {
        "pyrender_renderer": ("Pyrenderer", PyrenderRendererConfig()),
        "trimesh_renderer": ("Trimesh", TrimeshRendererConfig()),
        "blender_renderer": ("Blender with Objaverse scene parameters", BlenderRendererConfig()),
    },
    PipelineStage.SAMPLE_EMBEDDINGS: {
        "clip": ("CLIP embeddings only", (ClipAtScaleEmbeddingConfig(),)),
        "clip_and_dino": (
            "Both CLIP (default) and DINO embeddings",
            (ClipAtScaleEmbeddingConfig(), DinoEmbeddingConfig()),
        ),
    },
    PipelineStage.TRAIN_FIELD: {
        "lerf_standard_field": ("LERF standard method", FieldConfig(trainer=adapt_lerf_config(lerf_method.config))),
        "lerf_lite_field": ("LERF lite method", FieldConfig(trainer=adapt_lerf_config(lerf_method_lite.config))),
        "lerf_big_field": ("LERF big method", FieldConfig(trainer=adapt_lerf_config(lerf_method_big.config))),
    },
    PipelineStage.SAMPLE_POSITIONS: {
        "trimesh_sampler": ("Random face area weighted position sampler with Trimesh", TrimeshPositionSamplerConfig()),
    },
    PipelineStage.CLUSTER_EMBEDDINGS: {
        "kmeans_clustering": ("Fixed-k KMeans clustering", KMeansClusteringConfig()),
        "frac_elbow_kmeans_clustering": (
            "Variable-k KMeans clustering with elbow method where k is selected based on a fraction of the distortion range",
            FractionalElbowKMeansClusteringConfig(),
        ),
        "closest_elbow_kmeans_clustering": (
            "Variable-k KMeans clustering with elbow method where k is selected by the closest point to the origin",
            ClosestElbowKMeansClusteringConfig(),
        ),
    },
    PipelineStage.SELECT_VIEWS: {
        "best_training_selection": (
            "Views selected from training views that are most similar to cluster embeddings",
            BestTrainingViewSelectionConfig(),
        ),
    },
}

pipeline_configs: Dict[str, PipelineConfig] = dict()
pipeline_descriptions: Dict[str, str] = dict()

for views_method, (views_description, views_config) in pipeline_components[PipelineStage.SAMPLE_VIEWS].items():
    for renderer_method, (renderer_description, renderer_config) in pipeline_components[
        PipelineStage.RENDER_VIEWS
    ].items():
        for embeddings_method, (embeddings_description, embeddings_configs) in pipeline_components[
            PipelineStage.SAMPLE_EMBEDDINGS
        ].items():
            for field_method, (field_description, field_config) in pipeline_components[
                PipelineStage.TRAIN_FIELD
            ].items():
                for sampler_method, (sampler_description, sampler_config) in pipeline_components[
                    PipelineStage.SAMPLE_POSITIONS
                ].items():
                    for clustering_method, (clustering_description, clustering_config) in pipeline_components[
                        PipelineStage.CLUSTER_EMBEDDINGS
                    ].items():
                        for selection_method, (selection_description, selection_config) in pipeline_components[
                            PipelineStage.SELECT_VIEWS
                        ].items():
                            method = ".".join(
                                [
                                    views_method,
                                    renderer_method,
                                    embeddings_method,
                                    field_method,
                                    sampler_method,
                                    clustering_method,
                                    selection_method,
                                ]
                            )

                            description = (
                                f"1. {views_description}\n"
                                f"2. {renderer_description}\n"
                                f"3. {embeddings_description}\n"
                                f"4. {field_description}\n"
                                f"5. {sampler_description}\n"
                                f"6. {clustering_description}\n"
                                f"7. {selection_description}\n"
                            )

                            config = PipelineConfig(
                                method_name=method,
                                views=views_config,
                                renderer=renderer_config,
                                embeddings=embeddings_configs,
                                field=field_config,
                                sampler=sampler_config,
                                clustering=clustering_config,
                                selection=selection_config,
                            )

                            pipeline_configs[method] = config
                            pipeline_descriptions[method] = description

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=pipeline_configs, descriptions=pipeline_descriptions)
    ]
]
