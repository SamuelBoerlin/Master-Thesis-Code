from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Tuple

import tyro
from lerf.lerf_config import lerf_method, lerf_method_big, lerf_method_lite
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.engine.trainer import TrainerConfig
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from tyro._argparse_formatter import THEME

from rvs.lerf.lerf_datamanager import CustomLERFDataManagerConfig
from rvs.lerf.lerf_model import CustomLERFModelConfig
from rvs.pipeline.clustering import (
    ClosestElbowKMeansClusteringConfig,
    FractionalElbowKMeansClusteringConfig,
    KMeansClusteringConfig,
    LargestKMeansClusteringConfig,
    XMeansClusteringConfig,
)
from rvs.pipeline.embedding import ClipAtRandomScaleEmbeddingConfig, ClipAtScaleEmbeddingConfig, DinoEmbeddingConfig
from rvs.pipeline.pipeline import FieldConfig, PipelineConfig
from rvs.pipeline.renderer import BlenderRendererConfig, PyrenderRendererConfig, TrimeshRendererConfig
from rvs.pipeline.sampler import (
    BinarySearchDensityTrimeshPositonSamplerConfig,
    FarthestPointSamplingDensityTrimeshPositonSamplerConfig,
    MinDistanceTrimeshPositionSamplerConfig,
)
from rvs.pipeline.selection import MostSimilarToCentroidTrainingViewSelectionConfig, SpatialViewSelectionConfig
from rvs.pipeline.stage import PipelineStage
from rvs.pipeline.transform import FixedPCATransformConfig, IdentityTransformConfig, VariancePCATransformConfig
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
        "rand_clip": ("CLIP embeddings only at random scale", (ClipAtRandomScaleEmbeddingConfig(),)),
        "rand_clip_and_dino": (
            "Both CLIP at random scale (default) and DINO embeddings",
            (ClipAtRandomScaleEmbeddingConfig(), DinoEmbeddingConfig()),
        ),
    },
    PipelineStage.TRAIN_FIELD: {
        "lerf_standard_field": ("LERF standard method", FieldConfig(trainer=adapt_lerf_config(lerf_method.config))),
        "lerf_lite_field": ("LERF lite method", FieldConfig(trainer=adapt_lerf_config(lerf_method_lite.config))),
        "lerf_big_field": ("LERF big method", FieldConfig(trainer=adapt_lerf_config(lerf_method_big.config))),
    },
    PipelineStage.SAMPLE_POSITIONS: {
        "min_distance_sampler": (
            "Uniform geometry-based position sampler where samples are decimated until a minimum distance is reached",
            MinDistanceTrimeshPositionSamplerConfig(),
        ),
        "binary_search_density_sampler": (
            "Uniform geometry-based position sampler that attempts to sample with a given density per surface area by performing a binary search over minimum distance of min_distance_sampler",
            BinarySearchDensityTrimeshPositonSamplerConfig(),
        ),
        "fps_density_sampler": (
            "Uniform geometry-based position sampler that samples with a given density per surface area using Farthest Point Sampling",
            FarthestPointSamplingDensityTrimeshPositonSamplerConfig(),
        ),
    },
    PipelineStage.TRANSFORM_EMBEDDINGS: {
        "identity_transform": (
            "Identity transform, i.e. embeddings are not changed at all",
            IdentityTransformConfig(),
        ),
        "fixed_pca_transform": (
            "PCA transform with fixed number of principal components",
            FixedPCATransformConfig(),
        ),
        "variance_pca_transform": (
            "PCA transform with variable number of principal components based on the amount of explained variance",
            VariancePCATransformConfig(),
        ),
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
        "xmeans_bic_clustering": ("X-Means clustering using Bayesian Information Criterion", XMeansClusteringConfig()),
        "largest_kmeans_clustering": (
            "Largest n clusters of fixed-k KMeans clustering",
            LargestKMeansClusteringConfig(implementation=KMeansClusteringConfig()),
        ),
        "largest_frac_elbow_kmeans_clustering": (
            "Largest n clusters of Variable-k KMeans clustering with elbow method where k is selected based on a fraction of the distortion range",
            LargestKMeansClusteringConfig(implementation=FractionalElbowKMeansClusteringConfig()),
        ),
        "largest_closest_elbow_kmeans_clustering": (
            "Largest n clusters of variable-k KMeans clustering with elbow method where k is selected by the closest point to the origin",
            LargestKMeansClusteringConfig(implementation=ClosestElbowKMeansClusteringConfig()),
        ),
        "largest_xmeans_bic_clustering": (
            "Largest n clusters of X-Means clustering using Bayesian Information Criterion",
            LargestKMeansClusteringConfig(implementation=XMeansClusteringConfig()),
        ),
    },
    PipelineStage.SELECT_VIEWS: {
        "best_training_selection": (
            "Views selected from training views that are most similar to cluster embeddings",
            MostSimilarToCentroidTrainingViewSelectionConfig(),
        ),
        "spatial_view_selection": (
            "Views are selected based on the spatial positions of the clustered embeddings samples",
            SpatialViewSelectionConfig(),
        ),
    },
}


component_stages: List[Tuple[PipelineStage, str]] = [
    (PipelineStage.SAMPLE_VIEWS, "<views>"),
    (PipelineStage.RENDER_VIEWS, "<renderer>"),
    (PipelineStage.SAMPLE_EMBEDDINGS, "<embedding>"),
    (PipelineStage.TRAIN_FIELD, "<field>"),
    (PipelineStage.SAMPLE_POSITIONS, "<sampler>"),
    (PipelineStage.TRANSFORM_EMBEDDINGS, "<transform>"),
    (PipelineStage.CLUSTER_EMBEDDINGS, "<clustering>"),
    (PipelineStage.SELECT_VIEWS, "<selection>"),
]


def pipeline_method_format() -> str:
    return ".".join([slug for _, slug in component_stages])


def pipeline_method_format_arg() -> str:
    return pipeline_method_format().replace("<", "").replace(">", "")


def parse_pipeline_method(method: str) -> Tuple[PipelineConfig, str]:
    components = method.split(".")

    if len(components) != len(component_stages):
        raise ValueError(
            f"Invalid number of components {len(components)}, expected {len(component_stages)}: {pipeline_method_format()}"
        )

    component_configs: Dict[PipelineStage, InstantiateConfig] = dict()

    pipeline_description = ""

    for i, component in enumerate(components):
        stage, slug = component_stages[i]

        available_components = pipeline_components[stage]

        if component not in available_components:
            console = Console(theme=THEME.as_rich_theme(), stderr=True)
            console.print(
                Panel(
                    Group(
                        f"Invalid [bold]{slug}[/bold] pipeline component '{component}', expected one of: [bold]"
                        + ", ".join(sorted(available_components.keys()))
                        + "[/bold]",
                        Rule(style=Style(color="red")),
                        f"For full helptext, run [bold]{pipeline_method_format_arg()}[/bold]",
                    ),
                    title="[bold]Invalid pipeline[/bold]",
                    title_align="left",
                    border_style=Style(color="bright_red"),
                    expand=False,
                )
            )
            sys.exit(2)

        component_description, component_config = available_components[component]

        component_configs[stage] = component_config

        pipeline_description += f"{i + 1}. {component_description}\n"

    pipeline_config = PipelineConfig(
        model_file=tyro.MISSING,
        method_name=method,
        views=component_configs[PipelineStage.SAMPLE_VIEWS],
        renderer=component_configs[PipelineStage.RENDER_VIEWS],
        embeddings=component_configs[PipelineStage.SAMPLE_EMBEDDINGS],
        field=component_configs[PipelineStage.TRAIN_FIELD],
        sampler=component_configs[PipelineStage.SAMPLE_POSITIONS],
        transform=component_configs[PipelineStage.TRANSFORM_EMBEDDINGS],
        clustering=component_configs[PipelineStage.CLUSTER_EMBEDDINGS],
        selection=component_configs[PipelineStage.SELECT_VIEWS],
    )

    return pipeline_config, pipeline_description


def list_available_components() -> str:
    text = ""

    for stage, slug in component_stages:
        if text != "":
            text += "\n"

        text += f"{slug}:\n"

        for component, (description, _) in sorted(pipeline_components[stage].items(), key=lambda t: t[0]):
            text += f" {component}: {description}\n"

    return text


@dataclass
class MethodDummyConfig:
    pass


def setup_pipeline_tyro_union(args: List[str]) -> Any:
    pipeline_configs: Dict[str, PipelineConfig] = dict()
    pipeline_descriptions: Dict[str, str] = dict()

    pipeline_configs["help"] = pipeline_configs[pipeline_method_format_arg()] = MethodDummyConfig()
    pipeline_descriptions["help"] = ""
    pipeline_descriptions[pipeline_method_format_arg()] = (
        f"Runs a pipeline {pipeline_method_format()}\n\n{list_available_components()}"
    )

    if len(args) > 1:
        subcommand = args[1]

        if subcommand not in pipeline_configs and "." in subcommand:
            config, description = parse_pipeline_method(subcommand)

            pipeline_configs[subcommand] = config
            pipeline_descriptions[subcommand] = description

    assert len(pipeline_configs) == len(pipeline_descriptions)

    return tyro.conf.SuppressFixed[
        tyro.conf.FlagConversionOff[
            tyro.extras.subcommand_type_from_defaults(defaults=pipeline_configs, descriptions=pipeline_descriptions)
        ]
    ]
