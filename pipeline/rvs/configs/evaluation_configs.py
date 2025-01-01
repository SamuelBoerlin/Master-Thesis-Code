from dataclasses import replace
from typing import Dict

import tyro

from rvs.configs.pipeline_configs import pipeline_configs
from rvs.evaluation.evaluation import EvaluationConfig
from rvs.pipeline.pipeline import PipelineConfig


def adapt_pipeline_config(config: PipelineConfig) -> PipelineConfig:
    config = replace(config)
    config.render_sample_positions_of_views = [11]
    config.render_sample_clusters_of_views = [11]
    config.render_selected_views = True
    return config


evaluation_configs: Dict[str, EvaluationConfig] = {}
evaluation_descriptions = {
    "default": "Default model.",
    "default-lite": "Default big model.",
    "default-big": "Default lite model.",
}

evaluation_configs["default"] = EvaluationConfig(
    pipeline=adapt_pipeline_config(pipeline_configs["default"]),
)

evaluation_configs["default-lite"] = EvaluationConfig(
    pipeline=adapt_pipeline_config(pipeline_configs["default-lite"]),
)

evaluation_configs["default-big"] = EvaluationConfig(
    pipeline=adapt_pipeline_config(pipeline_configs["default-big"]),
)

all_methods, all_descriptions = evaluation_configs, evaluation_descriptions

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=all_methods, descriptions=all_descriptions)
    ]
]
