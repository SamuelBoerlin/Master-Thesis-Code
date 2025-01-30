from dataclasses import replace
from typing import Any, Dict

import tyro

from rvs.configs.pipeline_configs import pipeline_configs, pipeline_descriptions
from rvs.evaluation.evaluation import EvaluationConfig, EvaluationResumeConfig
from rvs.pipeline.pipeline import PipelineConfig


def adapt_pipeline_config(config: PipelineConfig) -> PipelineConfig:
    config = replace(config)
    # config.render_sample_positions_of_views = [11]
    # config.render_sample_clusters_of_views = [11]
    # config.render_selected_views = True
    return config


evaluation_configs: Dict[str, Any] = dict()
evaluation_descriptions: Dict[str, str] = dict()

for name, config in pipeline_configs.items():
    evaluation_descriptions[name] = pipeline_descriptions[name]
    evaluation_configs[name] = EvaluationConfig(
        pipeline=adapt_pipeline_config(config),
    )

evaluation_descriptions["resume"] = "Resume evaluation"
evaluation_configs["resume"] = EvaluationResumeConfig(config=None)

all_methods, all_descriptions = evaluation_configs, evaluation_descriptions

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=all_methods, descriptions=all_descriptions)
    ]
]
