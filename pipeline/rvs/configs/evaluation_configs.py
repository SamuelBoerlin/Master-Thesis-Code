from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tyro

from rvs.configs.pipeline_configs import (
    MethodDummyConfig,
    list_available_components,
    parse_pipeline_method,
    pipeline_method_format,
    pipeline_method_format_arg,
)
from rvs.evaluation.evaluation import EvaluationConfig, EvaluationResumeConfig
from rvs.pipeline.pipeline import PipelineConfig


def adapt_pipeline_config(config: PipelineConfig) -> PipelineConfig:
    config = replace(config)
    config.model_file = Path("<unknown>")
    return config


def parse_evaluation_method(method: str) -> Tuple[EvaluationConfig, str]:
    pipeline_config, pipeline_description = parse_pipeline_method(method)
    evaluation_config = EvaluationConfig(
        pipeline=adapt_pipeline_config(pipeline_config),
    )
    return evaluation_config, pipeline_description


def setup_evaluation_tyro_union(args: List[str]) -> Any:
    evaluation_configs: Dict[str, Any] = dict()
    evaluation_descriptions: Dict[str, str] = dict()

    evaluation_configs["help"] = evaluation_configs[pipeline_method_format_arg()] = MethodDummyConfig()
    evaluation_descriptions["help"] = ""
    evaluation_descriptions[pipeline_method_format_arg()] = (
        f"Runs evaluation for a pipeline {pipeline_method_format()}\n\n{list_available_components()}"
    )

    evaluation_descriptions["resume"] = "Resume evaluation"
    evaluation_configs["resume"] = EvaluationResumeConfig(config=tyro.MISSING)

    if len(args) > 1:
        subcommand = args[1]

        if subcommand not in evaluation_configs and "." in subcommand:
            config, description = parse_evaluation_method(subcommand)

            evaluation_configs[subcommand] = config
            evaluation_descriptions[subcommand] = description

    assert len(evaluation_configs) == len(evaluation_descriptions)

    return tyro.conf.SuppressFixed[
        tyro.conf.FlagConversionOff[
            tyro.extras.subcommand_type_from_defaults(defaults=evaluation_configs, descriptions=evaluation_descriptions)
        ]
    ]
