from dataclasses import dataclass

from rvs.pipeline.pipeline import PipelineConfig


@dataclass
class PipelineSpecification:
    config: PipelineConfig
    """Pipeline configuration"""

    description: str
    """Description shown in `rvs` help"""
