from dataclasses import dataclass, field
from typing import Dict, Type

import numpy as np
from nerfstudio.configs.base_config import InstantiateConfig
from numpy.typing import NDArray

from rvs.pipeline.state import PipelineState


@dataclass
class TransformConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Transform)
    """target class to instantiate"""


class Transform:
    config: TransformConfig

    def __init__(self, config: TransformConfig) -> None:
        self.config = config

    def create(self, samples: NDArray, pipeline_state: PipelineState) -> Dict[str, NDArray]:
        pass

    def apply(self, samples: NDArray, parameters: Dict[str, NDArray]) -> NDArray:
        pass

    def get_output_dimension(parameters: Dict[str, NDArray]) -> int:
        pass


@dataclass
class IdentityTransformConfig(TransformConfig):
    _target: Type = field(default_factory=lambda: IdentityTransform)


class IdentityTransform(Transform):
    def create(self, samples: NDArray, pipeline_state: PipelineState) -> Dict[str, NDArray]:
        return {"dim": np.array([samples.shape[1]])}

    def apply(self, samples: NDArray, parameters: Dict[str, NDArray]) -> NDArray:
        if "dim" not in parameters:
            raise ValueError("Missing dim parameter")

        dim: int = parameters["dim"][0]

        input_dim = samples.shape[1]

        if input_dim != dim:
            raise ValueError(f"Invalid input dimension {input_dim}, expected {dim}")

        return samples

    def get_output_dimension(parameters: Dict[str, NDArray]) -> int:
        if "dim" not in parameters:
            raise ValueError("Missing dim parameter")

        return parameters["dim"][0]
