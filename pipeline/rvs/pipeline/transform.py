import json
from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from nerfstudio.configs.base_config import InstantiateConfig
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from rvs.pipeline.state import PipelineState
from rvs.utils.plot import save_figure


@dataclass
class TransformConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Transform)
    """target class to instantiate"""


class Transform:
    config: TransformConfig

    def __init__(self, config: TransformConfig) -> None:
        self.config = config

    def create(self, samples: NDArray, sample_type: str, pipeline_state: PipelineState) -> Dict[str, NDArray]:
        pass

    def apply(self, samples: NDArray, parameters: Dict[str, NDArray]) -> NDArray:
        pass

    def get_output_dimension(self, parameters: Dict[str, NDArray]) -> int:
        pass


@dataclass
class IdentityTransformConfig(TransformConfig):
    _target: Type = field(default_factory=lambda: IdentityTransform)


class IdentityTransform(Transform):
    def create(self, samples: NDArray, sample_type: str, pipeline_state: PipelineState) -> Dict[str, NDArray]:
        return {"dim": np.array([samples.shape[1]])}

    def apply(self, samples: NDArray, parameters: Dict[str, NDArray]) -> NDArray:
        if "dim" not in parameters:
            raise ValueError("Missing dim parameter")

        dim: int = parameters["dim"][0]

        input_dim = samples.shape[1]

        if input_dim != dim:
            raise ValueError(f"Invalid input dimension {input_dim}, expected {dim}")

        return samples

    def get_output_dimension(self, parameters: Dict[str, NDArray]) -> int:
        if "dim" not in parameters:
            raise ValueError("Missing dim parameter")

        return parameters["dim"][0]


class SKLearnPCATransform(Transform):
    _normalize: bool = False

    def __init__(self, config: TransformConfig) -> None:
        super().__init__(config)

    def _setup_pca(self) -> PCA:
        pass

    def create(self, samples: NDArray, sample_type: str, pipeline_state: PipelineState) -> Dict[str, NDArray]:
        pca = self._setup_pca()

        pca.fit(samples)

        if pipeline_state.scratch_output_dir is not None:
            explained_variance = np.cumsum(pca.explained_variance_ratio_)

            variance_json_file = pipeline_state.scratch_output_dir / f"{sample_type.lower()}_variance.json"
            with variance_json_file.open("w") as f:
                json.dump(explained_variance.tolist(), f)

            x = np.arange(explained_variance.shape[0]) + 1
            y = explained_variance

            fig, ax = plt.subplots()

            ax.set_title("PCA Explained Variance Ratio")

            ax.plot(x, y)

            ax.set_xlabel("Number of Principal Components")
            ax.get_xaxis().set_major_locator(MaxNLocator(nbins="auto", integer=True))

            ax.set_ylabel("Cumulative Explained Variance Ratio")

            save_figure(fig, pipeline_state.scratch_output_dir / f"{sample_type.lower()}_variance.png")

        return {"mean": pca.mean_, "components": pca.components_}

    def _parse_parameters(self, parameters: Dict[str, NDArray]) -> Tuple[NDArray, NDArray]:
        if "mean" not in parameters:
            raise ValueError("Missing mean parameter")

        if "components" not in parameters:
            raise ValueError("Missing components parameter")

        return (parameters["mean"], parameters["components"])

    def apply(self, samples: NDArray, parameters: Dict[str, NDArray]) -> NDArray:
        mean, components = self._parse_parameters(parameters)

        samples = samples.copy()

        samples -= mean

        samples = np.dot(samples, components.T)

        if self._normalize:
            samples /= np.linalg.norm(samples, axis=1, keepdims=True)

        return samples

    def get_output_dimension(self, parameters: Dict[str, NDArray]) -> int:
        _, components = self._parse_parameters(parameters)
        return components.shape[0]


@dataclass
class FixedPCATransformConfig(TransformConfig):
    _target: Type = field(default_factory=lambda: FixedPCATransform)

    num_components: int = 32
    """Number of principal components"""

    normalize: bool = True
    """Whether to normalize the samples after PCA"""


class FixedPCATransform(SKLearnPCATransform):
    config: FixedPCATransformConfig

    def __init__(self, config: FixedPCATransformConfig) -> None:
        super().__init__(config)
        self._normalize = self.config.normalize

    def _setup_pca(self) -> PCA:
        return PCA(n_components=self.config.num_components)


@dataclass
class VariancePCATransformConfig(TransformConfig):
    _target: Type = field(default_factory=lambda: VariancePCATransform)

    explained_variance_threshold: float = 0.5
    """Number of principal components is increased until the specified percentage of variance is accounted for"""

    normalize: bool = True
    """Whether to normalize the samples after PCA"""


class VariancePCATransform(SKLearnPCATransform):
    config: VariancePCATransformConfig

    def __init__(self, config: VariancePCATransformConfig) -> None:
        super().__init__(config)
        self._normalize = self.config.normalize

    def _setup_pca(self) -> PCA:
        if self.config.explained_variance_threshold <= 0.0 or self.config.explained_variance_threshold >= 1.0:
            raise ValueError("Explained variance threshold must be in range (0.0, 1.0)")
        return PCA(n_components=self.config.explained_variance_threshold)
