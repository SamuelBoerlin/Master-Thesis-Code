from dataclasses import dataclass, field
from typing import Type

import numpy as np
from scipy.cluster.vq import kmeans, whiten
from trimesh.typed import NDArray

from nerfstudio.configs.base_config import InstantiateConfig


@dataclass
class ClusteringConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Clustering)

    num_clusters: int = 3
    """Target number of clusters"""


class Clustering:
    config: ClusteringConfig

    def __init__(self, config: ClusteringConfig):
        self.config = config

    def cluster(self, samples: NDArray) -> NDArray:
        pass


@dataclass
class KMeansClusteringConfig(ClusteringConfig):
    _target: Type = field(default_factory=lambda: KMeansClustering)


class KMeansClustering(Clustering):
    def cluster(self, samples: NDArray) -> NDArray:
        # Do "whitening" manually so we can undo it again afterwards
        # samples = whiten(samples)
        std_dev = samples.std(axis=0)
        samples /= std_dev

        centroids, _ = kmeans(samples, self.config.num_clusters)

        # Undo "whitening"
        centroids *= std_dev

        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

        return centroids
