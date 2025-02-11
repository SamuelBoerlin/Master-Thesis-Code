from dataclasses import dataclass, field
from typing import Tuple, Type

import numpy as np
from git import List, Optional
from matplotlib import pyplot as plt
from nerfstudio.configs.base_config import InstantiateConfig
from scipy.cluster.vq import kmeans
from trimesh.typed import NDArray

from rvs.pipeline.state import PipelineState
from rvs.utils.elbow import Elbow, save_elbow
from rvs.utils.plot import elbow_plot, save_figure


@dataclass
class ClusteringConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Clustering)


class Clustering:
    config: ClusteringConfig

    def __init__(self, config: ClusteringConfig):
        self.config = config

    def cluster(self, samples: NDArray, pipeline_state: PipelineState) -> NDArray:
        pass


@dataclass
class FixedKClusteringConfig(ClusteringConfig):
    num_clusters: int = 3
    """Target number of clusters"""


@dataclass
class RangedKClusteringConfig(ClusteringConfig):
    min_clusters: int = 1
    """Minimum number of clusters"""

    max_clusters: int = 5
    """Maximum number of clusters"""


@dataclass
class KMeansClusteringConfig(FixedKClusteringConfig):
    _target: Type = field(default_factory=lambda: KMeansClustering)

    num_clusters: int = 3
    """Target number of clusters"""

    whitening: bool = True
    """Whether whitening should be done before K-Means clustering"""


class KMeansClustering(Clustering):
    config: KMeansClusteringConfig

    def cluster(self, samples: NDArray, pipeline_state: PipelineState) -> NDArray:
        centroids, _ = KMeansClustering.kmeans_clustering(
            samples,
            self.config.num_clusters,
            self.config.whitening,
            pipeline_state.pipeline.config.machine.seed,
            True,
        )
        return centroids

    @staticmethod
    def kmeans_clustering(
        samples: NDArray, num_clusters: int, whitening: bool, seed: int, normalize_centroids: bool
    ) -> Tuple[NDArray, float]:
        std_dev: NDArray = None

        if whitening:
            # Do "whitening" manually so we can undo it again afterwards
            # samples = whiten(samples)
            std_dev = samples.std(axis=0)
            samples /= std_dev

        centroids, distortion = kmeans(
            samples,
            num_clusters,
            seed,
        )

        if whitening and std_dev is not None:
            # Undo "whitening"
            centroids *= std_dev

        if normalize_centroids:
            centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

        return (centroids, distortion)


@dataclass
class ElbowKMeansClusteringConfig(RangedKClusteringConfig):
    _target: Type = field(default_factory=lambda: ElbowKMeansClustering)

    last_probe_cluster_number: Optional[int] = 10
    """Number of clusters for last probe"""

    fraction_between_first_and_last_distortion: Optional[float] = 0.5
    """Selects number of clusters based on the specified fraction of distortion between the first and last probe distortion"""

    trials_per_k: int = 1  # TODO Multiple runs per k and avg. distortion incl. outlier rejection
    """Number of times clustering should be done per k. Distortion is averaged over the trials before elbow method and the clustering with lowest distortion for the selected k is used in the end."""

    whitening: bool = True
    """Whether whitening should be done before K-Means clustering"""

    # TODO Try method with closest point to origin (requires distortion / Y axis scaling parameter)


class ElbowKMeansClustering(Clustering):
    config: ElbowKMeansClusteringConfig

    def cluster(self, samples: NDArray, pipeline_state: PipelineState) -> NDArray:
        centroids: List[NDArray] = []
        distortions: List[float] = []

        num_clusters = list(range(self.config.min_clusters, self.config.max_clusters + 1))
        if self.config.last_probe_cluster_number is not None:
            num_clusters.append(self.config.last_probe_cluster_number)

        for k in num_clusters:
            k_centroids, distortion = KMeansClustering.kmeans_clustering(
                samples,
                k,
                self.config.whitening,
                pipeline_state.pipeline.config.machine.seed,
                True,
            )

            centroids.append(k_centroids)
            distortions.append(float(distortion))

        pred_frac_k_distortion = (
            distortions[-1]
            + (distortions[0] - distortions[-1]) * self.config.fraction_between_first_and_last_distortion
        )

        pred_k = 0
        pred_frac_k = 0.0

        for i in range(len(distortions) - 1):
            upper_distortion = distortions[i]
            lower_distortion = distortions[i + 1]

            if pred_frac_k_distortion >= lower_distortion and pred_frac_k_distortion <= upper_distortion:
                fraction = (pred_frac_k_distortion - lower_distortion) / (upper_distortion - lower_distortion)
                pred_frac_k = num_clusters[i + 1] + (num_clusters[i] - num_clusters[i + 1]) * fraction
                break

        pred_k = int(round(pred_frac_k))

        pred_k_centroids: NDArray = None
        pred_k_distortion: float = None

        try:
            pred_k_i = num_clusters.index(pred_k)
            pred_k_centroids = centroids[pred_k_i]
            pred_k_distortion = distortions[pred_k_i]
        except ValueError:
            pass

        pred_k_centroids, pred_k_distortion = KMeansClustering.kmeans_clustering(
            samples,
            pred_k,
            self.config.whitening,
            pipeline_state.pipeline.config.machine.seed,
            True,
        )
        pred_k_distortion = float(pred_k_distortion)

        if pipeline_state.scratch_output_dir is not None:
            elbow = Elbow(
                ks=num_clusters,
                ds=distortions,
                pred_k=pred_k,
                pred_k_d=pred_k_distortion,
                pred_frac_k=pred_frac_k,
                pred_frac_k_d=pred_frac_k_distortion,
            )

            save_elbow(pipeline_state.scratch_output_dir / "elbow.json", elbow)

            fig, ax = plt.subplots()

            elbow_plot(ax, elbow)

            save_figure(fig, pipeline_state.scratch_output_dir / "elbow.png")

        return pred_k_centroids
