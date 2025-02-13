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

    trials_per_k: int = 5
    """Number of times clustering should be done per k. Distortion is averaged over the trials before elbow method and the clustering with lowest distortion for the selected k is used in the end."""

    whitening: bool = True
    """Whether whitening should be done before K-Means clustering"""

    # TODO Try method with closest point to origin (requires distortion / Y axis scaling parameter)


class ElbowKMeansClustering(Clustering):
    config: ElbowKMeansClusteringConfig

    def __cluster(
        self, samples: NDArray, pipeline_state: PipelineState, k: int, rounds: int
    ) -> Tuple[List[NDArray], List[float]]:
        k_centroids: List[NDArray] = []
        k_distortions: List[float] = []

        for j in range(rounds):
            k_centroid, k_distortion = KMeansClustering.kmeans_clustering(
                samples,
                k,
                self.config.whitening,
                pipeline_state.pipeline.config.machine.seed + j,
                True,
            )

            k_centroids.append(k_centroid)
            k_distortions.append(float(k_distortion))

        return (k_centroids, k_distortions)

    def cluster(self, samples: NDArray, pipeline_state: PipelineState) -> NDArray:
        centroids: List[List[NDArray]] = []
        distortions: List[List[float]] = []

        num_clusters = list(range(self.config.min_clusters, self.config.max_clusters + 1))
        if self.config.last_probe_cluster_number is not None:
            num_clusters.append(self.config.last_probe_cluster_number)

        for k in num_clusters:
            k_centroids, k_distortions = self.__cluster(samples, pipeline_state, k, self.config.trials_per_k)
            centroids.append(k_centroids)
            distortions.append(k_distortions)

        avg_distortions = [np.mean(np.array(k_distortions)) for k_distortions in distortions]

        pred_frac_k_distortion = (
            avg_distortions[-1]
            + (avg_distortions[0] - avg_distortions[-1]) * self.config.fraction_between_first_and_last_distortion
        )

        pred_k = 0
        pred_frac_k = 0.0

        for i in range(len(distortions)):
            upper_distortion = avg_distortions[i]
            lower_distortion = avg_distortions[i + 1]

            if pred_frac_k_distortion >= lower_distortion and pred_frac_k_distortion <= upper_distortion:
                fraction = (pred_frac_k_distortion - lower_distortion) / (upper_distortion - lower_distortion)
                pred_frac_k = num_clusters[i + 1] + (num_clusters[i] - num_clusters[i + 1]) * fraction
                break

        pred_k = int(round(pred_frac_k))

        pred_k_centroids: List[NDArray] = None
        pred_k_distortions: List[float] = None

        try:
            pred_k_i = num_clusters.index(pred_k)
            pred_k_centroids = centroids[pred_k_i]
            pred_k_distortions = distortions[pred_k_i]
        except ValueError:
            pred_k_centroids, pred_k_distortions = self.__cluster(
                samples, pipeline_state, pred_k, self.config.trials_per_k
            )

        best_pred_k_centroid: NDArray
        best_pred_k_distortion: float

        best_trial_idx = np.argmin(pred_k_distortions)
        best_pred_k_centroid = pred_k_centroids[best_trial_idx]
        best_pred_k_distortion = pred_k_distortions[best_trial_idx]

        best_pred_k_centroid = best_pred_k_centroid / np.linalg.norm(best_pred_k_centroid, axis=1, keepdims=True)

        if pipeline_state.scratch_output_dir is not None:
            avg_elbow = Elbow(
                ks=num_clusters,
                ds=avg_distortions,
                pred_k=pred_k,
                pred_k_d=best_pred_k_distortion,
                pred_frac_k=pred_frac_k,
                pred_frac_k_d=pred_frac_k_distortion,
            )

            min_elbow = Elbow(
                ks=num_clusters,
                ds=[np.min(np.array(k_distortions)) for k_distortions in distortions],
                pred_k=pred_k,
                pred_k_d=best_pred_k_distortion,
                pred_frac_k=pred_frac_k,
                pred_frac_k_d=pred_frac_k_distortion,
            )

            max_elbow = Elbow(
                ks=num_clusters,
                ds=[np.max(np.array(k_distortions)) for k_distortions in distortions],
                pred_k=pred_k,
                pred_k_d=best_pred_k_distortion,
                pred_frac_k=pred_frac_k,
                pred_frac_k_d=pred_frac_k_distortion,
            )

            save_elbow(pipeline_state.scratch_output_dir / "elbow.json", avg_elbow)
            save_elbow(pipeline_state.scratch_output_dir / "min_elbow.json", min_elbow)
            save_elbow(pipeline_state.scratch_output_dir / "max_elbow.json", max_elbow)

            fig, ax = plt.subplots()

            flags = [True, False, False]
            linestyles = ["-", "--", "--"]

            elbow_plot(
                ax,
                elbow=[avg_elbow, min_elbow, max_elbow],
                pred_point=flags,
                pred_frac_point=flags,
                pred_hlines=flags,
                pred_vlines=flags,
                pred_frac_hlines=flags,
                pred_frac_vlines=flags,
                colors=None,
                linestyles=linestyles,
            )

            save_figure(fig, pipeline_state.scratch_output_dir / "elbow.png")

        return best_pred_k_centroid
