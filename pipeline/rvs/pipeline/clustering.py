import shutil
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

    trials_per_k: int = 5
    """Number of times clustering should be done per k. Distortion is averaged over the trials before elbow method and the clustering with lowest distortion for the selected k is used in the end."""

    whitening: bool = True
    """Whether whitening should be done before K-Means clustering"""


class ElbowKMeansClustering(Clustering):
    config: ElbowKMeansClusteringConfig

    _num_clusters: List[int]

    def __init__(
        self,
        config: ElbowKMeansClusteringConfig,
        num_clusters: Optional[List[int]] = None,
    ) -> None:
        super().__init__(config)

        if num_clusters is not None:
            self._num_clusters = list(num_clusters)
        else:
            self._num_clusters = list(range(self.config.min_clusters, self.config.max_clusters + 1))

    def cluster(self, samples: NDArray, pipeline_state: PipelineState) -> NDArray:
        centroids: List[List[NDArray]] = []
        distortions: List[List[float]] = []

        for k in self._num_clusters:
            k_centroids, k_distortions = self.__cluster(samples, pipeline_state, k, self.config.trials_per_k)
            centroids.append(k_centroids)
            distortions.append(k_distortions)

        avg_distortions = [np.mean(np.array(k_distortions)) for k_distortions in distortions]

        pred_frac_k_distortion = self._select_distortion(self._num_clusters, avg_distortions)

        pred_k = 0
        pred_frac_k = 0.0

        for i in range(len(distortions)):
            upper_distortion = avg_distortions[i]
            lower_distortion = avg_distortions[i + 1]

            if pred_frac_k_distortion >= lower_distortion and pred_frac_k_distortion <= upper_distortion:
                fraction = (pred_frac_k_distortion - lower_distortion) / (upper_distortion - lower_distortion)
                pred_frac_k = self._num_clusters[i + 1] + (self._num_clusters[i] - self._num_clusters[i + 1]) * fraction
                break

        pred_k = int(round(pred_frac_k))

        pred_k_centroids: List[NDArray] = None
        pred_k_distortions: List[float] = None

        try:
            pred_k_i = self._num_clusters.index(pred_k)
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
                ks=self._num_clusters,
                ds=avg_distortions,
                pred_k=pred_k,
                pred_k_d=best_pred_k_distortion,
                pred_frac_k=pred_frac_k,
                pred_frac_k_d=pred_frac_k_distortion,
            )

            min_elbow = Elbow(
                ks=self._num_clusters,
                ds=[np.min(np.array(k_distortions)) for k_distortions in distortions],
                pred_k=pred_k,
                pred_k_d=best_pred_k_distortion,
                pred_frac_k=pred_frac_k,
                pred_frac_k_d=pred_frac_k_distortion,
            )

            max_elbow = Elbow(
                ks=self._num_clusters,
                ds=[np.max(np.array(k_distortions)) for k_distortions in distortions],
                pred_k=pred_k,
                pred_k_d=best_pred_k_distortion,
                pred_frac_k=pred_frac_k,
                pred_frac_k_d=pred_frac_k_distortion,
            )

            elbows_dir = pipeline_state.scratch_output_dir / "elbow"

            if elbows_dir.exists():
                shutil.rmtree(elbows_dir)

            elbows_dir.mkdir(parents=True)

            save_elbow(elbows_dir / "avg.json", avg_elbow)
            save_elbow(elbows_dir / "min.json", min_elbow)
            save_elbow(elbows_dir / "max.json", max_elbow)

            all_elbows_dir = elbows_dir / "all"
            all_elbows_dir.mkdir(parents=True)

            for i in range(self.config.trials_per_k):
                i_elbow = Elbow(
                    ks=self._num_clusters,
                    ds=[k_distortions[i] for k_distortions in distortions],
                    pred_k=pred_k,
                    pred_k_d=best_pred_k_distortion,
                    pred_frac_k=pred_frac_k,
                    pred_frac_k_d=pred_frac_k_distortion,
                )

                save_elbow(all_elbows_dir / f"{i}.json", i_elbow)

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

    def _select_distortion(self, ks: List[int], ds: List[float]) -> int:
        pass


@dataclass
class FractionalElbowKMeansClusteringConfig(ElbowKMeansClusteringConfig):
    _target: Type = field(default_factory=lambda: FractionalElbowKMeansClustering)

    last_probe_cluster_number: Optional[int] = 10
    """Number of clusters for last probe"""

    fraction_between_first_and_last_distortion: Optional[float] = 0.5
    """Selects number of clusters based on the specified fraction of distortion between the first and last probe distortion"""


class FractionalElbowKMeansClustering(ElbowKMeansClustering):
    config: FractionalElbowKMeansClusteringConfig

    def __init__(self, config: FractionalElbowKMeansClusteringConfig) -> None:
        num_clusters = list(range(config.min_clusters, config.max_clusters + 1))

        if config.last_probe_cluster_number is not None:
            num_clusters.append(config.last_probe_cluster_number)

        super().__init__(config, num_clusters)

    def _select_distortion(self, ks: List[int], ds: List[float]) -> float:
        return ds[-1] + (ds[0] - ds[-1]) * self.config.fraction_between_first_and_last_distortion


@dataclass
class ClosestElbowKMeansClusteringConfig(ElbowKMeansClusteringConfig):
    _target: Type = field(default_factory=lambda: ClosestElbowKMeansClustering)

    x_scale: float = 1.0
    """Scaling factor for x axis (number of clusters)"""

    y_scale: float = 1.0
    """Scaling factor for y axis (distortion)"""

    normalize_range: bool = True
    """Whether the x and y axis should be normalized to a 0.0 - 1.0 range before applying scale"""


class ClosestElbowKMeansClustering(ElbowKMeansClustering):
    config: ClosestElbowKMeansClusteringConfig

    def _select_distortion(self, ks: List[int], ds: List[float]) -> float:
        x = np.array([float(k) for k in ks])
        y = np.array([float(d) for d in ds])

        points = np.stack([x, y], axis=1)

        min = np.min(points, axis=0)
        max = np.max(points, axis=0)

        points = points - min

        if self.config.normalize_range:
            range = max - min
            points /= range

        points[:, 0] *= self.config.x_scale
        points[:, 1] *= self.config.y_scale

        distances = np.linalg.norm(points, axis=1)

        return ds[np.argmin(distances)]
