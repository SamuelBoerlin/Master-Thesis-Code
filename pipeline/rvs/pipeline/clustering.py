import json
import shutil
from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

import numpy as np
from git import List, Optional
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from nerfstudio.configs.base_config import InstantiateConfig
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import splitting_type, xmeans
from scipy.cluster.vq import kmeans, vq
from trimesh.typed import NDArray

from rvs.pipeline.state import PipelineState
from rvs.utils.elbow import Elbow, save_elbow
from rvs.utils.plot import elbow_plot, save_figure
from rvs.utils.xmeans import CustomXMeans, XMeansCriterion


@dataclass
class ClusteringConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Clustering)


class Clustering:
    config: ClusteringConfig

    def __init__(self, config: ClusteringConfig):
        self.config = config

    def cluster(self, samples: NDArray, pipeline_state: PipelineState) -> Tuple[Dict[str, NDArray], NDArray[np.intp]]:
        pass

    def get_number_of_clusters(self, parameters: Dict[str, NDArray]) -> int:
        pass

    def hard_classifier(self, samples: NDArray, parameters: Dict[str, NDArray]) -> NDArray[np.intp]:
        pass

    def soft_classifier(self, samples: NDArray, parameters: Dict[str, NDArray]) -> NDArray:
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

    normalize: bool = True
    """Whether cluster centroids should be normalized at the end"""


class KMeansClustering(Clustering):
    config: KMeansClusteringConfig

    def cluster(self, samples: NDArray, pipeline_state: PipelineState) -> Tuple[Dict[str, NDArray], NDArray[np.intp]]:
        samples = samples.copy()

        normalization = KMeansClustering.kmeans_inplace_whitening(samples, whiten=self.config.whitening)

        centroids, indices, _ = KMeansClustering.kmeans_clustering(
            samples,
            self.config.num_clusters,
            pipeline_state.pipeline.config.machine.seed,
        )

        KMeansClustering.kmeans_inplace_unwhitening(centroids, normalization)

        if self.config.normalize:
            centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

        return {"centroids": centroids, "normalization": normalization}, indices

    def get_number_of_clusters(self, parameters: Dict[str, NDArray]) -> int:
        if "centroids" not in parameters:
            raise ValueError("Missing centroids parameter")

        return len(parameters["centroids"])

    def hard_classifier(self, samples: NDArray, parameters: Dict[str, NDArray]) -> NDArray[np.intp]:
        if "centroids" not in parameters:
            raise ValueError("Missing centroids parameter")

        if "normalization" not in parameters:
            raise ValueError("Missing normalization parameter")

        return KMeansClustering.kmeans_hard_classifier(
            samples, parameters["centroids"], normalization=parameters["normalization"]
        )

    def soft_classifier(self, samples: NDArray, parameters: Dict[str, NDArray]) -> NDArray:
        if "centroids" not in parameters:
            raise ValueError("Missing centroids parameter")

        if "normalization" not in parameters:
            raise ValueError("Missing normalization parameter")

        return KMeansClustering.kmeans_pseudo_soft_classifier(
            samples, parameters["centroids"], parameters["normalization"]
        )

    @staticmethod
    def kmeans_inplace_whitening(xs: NDArray, whiten: bool = True) -> NDArray:
        normalization = np.ones((xs.shape[1],))

        if whiten:
            # Do "whitening" manually so we can undo it again afterwards
            # samples = whiten(samples)

            if not np.isfinite(xs).all():
                raise ValueError("array must not contain infs or NaNs")

            # scipy.cluster.vq whiten
            std_dev = xs.std(axis=0)
            zero_std_mask = std_dev == 0
            if zero_std_mask.any():
                std_dev[zero_std_mask] = 1.0

            normalization = np.reciprocal(std_dev)

            xs *= normalization

        return normalization

    @staticmethod
    def kmeans_inplace_unwhitening(xs: NDArray, normalization: NDArray) -> None:
        xs /= normalization

    @staticmethod
    def kmeans_clustering(
        samples: NDArray,
        num_clusters: int,
        seed: int,
    ) -> Tuple[NDArray, NDArray[np.intp], float]:
        centroids, distortion = kmeans(
            samples,
            num_clusters,
            seed=seed,
        )

        indices = KMeansClustering.kmeans_hard_classifier(samples, centroids)

        return (centroids, indices, distortion)

    @staticmethod
    def kmeans_hard_classifier(
        samples: NDArray,
        centroids: NDArray,
        normalization: Optional[NDArray] = None,
    ) -> NDArray[np.intp]:
        if normalization is not None:
            samples = samples * normalization
            centroids = centroids * normalization

        indices, _ = vq(samples, centroids)

        return indices

    @staticmethod
    def kmeans_pseudo_soft_classifier(
        samples: NDArray,
        centroids: NDArray,
        normalization: Optional[NDArray] = None,
    ) -> NDArray:
        """This isn't really a soft classifier with proper probability distributions but should suffice for debugging purposes"""

        if normalization is not None:
            samples = samples * normalization
            centroids = centroids * normalization

        soft_labels = np.zeros((samples.shape[0], centroids.shape[0]))

        def dst(xs: NDArray, x: NDArray):
            return np.sqrt(np.sum((xs - x) ** 2, axis=1))

        for i in range(samples.shape[0]):
            sample = samples[i]

            distances = dst(centroids, sample)

            soft_labels[i] = distances / np.sum(distances)

        return soft_labels


@dataclass
class ElbowKMeansClusteringConfig(RangedKClusteringConfig):
    _target: Type = field(default_factory=lambda: ElbowKMeansClustering)

    trials_per_k: int = 5
    """Number of times clustering should be done per k. Distortion is averaged over the trials before elbow method and the clustering with lowest distortion for the selected k is used in the end."""

    whitening: bool = True
    """Whether whitening should be done before K-Means clustering"""

    normalize: bool = True
    """Whether cluster centroids should be normalized at the end"""


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

    def cluster(self, samples: NDArray, pipeline_state: PipelineState) -> Tuple[Dict[str, NDArray], NDArray[np.intp]]:
        samples = samples.copy()

        normalization = KMeansClustering.kmeans_inplace_whitening(samples, whiten=self.config.whitening)

        centroids: List[List[NDArray]] = []
        indices: List[List[NDArray[np.intp]]] = []
        distortions: List[List[float]] = []

        for k in self._num_clusters:
            k_centroids, k_indices, k_distortions = self.__cluster(samples, pipeline_state, k, self.config.trials_per_k)
            centroids.append(k_centroids)
            indices.append(k_indices)
            distortions.append(k_distortions)

        for centroids_list in centroids:
            for centroids_array in centroids_list:
                KMeansClustering.kmeans_inplace_unwhitening(centroids_array, normalization)

                if self.config.normalize:
                    centroids_array /= np.linalg.norm(centroids_array, axis=1, keepdims=True)

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
        pred_k_indices: List[NDArray[np.intp]] = None
        pred_k_distortions: List[float] = None

        try:
            pred_k_i = self._num_clusters.index(pred_k)
            pred_k_centroids = centroids[pred_k_i]
            pred_k_indices = indices[pred_k_i]
            pred_k_distortions = distortions[pred_k_i]
        except ValueError:
            pred_k_centroids, pred_k_indices, pred_k_distortions = self.__cluster(
                samples, pipeline_state, pred_k, self.config.trials_per_k
            )

        best_pred_k_centroids: NDArray
        best_pred_k_distortion: float

        best_trial_idx = np.argmin(pred_k_distortions)
        best_pred_k_centroids = pred_k_centroids[best_trial_idx]
        best_pred_k_indices = pred_k_indices[best_trial_idx]
        best_pred_k_distortion = pred_k_distortions[best_trial_idx]

        best_pred_k_centroids = best_pred_k_centroids / np.linalg.norm(best_pred_k_centroids, axis=1, keepdims=True)

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

        return {"centroids": best_pred_k_centroids, "normalization": normalization}, best_pred_k_indices

    def get_number_of_clusters(self, parameters: Dict[str, NDArray]) -> int:
        if "centroids" not in parameters:
            raise ValueError("Missing centroids parameter")

        return len(parameters["centroids"])

    def hard_classifier(self, samples: NDArray, parameters: Dict[str, NDArray]) -> NDArray[np.intp]:
        if "centroids" not in parameters:
            raise ValueError("Missing centroids parameter")

        if "normalization" not in parameters:
            raise ValueError("Missing normalization parameter")

        return KMeansClustering.kmeans_hard_classifier(
            samples, parameters["centroids"], normalization=parameters["normalization"]
        )

    def soft_classifier(self, samples: NDArray, parameters: Dict[str, NDArray]) -> NDArray:
        if "centroids" not in parameters:
            raise ValueError("Missing centroids parameter")

        if "normalization" not in parameters:
            raise ValueError("Missing normalization parameter")

        return KMeansClustering.kmeans_pseudo_soft_classifier(
            samples, parameters["centroids"], parameters["normalization"]
        )

    def __cluster(
        self,
        samples: NDArray,
        pipeline_state: PipelineState,
        k: int,
        rounds: int,
    ) -> Tuple[List[NDArray], List[NDArray[np.intp]], List[float]]:
        k_centroids: List[NDArray] = []
        k_indices: List[NDArray[np.intp]] = []
        k_distortions: List[float] = []

        for j in range(rounds):
            round_k_centroids, round_k_indices, round_k_distortion = KMeansClustering.kmeans_clustering(
                samples,
                k,
                pipeline_state.pipeline.config.machine.seed + j,
            )

            k_centroids.append(round_k_centroids)
            k_indices.append(round_k_indices)
            k_distortions.append(float(round_k_distortion))

        return (k_centroids, k_indices, k_distortions)

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


@dataclass
class XMeansClusteringConfig(RangedKClusteringConfig):
    _target: Type = field(default_factory=lambda: XMeansClustering)

    criterion: XMeansCriterion = XMeansCriterion.BIC
    """Criterion used for scoring cluster splits and for final selection"""

    kmeans_iterations: int = 10
    """Number of K-Means iterations in each Improve-Params step"""

    select_best: bool = True
    """Whether the best K should be selected given the global score of the criterion"""

    whitening: bool = True
    """Whether whitening should be done before K-Means clustering"""

    normalize: bool = True
    """Whether cluster centroids should be normalized at the end"""


# Pelleg, Dan and Moore, Andrew W. (2000), X-means: Extending K-means with Efficient Estimation of the Number of Clusters.
class XMeansClustering(Clustering):
    config: XMeansClusteringConfig

    def cluster(self, samples: NDArray, pipeline_state: PipelineState) -> Tuple[Dict[str, NDArray], NDArray[np.intp]]:
        samples = samples.copy()

        normalization = KMeansClustering.kmeans_inplace_whitening(samples, whiten=self.config.whitening)

        initial_centers = kmeans_plusplus_initializer(samples, self.config.min_clusters).initialize()

        initial_xmeans_instance = CustomXMeans(
            samples.tolist(),
            initial_centers,
            kmax=self.config.max_clusters,
            repeat=self.config.kmeans_iterations,
            criterion=self.config.criterion,
            random_state=pipeline_state.pipeline.config.machine.seed,
        )

        initial_xmeans_instance.process()

        final_xmeans_instance = initial_xmeans_instance

        if self.config.select_best:
            best_idx = np.argmax(initial_xmeans_instance.global_scores)

            best_k = initial_xmeans_instance.pre_split_clusters[best_idx]

            final_xmeans_instance = CustomXMeans(
                samples.tolist(),
                initial_centers,
                kmax=best_k,
                repeat=self.config.kmeans_iterations,
                criterion=self.config.criterion,
                random_state=pipeline_state.pipeline.config.machine.seed,
            )

            final_xmeans_instance.process()

        centroids = np.array(final_xmeans_instance.get_centers())

        KMeansClustering.kmeans_inplace_unwhitening(centroids, normalization)

        if self.config.normalize:
            centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

        indices = -np.ones((samples.shape[0],))
        for cluster_idx, cluster in enumerate(final_xmeans_instance.get_clusters()):
            for sample_idx in cluster:
                indices[sample_idx] = cluster_idx

        assert not np.any(indices < 0)

        if pipeline_state.scratch_output_dir is not None:
            json_file = pipeline_state.scratch_output_dir / "xmeans_iterations.json"

            json_iters_arr = []

            for i, score in enumerate(initial_xmeans_instance.global_scores):
                json_iter_obj = {
                    "score": score,
                    "splits": [],
                    "pre_clusters": initial_xmeans_instance.pre_split_clusters[i],
                    "post_clusters": initial_xmeans_instance.post_split_clusters[i],
                }

                pre_split_scores = initial_xmeans_instance.local_pre_split_scores[i]
                post_split_scores = initial_xmeans_instance.local_post_split_scores[i]

                json_iter_obj["splits"] = list(
                    {
                        "pre_score": pre_split_scores[j],
                        "post_score": post_split_scores[j],
                    }
                    for j in range(len(pre_split_scores))
                )

                json_iters_arr.append(json_iter_obj)

            with json_file.open("w") as f:
                json.dump(json_iters_arr, f)

            x = np.arange(len(initial_xmeans_instance.global_scores))

            fig, ax = plt.subplots()
            ax1 = ax

            ax.set_title("X-Means BIC")

            ax.set_xlabel("Iteration")
            ax.get_xaxis().set_major_locator(MaxNLocator(nbins="auto", integer=True))

            ax1.plot(x, np.array(initial_xmeans_instance.global_scores), color="blue", label="BIC$_{i}$")
            ax1.set_ylabel("BIC", color="blue")
            ax1.get_yaxis().set_tick_params(colors="blue")
            ax1.legend(loc="center left")

            ax2 = ax.twinx()

            ax2.plot(
                x, np.array(initial_xmeans_instance.pre_split_clusters), color="red", linestyle="--", label="K$_{i}$"
            )
            ax2.set_ylabel("Number of Clusters", color="red")
            ax2.get_yaxis().set_tick_params(colors="red")
            ax2.legend(loc="center right")

            xlim1 = ax1.get_xlim()
            ylim1 = ax1.get_ylim()

            xlim2 = ax2.get_xlim()
            ylim2 = ax2.get_ylim()

            ax1.vlines(
                len(final_xmeans_instance.global_scores) - 1,
                -1000000000,
                1000000000,
                color="blue",
                linestyle=":",
                alpha=0.5,
                zorder=0,
            )

            ax2.hlines(
                len(centroids),
                -1000000000,
                1000000000,
                color="red",
                linestyle=":",
                alpha=0.5,
                zorder=0,
            )

            ax1.set_xlim(xlim1)
            ax1.set_ylim(ylim1)

            ax2.set_xlim(xlim2)
            ax2.set_ylim(ylim2)

            save_figure(fig, pipeline_state.scratch_output_dir / "xmeans_global_scores.png")

        return {"centroids": centroids, "normalization": normalization}, indices

    def get_number_of_clusters(self, parameters: Dict[str, NDArray]) -> int:
        if "centroids" not in parameters:
            raise ValueError("Missing centroids parameter")

        return len(parameters["centroids"])

    def hard_classifier(self, samples: NDArray, parameters: Dict[str, NDArray]) -> NDArray[np.intp]:
        if "centroids" not in parameters:
            raise ValueError("Missing centroids parameter")

        if "normalization" not in parameters:
            raise ValueError("Missing normalization parameter")

        return KMeansClustering.kmeans_hard_classifier(
            samples, parameters["centroids"], normalization=parameters["normalization"]
        )

    def soft_classifier(self, samples: NDArray, parameters: Dict[str, NDArray]) -> NDArray:
        if "centroids" not in parameters:
            raise ValueError("Missing centroids parameter")

        if "normalization" not in parameters:
            raise ValueError("Missing normalization parameter")

        return KMeansClustering.kmeans_pseudo_soft_classifier(
            samples, parameters["centroids"], parameters["normalization"]
        )
