import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
from git import List, Optional
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from nerfstudio.configs.base_config import InstantiateConfig
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from scipy.cluster.vq import kmeans, vq
from trimesh.typed import NDArray

from rvs.pipeline.state import PipelineState
from rvs.utils.elbow import Elbow, save_elbow
from rvs.utils.plot import elbow_plot, save_figure
from rvs.utils.xmeans import (
    CustomXMeans,
    CustomXMeansData,
    CustomXMeansSolution,
    XMeansCriterion,
    save_xmeans_data,
    save_xmeans_solution,
)


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

    select_k_by_best_score: bool = False
    """Whether the best K should be selected given the global score of the criterion (implies evaluate_iterations=True)"""

    rerun_with_selected_k: bool = False
    """Whether to rerun the clustering with the selected K as maximum K"""

    evaluate_iterations: bool = False
    """Whether the global score should be evaluated for each iteration"""

    evaluate_clusters: bool = False
    """Whether the global score should be evaluated for all tried cluster configurations, making select_k_by_best_score more fine-grained"""

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

        xmeans_instance = CustomXMeans(
            samples.tolist(),
            initial_centers,
            kmax=self.config.max_clusters,
            repeat=self.config.kmeans_iterations,
            criterion=self.config.criterion,
            random_state=pipeline_state.pipeline.config.machine.seed,
            evaluate_iterations=self.config.evaluate_iterations or self.config.select_k_by_best_score,
            evaluate_clusters=self.config.evaluate_clusters,
        )

        xmeans_instance.process()

        xmeans_data = xmeans_instance.data

        xmeans_solution = xmeans_instance.solution

        if self.config.select_k_by_best_score:
            assert len(xmeans_data.clusters_global_scores) > 0

            best_idx = -1

            if self.config.criterion == XMeansCriterion.BIC:
                best_idx = np.argmax(xmeans_data.clusters_global_scores)
            elif self.config.criterion == XMeansCriterion.MNDL:
                best_idx = np.argmin(xmeans_data.clusters_global_scores)

            assert best_idx >= 0

            if self.config.rerun_with_selected_k:
                best_k = len(xmeans_data.clusters_centers[best_idx])

                rerun_xmeans_instance = CustomXMeans(
                    samples.tolist(),
                    initial_centers,
                    kmax=best_k,
                    repeat=self.config.kmeans_iterations,
                    criterion=self.config.criterion,
                    random_state=pipeline_state.pipeline.config.machine.seed,
                    evaluate_clusters=False,
                    evaluate_iterations=False,
                )

                rerun_xmeans_instance.process()

                xmeans_solution = rerun_xmeans_instance.solution
            else:
                xmeans_solution = CustomXMeansSolution(
                    indices=xmeans_data.clusters_indices[best_idx],
                    centers=xmeans_data.clusters_centers[best_idx],
                    score=xmeans_data.clusters_global_scores[best_idx],
                )

        centroids = np.array(xmeans_solution.centers)

        KMeansClustering.kmeans_inplace_unwhitening(centroids, normalization)

        if self.config.normalize:
            centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

        indices = np.zeros((samples.shape[0],), dtype=np.intp)
        for cluster_idx, cluster in enumerate(xmeans_solution.indices):
            assert cluster_idx >= 0
            for sample_idx in cluster:
                assert sample_idx >= 0
                indices[sample_idx] = int(cluster_idx) + 1

        assert not np.any(indices <= 0)

        indices -= 1

        assert np.any(indices == 0)

        if pipeline_state.scratch_output_dir is not None:
            save_xmeans_data(pipeline_state.scratch_output_dir / "xmeans_data.json", xmeans_data)

            save_xmeans_solution(pipeline_state.scratch_output_dir / "xmeans_solution.json", xmeans_solution)

            self.__save_xmeans_iterations_global_scores_plot(
                xmeans_data, xmeans_solution, pipeline_state.scratch_output_dir / "xmeans_iterations_global_scores.png"
            )

            self.__save_xmeans_clusters_global_scores_plot(
                xmeans_data, xmeans_solution, pipeline_state.scratch_output_dir / "xmeans_clusters_global_scores.png"
            )

        return {"centroids": centroids, "normalization": normalization}, indices

    def __save_xmeans_iterations_global_scores_plot(
        self, xmeans_data: CustomXMeansData, xmeans_solution: CustomXMeansSolution, file: Path
    ):
        solution_k = len(xmeans_solution.centers)

        solution_iteration: Optional[int] = None
        for i, it in enumerate(xmeans_data.clusters_iterations):
            if len(xmeans_data.clusters_centers[i]) == solution_k:
                solution_iteration = it

        x = np.arange(len(xmeans_data.iterations_global_scores), dtype=np.float32)

        x_frac_solution_iteration: Optional[float] = None
        if solution_iteration is not None and solution_iteration < len(xmeans_data.iterations_pre_split_clusters) - 1:
            min_k = xmeans_data.iterations_pre_split_clusters[solution_iteration]
            max_k = xmeans_data.iterations_pre_split_clusters[solution_iteration + 1]

            assert min_k <= solution_k
            assert max_k >= solution_k

            if min_k == max_k:
                x_frac_solution_iteration = solution_iteration
            else:
                x_frac_solution_iteration = solution_iteration + (solution_k - min_k) / (max_k - min_k)

            assert x_frac_solution_iteration >= 0.0

        y_global_scores = np.array(xmeans_data.iterations_global_scores)
        y_num_clusters = np.array(xmeans_data.iterations_pre_split_clusters)

        y_solution_global_score: Optional[float] = None
        y_solution_num_clusters: Optional[float] = None

        if x_frac_solution_iteration is not None:
            ins_idx = np.argmax(x == solution_iteration) + 1

            if ins_idx < x.shape[0]:
                x = np.insert(x, ins_idx, x_frac_solution_iteration)

                y_solution_global_score = xmeans_solution.score
                y_solution_num_clusters = len(xmeans_solution.centers)

                y_global_scores = np.insert(y_global_scores, ins_idx, y_solution_global_score)
                y_num_clusters = np.insert(y_num_clusters, ins_idx, y_solution_num_clusters)

        fig, ax = plt.subplots()
        ax1 = ax

        ax.set_title("X-Means BIC")

        ax.set_xlabel("Iteration")
        ax.get_xaxis().set_major_locator(MaxNLocator(nbins="auto", integer=True))

        ax1.plot(x, y_global_scores, color="C0", label="BIC$_{i}$")

        if x_frac_solution_iteration is not None and y_solution_global_score is not None:
            ax1.plot(x_frac_solution_iteration, y_solution_global_score, "o", color="C0", fillstyle="full")

        ax1.set_ylabel("BIC", color="C0")
        ax1.get_yaxis().set_tick_params(colors="C0")
        ax1.legend(loc="center left")

        ax2 = ax.twinx()

        ax2.plot(
            x,
            y_num_clusters,
            color="C1",
            linestyle="--",
            label="K$_{i}$",
        )

        if x_frac_solution_iteration is not None and y_solution_num_clusters is not None:
            ax2.plot(x_frac_solution_iteration, y_solution_num_clusters, "o", color="C1", fillstyle="full")

        ax2.set_ylabel("Number of Clusters", color="C1")
        ax2.get_yaxis().set_tick_params(colors="C1")
        ax2.legend(loc="center right")

        xlim1 = ax1.get_xlim()
        ylim1 = ax1.get_ylim()

        xlim2 = ax2.get_xlim()
        ylim2 = ax2.get_ylim()

        if y_solution_global_score is not None:
            ax1.hlines(
                y_solution_global_score,
                -1000000000,
                x_frac_solution_iteration,
                color="gray",
                linestyle=":",
                alpha=0.5,
                zorder=0,
            )
            ax1.vlines(
                x_frac_solution_iteration,
                -1000000000,
                y_solution_global_score,
                color="gray",
                linestyle=":",
                alpha=0.5,
                zorder=0,
            )

        if y_solution_num_clusters is not None:
            ax2.hlines(
                y_solution_num_clusters,
                x_frac_solution_iteration,
                1000000000,
                color="gray",
                linestyle=":",
                alpha=0.5,
                zorder=0,
            )
            ax2.vlines(
                x_frac_solution_iteration,
                -1000000000,
                y_solution_num_clusters,
                color="gray",
                linestyle=":",
                alpha=0.5,
                zorder=0,
            )

        ax1.set_xlim(xlim1)
        ax1.set_ylim(ylim1)

        ax2.set_xlim(xlim2)
        ax2.set_ylim(ylim2)

        save_figure(fig, file)

    def __save_xmeans_clusters_global_scores_plot(
        self, xmeans_data: CustomXMeansData, xmeans_solution: CustomXMeansSolution, file: Path
    ):
        min_k = 100000
        max_k = -1

        if len(xmeans_data.clusters_centers) > 0:
            for centers in xmeans_data.clusters_centers:
                k = len(centers)
                min_k = min(min_k, k)
                max_k = max(max_k, k)
        else:
            min_k = 0
            max_k = 0

        assert min_k < 100000
        assert max_k >= 0

        x: NDArray = np.array(list(range(min_k, max_k + 1)))

        y = np.ones((x.shape[0],)) * np.nan
        for i, centers in enumerate(xmeans_data.clusters_centers):
            k = len(centers)
            yi = k - min_k
            if yi < y.shape[0]:
                y[yi] = xmeans_data.clusters_global_scores[i]
        y_indices = np.nonzero(np.isfinite(y))[0]
        y_values = y[y_indices]
        y = np.interp(np.arange(y.shape[0]), y_indices, y_values)

        x_solution = len(xmeans_solution.centers)
        y_solution = xmeans_solution.score

        fig, ax = plt.subplots()

        ax.set_title("X-Means BIC")

        ax.plot(x, y, color="C0")

        ax.plot(x_solution, y_solution, "o", color="C0", fillstyle="full")

        ax.set_xlabel("Number of Clusters")
        ax.get_xaxis().set_major_locator(MaxNLocator(nbins="auto", integer=True))

        ax.set_ylabel("BIC")

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if x_solution is not None and y_solution is not None:
            ax.hlines(
                y_solution,
                -1000000000,
                x_solution,
                color="gray",
                linestyle=":",
                alpha=0.5,
                zorder=0,
            )
            ax.vlines(
                x_solution,
                -1000000000,
                y_solution,
                color="gray",
                linestyle=":",
                alpha=0.5,
                zorder=0,
            )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        save_figure(fig, file)

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


@dataclass
class LargestKMeansClusteringConfig(FixedKClusteringConfig):
    _target: Type = field(default_factory=lambda: LargestKMeansClustering)

    num_clusters: int = 1
    """Number of largest clusters"""

    implementation: ClusteringConfig = field(default_factory=lambda: ClusteringConfig)
    """Acutal clustering implementation to be used"""

    def setup(self, **kwargs) -> Any:
        implementation = self.implementation.setup(**kwargs)
        return self._target(self, implementation=implementation, **kwargs)


class LargestKMeansClustering(Clustering):
    config: LargestKMeansClusteringConfig

    __implementation: Clustering

    def __init__(self, config: LargestKMeansClusteringConfig, implementation: ClusteringConfig) -> None:
        super().__init__(config)
        self.__implementation = implementation

    def cluster(self, samples: NDArray, pipeline_state: PipelineState) -> Tuple[Dict[str, NDArray], NDArray[np.intp]]:
        parameters, indices = self.__implementation.cluster(samples, pipeline_state)

        if "centroids" not in parameters:
            raise ValueError("Missing centroids parameter in parameters returned by implementation")

        centroids = parameters["centroids"]

        if "normalization" not in parameters:
            raise ValueError("Missing normalization parameter in parameters returned by implementation")

        normalization = parameters["normalization"]

        cluster_sizes = np.zeros((self.get_number_of_clusters(parameters),), dtype=np.intp)
        for i in range(indices.shape[0]):
            cluster_sizes[indices[i]] += 1

        sorted_clusters = sorted(
            [(i, int(cluster_sizes[i])) for i in range(cluster_sizes.shape[0])], key=lambda t: t[1], reverse=True
        )

        selected_cluster_indices = [index for index, _ in sorted_clusters[: self.config.num_clusters]]

        selected_centroids_list: List[NDArray] = []
        for cluster_index in selected_cluster_indices:
            selected_centroids_list.append(centroids[cluster_index])
        selected_centroids = np.stack(selected_centroids_list, axis=0)

        reassigned_indices = KMeansClustering.kmeans_hard_classifier(
            samples, selected_centroids, normalization=normalization
        )

        return {
            "centroids": selected_centroids,
            "input_centroids": centroids,
            "normalization": normalization,
        }, reassigned_indices

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
