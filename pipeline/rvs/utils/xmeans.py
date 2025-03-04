import json
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from pyclustering.cluster.xmeans import splitting_type, xmeans

# Fix: xmeans triggers AttributeError: module 'numpy' has no attribute 'warnings'
np.warnings = warnings


class XMeansCriterion(str, Enum):
    BIC = "BIC"
    MNDL = "MNDL"


@dataclass
class CustomXMeansData:
    improve_parameters_count: int = 0
    improve_structure_count: int = 0

    iterations_global_scores: List[float] = field(default_factory=lambda: [])
    iterations_pre_split_clusters: List[int] = field(default_factory=lambda: [])
    iterations_post_split_clusters: List[int] = field(default_factory=lambda: [])
    iterations_local_pre_split_scores: List[List[float]] = field(default_factory=lambda: [])
    iterations_local_post_split_scores: List[List[float]] = field(default_factory=lambda: [])

    clusters_iterations: List[int] = field(default_factory=lambda: [])
    clusters_global_scores: List[float] = field(default_factory=lambda: [])
    clusters_indices: List[List[int]] = field(default_factory=lambda: [])
    clusters_centers: List[List[float]] = field(default_factory=lambda: [])


def save_xmeans_data(
    file: Path,
    data: CustomXMeansData,
) -> None:
    with file.open("w") as f:
        json.dump(asdict(data), f)


def load_xmeans_data(
    file: Path,
) -> CustomXMeansData:
    with file.open("r") as f:
        return CustomXMeansData(**json.load(f))


@dataclass
class CustomXMeansSolution:
    indices: List[List[int]] = field(default_factory=lambda: [])
    centers: List[List[float]] = field(default_factory=lambda: [])
    score: float = 0.0


def save_xmeans_solution(
    file: Path,
    data: CustomXMeansSolution,
) -> None:
    with file.open("w") as f:
        json.dump(asdict(data), f)


def load_xmeans_solution(
    file: Path,
) -> CustomXMeansSolution:
    with file.open("r") as f:
        return CustomXMeansSolution(**json.load(f))


class CustomXMeans(xmeans):
    evaluate_clusters: bool
    evaluate_iterations: bool

    data: CustomXMeansData = CustomXMeansData()

    solution: CustomXMeansSolution = CustomXMeansSolution()

    def __init__(
        self,
        data,
        initial_centers=None,
        kmax=20,
        tolerance=0.025,
        criterion=splitting_type.BAYESIAN_INFORMATION_CRITERION,
        evaluate_iterations: bool = False,
        evaluate_clusters: bool = False,
        **kwargs,
    ) -> None:
        if criterion in (XMeansCriterion.BIC, XMeansCriterion.BIC.value):
            criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION
        elif criterion in (XMeansCriterion.MNDL, XMeansCriterion.MNDL.value):
            criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH

        self.evaluate_iterations = evaluate_iterations
        self.evaluate_clusters = evaluate_clusters

        super().__init__(
            data,
            initial_centers=initial_centers,
            kmax=kmax,
            tolerance=tolerance,
            criterion=criterion,
            ccore=False,  # Using the python implementation to intercept iterations
            **kwargs,
        )

    @property
    def kmax(self) -> int:
        return self._xmeans__kmax

    def process(self):
        self.data = CustomXMeansData()

        self.solution = CustomXMeansSolution()

        super().process()

        if self.evaluate_iterations:
            assert len(self.data.iterations_global_scores) == self.data.improve_parameters_count

            assert len(self.data.iterations_pre_split_clusters) == len(self.data.iterations_post_split_clusters)
            assert len(self.data.iterations_pre_split_clusters) == len(self.data.iterations_local_pre_split_scores)
            assert len(self.data.iterations_local_pre_split_scores) == len(self.data.iterations_local_post_split_scores)
            assert len(self.data.iterations_pre_split_clusters) == self.data.improve_structure_count

            self.data.iterations_pre_split_clusters.append(self.data.iterations_post_split_clusters[-1])
            self.data.iterations_post_split_clusters.append(self.data.iterations_post_split_clusters[-1])
            self.data.iterations_local_pre_split_scores.append([])
            self.data.iterations_local_post_split_scores.append([])

        if self.evaluate_iterations or self.evaluate_clusters:
            assert len(self.data.clusters_iterations) == len(self.data.clusters_global_scores)
            assert len(self.data.clusters_global_scores) == len(self.data.clusters_indices)
            assert len(self.data.clusters_indices) == len(self.data.clusters_centers)
            assert len(self.data.clusters_global_scores) >= self.data.improve_structure_count

        self.solution = CustomXMeansSolution(
            indices=self.get_clusters(),
            centers=self.get_centers(),
            score=super()._xmeans__splitting_criterion(self.get_clusters(), self.get_centers()),
        )

        return self

    # Overriding the name-mangled method __improve_parameters in xmeans
    def _xmeans__improve_parameters(self, centers, available_indexes=None):
        clusters, local_centers, local_wce = super()._xmeans__improve_parameters(
            centers, available_indexes=available_indexes
        )

        if centers is not None:
            if self.evaluate_iterations:
                score = super()._xmeans__splitting_criterion(clusters, local_centers)

                self.data.iterations_global_scores.append(score)

                if not self.evaluate_clusters:
                    self.data.clusters_iterations.append(self.data.improve_structure_count)
                    self.data.clusters_global_scores.append(score)
                    self.data.clusters_indices.append(clusters)
                    self.data.clusters_centers.append(local_centers)

            self.data.improve_parameters_count += 1

        return clusters, local_centers, local_wce

    # Overriding the name-mangled method __improve_structure in xmeans
    def _xmeans__improve_structure(self, clusters, centers):
        if self.evaluate_iterations:
            self.data.iterations_local_pre_split_scores.append([])
            self.data.iterations_local_post_split_scores.append([])

            self.data.iterations_pre_split_clusters.append(len(centers))

        # allocated_centers = super()._xmeans__improve_structure(clusters, centers)
        allocated_centers = self.__modified__improve_structure(clusters, centers)

        if self.evaluate_iterations:
            self.data.iterations_post_split_clusters.append(len(allocated_centers))

        self.data.improve_structure_count += 1

        return allocated_centers

    def __modified__improve_structure(self, clusters, centers):
        allocated_centers = []
        amount_free_centers = self._xmeans__kmax - len(centers)

        split_configurations: List[Tuple[int, List[List[float]], float, float]] = []

        eval_modified_centers: Optional[List] = None

        if self.evaluate_clusters:
            eval_modified_centers = [[c] for c in centers]

        def flatten_eval_modified_centers() -> List[List[float]]:
            return [] if eval_modified_centers is None else [c for cs in eval_modified_centers for c in cs]

        for index_cluster in range(len(clusters)):
            # solve k-means problem for children where data of parent are used.
            (parent_child_clusters, parent_child_centers, _) = self._xmeans__improve_parameters(
                None, clusters[index_cluster]
            )

            # If it's possible to split current data
            if len(parent_child_clusters) > 1:
                # Calculate splitting criterion
                parent_scores = self._xmeans__splitting_criterion([clusters[index_cluster]], [centers[index_cluster]])
                child_scores = self._xmeans__splitting_criterion(
                    [parent_child_clusters[0], parent_child_clusters[1]], parent_child_centers
                )

                split_require = False

                # Reallocate number of centers (clusters) in line with scores
                if self._xmeans__criterion == splitting_type.BAYESIAN_INFORMATION_CRITERION:
                    if parent_scores < child_scores:
                        split_require = True

                elif self._xmeans__criterion == splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH:
                    # If its score for the split structure with two children is smaller than that for the parent structure,
                    # then representing the data samples with two clusters is more accurate in comparison to a single parent cluster.
                    if parent_scores > child_scores:
                        split_require = True

                # Modifiation: append to list and only add splits after sorting
                # if (split_require is True) and (amount_free_centers > 0):
                #    allocated_centers.append(parent_child_centers[0])
                #    allocated_centers.append(parent_child_centers[1])
                #
                #    amount_free_centers -= 1
                # else:
                #    allocated_centers.append(centers[index_cluster])
                if split_require is True:
                    split_configurations.append(
                        (index_cluster, parent_child_clusters, parent_child_centers, parent_scores, child_scores)
                    )
                else:
                    allocated_centers.append(centers[index_cluster])

            else:
                allocated_centers.append(centers[index_cluster])

        # Modification: prioritize splits by their contribution to the score
        if self._xmeans__criterion == splitting_type.BAYESIAN_INFORMATION_CRITERION:
            sort_descending = True
        elif self._xmeans__criterion == splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH:
            sort_descending = False
        split_configurations = sorted(
            split_configurations, key=lambda configuration: configuration[4] - configuration[3], reverse=sort_descending
        )
        for (
            index_cluster,
            parent_child_clusters,
            parent_child_centers,
            parent_scores,
            child_scores,
        ) in split_configurations:
            if amount_free_centers > 0:
                allocated_centers.append(parent_child_centers[0])
                allocated_centers.append(parent_child_centers[1])

                amount_free_centers -= 1

                # Modification: keep track of all modified configurations and corresponding global score.
                # This is slow and removes much of the computational benefit over regular k-means with BIC...
                if eval_modified_centers is not None:
                    # Replace the previous single cluster with the two children
                    eval_modified_centers[index_cluster] = [parent_child_centers[0], parent_child_centers[1]]

                    modified_improved_clusters, modified_improved_centers, _ = super()._xmeans__improve_parameters(
                        flatten_eval_modified_centers()
                    )

                    modified_improved_global_score = super()._xmeans__splitting_criterion(
                        modified_improved_clusters, modified_improved_centers
                    )

                    self.data.clusters_iterations.append(self.data.improve_structure_count)
                    self.data.clusters_global_scores.append(modified_improved_global_score)
                    self.data.clusters_indices.append(modified_improved_clusters)
                    self.data.clusters_centers.append(modified_improved_centers)
            else:
                allocated_centers.append(centers[index_cluster])

        return allocated_centers

    # Overriding the name-mangled method __splitting_criterion in xmeans
    def _xmeans__splitting_criterion(self, clusters, centers):
        score = super()._xmeans__splitting_criterion(clusters, centers)

        if self.evaluate_iterations:
            if len(clusters) == 1:
                self.data.iterations_local_pre_split_scores[-1].append(score)
            elif len(clusters) == 2:
                self.data.iterations_local_post_split_scores[-1].append(score)
            else:
                assert False

        return score
