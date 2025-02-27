import warnings
from enum import Enum
from typing import List

import numpy as np
from pyclustering.cluster.xmeans import splitting_type, xmeans

# Fix: xmeans triggers AttributeError: module 'numpy' has no attribute 'warnings'
np.warnings = warnings


class XMeansCriterion(str, Enum):
    BIC = "BIC"
    MNDL = "MNDL"


class CustomXMeans(xmeans):
    global_scores: List[float] = []
    pre_split_clusters: List[int] = []
    post_split_clusters: List[int] = []
    local_pre_split_scores: List[List[float]] = []
    local_post_split_scores: List[List[float]] = []

    def __init__(
        self,
        data,
        initial_centers=None,
        kmax=20,
        tolerance=0.025,
        criterion=splitting_type.BAYESIAN_INFORMATION_CRITERION,
        **kwargs,
    ) -> None:
        if criterion in (XMeansCriterion.BIC, XMeansCriterion.BIC.value):
            criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION
        elif criterion in (XMeansCriterion.MNDL, XMeansCriterion.MNDL.value):
            criterion = splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH

        super().__init__(
            data,
            initial_centers=initial_centers,
            kmax=kmax,
            tolerance=tolerance,
            criterion=criterion,
            ccore=False,  # Using the python implementation to intercept iterations
            **kwargs,
        )

    def process(self):
        self.global_scores = []
        self.pre_split_clusters = []
        self.post_split_clusters = []
        self.local_pre_split_scores = []
        self.local_post_split_scores = []

        super().process()

        assert len(self.pre_split_clusters) == len(self.post_split_clusters)
        assert len(self.pre_split_clusters) == len(self.local_pre_split_scores)
        assert len(self.local_pre_split_scores) == len(self.local_post_split_scores)
        assert len(self.pre_split_clusters) == len(self.global_scores) - 1

        self.pre_split_clusters.append(self.post_split_clusters[-1])
        self.post_split_clusters.append(self.post_split_clusters[-1])
        self.local_pre_split_scores.append([])
        self.local_post_split_scores.append([])

        return self

    # Overriding the name-mangled method __improve_parameters in xmeans
    def _xmeans__improve_parameters(self, centers, available_indexes=None):
        clusters, local_centers, local_wce = super()._xmeans__improve_parameters(
            centers, available_indexes=available_indexes
        )

        if centers is not None:
            score = super()._xmeans__splitting_criterion(clusters, local_centers)
            self.global_scores.append(score)

        return clusters, local_centers, local_wce

    # Overriding the name-mangled method __improve_structure in xmeans
    def _xmeans__improve_structure(self, clusters, centers):
        self.local_pre_split_scores.append([])
        self.local_post_split_scores.append([])

        self.pre_split_clusters.append(len(centers))

        allocated_centers = super()._xmeans__improve_structure(clusters, centers)

        self.post_split_clusters.append(len(allocated_centers))

        return allocated_centers

    # Overriding the name-mangled method __splitting_criterion in xmeans
    def _xmeans__splitting_criterion(self, clusters, centers):
        score = super()._xmeans__splitting_criterion(clusters, centers)

        if len(clusters) == 1:
            self.local_pre_split_scores[-1].append(score)
        elif len(clusters) == 2:
            self.local_post_split_scores[-1].append(score)
        else:
            assert False

        return score
