import json
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from rvs.evaluation.analysis.histogram import (
    calculate_avg_per_category,
    calculate_discrete_histogram_per_category,
    plot_avg_per_category,
    plot_histogram,
)
from rvs.evaluation.lvis import Category, LVISDataset, Uid
from rvs.evaluation.pipeline import PipelineEvaluationInstance


def count_clusters(
    lvis: LVISDataset,
    uids: Set[Uid],
    instance: PipelineEvaluationInstance,
) -> Dict[Uid, int]:
    counts: Dict[Uid, int] = dict()

    for uid in tqdm(sorted(uids)):
        model_file = Path(lvis.uid_to_file[uid])

        pipeline_config = instance.create_pipeline_config(model_file)

        clusters_file = pipeline_config.get_base_dir() / "clustering" / "clusters.json"

        clusters: NDArray = None

        try:
            with clusters_file.open(mode="r") as f:
                clusters = np.array(json.load(f))
        except Exception:
            pass

        if clusters is not None:
            counts[uid] = clusters.shape[0]

    return counts


def calculate_clusters_avg_per_category(
    lvis: LVISDataset,
    uids: Set[Uid],
    instance: PipelineEvaluationInstance,
) -> Dict[Category, float]:
    return calculate_avg_per_category(lvis, count_clusters(lvis, uids, instance))


def calculte_clusters_histogram_per_category(
    lvis: LVISDataset,
    uids: Set[Uid],
    instance: PipelineEvaluationInstance,
) -> Dict[Category, NDArray]:
    return calculate_discrete_histogram_per_category(lvis, count_clusters(lvis, uids, instance))


def plot_clusters_avg_per_category(
    avg_counts: Dict[Category, float],
    file: Path,
    category_names: Optional[Dict[Category, str]] = None,
    category_filter: Optional[Set[str]] = None,
) -> None:
    plot_avg_per_category(
        avg_counts,
        file,
        title="Average Number of Clusters",
        xlabel="Objaverse 1.0 LVIS Category\n(size of category in parentheses)",
        category_names=category_names,
        category_filter=category_filter,
    )


def plot_clusters_histogram(
    histograms: Dict[Category, NDArray],
    file: Path,
    category_names: Optional[Dict[Category, str]] = None,
    category_filter: Optional[Set[str]] = None,
) -> None:
    plot_histogram(
        histograms,
        file,
        title="Number of Clusters for Objaverse 1.0 LVIS Category",
        xlabel="Number of Clusters",
        category_names=category_names,
        category_filter=category_filter,
    )
