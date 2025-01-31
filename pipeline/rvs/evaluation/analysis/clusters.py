import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import Generator
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
from rvs.pipeline.pipeline import PipelineStage
from rvs.utils.elbow import Elbow, load_elbow
from rvs.utils.plot import elbow_plot, save_figure


def count_clusters(
    lvis: LVISDataset,
    uids: Set[Uid],
    instance: PipelineEvaluationInstance,
) -> Dict[Uid, int]:
    counts: Dict[Uid, int] = dict()

    for uid in tqdm(sorted(uids)):
        model_file = Path(lvis.uid_to_file[uid])

        clusters_file = instance.create_pipeline_io(model_file).get_path(
            PipelineStage.CLUSTER_EMBEDDINGS,
            Path("clustering") / "clusters.json",
        )

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


def plot_elbows_samples(
    lvis: LVISDataset,
    uids: Set[Uid],
    category: Category,
    instance: PipelineEvaluationInstance,
    rng: Generator,
    number_of_samples: int,
    file: Path,
    category_names: Optional[Dict[Category, str]] = None,
) -> None:
    category_name = category_names[category] if category_names is not None else category

    samples: List[Elbow] = []

    selection_rng = np.random.default_rng(seed=rng)

    all_elbows: List[Tuple[Path, Uid]] = []

    for uid in lvis.dataset[category]:
        if uid in uids:
            model_file = Path(lvis.uid_to_file[uid])

            elbow_file = instance.create_pipeline_io(model_file).get_path(
                PipelineStage.CLUSTER_EMBEDDINGS,
                Path("clustering") / "scratch" / "elbow.json",
            )

            if elbow_file.exists() and elbow_file.is_file():
                all_elbows.append((elbow_file, uid))

    selection_rng.shuffle(all_elbows)

    for elbow_file, uid in all_elbows[:number_of_samples]:
        samples.append(load_elbow(elbow_file))

    fig, ax = plt.subplots()

    ax.set_title(f'Random Sample of Cluster Distortion for Objaverse 1.0 LVIS Category\n"{category_name}"')

    elbow_plot(
        ax,
        samples,
        pred_frac_point=False,
        pred_frac_hlines=False,
        pred_frac_vlines=False,
        xlabel="Number of clusters",
        ylabel="Distortion",
    )

    fig.tight_layout()

    save_figure(fig, file)
