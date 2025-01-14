from pathlib import Path
from typing import Dict, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
from nerfstudio.utils.rich_utils import CONSOLE
from numpy.random import Generator
from numpy.typing import NDArray
from tqdm import tqdm

from rvs.evaluation.analysis.utils import rename_categories_tuple
from rvs.evaluation.embedder import Embedder
from rvs.evaluation.lvis import Category, LVISDataset, Uid
from rvs.evaluation.pipeline import PipelineEvaluationInstance
from rvs.utils.console import file_link
from rvs.utils.map import convert_map_to_tuple
from rvs.utils.plot import bar_plot, discrete_histogram_plot, save_figure


def count_selected_views(
    lvis: LVISDataset,
    uids: Set[Uid],
    instance: PipelineEvaluationInstance,
) -> Dict[Uid, int]:
    counts: Dict[Uid, int] = dict()

    for uid in tqdm(sorted(uids)):
        count = 0

        model_file = Path(lvis.uid_to_file[uid])

        results_dir = instance.results_dir / model_file.name

        if results_dir.exists() and results_dir.is_dir():
            for image_file in results_dir.iterdir():
                if image_file.is_file() and image_file.name.endswith(".png"):
                    count += 1

        counts[uid] = count

    return counts


def calculate_selected_views_avg_per_category(
    lvis: LVISDataset,
    uids: Set[Uid],
    instance: PipelineEvaluationInstance,
) -> Dict[Category, float]:
    counts = count_selected_views(lvis, uids, instance)

    category_total: Dict[Category, float] = dict()
    category_items: Dict[Category, int] = dict()

    for uid, count in counts.items():
        category = lvis.uid_to_category[uid]

        if category not in category_total:
            category_total[category] = 0.0

        if category not in category_items:
            category_items[category] = 0

        category_total[category] += count
        category_items[category] += 1

    for category in category_total.keys():
        category_total[category] /= category_items[category]

    return category_total


def calculte_selected_views_histogram_per_category(
    lvis: LVISDataset,
    uids: Set[Uid],
    instance: PipelineEvaluationInstance,
) -> Dict[Category, NDArray]:
    counts = count_selected_views(lvis, uids, instance)

    dict_histograms: Dict[Category, Dict[int, int]] = dict()

    for uid, count in counts.items():
        category = lvis.uid_to_category[uid]
        count = counts[uid]

        if category not in dict_histograms:
            dict_histograms[category] = dict()

        dict_histogram = dict_histograms[category]

        if count not in dict_histogram:
            dict_histogram[count] = 0

        dict_histogram[count] += 1

    histograms: Dict[Category, NDArray] = dict()

    for category in dict_histograms.keys():
        dict_histogram = dict_histograms[category]

        histogram = np.zeros((max(dict_histogram.keys()) + 1,))

        for bucket, value in dict_histogram.items():
            histogram[bucket] = value

        histograms[category] = histogram

    return histograms


def plot_selected_views_avg_per_category(
    avg_counts: Dict[Category, float],
    file: Path,
    category_names: Optional[Dict[Category, str]] = None,
    category_filter: Optional[Set[str]] = None,
) -> None:
    categories = tuple(
        sorted([category for category in avg_counts.keys() if category_filter is None or category in category_filter])
    )

    fig, ax = plt.subplots()

    ax.set_title("Average Number of Selected Views")

    bar_plot(
        ax,
        names=rename_categories_tuple(categories, category_names),
        values=convert_map_to_tuple(avg_counts, key_order=categories, default=lambda _: 0.0),
        xlabel="Objaverse 1.0 LVIS Category\n(size of category in parentheses)",
        ylabel="Average Number of Selected Views",
    )

    fig.tight_layout()

    save_figure(fig, file)


def plot_selected_views_histogram(
    histograms: Dict[Category, NDArray],
    file: Path,
    category_names: Optional[Dict[Category, str]] = None,
    category_filter: Optional[Set[str]] = None,
) -> None:
    categories = tuple(
        sorted([category for category in histograms.keys() if category_filter is None or category in category_filter])
    )

    fig = plt.figure()

    for category in categories:
        category_name = category_names[category] if category_names is not None else category

        histogram = histograms[category]

        ax = fig.subplots()

        ax.set_title(f'Number of Selected Views for Objaverse 1.0 LVIS Category\n"{category_name}"')

        discrete_histogram_plot(
            ax,
            values=histogram,
            xlabel="Number of Selected Views",
            ylabel="Count",
        )

    fig.tight_layout()

    save_figure(fig, file)


def embed_selected_views_avg(
    lvis: LVISDataset,
    uids: Set[Uid],
    embedder: Embedder,
    instance: PipelineEvaluationInstance,
) -> Dict[Uid, NDArray]:
    embeddings: Dict[Uid, NDArray] = dict()

    for uid in tqdm(sorted(uids)):
        model_file = Path(lvis.uid_to_file[uid])

        results_dir = instance.results_dir / model_file.name

        if results_dir.exists() and results_dir.is_dir():
            avg_embedding = None

            for image_file in results_dir.iterdir():
                if image_file.is_file() and image_file.name.endswith(".png"):
                    # CONSOLE.log(f"Embedding selected view {file_link(image_file)}")

                    embedding = embedder.embed_image_numpy(image_file)

                    avg_embedding = embedding if avg_embedding is None else avg_embedding + embedding

            if avg_embedding is not None:
                avg_embedding /= np.linalg.norm(avg_embedding)

                embeddings[uid] = avg_embedding

                # CONSOLE.log(f"Embedded selected views of {file_link(model_file)}")

    return embeddings


def embed_random_views_avg(
    lvis: LVISDataset,
    uids: Set[Uid],
    embedder: Embedder,
    instance: PipelineEvaluationInstance,
    rng: Generator,
    number_of_views: int,
) -> Dict[Uid, NDArray]:
    embeddings = dict()

    for uid in tqdm(sorted(uids)):
        selection_rng = np.random.default_rng(seed=rng)

        model_file = Path(lvis.uid_to_file[uid])

        pipeline_config = instance.create_pipeline_config(model_file)

        views_dir = pipeline_config.get_base_dir() / "renderer" / "images"

        if views_dir.exists() and views_dir.is_dir():
            available_image_files = [
                file for file in views_dir.iterdir() if file.is_file() and file.name.endswith(".png")
            ]
            random_image_files = []

            for i in range(min(len(available_image_files), number_of_views)):
                idx = selection_rng.integers(low=0, high=len(available_image_files))
                random_image_files.append(available_image_files[idx])
                del available_image_files[idx]

            if len(random_image_files) != number_of_views:
                CONSOLE.log(
                    f"[bold yellow]WARNING: Only {len(random_image_files)} views available of {file_link(model_file)} for random selection of {number_of_views} views"
                )
                continue

            avg_embedding = None

            for image_file in random_image_files:
                # CONSOLE.log(f"Embedding random view {file_link(image_file)}")

                embedding = embedder.embed_image_numpy(image_file)

                avg_embedding = embedding if avg_embedding is None else avg_embedding + embedding

            if avg_embedding is not None:
                avg_embedding /= np.linalg.norm(avg_embedding)

                embeddings[uid] = avg_embedding

                # CONSOLE.log(f"Embedded random views of {file_link(model_file)}")

    return embeddings
