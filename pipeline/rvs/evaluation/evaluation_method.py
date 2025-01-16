from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from nerfstudio.utils.rich_utils import CONSOLE
from numpy.typing import NDArray
from tqdm import tqdm

from rvs.evaluation.analysis.clusters import (
    calculate_clusters_avg_per_category,
    calculte_clusters_histogram_per_category,
    plot_clusters_avg_per_category,
    plot_clusters_histogram,
)
from rvs.evaluation.analysis.precision_recall import (
    calculate_precision_recall,
    calculate_precision_recall_auc,
    plot_precision_recall,
    plot_precision_recall_auc,
)
from rvs.evaluation.analysis.similarity import (
    calculate_avg_similarity_between_all_models_and_category_ground_truths,
    calculate_similarity_to_ground_truth,
    plot_avg_similarities_per_category,
    plot_avg_similariy_between_models_and_categories,
)
from rvs.evaluation.analysis.utils import count_category_items
from rvs.evaluation.analysis.views import (
    calculate_selected_views_avg_per_category,
    calculte_selected_views_histogram_per_category,
    embed_random_views_avg,
    embed_selected_views_avg,
    plot_selected_views_avg_per_category,
    plot_selected_views_histogram,
    plot_selected_views_samples,
)
from rvs.evaluation.embedder import Embedder
from rvs.evaluation.lvis import Category, LVISDataset, Uid
from rvs.evaluation.pipeline import PipelineEvaluationInstance


def evaluate_results(
    lvis: LVISDataset, embedder: Embedder, instance: PipelineEvaluationInstance, output_dir: Path
) -> None:
    CONSOLE.rule("Embedding prompts for categories...")
    categories_embeddings = embed_categories(lvis.dataset.keys(), embedder)

    available_uids: Set[Uid] = set()
    for category in lvis.dataset.keys():
        for uid in lvis.dataset[category]:
            available_uids.add(uid)

    number_of_views = 3  # FIXME: This should come from the config

    CONSOLE.rule("Embedding selected views...")
    avg_selected_views_embeddings = embed_selected_views_avg(
        lvis,
        available_uids,
        embedder,
        instance,
    )
    available_uids = avg_selected_views_embeddings.keys()

    CONSOLE.rule("Embedding random views...")
    avg_random_views_embeddings = embed_random_views_avg(
        lvis,
        available_uids,
        embedder,
        instance,
        np.random.default_rng(seed=238947978),
        number_of_views,
    )
    available_uids = avg_selected_views_embeddings.keys()

    CONSOLE.rule("Calculate number of clusters...")
    avg_number_of_clusters_per_category = calculate_clusters_avg_per_category(lvis, available_uids, instance)
    histogram_of_clusters_per_category = calculte_clusters_histogram_per_category(lvis, available_uids, instance)

    CONSOLE.rule("Calculate number of selected views...")
    avg_number_of_views_per_category = calculate_selected_views_avg_per_category(lvis, available_uids, instance)
    histogram_of_views_per_category = calculte_selected_views_histogram_per_category(lvis, available_uids, instance)

    CONSOLE.rule("Calculate similarities...")
    similarities = calculate_similarity_to_ground_truth(
        lvis,
        {
            f"Method 1: Average Embedding of Selected Views ($N \leq {number_of_views}$)": avg_selected_views_embeddings,
            f"Method 2: Average Embedding of Random Views ($N = {number_of_views}$)": avg_random_views_embeddings,
        },
        categories_embeddings,
    )
    cross_similarities = calculate_avg_similarity_between_all_models_and_category_ground_truths(
        lvis,
        {
            f"Method 1: Average Embedding of Selected Views ($N \leq {number_of_views}$)": avg_selected_views_embeddings,
            f"Method 2: Average Embedding of Random Views ($N = {number_of_views}$)": avg_random_views_embeddings,
        },
        categories_embeddings,
    )

    CONSOLE.rule("Calculate precision/recall...")
    precision_recall = calculate_precision_recall(
        {
            f"Method 1: Average Embedding of Selected Views ($N \leq {number_of_views}$)": avg_selected_views_embeddings,
            f"Method 2: Average Embedding of Random Views ($N = {number_of_views}$)": avg_random_views_embeddings,
        },
        categories_embeddings,
        lvis.uid_to_category,
    )
    precision_recall_auc = calculate_precision_recall_auc(precision_recall)

    CONSOLE.rule("Create results...")

    category_names_with_sizes = {
        category: category + " (" + str(count_category_items(lvis.uid_to_category, available_uids, category)) + ")"
        for category in lvis.categories
    }

    plot_clusters_avg_per_category(
        avg_number_of_clusters_per_category,
        output_dir / "nr_of_clusters_avg.png",
        category_names=category_names_with_sizes,
    )

    for category in categories_embeddings.keys():
        plot_clusters_histogram(
            histogram_of_clusters_per_category,
            output_dir / "nr_of_clusters_histogram" / f"{category}.png",
            category_filter={category},
        )

    plot_selected_views_avg_per_category(
        avg_number_of_views_per_category,
        output_dir / "nr_of_views_avg.png",
        category_names=category_names_with_sizes,
    )

    for category in categories_embeddings.keys():
        plot_selected_views_histogram(
            histogram_of_views_per_category,
            output_dir / "nr_of_views_histogram" / f"{category}.png",
            category_filter={category},
        )

    plot_avg_similarities_per_category(
        lvis,
        similarities,
        output_dir / "similarities.png",
        category_names=category_names_with_sizes,
    )

    for category in cross_similarities.keys():
        plot_avg_similariy_between_models_and_categories(
            cross_similarities[category],
            category,
            output_dir / "cross_similarity" / f"{category}.png",
            category_names=category_names_with_sizes,
        )

    for category in categories_embeddings.keys():
        plot_precision_recall(
            precision_recall,
            len(available_uids),
            output_dir / "precision_recall" / f"{category}.png",
            category_filter={category},
        )

    plot_precision_recall_auc(
        precision_recall_auc,
        output_dir / "precision_recall_auc.png",
        category_names=category_names_with_sizes,
    )

    CONSOLE.rule("Create selected views samples...")

    for category in tqdm(categories_embeddings.keys()):
        plot_selected_views_samples(
            lvis,
            available_uids,
            category,
            instance,
            np.random.default_rng(seed=238947978),
            5 * 5,
            output_dir / "views" / f"{category}.png",
        )


def embed_categories(categories: List[Category], embedder: Embedder) -> Dict[Category, NDArray]:
    embeddings = dict()

    for category in tqdm(categories):
        prompt = category_name_to_embedding_prompt(category)
        # CONSOLE.log(f"Embedding category {category} as '{prompt}'")
        embeddings[category] = embedder.embed_text_numpy(prompt)

    return embeddings


def category_name_to_embedding_prompt(category: Category) -> None:
    return category.replace("_", " ")
