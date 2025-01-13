from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from rvs.evaluation.analysis.utils import Method, count_category_items
from rvs.evaluation.lvis import Category, Uid
from rvs.utils.map import extract_nested_maps, get_keys_of_nested_maps
from rvs.utils.plot import Precision, Recall, precision_recall_plot, save_figure


def calculate_similarity_to_category(
    embeddings: Dict[Method, Dict[Uid, NDArray]],
    categories_embeddings: Dict[Category, NDArray],
    category_filter: Optional[Set[str]] = None,
) -> Dict[Method, Dict[Uid, Dict[Category, float]]]:
    similarities: Dict[Method, Dict[Uid, Dict[Category, float]]] = dict()

    for method in embeddings.keys():
        method_embeddings = embeddings[method]

        method_similarities: Dict[Uid, Dict[Category, float]] = dict()

        for uid in method_embeddings.keys():
            embedding = method_embeddings[uid]

            item_similarities: Dict[Category, float] = dict()

            for category, category_embedding in categories_embeddings.items():
                if category_filter is None or category in category_filter:
                    item_similarities[category] = np.dot(embedding, category_embedding)

            method_similarities[uid] = item_similarities

        similarities[method] = method_similarities

    return similarities


def calculate_precision_recall(
    embeddings: Dict[Method, Dict[Uid, NDArray]],
    categories_embeddings: Dict[Category, NDArray],
    ground_truth: Dict[Uid, Category],
    category_filter: Optional[Set[str]] = None,
) -> Dict[Method, Dict[Category, Tuple[Precision, Recall, int]]]:
    if category_filter is not None:
        categories_embeddings = {
            category: categories_embeddings[category]
            for category in categories_embeddings.keys()
            if category in category_filter
        }

    uids = get_keys_of_nested_maps(embeddings)

    scores = calculate_similarity_to_category(embeddings, categories_embeddings)

    category_sizes = {
        category: count_category_items(ground_truth, uids, category) for category in categories_embeddings.keys()
    }

    results: Dict[Method, Dict[Category, Tuple[Precision, Recall]]] = dict()

    for method in embeddings.keys():
        method_scores = scores[method]

        method_pr: Dict[Category, Tuple[Precision, Recall]] = dict()

        for category in categories_embeddings.keys():
            category_size = category_sizes[category]

            # UIDs ranked by scores in descending order
            ranking = sorted(list(method_scores.keys()), key=lambda uid: method_scores[uid][category], reverse=True)

            true_positives = np.array([1 if ground_truth[uid] == category else 0 for uid in ranking])
            true_positives_cumsum = np.cumsum(true_positives)

            # True Positives / Total Retrieved
            precision = true_positives_cumsum * 1.0 / (np.arange(len(ranking)) + 1.0)

            # True Positives / Total Relevant
            recall = true_positives_cumsum * 1.0 / category_size

            method_pr[category] = (precision, recall, category_size)

        results[method] = method_pr

    return results


def plot_precision_recall(
    precision_recall: Dict[Method, Dict[Category, Tuple[Precision, Recall, int]]],
    total_count: int,
    file: Path,
    category_names: Optional[Dict[Category, str]] = None,
    category_filter: Optional[Set[str]] = None,
) -> None:
    categories = get_keys_of_nested_maps(precision_recall)

    fig = plt.figure()

    for category in categories:
        if category_filter is not None and category not in category_filter:
            continue

        category_name = category_names[category] if category_names is not None else category
        _, _, category_size = precision_recall[next(iter(precision_recall.keys()))][category]

        ax = fig.subplots()

        ax.set_title(
            f'Precision Recall Curve for Objaverse 1.0 LVIS Category\n"{category_name}"\n(relevant: {category_size}, total: {total_count})'
        )

        precision_recall_plot(
            ax,
            values=extract_nested_maps(precision_recall, category),
            xlabel="Recall",
            ylabel="Precision",
        )

    fig.tight_layout()

    save_figure(fig, file)
