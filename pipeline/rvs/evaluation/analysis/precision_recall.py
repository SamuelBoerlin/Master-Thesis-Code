from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from rvs.evaluation.analysis.utils import (
    Method,
    count_category_items,
    rename_categories_dict,
    rename_categories_tuple,
    rename_methods_dict,
)
from rvs.evaluation.lvis import Category, Uid
from rvs.utils.map import convert_nested_maps_to_tuples, extract_nested_maps, get_keys_of_nested_maps
from rvs.utils.plot import (
    Precision,
    Recall,
    comparison_grid_plot,
    grouped_bar_plot,
    place_legend_outside,
    precision_recall_plot,
    save_figure,
)


def calculate_best_similarity_to_category(
    embeddings: Dict[Method, Dict[Uid, Union[NDArray, List[NDArray]]]],
    categories_embeddings: Dict[Category, NDArray],
    category_filter: Optional[Set[str]] = None,
) -> Dict[Method, Dict[Uid, Dict[Category, float]]]:
    similarities: Dict[Method, Dict[Uid, Dict[Category, float]]] = dict()

    for method in embeddings.keys():
        method_embeddings = embeddings[method]

        method_similarities: Dict[Uid, Dict[Category, float]] = dict()

        for uid in method_embeddings.keys():
            uid_embeddings = method_embeddings[uid]

            if not isinstance(uid_embeddings, List):
                uid_embeddings = [uid_embeddings]

            item_similarities: Dict[Category, float] = dict()

            for category, category_embedding in categories_embeddings.items():
                if category_filter is None or category in category_filter:
                    uid_similarities: List[float] = []

                    for embedding in uid_embeddings:
                        uid_similarities.append(np.dot(embedding, category_embedding))

                    assert len(uid_similarities) > 0

                    item_similarities[category] = max(uid_similarities)

            method_similarities[uid] = item_similarities

        similarities[method] = method_similarities

    return similarities


def calculate_precision_recall(
    embeddings: Dict[Method, Dict[Uid, Union[NDArray, List[NDArray]]]],
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

    scores = calculate_best_similarity_to_category(embeddings, categories_embeddings)

    category_sizes = {
        category: count_category_items(ground_truth, uids, category) for category in categories_embeddings.keys()
    }

    results: Dict[Method, Dict[Category, Tuple[Precision, Recall]]] = dict()

    for method in embeddings.keys():
        method_scores = scores[method]

        method_pr: Dict[Category, Tuple[Precision, Recall]] = dict()

        for category in categories_embeddings.keys():
            category_size = category_sizes[category]

            if category_size <= 0:
                continue

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


def calculate_precision_recall_auc(
    precision_recall: Dict[Method, Dict[Category, Tuple[Precision, Recall, int]]],
) -> Dict[Method, Dict[Category, float]]:
    auc: Dict[Method, Dict[Category, float]] = dict()

    for method, method_pr in precision_recall.items():
        method_auc: Dict[Category, float] = dict()

        for category, pr in method_pr.items():
            precision, recall, _ = pr

            total_area = 0.0

            if precision.shape[0] >= 2:
                start_precision = np.clip(precision[0], 0.0, 1.0)
                start_recall = np.clip(recall[0], 0.0, 1.0)

                for i in range(1, precision.shape[0]):
                    end_precision = np.clip(precision[i], 0.0, 1.0)
                    end_recall = np.clip(recall[i], 0.0, 1.0)

                    dx = np.clip(end_recall - start_recall, 0.0, 1.0)

                    min_y = min(start_precision, end_precision)
                    max_y = max(start_precision, end_precision)

                    total_area += dx * min_y + 0.5 * dx * (max_y - min_y)

                    start_precision = end_precision
                    start_recall = end_recall

            method_auc[category] = total_area

        auc[method] = method_auc

    return auc


def plot_precision_recall(
    precision_recall: Dict[Method, Dict[Category, Tuple[Precision, Recall, int]]],
    total_count: int,
    file: Path,
    category_names: Optional[Dict[Category, str]] = None,
    category_filter: Optional[Set[str]] = None,
    method_names: Optional[Dict[Method, str]] = None,
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
            values=rename_methods_dict(extract_nested_maps(precision_recall, category), method_names),
            xlabel="Recall",
            ylabel="Precision",
            legend_loc="upper left",
        )

        place_legend_outside(ax)

    fig.tight_layout()

    save_figure(fig, file)


def plot_precision_recall_auc(
    precision_recall_auc: Dict[Method, Dict[Category, float]],
    file: Path,
    category_names: Optional[Dict[Category, str]] = None,
    category_filter: Optional[Set[str]] = None,
    method_names: Optional[Dict[Method, str]] = None,
) -> None:
    categories = tuple(
        sorted(
            [
                category
                for category in get_keys_of_nested_maps(precision_recall_auc)
                if category_filter is None or category in category_filter
            ]
        )
    )

    fig, ax = plt.subplots()

    ax.set_title("Area Under the Precision Recall Curve")

    grouped_bar_plot(
        ax,
        groups=rename_categories_tuple(categories, category_names),
        values=rename_methods_dict(
            convert_nested_maps_to_tuples(precision_recall_auc, key_order=categories, default=lambda _: 0.0),
            method_names,
        ),
        xlabel="Objaverse 1.0 LVIS Category\n(size of category in parentheses)",
        ylabel="PR AUC",
        legend_loc="upper left",
    )

    place_legend_outside(ax)

    ax.set_ylim(ymin=0.0, ymax=1.0)

    fig.tight_layout()

    save_figure(fig, file)


def plot_precision_recall_auc_grid(
    precision_recall_auc: Dict[Method, Dict[Category, float]],
    file: Path,
    category_names: Optional[Dict[Category, str]] = None,
    category_filter: Optional[Set[str]] = None,
    method_names: Optional[Dict[Method, str]] = None,
) -> None:
    methods: Dict[Method, int] = dict()
    categories: Dict[Category, int] = dict()

    for method, category_pr_auc in precision_recall_auc.items():
        for category, value in category_pr_auc.items():
            if category_filter is not None and category not in category_filter:
                continue

            if method not in methods:
                methods[method] = len(methods)

            if category not in categories:
                categories[category] = len(categories)

    num_methods = len(methods)
    num_categories = len(categories)

    values = np.ones((num_methods, num_categories)) * np.nan
    for method, category_pr_auc in precision_recall_auc.items():
        for category, value in category_pr_auc.items():
            if category_filter is not None and category not in category_filter:
                continue

            values[methods[method]][categories[category]] = value

    fig, ax = plt.subplots(layout="constrained")

    ax.set_title("Area Under the Precision Recall Curve")

    comparison_grid_plot(
        fig,
        ax,
        values,
        xlabels=list(rename_categories_dict(categories, category_names).keys()),
        ylabels=list(rename_methods_dict(methods, method_names).keys()),
        colorbar_label="PR AUC",
    )

    save_figure(fig, file)
