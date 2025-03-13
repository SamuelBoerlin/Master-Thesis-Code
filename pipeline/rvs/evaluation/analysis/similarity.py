from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from rvs.evaluation.analysis.utils import Method, get_categories_tuple, rename_categories_tuple, rename_methods_dict
from rvs.evaluation.lvis import Category, LVISDataset, Uid
from rvs.utils.map import convert_nested_maps_to_tuples, get_keys_of_nested_maps
from rvs.utils.plot import grouped_bar_plot, save_figure


def calculate_similarity_to_ground_truth(
    lvis: LVISDataset,
    embeddings: Dict[Method, Dict[Uid, NDArray]],
    ground_truth: Dict[Category, NDArray],
    category_filter: Optional[Set[str]] = None,
) -> Dict[Method, Dict[Uid, float]]:
    similarities: Dict[Method, Dict[Uid, float]] = dict()

    for method in embeddings.keys():
        method_embeddings = embeddings[method]

        method_similarities: Dict[Uid, float] = dict()

        for uid in method_embeddings.keys():
            category = lvis.uid_to_category[uid]

            if category_filter is not None and category not in category_filter:
                continue

            method_embedding = method_embeddings[uid]
            ground_truth_embedding = ground_truth[category]

            method_similarities[uid] = np.dot(ground_truth_embedding, method_embedding)

        similarities[method] = method_similarities

    return similarities


def plot_avg_similarities_per_category(
    lvis: LVISDataset,
    similarities: Dict[Method, Dict[Uid, float]],
    file: Path,
    category_names: Optional[Dict[Category, str]] = None,
    method_names: Optional[Dict[Method, str]] = None,
) -> None:
    categories = get_categories_tuple(lvis, similarities)

    avg_similarities: Dict[Method, Dict[Category, float]] = dict()

    for method in similarities.keys():
        method_similarities = similarities[method]

        method_avg_similarities: Dict[Category, float] = {category: 0.0 for category in categories}
        method_avg_counts: Dict[Category, int] = {category: 0 for category in categories}

        for uid in method_similarities.keys():
            similarity = method_similarities[uid]

            category = lvis.uid_to_category[uid]

            method_avg_similarities[category] += similarity
            method_avg_counts[category] += 1

        for category in method_avg_similarities:
            count = method_avg_counts[category]
            if count > 0:
                method_avg_similarities[category] /= count

        avg_similarities[method] = method_avg_similarities

    fig, ax = plt.subplots()

    ax.set_title(
        "Similarity Between CLIP Embeddings of Images and Their\n Corresponding Ground Truth Category Text Prompts"
    )

    grouped_bar_plot(
        ax,
        groups=rename_categories_tuple(categories, category_names),
        values=convert_nested_maps_to_tuples(
            rename_methods_dict(avg_similarities, method_names), key_order=categories, default=lambda _: 0.0
        ),
        xlabel="Objaverse 1.0 LVIS Category\n(size of category in parentheses)",
        ylabel="Average Embedding Cosine-Similarity",
    )

    ax.set_ylim(ymin=0.0, ymax=1.0)

    fig.tight_layout()

    save_figure(fig, file)


def calculate_avg_similarity_between_category_models_and_category_ground_truths(
    lvis: LVISDataset,
    model_embeddings: Dict[Method, Dict[Uid, NDArray]],
    model_category: Category,
    category_embeddings: Dict[Category, NDArray],
) -> Dict[Method, Dict[Category, float]]:
    similarities: Dict[Method, Dict[Category, float]] = dict()

    for method in model_embeddings.keys():
        method_embeddings = model_embeddings[method]

        method_similarities: Dict[Category, float] = dict()

        for category, category_embedding in category_embeddings.items():
            total: float = 0.0
            count: int = 0

            for uid in method_embeddings.keys():
                uid_category = lvis.uid_to_category[uid]

                if uid_category == model_category:
                    total += np.dot(method_embeddings[uid], category_embedding)
                    count += 1

            if count > 0:
                method_similarities[category] = total / count
            else:
                method_similarities[category] = 0

        similarities[method] = method_similarities

    return similarities


def calculate_avg_similarity_between_all_models_and_category_ground_truths(
    lvis: LVISDataset,
    model_embeddings: Dict[Method, Dict[Uid, NDArray]],
    category_embeddings: Dict[Category, NDArray],
) -> Dict[Category, Dict[Method, Dict[Category, float]]]:
    return {
        category: calculate_avg_similarity_between_category_models_and_category_ground_truths(
            lvis, model_embeddings, category, category_embeddings
        )
        for category in category_embeddings.keys()
    }


def plot_avg_similariy_between_models_and_categories(
    similarities: Dict[Method, Dict[Category, float]],
    main_category: str,
    file: Path,
    category_names: Optional[Dict[Category, str]] = None,
    method_names: Optional[Dict[Method, str]] = None,
) -> None:
    main_category_name = rename_categories_tuple((main_category,), category_names)[0]

    categories = tuple(sorted(list(get_keys_of_nested_maps(similarities))))

    fig, ax = plt.subplots()

    ax.set_title(
        f'Similarity Between CLIP Embeddings of Images in Category\n"{main_category_name}"\nand Embeddings of Category Text Prompts'
    )

    grouped_bar_plot(
        ax,
        groups=rename_categories_tuple(categories, category_names),
        values=convert_nested_maps_to_tuples(
            rename_methods_dict(similarities, method_names), key_order=categories, default=lambda _: 0.0
        ),
        xlabel="Objaverse 1.0 LVIS Category\n(size of category in parentheses)",
        ylabel="Average Embedding Cosine-Similarity",
    )

    ax.set_ylim(ymin=0.0, ymax=1.0)

    fig.tight_layout()

    save_figure(fig, file)
