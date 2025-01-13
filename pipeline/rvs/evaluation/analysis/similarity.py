from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from rvs.evaluation.analysis.utils import Method, get_categories_tuple, rename_categories_tuple
from rvs.evaluation.lvis import Category, LVISDataset, Uid
from rvs.utils.map import convert_nested_maps_to_tuples
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

    ax.set_title("Similarity Between Image and Prompt CLIP Embedding")

    grouped_bar_plot(
        ax,
        groups=rename_categories_tuple(categories, category_names),
        values=convert_nested_maps_to_tuples(avg_similarities, categories),
        xlabel="Prompts for Objaverse 1.0 LVIS Categories\n(size of categories in parentheses not part of prompt)",
        ylabel="Average Embedding Cosine-Similarity",
    )

    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylim(ymin=0.0, ymax=1.0)

    fig.tight_layout()

    save_figure(fig, file)
