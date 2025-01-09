from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from matplotlib import pyplot as plt
from nerfstudio.utils.rich_utils import CONSOLE
from numpy.random import Generator
from numpy.typing import NDArray
from tqdm import tqdm

from rvs.evaluation.embedder import Embedder
from rvs.evaluation.lvis import LVISDataset
from rvs.evaluation.pipeline import PipelineEvaluationInstance
from rvs.utils.console import file_link
from rvs.utils.map import convert_nested_maps_to_tuples, get_keys_of_nested_maps
from rvs.utils.plot import grouped_bar_plot, save_figure


def evaluate_results(
    lvis: LVISDataset, embedder: Embedder, instance: PipelineEvaluationInstance, output_dir: Path
) -> None:
    CONSOLE.rule("Embedding prompts for categories...")
    categories_embeddings = embed_categories(lvis.dataset.keys(), embedder)

    available_uids = set()
    for category in lvis.dataset.keys():
        for uid in lvis.dataset[category]:
            available_uids.add(uid)

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
        3,  # FIXME: This should come from the config
    )
    available_uids = avg_selected_views_embeddings.keys()

    CONSOLE.rule("Calculate similarities...")
    similarities = calculate_similarities(
        lvis,
        {
            "Average Embedding of Selected Views": avg_selected_views_embeddings,
            "Average Embedding of Random Views": avg_random_views_embeddings,
        },
        categories_embeddings,
    )

    CONSOLE.rule("Create results...")

    category_names = {
        category: category + " (" + str(count_category_items(lvis, category, available_uids)) + ")"
        for category in lvis.categories
    }

    plot_avg_similarities_per_category(
        lvis, similarities, output_dir / "similarities.png", category_names=category_names
    )


def embed_selected_views_avg(
    lvis: LVISDataset,
    uids: Set[str],
    embedder: Embedder,
    instance: PipelineEvaluationInstance,
) -> Dict[str, NDArray]:
    embeddings = dict()

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
    uids: Set[str],
    embedder: Embedder,
    instance: PipelineEvaluationInstance,
    rng: Generator,
    number_of_views: int,
) -> Dict[str, NDArray]:
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


def embed_categories(categories: List[str], embedder: Embedder) -> Dict[str, NDArray]:
    embeddings = dict()

    for category in tqdm(categories):
        prompt = category_name_to_embedding_prompt(category)
        # CONSOLE.log(f"Embedding category {category} as '{prompt}'")
        embeddings[category] = embedder.embed_text_numpy(prompt)

    return embeddings


def category_name_to_embedding_prompt(category: str) -> None:
    return category.replace("_", " ")


def get_categories_of_uids(lvis: LVISDataset, uids: List[str]) -> Set[str]:
    categories = set()
    for uid in uids:
        categories.add(lvis.uid_to_category[uid])
    return categories


def get_categories_tuple(lvis: LVISDataset, map: Dict[str, Dict[str, Any]]) -> Tuple[str, ...]:
    return tuple(sorted(list(get_categories_of_uids(lvis, get_keys_of_nested_maps(map)))))


def rename_categories_tuple(
    categories: Tuple[str, ...],
    category_names: Optional[Dict[str, str]],
) -> Tuple[str, ...]:
    return tuple([category_names[category] if category in category_names else category for category in categories])


def calculate_similarities(
    lvis: LVISDataset, embeddings: Dict[str, Dict[str, NDArray]], ground_truth: Dict[str, Dict[str, NDArray]]
) -> Dict[str, Dict[str, float]]:
    similarities: Dict[str, Dict[str, float]] = dict()  # method, uid, similarity

    for method in embeddings.keys():
        method_embeddings = embeddings[method]

        method_similarities: Dict[str, float] = dict()

        for uid in method_embeddings.keys():
            category = lvis.uid_to_category[uid]

            method_embedding = method_embeddings[uid]
            ground_truth_embedding = ground_truth[category]

            method_similarities[uid] = np.dot(ground_truth_embedding, method_embedding)

        similarities[method] = method_similarities

    return similarities


def count_category_items(lvis: LVISDataset, category: str, uids: List[str]) -> int:
    count = 0
    for uid in uids:
        item_category = lvis.uid_to_category[uid]
        if item_category == category:
            count += 1
    return count


def plot_avg_similarities_per_category(
    lvis: LVISDataset,
    similarities: Dict[str, Dict[str, float]],
    file: Path,
    category_names: Optional[Dict[str, str]] = None,
) -> None:
    categories = get_categories_tuple(lvis, similarities)

    avg_similarities: Dict[str, Dict[str, float]] = dict()  # method, category, similarity

    for method in similarities.keys():
        method_similarities = similarities[method]  # uid, similarity

        method_avg_similarities: Dict[str, float] = {category: 0.0 for category in categories}  # category, similarity
        method_avg_counts: Dict[str, int] = {category: 0 for category in categories}  # category, similarity

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
        xlabel="Objaverse 1.0 LVIS Categories",
        ylabel="Embedding Cosine-Similarity",
    )

    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylim(bottom=0.0, top=1.0)

    fig.tight_layout()

    save_figure(fig, file)
