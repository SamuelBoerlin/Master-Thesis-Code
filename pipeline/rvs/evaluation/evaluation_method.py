from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from nerfstudio.utils.rich_utils import CONSOLE
from numpy.random import Generator
from numpy.typing import NDArray
from tqdm import tqdm

from rvs.evaluation.analysis.precision_recall import calculate_precision_recall, plot_precision_recall
from rvs.evaluation.analysis.similarity import calculate_similarity_to_ground_truth, plot_avg_similarities_per_category
from rvs.evaluation.analysis.utils import count_category_items
from rvs.evaluation.embedder import Embedder
from rvs.evaluation.lvis import Category, LVISDataset, Uid
from rvs.evaluation.pipeline import PipelineEvaluationInstance
from rvs.utils.console import file_link


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

    CONSOLE.rule("Calculate similarities...")
    similarities = calculate_similarity_to_ground_truth(
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

    CONSOLE.rule("Create results...")

    plot_avg_similarities_per_category(
        lvis,
        similarities,
        output_dir / "similarities.png",
        category_names={
            category: category + " (" + str(count_category_items(lvis.uid_to_category, available_uids, category)) + ")"
            for category in lvis.categories
        },
    )

    for category in categories_embeddings.keys():
        plot_precision_recall(
            precision_recall,
            len(available_uids),
            output_dir / f"precision_recall_{category}.png",
            category_filter={category},
        )


def embed_selected_views_avg(
    lvis: LVISDataset,
    uids: Set[Uid],
    embedder: Embedder,
    instance: PipelineEvaluationInstance,
) -> Dict[Uid, NDArray]:
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


def embed_categories(categories: List[Category], embedder: Embedder) -> Dict[Category, NDArray]:
    embeddings = dict()

    for category in tqdm(categories):
        prompt = category_name_to_embedding_prompt(category)
        # CONSOLE.log(f"Embedding category {category} as '{prompt}'")
        embeddings[category] = embedder.embed_text_numpy(prompt)

    return embeddings


def category_name_to_embedding_prompt(category: Category) -> None:
    return category.replace("_", " ")
