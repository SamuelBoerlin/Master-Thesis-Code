import pickle
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
from nerfstudio.utils.rich_utils import CONSOLE
from numpy.typing import NDArray
from tqdm import tqdm

from rvs.evaluation.analysis.clusters import (
    calculate_clusters_avg_per_category,
    calculte_clusters_histogram_per_category,
    plot_clusters_avg_per_category,
    plot_clusters_histogram,
    plot_elbows_samples,
)
from rvs.evaluation.analysis.loss import calculate_avg_loss, plot_avg_loss
from rvs.evaluation.analysis.precision_recall import (
    calculate_precision_recall,
    calculate_precision_recall_auc,
    plot_precision_recall,
    plot_precision_recall_auc,
    plot_precision_recall_auc_grid,
)
from rvs.evaluation.analysis.similarity import (
    calculate_avg_similarity_between_all_models_and_category_ground_truths,
    calculate_similarity_to_ground_truth,
    plot_avg_similarities_per_category,
    plot_avg_similarities_per_category_grid,
    plot_avg_similariy_between_models_and_categories,
    plot_avg_similariy_between_models_and_categories_grid,
)
from rvs.evaluation.analysis.utils import Method, count_category_items
from rvs.evaluation.analysis.views import (
    calculate_selected_views_avg_per_category,
    calculate_views_histogram_avg,
    calculte_selected_views_histogram_per_category,
    embed_best_views,
    embed_random_views,
    embed_selected_views,
    plot_selected_views_avg_per_category,
    plot_selected_views_histogram,
    plot_selected_views_samples,
)
from rvs.evaluation.embedder import CachedEmbedder
from rvs.evaluation.lvis import Category, LVISDataset, Uid
from rvs.evaluation.pipeline import PipelineEvaluationInstance
from rvs.utils.cache import get_evaluation_prompt_embedding_cache_key
from rvs.utils.random import discrete_distribution


def evaluate_results(
    lvis: LVISDataset,
    embedder: CachedEmbedder,
    instance: PipelineEvaluationInstance,
    seed: int,
    output_dir: Path,
    skip_validation: bool,
) -> None:
    dumps_dir = output_dir / "dumps"
    dumps_dir.mkdir(parents=True, exist_ok=True)

    CONSOLE.rule("Embedding prompts for categories...")
    categories_embeddings = embed_categories(lvis.dataset.keys(), lvis, embedder)

    available_uids: Set[Uid] = set()
    for category in lvis.dataset.keys():
        for uid in lvis.dataset[category]:
            available_uids.add(uid)

    CONSOLE.rule("Calculate losses...")
    avg_clip_loss = calculate_avg_loss(lvis, available_uids, instance, "clip_loss")
    dump_result(avg_clip_loss, dumps_dir / "avg_clip_loss.pkl")
    avg_dino_loss = calculate_avg_loss(lvis, available_uids, instance, "dino_loss")
    dump_result(avg_dino_loss, dumps_dir / "avg_dino_loss.pkl")
    avg_rgb_loss = calculate_avg_loss(lvis, available_uids, instance, "rgb_loss")
    dump_result(avg_rgb_loss, dumps_dir / "avg_rgb_loss.pkl")
    avg_image_psnr = calculate_avg_loss(lvis, available_uids, instance, "image_psnr")
    dump_result(avg_image_psnr, dumps_dir / "avg_image_psnr.pkl")
    avg_train_loss = calculate_avg_loss(lvis, available_uids, instance, "train_loss")
    dump_result(avg_train_loss, dumps_dir / "avg_train_loss.pkl")

    CONSOLE.rule("Calculate number of selected views...")
    avg_number_of_views_per_category = calculate_selected_views_avg_per_category(
        lvis, available_uids, instance, skip_validation=skip_validation
    )
    dump_result(avg_number_of_views_per_category, dumps_dir / "avg_number_of_views_per_category.pkl")
    histogram_of_views_per_category = calculte_selected_views_histogram_per_category(
        lvis, available_uids, instance, skip_validation=skip_validation
    )
    dump_result(histogram_of_views_per_category, dumps_dir / "histogram_of_views_per_category.pkl")

    min_number_of_views: int = None
    max_number_of_views: int = None
    for histogram in histogram_of_views_per_category.values():
        nonzero_i = np.nonzero(histogram)[0]
        if nonzero_i.shape[0] > 0:
            if min_number_of_views is None:
                min_number_of_views = nonzero_i[0]
            else:
                min_number_of_views = min(min_number_of_views, nonzero_i[0])
            if max_number_of_views is None:
                max_number_of_views = nonzero_i[-1]
            else:
                max_number_of_views = max(max_number_of_views, nonzero_i[-1])
    if min_number_of_views is None or max_number_of_views is None:
        min_number_of_views = 0
        max_number_of_views = 0

    number_of_views_str = f"${min_number_of_views} \leq N \leq {max_number_of_views}$"
    if min_number_of_views == max_number_of_views:
        number_of_views_str = f"$N \equal {max_number_of_views}$"

    method_titles: Dict[Method, str] = {
        "best_embedding_of_views_wrt_ground_truth": "Best Embedding of Views w.r.t. Ground Truth",
        "avg_embedding_of_selected_views": f"Average Embedding of Selected Views ({number_of_views_str})",
        "best_embedding_of_selected_views_wrt_query": f"Best Embedding of Selected Views w.r.t. Query ({number_of_views_str})",
        "avg_embedding_of_random_views": f"Average Embedding of Random Views ({number_of_views_str})",
        "best_embedding_of_random_views_wrt_query": f"Best Embedding of Random Views w.r.t. Query ({number_of_views_str})",
    }

    # avg_number_of_views = np.average([value for _, value in avg_number_of_views_per_category.items()])
    avg_histogram_of_views = calculate_views_histogram_avg(histogram_of_views_per_category)
    dump_result(avg_histogram_of_views, dumps_dir / "avg_histogram_of_views.pkl")

    CONSOLE.rule("Embedding selected views...")
    avg_selected_views_embeddings, all_selected_views_embeddings = embed_selected_views(
        lvis,
        available_uids,
        embedder,
        instance,
        skip_validation=skip_validation,
    )
    dump_result(avg_selected_views_embeddings, dumps_dir / "avg_selected_views_embeddings.pkl")
    dump_result(all_selected_views_embeddings, dumps_dir / "all_selected_views_embeddings.pkl")
    available_uids = avg_selected_views_embeddings.keys()

    CONSOLE.rule("Embedding equivalent distribution of random views...")
    avg_equiv_random_views_embeddings, all_equiv_random_views_embeddings, all_equiv_random_views_indices = (
        embed_random_views(
            lvis,
            available_uids,
            embedder,
            instance,
            np.random.default_rng(seed=seed),
            discrete_distribution(np.arange(avg_histogram_of_views.shape[0]), avg_histogram_of_views),
        )
    )
    dump_result(avg_equiv_random_views_embeddings, dumps_dir / "avg_equiv_random_views_embeddings.pkl")
    dump_result(all_equiv_random_views_embeddings, dumps_dir / "all_equiv_random_views_embeddings.pkl")
    dump_result(all_equiv_random_views_indices, dumps_dir / "all_equiv_random_views_indices.pkl")
    available_uids = avg_equiv_random_views_embeddings.keys()

    CONSOLE.rule("Embedding best views w.r.t. ground truth...")
    best_views_embeddings, best_views_indices = embed_best_views(
        lvis,
        available_uids,
        embedder,
        instance,
        categories_embeddings,
    )
    dump_result(best_views_embeddings, dumps_dir / "best_views_embeddings.pkl")
    dump_result(best_views_indices, dumps_dir / "best_views_indices.pkl")
    available_uids = best_views_embeddings.keys()

    CONSOLE.rule("Calculate number of clusters...")
    avg_number_of_clusters_per_category = calculate_clusters_avg_per_category(lvis, available_uids, instance)
    dump_result(avg_number_of_clusters_per_category, dumps_dir / "avg_number_of_clusters_per_category.pkl")
    histogram_of_clusters_per_category = calculte_clusters_histogram_per_category(lvis, available_uids, instance)
    dump_result(histogram_of_clusters_per_category, dumps_dir / "histogram_of_clusters_per_category.pkl")

    CONSOLE.rule("Calculate similarities...")
    similarities = calculate_similarity_to_ground_truth(
        lvis,
        {
            "best_embedding_of_views_wrt_ground_truth": best_views_embeddings,
            "avg_embedding_of_selected_views": avg_selected_views_embeddings,
            "avg_embedding_of_random_views": avg_equiv_random_views_embeddings,
        },
        categories_embeddings,
    )
    dump_result(similarities, dumps_dir / "similarities.pkl")
    cross_similarities = calculate_avg_similarity_between_all_models_and_category_ground_truths(
        lvis,
        {
            "best_embedding_of_views_wrt_ground_truth": best_views_embeddings,
            "avg_embedding_of_selected_views": avg_selected_views_embeddings,
            "avg_embedding_of_random_views": avg_equiv_random_views_embeddings,
        },
        categories_embeddings,
    )
    dump_result(cross_similarities, dumps_dir / "cross_similarities.pkl")

    CONSOLE.rule("Calculate precision/recall...")
    precision_recall = calculate_precision_recall(
        {
            "best_embedding_of_views_wrt_ground_truth": best_views_embeddings,
            "avg_embedding_of_selected_views": avg_selected_views_embeddings,
            "best_embedding_of_selected_views_wrt_query": all_selected_views_embeddings,
            "avg_embedding_of_random_views": avg_equiv_random_views_embeddings,
            "best_embedding_of_random_views_wrt_query": all_equiv_random_views_embeddings,
        },
        categories_embeddings,
        lvis.uid_to_category,
    )
    dump_result(precision_recall, dumps_dir / "precision_recall.pkl")
    precision_recall_auc = calculate_precision_recall_auc(precision_recall)
    dump_result(precision_recall_auc, dumps_dir / "precision_recall_auc.pkl")

    CONSOLE.rule("Create results...")

    category_names_with_sizes = {
        category: category + " (" + str(count_category_items(lvis.uid_to_category, available_uids, category)) + ")"
        for category in lvis.categories
    }

    plot_avg_loss(
        avg_clip_loss, "CLIP Loss", "Huber Loss ($\delta \equal 1.25$)", output_dir / "losses" / "clip_loss.png"
    )
    plot_avg_loss(avg_dino_loss, "DINO Loss", "Mean Squared Error", output_dir / "losses" / "dino_loss.png")
    plot_avg_loss(avg_train_loss, "Train Loss", "Loss", output_dir / "losses" / "train_loss.png")

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

    for category in tqdm(categories_embeddings.keys()):
        plot_elbows_samples(
            lvis,
            available_uids,
            category,
            instance,
            np.random.default_rng(seed=seed),
            10,
            output_dir / "elbows" / f"{category}.png",
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
        method_names=method_titles,
    )
    plot_avg_similarities_per_category_grid(
        lvis,
        similarities,
        output_dir / "similarities_grid.png",
        category_names=category_names_with_sizes,
        method_names=method_titles,
    )

    for category in cross_similarities.keys():
        plot_avg_similariy_between_models_and_categories(
            cross_similarities[category],
            category,
            output_dir / "cross_similarity" / f"{category}.png",
            category_names=category_names_with_sizes,
            method_names=method_titles,
        )
        plot_avg_similariy_between_models_and_categories_grid(
            cross_similarities[category],
            category,
            output_dir / "cross_similarity" / f"{category}_grid.png",
            category_names=category_names_with_sizes,
            method_names=method_titles,
        )

    for category in categories_embeddings.keys():
        plot_precision_recall(
            precision_recall,
            len(available_uids),
            output_dir / "precision_recall" / f"{category}.png",
            category_filter={category},
            method_names=method_titles,
        )

    plot_precision_recall_auc(
        precision_recall_auc,
        output_dir / "precision_recall_auc.png",
        category_names=category_names_with_sizes,
        method_names=method_titles,
    )
    plot_precision_recall_auc_grid(
        precision_recall_auc,
        output_dir / "precision_recall_auc_grid.png",
        category_names=category_names_with_sizes,
        method_names=method_titles,
    )

    CONSOLE.rule("Create selected views samples...")

    for category in tqdm(categories_embeddings.keys()):
        plot_selected_views_samples(
            lvis,
            available_uids,
            category,
            instance,
            np.random.default_rng(seed=seed),
            5 * 5,
            output_dir / "views" / f"{category}.png",
            skip_validation=skip_validation,
        )


def embed_categories(
    categories: List[Category], lvis: LVISDataset, embedder: CachedEmbedder
) -> Dict[Category, NDArray]:
    embeddings = dict()

    for category in tqdm(categories):
        prompt = category_name_to_embedding_prompt(category, lvis)
        CONSOLE.log(f"Embedding category '{category}' as '{prompt}'")
        embeddings[category] = embedder.embed_text_numpy(
            prompt, cache_key=get_evaluation_prompt_embedding_cache_key(prompt)
        )

    return embeddings


def category_name_to_embedding_prompt(category: Category, lvis: LVISDataset) -> str:
    category_name = lvis.get_category_name(category).replace("_", " ")
    return f"a photo of a {category_name}."


def dump_result(result: Any, file: Path) -> None:
    with file.open("wb") as f:
        pickle.dump(result, f)


def load_result(file: Path) -> Any:
    with file.open("rb") as f:
        return pickle.load(f)
