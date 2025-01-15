from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from matplotlib import pyplot as plt
from nerfstudio.utils.rich_utils import CONSOLE
from numpy.random import Generator
from numpy.typing import NDArray
from PIL import Image as im
from tqdm import tqdm

from rvs.evaluation.analysis.histogram import (
    calculate_avg_per_category,
    calculate_discrete_histogram_per_category,
    plot_avg_per_category,
    plot_histogram,
)
from rvs.evaluation.embedder import Embedder
from rvs.evaluation.index import load_index
from rvs.evaluation.lvis import Category, LVISDataset, Uid
from rvs.evaluation.pipeline import PipelineEvaluationInstance
from rvs.utils.console import file_link
from rvs.utils.plot import image_grid_plot, save_figure


def count_selected_views(
    lvis: LVISDataset,
    uids: Set[Uid],
    instance: PipelineEvaluationInstance,
) -> Dict[Uid, int]:
    counts: Dict[Uid, int] = dict()

    for uid in tqdm(sorted(uids)):
        model_file = Path(lvis.uid_to_file[uid])

        index_file = instance.get_index_file(model_file)

        try:
            images, _ = load_index(index_file)
            counts[uid] = len(images)
        except Exception:
            pass

    return counts


def calculate_selected_views_avg_per_category(
    lvis: LVISDataset,
    uids: Set[Uid],
    instance: PipelineEvaluationInstance,
) -> Dict[Category, float]:
    return calculate_avg_per_category(lvis, count_selected_views(lvis, uids, instance))


def calculte_selected_views_histogram_per_category(
    lvis: LVISDataset,
    uids: Set[Uid],
    instance: PipelineEvaluationInstance,
) -> Dict[Category, NDArray]:
    return calculate_discrete_histogram_per_category(lvis, count_selected_views(lvis, uids, instance))


def plot_selected_views_avg_per_category(
    avg_counts: Dict[Category, float],
    file: Path,
    category_names: Optional[Dict[Category, str]] = None,
    category_filter: Optional[Set[str]] = None,
) -> None:
    plot_avg_per_category(
        avg_counts,
        file,
        title="Average Number of Selected Views",
        xlabel="Objaverse 1.0 LVIS Category\n(size of category in parentheses)",
        category_names=category_names,
        category_filter=category_filter,
    )


def plot_selected_views_histogram(
    histograms: Dict[Category, NDArray],
    file: Path,
    category_names: Optional[Dict[Category, str]] = None,
    category_filter: Optional[Set[str]] = None,
) -> None:
    plot_histogram(
        histograms,
        file,
        title="Number of Selected Views for Objaverse 1.0 LVIS Category",
        xlabel="Number of Selected Views",
        category_names=category_names,
        category_filter=category_filter,
    )


def embed_selected_views_avg(
    lvis: LVISDataset,
    uids: Set[Uid],
    embedder: Embedder,
    instance: PipelineEvaluationInstance,
) -> Dict[Uid, NDArray]:
    embeddings: Dict[Uid, NDArray] = dict()

    for uid in tqdm(sorted(uids)):
        model_file = Path(lvis.uid_to_file[uid])

        index_file = instance.get_index_file(model_file)

        images: List[Path] = None

        try:
            images, _ = load_index(index_file)
        except Exception:
            pass

        if images is not None:
            avg_embedding = None

            for image_file in images:
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


def plot_selected_views_samples(
    lvis: LVISDataset,
    uids: Set[Uid],
    category: Category,
    instance: PipelineEvaluationInstance,
    rng: Generator,
    number_of_views: int,
    file: Path,
) -> None:
    samples: List[Tuple[im.Image, Uid]] = []

    selection_rng = np.random.default_rng(seed=rng)

    try:
        all_views: List[Tuple[Path, Uid]] = []

        for uid in lvis.dataset[category]:
            if uid in uids:
                model_file = Path(lvis.uid_to_file[uid])

                index_file = instance.get_index_file(model_file)

                try:
                    selected_views, _ = load_index(index_file)
                    for view in selected_views:
                        all_views.append((view, uid))
                except Exception:
                    pass

        selection_rng.shuffle(all_views)

        for view, uid in all_views[:number_of_views]:
            image = im.open(view)
            image.load()
            samples.append((image, uid))

        fig = plt.figure()

        fig_size = fig.get_size_inches()
        fig.set_size_inches(fig_size[0], fig_size[0])

        image_grid_plot(
            fig,
            images=[sample[0] for sample in samples],
            labels=[sample[1] for sample in samples],
            label_face_alpha=0.5,
            border_color="black",
        )

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.set_facecolor((0, 0, 0, 0))

        fig.tight_layout()

        save_figure(fig, file)

    finally:
        for image, _ in samples:
            image.close()
