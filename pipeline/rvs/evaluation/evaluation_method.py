from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from nerfstudio.utils.rich_utils import CONSOLE
from numpy.random import Generator
from numpy.typing import NDArray
from tqdm import tqdm

from rvs.evaluation.embedder import Embedder
from rvs.evaluation.lvis import LVISDataset
from rvs.evaluation.pipeline import PipelineEvaluationInstance
from rvs.utils.console import file_link


def evaluate_results(
    lvis: LVISDataset, embedder: Embedder, instance: PipelineEvaluationInstance, output_dir: Path
) -> None:
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
    avg_random_views_embeddings = embed_randomn_views_avg(
        lvis,
        available_uids,
        embedder,
        instance,
        np.random.default_rng(seed=238947978),
        3,  # FIXME: This should come from the config
    )
    available_uids = avg_selected_views_embeddings.keys()


def embed_selected_views_avg(
    lvis: LVISDataset,
    uids: Set[str],
    embedder: Embedder,
    instance: PipelineEvaluationInstance,
) -> Dict[str, NDArray]:
    embeddings = dict()

    for uid in tqdm(uids):
        model_file = Path(lvis.uid_to_file[uid])

        results_dir = instance.results_dir / model_file.name

        if results_dir.exists() and results_dir.is_dir():
            avg_embedding = None

            for image_file in results_dir.iterdir():
                if image_file.is_file() and image_file.name.endswith(".png"):
                    CONSOLE.log(f"Embedding selected view {file_link(image_file)}")

                    embedding = embedder.embed_image(image_file).detach().cpu().numpy()

                    avg_embedding = embedding if avg_embedding is None else avg_embedding + embedding

            if avg_embedding is not None:
                avg_embedding /= np.linalg.norm(avg_embedding, axis=1, keepdims=True)

                embeddings[uid] = avg_embedding

                CONSOLE.log(f"Embedded selected views of {file_link(model_file)}")

    return embeddings


def embed_randomn_views_avg(
    lvis: LVISDataset,
    uids: Set[str],
    embedder: Embedder,
    instance: PipelineEvaluationInstance,
    rng: Generator,
    number_of_views: int,
) -> Dict[str, NDArray]:
    embeddings = dict()

    for uid in tqdm(uids):
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
                CONSOLE.log(f"Embedding random view {file_link(image_file)}")

                embedding = embedder.embed_image(image_file).detach().cpu().numpy()

                avg_embedding = embedding if avg_embedding is None else avg_embedding + embedding

            if avg_embedding is not None:
                avg_embedding /= np.linalg.norm(avg_embedding, axis=1, keepdims=True)

                embeddings[uid] = avg_embedding

                CONSOLE.log(f"Embedded random views of {file_link(model_file)}")

    return embeddings
