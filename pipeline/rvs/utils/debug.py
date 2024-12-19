from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from PIL import Image as im

from rvs.pipeline.renderer import TrimeshRenderer, TrimeshRendererConfig, View


def render_sample_positions(
    file: Path, view: View, sample_positions: NDArray, callback: Callable[[View, im.Image], None]
) -> None:
    renderer = TrimeshRenderer(TrimeshRendererConfig())
    renderer.render(file, [view], callback, sample_positions=sample_positions)


def render_sample_clusters(
    file: Path,
    view: View,
    sample_positions: NDArray,
    sample_embeddings: NDArray,
    sample_clusters: NDArray,
    callback: Callable[[View, im.Image], None],
    hard_assignments: bool = False,
) -> None:
    if sample_clusters.shape[0] != 3:
        raise Exception("Can only render exactly 3 clusters")

    r_embedding = sample_clusters[0]
    g_embedding = sample_clusters[1]
    b_embedding = sample_clusters[2]

    sample_colors = np.zeros((sample_embeddings.shape[0], 3))

    for i in range(sample_embeddings.shape[0]):
        sample_embedding = sample_embeddings[i] / np.linalg.norm(sample_embeddings[i])

        if not hard_assignments:
            sample_colors[i, 0] = np.dot(r_embedding, sample_embedding)
            sample_colors[i, 1] = np.dot(g_embedding, sample_embedding)
            sample_colors[i, 2] = np.dot(b_embedding, sample_embedding)
        else:
            best = -1.0

            r_sim = np.dot(r_embedding, sample_embedding)
            g_sim = np.dot(g_embedding, sample_embedding)
            b_sim = np.dot(b_embedding, sample_embedding)

            if r_sim > best:
                best = r_sim
                sample_colors[i, 0] = 1.0
                sample_colors[i, 1] = 0.0
                sample_colors[i, 2] = 0.0

            if g_sim > best:
                best = g_sim
                sample_colors[i, 0] = 0.0
                sample_colors[i, 1] = 1.0
                sample_colors[i, 2] = 0.0

            if b_sim > best:
                best = b_sim
                sample_colors[i, 0] = 0.0
                sample_colors[i, 1] = 0.0
                sample_colors[i, 2] = 1.0

    renderer = TrimeshRenderer(TrimeshRendererConfig())
    renderer.render(file, [view], callback, sample_positions=sample_positions, sample_colors=sample_colors)
