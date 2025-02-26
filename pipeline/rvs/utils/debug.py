from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Patch
from numpy.typing import NDArray
from PIL import Image as im

from rvs.pipeline.renderer import RenderOutput, TrimeshRenderer, TrimeshRendererConfig, View
from rvs.pipeline.state import Normalization, PipelineState
from rvs.utils.plot import cluster_colors


def render_sample_positions(
    file: Path,
    view: View,
    normalization: Normalization,
    sample_positions: NDArray,
    callback: Callable[[View, im.Image], None],
    render_as_plot: bool = True,
) -> None:
    output = RenderOutput(
        path=None,
        callback=lambda view, image: render_image_plot(view, image, callback)
        if render_as_plot
        else callback(view, image),
    )

    state = PipelineState(None)
    state.model_normalization = normalization

    renderer = TrimeshRenderer(TrimeshRendererConfig())
    renderer.render(
        file,
        [view],
        output,
        state,
        sample_positions=sample_positions,
    )


def render_sample_clusters(
    file: Path,
    view: View,
    normalization: Normalization,
    sample_positions: NDArray,
    sample_embeddings: NDArray,
    num_clusters: int,
    hard_classifier: Callable[[NDArray], NDArray[np.intp]],
    soft_classifier: Callable[[NDArray], NDArray],
    callback: Callable[[View, im.Image], None],
    hard_assignments: bool = False,
    render_as_plot: bool = True,
) -> None:
    num_samples = sample_embeddings.shape[0]

    colors = cluster_colors(num_clusters)
    labels = [str(i + 1) for i in range(num_clusters)]

    sample_colors = np.zeros((num_samples, 3))

    for i in range(num_samples):
        sample_embedding = sample_embeddings[i] / np.linalg.norm(sample_embeddings[i])

        if hard_assignments:
            cluster_idx = hard_classifier(np.array([sample_embedding]))[0]
            sample_colors[i] = colors[cluster_idx]
        else:
            soft_labels = soft_classifier(np.array([sample_embedding]))[0]
            sample_colors[i] += soft_labels * colors

    output = RenderOutput(
        path=None,
        callback=lambda view, image: render_image_plot(
            view, image, callback, figure_setup=color_legend(colors, labels, "Clusters")
        )
        if render_as_plot
        else callback(view, image),
    )

    state = PipelineState(None)
    state.model_normalization = normalization

    renderer = TrimeshRenderer(TrimeshRendererConfig())
    renderer.render(
        file,
        [view],
        output,
        state,
        sample_positions=sample_positions,
        sample_colors=sample_colors,
    )


def render_sample_kmeans_clusters_with_cosine_similarity(
    file: Path,
    view: View,
    normalization: Normalization,
    sample_positions: NDArray,
    sample_embeddings: NDArray,
    cluster_centroids: NDArray,
    callback: Callable[[View, im.Image], None],
    hard_assignments: bool = False,
    render_as_plot: bool = True,
) -> None:
    num_clusters = cluster_centroids.shape[0]
    num_samples = sample_embeddings.shape[0]

    colors = cluster_colors(num_clusters)
    labels = [str(i + 1) for i in range(num_clusters)]

    sample_colors = np.zeros((num_samples, 3))

    for i in range(num_samples):
        sample_embedding = sample_embeddings[i] / np.linalg.norm(sample_embeddings[i])

        if not hard_assignments:
            sum = 0.0
            for j in range(num_clusters):
                weight = np.dot(cluster_centroids[j], sample_embedding)
                sample_colors[i] += weight * colors[j]
                sum += weight
            if sum > 0.0:
                sample_colors[i] /= sum
        else:
            best = -1.0
            for j in range(num_clusters):
                sim = np.dot(cluster_centroids[j], sample_embedding)
                if sim > best:
                    best = sim
                    sample_colors[i] = colors[j]

    output = RenderOutput(
        path=None,
        callback=lambda view, image: render_image_plot(
            view, image, callback, figure_setup=color_legend(colors, labels, "Clusters")
        )
        if render_as_plot
        else callback(view, image),
    )

    state = PipelineState(None)
    state.model_normalization = normalization

    renderer = TrimeshRenderer(TrimeshRendererConfig())
    renderer.render(
        file,
        [view],
        output,
        state,
        sample_positions=sample_positions,
        sample_colors=sample_colors,
    )


def render_image_plot(
    view: View,
    image: im.Image,
    callback: Callable[[View, im.Image], None],
    figure_setup: Optional[Callable[[plt.Figure, plt.Axes], None]] = None,
) -> None:
    if callback is None:
        return

    fig, ax = plt.subplots()
    try:
        ax.imshow(image)

        ax.axis("off")
        ax.set_aspect("equal")

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.set_facecolor((0, 0, 0, 0))

        if figure_setup is not None:
            figure_setup(fig, ax)

        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        data = canvas.buffer_rgba()

        with im.frombuffer("RGBA", canvas.get_width_height(), data, "raw", "RGBA", 0, 1) as plot_image:
            callback(view, plot_image)
    finally:
        plt.close(fig)


def color_legend(colors: List[Any], labels: List[str], title: str) -> Callable[[plt.Figure, plt.Axes], None]:
    def figure_setup(fig: plt.Figure, ax: plt.Axes) -> None:
        patches = [Patch(color=color, label=label) for color, label in zip(colors, labels)]
        ax.legend(handles=patches, title=title)

    return figure_setup
