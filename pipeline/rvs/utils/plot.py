from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, NewType, Optional, Tuple, Union

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import get_cmap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from numpy.typing import NDArray
from PIL import Image as im

from rvs.utils.elbow import Elbow

Precision = NewType("Precision", NDArray)
Recall = NewType("Recall", NDArray)
Label = NewType("Label", str)
UnitToInches = NamedTuple("UnitToInches", [("x", float), ("y", float)])
Inches = NamedTuple("Inches", [("width", float), ("height", float)])


def render_figure(fig: Figure, callback: Callable[[im.Image], None]) -> None:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    data = canvas.buffer_rgba()

    with im.frombuffer("RGBA", canvas.get_width_height(), data, "raw", "RGBA", 0, 1) as plot_image:
        callback(plot_image)


def save_figure(fig: Figure, file: Path) -> None:
    def save(image: im.Image) -> None:
        file.parent.mkdir(parents=True, exist_ok=True)
        image.save(file)

    render_figure(fig, save)


def measure_figure(fig: Figure) -> Tuple[Bbox, Inches, UnitToInches]:
    fig.canvas.draw()

    fig_bb: Bbox = fig.get_window_extent()
    fig_size_inches: NDArray = fig.get_size_inches()

    return (
        fig_bb,
        Inches(fig_size_inches[0], fig_size_inches[1]),
        UnitToInches(fig_size_inches[0] / fig_bb.width, fig_size_inches[1] / fig_bb.height),
    )


def measure_artist(artist: Artist) -> Tuple[Bbox, Inches, UnitToInches]:
    fig = artist.get_figure()

    if fig is None:
        raise ValueError("Artist must belong to a figure")

    _, _, unit_to_inches = measure_figure(fig)

    artist_bb: Bbox = artist.get_window_extent()

    return (artist_bb, Inches(artist_bb.width * unit_to_inches.x, artist_bb.height * unit_to_inches.y), unit_to_inches)


def bar_plot(
    ax: Axes,
    names: Tuple[str, ...],
    values: Union[Tuple[float, ...], NDArray],
    xlabel: str = None,
    ylabel: str = None,
) -> None:
    ax.bar(names, values)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")


def histogram_plot(
    ax: Axes,
    buckets: Tuple[str, ...],
    values: Union[Tuple[float, ...], NDArray],
    xlabel: str = None,
    ylabel: str = None,
) -> None:
    ax.bar(buckets, values)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.yaxis.get_major_locator().set_params(integer=True)


def discrete_histogram_plot(
    ax: Axes,
    values: NDArray,
    xlabel: str = None,
    ylabel: str = None,
    trim: bool = True,
) -> None:
    start = 0

    if trim:
        start = (values > 0).argmax()
        values = np.trim_zeros(values)

    buckets = tuple([str(nr) for nr in range(start, start + len(values))])

    histogram_plot(ax, buckets, values, xlabel=xlabel, ylabel=ylabel)


def grouped_bar_plot(
    ax: Axes,
    groups: Tuple[str, ...],
    values: Dict[Label, Union[Tuple[float, ...], NDArray]],
    bar_width=0.2,
    xlabel: str = None,
    ylabel: str = None,
    legend_loc: Union[str, int] = None,
) -> None:
    xs = np.arange(len(groups))
    ys = values

    x_offset = -bar_width * (len(ys.values()) - 1) * 0.5
    for y in ys.values():
        ax.bar(xs + x_offset, y, bar_width)
        x_offset += bar_width

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_xticks(xs, groups)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.legend(ys.keys(), loc=legend_loc)

    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")


def precision_recall_plot(
    ax: Axes,
    values: Dict[Label, Tuple[Precision, Recall]],
    xlabel: str = None,
    ylabel: str = None,
    legend_loc: Union[str, int] = None,
) -> None:
    for label in values.keys():
        ys, xs, *_ = values[label]

        ax.plot(xs, ys, "-o")

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_xlim(xmin=0.0, xmax=1.0)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_ylim(ymin=0.0, ymax=1.0)

    ax.legend(values.keys(), loc=legend_loc)


def place_legend_outside(
    ax: Axes,
    enlarge_figure: bool = True,
) -> None:
    ax.get_legend().set_bbox_to_anchor((1.0, 1.0))

    ax.figure.canvas.draw()

    legend_bb: Bbox = ax.get_legend().get_window_extent()

    _, fig_size_inches, unit_to_inches = measure_figure(ax.figure)

    if enlarge_figure:
        ax.figure.set_size_inches(fig_size_inches.width + legend_bb.width * unit_to_inches.x, fig_size_inches.height)


def image_grid_plot(
    fig: Figure,
    images: List[im.Image],
    columns: Optional[int] = None,
    enlarge_figure_for_more_columns_than: Optional[int] = 1,
    labels: Optional[List[str]] = None,
    label_face_color: Any = "auto",
    label_face_alpha: float = 1.0,
    label_font_weight: Any = "bold",
    border_color: Any = "auto",
    border_alpha: float = 0.5,
) -> List[List[Axes]]:
    if labels is not None and len(labels) != len(images):
        raise ValueError(f"Length of len(labels) ({len(labels)}) != len(images) ({len(images)})")

    if len(images) == 0:
        return

    if columns is None:
        columns = int(np.ceil(np.sqrt(len(images))))

    assert columns > 0

    rows = int(np.ceil(len(images) * 1.0 / columns))

    assert rows > 0 and rows <= columns

    if (
        enlarge_figure_for_more_columns_than is not None
        and enlarge_figure_for_more_columns_than > 0
        and columns > enlarge_figure_for_more_columns_than
    ):
        _, fig_size_inches, _ = measure_figure(fig)

        col_width_inches = fig_size_inches.width / enlarge_figure_for_more_columns_than
        row_height_inches = fig_size_inches.height / enlarge_figure_for_more_columns_than

        additional_width_inches = col_width_inches * (columns - enlarge_figure_for_more_columns_than)
        additional_height_inches = max(0, row_height_inches * (rows - enlarge_figure_for_more_columns_than))

        fig.set_size_inches(
            fig_size_inches.width + additional_width_inches,
            fig_size_inches.height + additional_height_inches,
        )

    axes: List[List[Axes]] = fig.subplots(nrows=rows, ncols=columns)

    if isinstance(axes, Axes):
        axes = [[axes]]

    if isinstance(axes[0], Axes):
        axes = [axes]

    i = 0

    for r in range(rows):
        for c in range(columns):
            ax = axes[r][c]

            ax.axis("off")
            ax.set_aspect("equal")

            if i < len(images):
                ax.imshow(images[i])

                if labels is not None:
                    ax.annotate(
                        labels[i],
                        (0.5, -0.015),
                        horizontalalignment="center",
                        verticalalignment="top",
                        xycoords="axes fraction",
                        fontweight=label_font_weight,
                        bbox={
                            "facecolor": fig.get_facecolor() if label_face_color == "auto" else label_face_color,
                            "alpha": label_face_alpha,
                            "boxstyle": "square",
                            "edgecolor": "none",
                            "linewidth": 0,
                        },
                    )

                if border_color is not None:
                    ax.add_patch(
                        Rectangle(
                            (0.0, 0.0),
                            1.0,
                            1.0,
                            transform=ax.transAxes,
                            linewidth=4,
                            edgecolor=fig.get_edgecolor() if border_color == "auto" else border_color,
                            facecolor="none",
                            alpha=border_alpha,
                            zorder=2,
                        )
                    )

            i += 1

    return axes


def cluster_colors(num_clusters: int) -> List[Any]:
    assert num_clusters >= 0
    if num_clusters == 0:
        return []
    elif num_clusters <= 3:
        return [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
        ][:num_clusters]
    elif num_clusters <= 10:
        cmap = get_cmap("tab10")
        return [cmap(i) for i in range(num_clusters)]
    else:
        cmap = get_cmap("gist_rainbow", num_clusters)
        return [cmap(i) for i in range(num_clusters)]


def elbow_plot(
    ax: Axes,
    elbow: Union[Elbow, List[Elbow]],
    xlabel: str = None,
    ylabel: str = None,
    pred_point: bool = True,
    pred_frac_point: bool = True,
    pred_hlines: bool = True,
    pred_vlines: bool = True,
    pred_frac_hlines: bool = True,
    pred_frac_vlines: bool = True,
    colors: List[Any] = "auto",
) -> None:
    if isinstance(elbow, Elbow):
        elbow = [elbow]

    if colors == "auto":
        if len(elbow) > 1:
            colors = cluster_colors(len(elbow))
        else:
            colors = None

    if colors is not None and len(colors) != len(elbow):
        raise ValueError("len(colors) != len(elbow)")

    for i, e in enumerate(elbow):
        color = "C0"
        if colors is not None:
            color = colors[i]

        ax.plot(e.ks, e.ds, color=color)

        if pred_frac_point:
            ax.plot(e.pred_frac_k, e.pred_frac_k_d, "o", color=color, fillstyle="none")

        if pred_point:
            ax.plot(e.pred_k, e.pred_k_d, "o", color=color, fillstyle="full")

        # FIXME
        # ax.set_xticks(e.ks)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for e in elbow:
        if pred_frac_hlines:
            ax.hlines(e.pred_frac_k_d, -1000, e.pred_frac_k, color="gray", linestyle="--", alpha=0.5, zorder=0)

        if pred_hlines:
            ax.hlines(e.pred_k_d, -1000, e.pred_k, color="gray", linestyle="--", alpha=0.5, zorder=0)

        if pred_frac_vlines:
            ax.vlines(e.pred_frac_k, -1000, e.pred_frac_k_d, color="gray", linestyle="--", alpha=0.5, zorder=0)

        if pred_vlines:
            ax.vlines(e.pred_k, -1000, e.pred_k_d, color="gray", linestyle="--", alpha=0.5, zorder=0)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_xlim(xlim)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
