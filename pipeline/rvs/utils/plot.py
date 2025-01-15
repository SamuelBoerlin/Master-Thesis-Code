from pathlib import Path
from typing import Callable, Dict, NewType, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from numpy.typing import NDArray
from PIL import Image as im

Precision = NewType("Precision", NDArray)
Recall = NewType("Recall", NDArray)
Label = NewType("Label", str)


def render_figure(fig: plt.Figure, callback: Callable[[im.Image], None]) -> None:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    data = canvas.buffer_rgba()

    with im.frombuffer("RGBA", canvas.get_width_height(), data, "raw", "RGBA", 0, 1) as plot_image:
        callback(plot_image)


def save_figure(fig: plt.Figure, file: Path) -> None:
    def save(image: im.Image) -> None:
        file.parent.mkdir(parents=True, exist_ok=True)
        image.save(file)

    render_figure(fig, save)


def bar_plot(
    ax: plt.Axes,
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
    ax: plt.Axes,
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
    ax: plt.Axes,
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
    ax: plt.Axes,
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
    ax: plt.Axes,
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
    ax: plt.Axes,
    enlarge_figure: bool = True,
) -> None:
    ax.get_legend().set_bbox_to_anchor((1.0, 1.0))

    plt.draw()

    legend_bb = ax.get_legend().get_window_extent()

    fig_bb = ax.figure.get_window_extent()
    fig_size = ax.figure.get_size_inches()

    unit_to_inches = fig_size[0] / fig_bb.width

    if enlarge_figure:
        ax.figure.set_size_inches(fig_size[0] + legend_bb.width * unit_to_inches, fig_size[1])
