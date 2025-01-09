from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image as im


def render_figure(fig: plt.Figure, callback: Callable[[im.Image], None]) -> None:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    data = canvas.buffer_rgba()

    with im.frombuffer("RGBA", canvas.get_width_height(), data, "raw", "RGBA", 0, 1) as plot_image:
        callback(plot_image)


def save_figure(fig: plt.Figure, file: Path) -> None:
    def save(image: im.Image) -> None:
        image.save(file)

    render_figure(fig, save)


def grouped_bar_plot(
    ax: plt.Axes,
    groups: Tuple[str, ...],
    values: Dict[str, Tuple[float, ...]],
    bar_width=0.2,
    xlabel: str = None,
    ylabel: str = None,
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
    ax.legend(ys.keys())
