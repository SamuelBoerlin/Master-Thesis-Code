from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, NewType, Optional, Tuple, TypeVar, Union

import numpy as np
from matplotlib import patheffects as pe
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import get_cmap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d.axes3d import Axes3D
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
    markers: Optional[str] = "ov^+xD*",
    marker_sizes: List[float] = [10, 9, 9, 8, 8, 11, 6],
    marker_only_at_recall_change: bool = True,
    fillstyle: str = "none",
    alpha: float = 0.8,
    fill_alpha: float = 0.05,
) -> None:
    zoffsets: NDArray = np.zeros((len(marker_sizes),))

    if markers is not None:
        if len(markers) != len(marker_sizes):
            raise ValueError("len(markers) != len(marker_sizes)")

        sorted_indices = np.argsort(marker_sizes)
        for i in range(sorted_indices.shape[0]):
            idx = sorted_indices[i]
            zoffsets[idx] = 1.0 - float(i) / sorted_indices.shape[0]

    mi = 0

    zoffset = 0.0

    lines: List[Tuple[Line2D, Recall, Precision]] = []

    for label in values.keys():
        ys, xs, *_ = values[label]

        xs = np.insert(xs, 0, 0.0)
        ys = np.insert(ys, 0, 1.0)

        marker = "o"
        marker_size = 13

        if markers is not None:
            marker = markers[mi]
            marker_size = marker_sizes[mi]

            zoffset = zoffsets[mi]

            mi += 1
            if mi >= len(markers):
                mi = 0
                zoffsets += 1.0
        else:
            zoffset += 1.0

        line = ax.plot(
            xs,
            ys,
            "-" + marker,
            alpha=alpha,
            fillstyle=fillstyle,
            markersize=marker_size,
            markeredgewidth=2,
            zorder=zoffset,
        )[0]

        visible_markers = list(range(0, len(xs)))

        if len(visible_markers) > 0:
            # Hide first marker
            visible_markers[0] = -1

        if marker_only_at_recall_change:
            for i in range(1, len(xs) - 1):
                if abs(xs[i - 1] - xs[i]) > 0.0000001 or abs(xs[i + 1] - xs[i]) > 0.0000001:
                    pass
                else:
                    visible_markers[i] = -1

        line.set_markevery(every=[index for index in visible_markers if index >= 0])

        lines.append((line, xs, ys))

    for line, xs, ys in lines:
        ax.fill_between(xs, ys, color=line.get_color(), alpha=fill_alpha, zorder=-1.0)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_xlim(xmin=0.0, xmax=1.025)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_ylim(ymin=0.0, ymax=1.025)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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
    labels: Optional[List[Optional[str]]] = None,
    label_face_color: Any = "auto",
    label_face_alpha: float = 1.0,
    label_font_weight: Any = "bold",
    row_labels: Optional[List[Optional[str]]] = None,
    row_label_offsets: Optional[List[int]] = None,
    col_labels: Optional[List[Optional[str]]] = None,
    col_label_offsets: Optional[List[int]] = None,
    border_color: Union[Any, List[Any]] = "auto",
    border_alpha: Union[float, List[float]] = 0.5,
) -> List[List[Axes]]:
    if not isinstance(border_alpha, List):
        border_alpha = [border_alpha] * len(images)

    if not isinstance(border_color, List):
        border_color = [border_color] * len(images)

    if len(border_color) != len(images):
        raise ValueError("len(border_color) != len(images)")

    if len(border_alpha) != len(images):
        raise ValueError("len(border_alpha) != len(images)")

    if labels is not None and len(labels) != len(images):
        raise ValueError(f"Length of len(labels) ({len(labels)}) != len(images) ({len(images)})")

    if len(images) == 0:
        return

    auto_cols = False

    if columns is None:
        columns = int(np.ceil(np.sqrt(len(images))))
        auto_cols = True

    assert columns > 0

    rows = int(np.ceil(len(images) * 1.0 / columns))

    assert rows > 0

    if auto_cols:
        assert rows <= columns

    if row_labels is not None and len(row_labels) != rows:
        raise ValueError("len(row_labels) != rows")

    if row_label_offsets is None:
        row_label_offsets = [0] * rows

    if len(row_label_offsets) != rows:
        raise ValueError("len(row_label_offsets) != rows")

    if col_labels is not None and len(col_labels) != columns:
        raise ValueError("len(col_labels) != columns")

    if col_label_offsets is None:
        col_label_offsets = [0] * columns

    if len(col_label_offsets) != columns:
        raise ValueError("len(col_label_offsets) != columns")

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

                if labels is not None and labels[i] is not None:
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

                if col_labels is not None and r == col_label_offsets[c] and col_labels[c] is not None:
                    ax.annotate(
                        col_labels[c],
                        (0.5, 1.015),
                        horizontalalignment="center",
                        verticalalignment="bottom",
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

                if row_labels is not None and c == row_label_offsets[r] and row_labels[r] is not None:
                    ax.annotate(
                        row_labels[r],
                        (-0.015, 0.5),
                        horizontalalignment="right",
                        verticalalignment="center",
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
                            edgecolor=fig.get_edgecolor() if border_color[i] == "auto" else border_color[i],
                            facecolor="none",
                            alpha=border_alpha[i],
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
        return [np.array(cmap(i)[:3]) for i in range(num_clusters)]
    else:
        cmap = get_cmap("gist_rainbow", num_clusters)
        return [np.array(cmap(i)[:3]) for i in range(num_clusters)]


def elbow_plot(
    ax: Axes,
    elbow: Union[Elbow, List[Elbow]],
    xlabel: str = None,
    ylabel: str = None,
    pred_point: List[bool] = True,
    pred_frac_point: List[bool] = True,
    pred_hlines: List[bool] = True,
    pred_vlines: List[bool] = True,
    pred_frac_hlines: List[bool] = True,
    pred_frac_vlines: List[bool] = True,
    colors: List[Any] = "auto",
    linestyles: List[Any] = "auto",
) -> None:
    if isinstance(elbow, Elbow):
        elbow = [elbow]

    if isinstance(pred_point, bool):
        pred_point = [pred_point] * len(elbow)

    if isinstance(pred_frac_point, bool):
        pred_frac_point = [pred_frac_point] * len(elbow)

    if isinstance(pred_hlines, bool):
        pred_hlines = [pred_hlines] * len(elbow)

    if isinstance(pred_vlines, bool):
        pred_vlines = [pred_vlines] * len(elbow)

    if isinstance(pred_frac_hlines, bool):
        pred_frac_hlines = [pred_frac_hlines] * len(elbow)

    if isinstance(pred_frac_vlines, bool):
        pred_frac_vlines = [pred_frac_vlines] * len(elbow)

    if colors == "auto":
        if len(elbow) > 1:
            colors = cluster_colors(len(elbow))
        else:
            colors = None

    if linestyles == "auto":
        linestyles = None

    if len(pred_point) != len(elbow):
        raise ValueError("len(pred_point) != len(elbow)")

    if len(pred_frac_point) != len(elbow):
        raise ValueError("len(pred_frac_point) != len(elbow)")

    if len(pred_hlines) != len(elbow):
        raise ValueError("len(pred_hlines) != len(elbow)")

    if len(pred_vlines) != len(elbow):
        raise ValueError("len(pred_vlines) != len(elbow)")

    if len(pred_frac_hlines) != len(elbow):
        raise ValueError("len(pred_frac_hlines) != len(elbow)")

    if len(pred_frac_vlines) != len(elbow):
        raise ValueError("len(pred_frac_vlines) != len(elbow)")

    if colors is not None and len(colors) != len(elbow):
        raise ValueError("len(colors) != len(elbow)")

    if linestyles is not None and len(linestyles) != len(elbow):
        raise ValueError("len(linestyles) != len(elbow)")

    for i, e in enumerate(elbow):
        color = "C0"
        if colors is not None:
            color = colors[i]

        linestyle = None
        if linestyles is not None:
            linestyle = linestyles[i]

        ax.plot(e.ks, e.ds, color=color, linestyle=linestyle)

        if pred_frac_point[i]:
            ax.plot(e.pred_frac_k, e.pred_frac_k_d, "o", color=color, fillstyle="none")

        if pred_point[i]:
            ax.plot(e.pred_k, e.pred_k_d, "o", color=color, fillstyle="full")

        # FIXME
        # ax.set_xticks(e.ks)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for e in elbow:
        if pred_frac_hlines[i]:
            ax.hlines(e.pred_frac_k_d, -1000, e.pred_frac_k, color="gray", linestyle=":", alpha=0.5, zorder=0)

        if pred_hlines[i]:
            ax.hlines(e.pred_k_d, -1000, e.pred_k, color="gray", linestyle=":", alpha=0.5, zorder=0)

        if pred_frac_vlines[i]:
            ax.vlines(e.pred_frac_k, -1000, e.pred_frac_k_d, color="gray", linestyle=":", alpha=0.5, zorder=0)

        if pred_vlines[i]:
            ax.vlines(e.pred_k, -1000, e.pred_k_d, color="gray", linestyle=":", alpha=0.5, zorder=0)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_xlim(xlim)
    ax.get_xaxis().set_major_locator(MaxNLocator(nbins="auto", integer=True))

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)


def comparison_grid_plot(
    fig: Figure,
    ax: Axes,
    values: NDArray,
    xlabels: List[str],
    ylabels: List[str],
    colorbar_label: Optional[str] = None,
    colorbar_orientation: str = "vertical",
    value_format: str = "{0:.2f}",
    inches_per_column: Optional[float] = 0.5,
    inches_per_row: Optional[float] = None,
) -> None:
    if len(ylabels) != values.shape[0]:
        raise ValueError("len(ylabels) != values.shape[0]")

    if len(xlabels) != values.shape[1]:
        raise ValueError("len(xlabels) != values.shape[1]")

    if inches_per_column is not None:
        fig.set_size_inches([fig.get_size_inches()[0] + inches_per_column * values.shape[1], fig.get_size_inches()[1]])

    if inches_per_row is not None:
        fig.set_size_inches([fig.get_size_inches()[0], fig.get_size_inches()[1] + inches_per_row * values.shape[0]])

    image = ax.imshow(values)

    if colorbar_orientation == "vertical":
        aspect = float(values.shape[0]) / values.shape[1] if values.shape[1] > 0 else 1.0
        cbar = fig.colorbar(
            image,
            ax=ax,
            fraction=0.0492 * aspect,
            pad=0.04 * aspect,
        )
        if colorbar_label is not None:
            cbar.ax.set_ylabel(colorbar_label, rotation=-90, va="bottom")
    elif colorbar_orientation == "horizontal":
        cbar = fig.colorbar(image, ax=ax, orientation="horizontal", location="top")
        if colorbar_label is not None:
            cbar.ax.set_ylabel(colorbar_label, rotation=0, va="center", ha="right")

    ax.set_xticks(range(values.shape[1]), labels=xlabels, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(values.shape[0]), labels=ylabels)

    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(values.shape[1] + 1) - 0.49, minor=True)
    ax.set_yticks(np.arange(values.shape[0] + 1) - 0.49, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            value_str = value_format.format(value) if np.isfinite(value) else "n/a"
            ax.text(
                j,
                i,
                value_str,
                ha="center",
                va="center",
                color="w",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
                fontweight="bold",
            )


T = TypeVar("T")


def fit_suptitle(fig: Figure, body: Callable[[], T], suptitle: str, suptitle_font_size: int = 24) -> T:
    suptitle = fig.suptitle(
        suptitle,
        bbox={
            "facecolor": fig.get_facecolor(),
            "alpha": 0.5,
            "boxstyle": "square",
            "edgecolor": "none",
            "linewidth": 0,
        },
        fontsize=suptitle_font_size,
        verticalalignment="top",
        y=1.0,
    )

    fig_size = fig.get_size_inches()
    fig_size[1] = fig_size[0]

    fig.set_size_inches(fig_size[0], fig_size[1])

    result = body()

    for ax in fig.axes:
        ax.set_anchor("S")

    fig_size = fig.get_size_inches()

    figure_bb, _, _ = measure_figure(fig)
    suptitle_bb, _, _ = measure_artist(suptitle)

    top_padding_inches = (1.0 - suptitle_bb.ymin / figure_bb.height) * fig_size[1]

    fig_size[1] += top_padding_inches

    fig.set_size_inches(fig_size[0], fig_size[1])

    fig.subplots_adjust(left=0.0, right=1.0, top=(1.0 - top_padding_inches / fig_size[1]), bottom=0.0)

    return result


def camera_transforms_plot(
    ax3d: Axes3D,
    transforms: List[NDArray],
    sphere_mesh_radius: Optional[float] = None,
    cull_behind_origin: Optional[bool] = False,
    frustum_width: float = 0.2,
    frustum_height: float = 0.15,
    frustum_depth: float = 0.15,
    frustum_line_width: float = 0.5,
    frustum_colors: Optional[Union[str, List[str]]] = None,
    show_world_axes: bool = True,
    world_axes_size: float = 0.25,
    world_axes_line_width: float = 3.0,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    zlim: Optional[Tuple[float, float]] = None,
) -> None:
    if frustum_colors is None:
        frustum_colors = "black"

    if not isinstance(frustum_colors, List):
        frustum_colors = [frustum_colors] * len(transforms)

    if len(frustum_colors) != len(transforms):
        raise ValueError("len(frustum_colors) != len(transforms)")

    ax: Axes = ax3d  # type hints not working for Axes3D?

    ax.set_box_aspect([1, 1, 1])

    if sphere_mesh_radius is not None:
        u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 15j]

        x = np.cos(u) * np.sin(v) * sphere_mesh_radius
        y = np.sin(u) * np.sin(v) * sphere_mesh_radius
        z = np.cos(v) * sphere_mesh_radius

        ax3d.plot_wireframe(x, y, z, color="lightgray", alpha=0.5)

    rot_x = 0.5 * np.pi
    to_z_up = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(rot_x), -np.sin(rot_x), 0],
            [0, np.sin(rot_x), np.cos(rot_x), 0],
            [0, 0, 0, 1],
        ]
    )

    camera_dir = np.array(
        [
            -np.cos(np.deg2rad(ax3d.elev)) * np.cos(np.deg2rad(ax3d.azim)),
            -np.cos(np.deg2rad(ax3d.elev)) * np.sin(np.deg2rad(ax3d.azim)),
            -np.sin(np.deg2rad(ax3d.elev)),
        ]
    )

    if show_world_axes:
        world_origin = np.array([0, 0, 0])

        world_axis_x = np.array([1, 0, 0]) * world_axes_size
        world_axis_y = -np.array([0, 1, 0]) * world_axes_size
        world_axis_z = np.array([0, 0, 1]) * world_axes_size

        ax.plot(*[[world_origin[i], world_axis_x[i]] for i in range(3)], color="red", linewidth=world_axes_line_width)
        ax.plot(*[[world_origin[i], world_axis_y[i]] for i in range(3)], color="blue", linewidth=world_axes_line_width)
        ax.plot(*[[world_origin[i], world_axis_z[i]] for i in range(3)], color="green", linewidth=world_axes_line_width)

    for i, transform in enumerate(transforms):
        transform = to_z_up @ transform

        dx = transform.T[0, :3]
        dy = transform.T[1, :3]
        dz = transform.T[2, :3]
        point = transform.T[3, :3]

        def get_color(color: Any) -> Any:
            if cull_behind_origin and np.dot(point, camera_dir) > 0:
                return "none"
            return color

        frustum_corners = [
            point - dx * frustum_width + dy * frustum_height - dz * frustum_depth,
            point + dx * frustum_width + dy * frustum_height - dz * frustum_depth,
            point + dx * frustum_width - dy * frustum_height - dz * frustum_depth,
            point - dx * frustum_width - dy * frustum_height - dz * frustum_depth,
        ]

        frustum_lines = [
            [frustum_corners[0], frustum_corners[1]],
            [frustum_corners[1], frustum_corners[2]],
            [frustum_corners[2], frustum_corners[3]],
            [frustum_corners[3], frustum_corners[0]],
            [frustum_corners[0], point],
            [frustum_corners[1], point],
            [frustum_corners[2], point],
            [frustum_corners[3], point],
        ]

        for frustum_line in frustum_lines:
            ax.plot(
                *[[frustum_line[0][i], frustum_line[1][i]] for i in range(3)],
                color=get_color(frustum_colors[i]),
                linewidth=frustum_line_width,
            )

        ax.scatter(point[0], point[1], point[2], color=get_color(frustum_colors[i]), s=10)

    if xlim is not None:
        ax3d.set_xlim(xlim[0], xlim[1])

    if ylim is not None:
        ax3d.set_ylim(ylim[0], ylim[1])

    if zlim is not None:
        ax3d.set_zlim(zlim[0], zlim[1])
