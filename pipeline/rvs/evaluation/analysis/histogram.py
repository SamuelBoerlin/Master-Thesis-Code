from numbers import Number
from pathlib import Path
from typing import Dict, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from rvs.evaluation.analysis.utils import rename_categories_tuple
from rvs.evaluation.lvis import Category, LVISDataset, Uid
from rvs.utils.map import convert_map_to_tuple
from rvs.utils.plot import bar_plot, discrete_histogram_plot, save_figure


def calculate_avg_per_category(
    lvis: LVISDataset,
    values: Dict[Uid, Number],
) -> Dict[Category, float]:
    category_total: Dict[Category, float] = dict()
    category_items: Dict[Category, int] = dict()

    for uid, count in values.items():
        category = lvis.uid_to_category[uid]

        if category not in category_total:
            category_total[category] = 0.0

        if category not in category_items:
            category_items[category] = 0

        category_total[category] += count
        category_items[category] += 1

    for category in category_total.keys():
        category_total[category] /= category_items[category]

    return category_total


def calculate_discrete_histogram_per_category(
    lvis: LVISDataset,
    counts: Dict[Uid, int],
) -> Dict[Category, NDArray]:
    dict_histograms: Dict[Category, Dict[int, int]] = dict()

    for uid, count in counts.items():
        category = lvis.uid_to_category[uid]
        count = counts[uid]

        if category not in dict_histograms:
            dict_histograms[category] = dict()

        dict_histogram = dict_histograms[category]

        if count not in dict_histogram:
            dict_histogram[count] = 0

        dict_histogram[count] += 1

    histograms: Dict[Category, NDArray] = dict()

    for category in dict_histograms.keys():
        dict_histogram = dict_histograms[category]

        histogram = np.zeros((max(dict_histogram.keys()) + 1,))

        for bucket, value in dict_histogram.items():
            histogram[bucket] = value

        histograms[category] = histogram

    return histograms


def plot_avg_per_category(
    avg_counts: Dict[Category, float],
    file: Path,
    title: str,
    xlabel: str,
    category_names: Optional[Dict[Category, str]] = None,
    category_filter: Optional[Set[str]] = None,
) -> None:
    categories = tuple(
        sorted([category for category in avg_counts.keys() if category_filter is None or category in category_filter])
    )

    fig, ax = plt.subplots()

    ax.set_title(title)

    bar_plot(
        ax,
        names=rename_categories_tuple(categories, category_names),
        values=convert_map_to_tuple(avg_counts, key_order=categories, default=lambda _: 0.0),
        xlabel=xlabel,
        ylabel=None,
    )

    fig.tight_layout()

    save_figure(fig, file)


def plot_histogram(
    histograms: Dict[Category, NDArray],
    file: Path,
    title: str,
    xlabel: str,
    category_names: Optional[Dict[Category, str]] = None,
    category_filter: Optional[Set[str]] = None,
) -> None:
    categories = tuple(
        sorted([category for category in histograms.keys() if category_filter is None or category in category_filter])
    )

    fig = plt.figure()

    for category in categories:
        category_name = category_names[category] if category_names is not None else category

        histogram = histograms[category]

        ax = fig.subplots()

        ax.set_title(f'{title}\n"{category_name}"')

        discrete_histogram_plot(
            ax,
            values=histogram,
            xlabel=xlabel,
            ylabel=None,
        )

    fig.tight_layout()

    save_figure(fig, file)
