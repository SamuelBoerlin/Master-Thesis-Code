from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tyro
from matplotlib import pyplot as plt
from nerfstudio.utils.rich_utils import CONSOLE
from tqdm import tqdm

from rvs.evaluation.analysis.similarity import Method
from rvs.evaluation.analysis.utils import rename_methods_dict
from rvs.evaluation.evaluation import EvaluationConfig
from rvs.evaluation.evaluation_method import load_result
from rvs.evaluation.lvis import Category
from rvs.utils.config import load_config, run_in_config_working_dir
from rvs.utils.plot import comparison_grid_plot, save_figure


@dataclass
class Args:
    configs: List[Path] = tyro.MISSING

    names: Optional[List[str]] = None

    category: Optional[str] = None

    output_dir: Path = tyro.MISSING

    value_format: str = "{0:.2f}"


@dataclass
class Eval:
    index: int
    dir: Path
    config: EvaluationConfig
    name: str


def main(args: Args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.names is not None and len(args.names) != len(args.configs):
        raise ValueError("len(names) != len(configs)")

    evals: List[Eval] = []

    for i, config_file in enumerate(args.configs):
        eval_config = load_config(config_file, EvaluationConfig)

        eval_dir = run_in_config_working_dir(config_file, eval_config.output_dir, eval_config.output_dir.resolve)

        eval_name: str = None
        if args.names is None:
            eval_name = eval_dir.name
        else:
            eval_name = args.names[i]
        assert eval_name is not None

        evals.append(
            Eval(
                index=i,
                dir=eval_dir,
                config=eval_config,
                name=eval_name,
            )
        )

    method_titles: Dict[Method, str] = {
        "best_embedding_of_views_wrt_ground_truth": "Best Embedding of Views w.r.t. Ground Truth",
        "avg_embedding_of_selected_views": "Average Embedding of Selected Views",
        "best_embedding_of_selected_views_wrt_query": "Best Embedding of Selected Views w.r.t. Query",
        "avg_embedding_of_random_views": "Average Embedding of Random Views",
        "best_embedding_of_random_views_wrt_query": "Best Embedding of Random Views w.r.t. Query",
    }

    CONSOLE.rule("Calculate average precision/recall")
    plot_avg_precision_recall_auc_grid(
        args.output_dir / "precision_recall_auc.png",
        evals,
        args.category,
        method_titles=method_titles,
        value_format=args.value_format,
    )

    plot_avg_precision_recall_auc_dots(
        args.output_dir / "precision_recall_auc_avg_dots.png",
        evals,
        "avg_embedding_of_selected_views",
        "avg_embedding_of_random_views",
        "Average Embedding",
        args.category,
        method_titles={
            "avg_embedding_of_selected_views": "Selected Views",
            "avg_embedding_of_random_views": "Random Views",
        },
    )

    plot_avg_precision_recall_auc_dots(
        args.output_dir / "precision_recall_auc_best_dots.png",
        evals,
        "best_embedding_of_selected_views_wrt_query",
        "best_embedding_of_random_views_wrt_query",
        "Best Embedding w.r.t. Query",
        args.category,
        method_titles={
            "best_embedding_of_selected_views_wrt_query": "Selected Views",
            "best_embedding_of_random_views_wrt_query": "Random Views",
        },
    )


def plot_avg_precision_recall_auc_grid(
    file: Path,
    evals: List[Eval],
    category: Optional[str],
    method_titles: Optional[Dict[Method, str]] = None,
    value_format="{0:.2f}",
) -> None:
    all_avg_precision_recall_auc: Dict[int, Dict[Method, float]] = dict()

    methods: Dict[Method, int] = dict()

    for eval in tqdm(evals):
        precision_recall_auc: Dict[Method, Dict[Category, float]] = load_result(
            eval.dir / "results" / "dumps" / "precision_recall_auc.pkl"
        )

        avg_precision_recall_auc: Dict[Method, float] = dict()

        for method, category_pr_auc in precision_recall_auc.items():
            values = list(category_pr_auc.values()) if category is None else [category_pr_auc.get(category, np.nan)]

            avg_precision_recall_auc[method] = np.average(values)

            if method not in methods:
                methods[method] = len(methods)

        all_avg_precision_recall_auc[eval.index] = avg_precision_recall_auc

    num_methods = len(methods)
    num_experiments = len(evals)

    values = np.ones((num_methods, num_experiments)) * np.nan
    for experiment, results in all_avg_precision_recall_auc.items():
        for method, value in results.items():
            values[methods[method]][experiment] = value

    fig, ax = plt.subplots(layout="constrained")

    ax.set_title("Average Area Under the Precision Recall Curve")

    comparison_grid_plot(
        fig,
        ax,
        values,
        xlabels=[eval.name for eval in evals],
        ylabels=list(rename_methods_dict(methods, method_titles).keys()),
        colorbar_label="Average PR AUC",
        value_format=value_format,
    )

    save_figure(fig, file)


def plot_avg_precision_recall_auc_dots(
    file: Path,
    evals: List[Eval],
    method1: Method,
    method2: Method,
    title: str,
    category: Optional[str],
    method_titles: Optional[Dict[Method, str]] = None,
) -> None:
    m1_name = method1
    if method_titles is not None and method1 in method_titles:
        m1_name = method_titles[method1]

    m2_name = method2
    if method_titles is not None and method2 in method_titles:
        m2_name = method_titles[method2]

    m1_all_avg_precision_recall_auc: List[float] = []
    m1_all_std_precision_recall_auc: List[float] = []

    m2_all_avg_precision_recall_auc: List[float] = []
    m2_all_std_precision_recall_auc: List[float] = []

    for eval in tqdm(evals):
        precision_recall_auc: Dict[Method, Dict[Category, float]] = load_result(
            eval.dir / "results" / "dumps" / "precision_recall_auc.pkl"
        )

        m1_avg_precision_recall_auc: float = None
        m1_std_precision_recall_auc: float = None

        m2_avg_precision_recall_auc: float = None
        m2_std_precision_recall_auc: float = None

        for m, category_pr_auc in precision_recall_auc.items():
            values = list(category_pr_auc.values()) if category is None else [category_pr_auc.get(category, np.nan)]

            if m == method1:
                m1_avg_precision_recall_auc = np.average(values)
                m1_std_precision_recall_auc = np.std(values)

            if m == method2:
                m2_avg_precision_recall_auc = np.average(values)
                m2_std_precision_recall_auc = np.std(values)

        assert m1_avg_precision_recall_auc is not None
        assert m1_std_precision_recall_auc is not None

        assert m2_avg_precision_recall_auc is not None
        assert m2_std_precision_recall_auc is not None

        m1_all_avg_precision_recall_auc.append(m1_avg_precision_recall_auc)
        m1_all_std_precision_recall_auc.append(m1_std_precision_recall_auc)

        m2_all_avg_precision_recall_auc.append(m2_avg_precision_recall_auc)
        m2_all_std_precision_recall_auc.append(m2_std_precision_recall_auc)

    assert len(m1_all_avg_precision_recall_auc) == len(m1_all_std_precision_recall_auc)
    assert len(m2_all_avg_precision_recall_auc) == len(m2_all_std_precision_recall_auc)
    assert len(m1_all_avg_precision_recall_auc) == len(evals)
    assert len(m1_all_avg_precision_recall_auc) == len(m2_all_avg_precision_recall_auc)

    fig, ax = plt.subplots(layout="constrained")

    ax.set_title(title)

    x = np.arange(len(evals))

    offset = 0.15

    ax.errorbar(
        x - offset,
        m1_all_avg_precision_recall_auc,
        yerr=m1_all_std_precision_recall_auc,
        fmt="o",
        capsize=5,
        linestyle="none",
        label=m1_name,
    )

    ax.errorbar(
        x + offset,
        m2_all_avg_precision_recall_auc,
        yerr=m2_all_std_precision_recall_auc,
        fmt="o",
        capsize=5,
        linestyle="none",
        label=m2_name,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([eval.name for eval in evals])

    ax.legend()

    ax.set_ylim(bottom=0.0, top=1.0)

    ax.set_ylabel("Average PR AUC")

    save_figure(fig, file)


def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(Args))


if __name__ == "__main__":
    entrypoint()
