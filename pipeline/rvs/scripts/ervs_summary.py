from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tyro
from matplotlib import pyplot as plt
from nerfstudio.utils.rich_utils import CONSOLE
from tqdm import tqdm

from rvs.evaluation.analysis.similarity import Method
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

    CONSOLE.rule("Calculate average precision/recall")
    plot_avg_precision_recall_auc(args.output_dir / "precision_recall_auc.png", evals, args.category)


def plot_avg_precision_recall_auc(file: Path, evals: List[Eval], category: Optional[str]) -> None:
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

    ax.set_title("Average PR AUC")

    comparison_grid_plot(
        fig,
        ax,
        values,
        xlabels=[eval.name for eval in evals],
        ylabels=list(methods.keys()),
        colorbarlabel="Average PR AUC",
    )

    save_figure(fig, file)


def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(Args))


if __name__ == "__main__":
    entrypoint()
