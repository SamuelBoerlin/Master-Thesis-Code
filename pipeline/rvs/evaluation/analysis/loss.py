import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from rvs.evaluation.lvis import LVISDataset, Uid
from rvs.evaluation.pipeline import PipelineEvaluationInstance
from rvs.pipeline.stage import PipelineStage
from rvs.utils.plot import save_figure


def calculate_avg_loss(
    lvis: LVISDataset,
    uids: Set[Uid],
    instance: PipelineEvaluationInstance,
    loss: str,
) -> List[Tuple[int, float, float]]:
    losses: Dict[int, List[float]] = dict()

    for uid in tqdm(sorted(uids)):
        if uid in uids:
            model_file = Path(lvis.uid_to_file[uid])

            loss_file = instance.create_pipeline_io(model_file).get_path(
                PipelineStage.TRAIN_FIELD,
                Path("field") / "scratch" / "tracking" / f"{loss}.json",
            )

            if loss_file.exists() and loss_file.is_file():
                with loss_file.open("r") as f:
                    json_arr: List[Dict[str, Any]] = json.load(f)

                    for obj in json_arr:
                        step: int = obj["step"]
                        value: float = obj["value"]

                        if np.isfinite(value):
                            if step not in losses:
                                losses[step] = []

                            losses[step].append(value)

    return [
        (step, np.mean(np.array(losses[step])), np.std(np.array(losses[step]))) for step in sorted(list(losses.keys()))
    ]


def plot_avg_loss(
    losses: List[Tuple[int, float, float]],
    loss_name: str,
    loss_metric: str,
    file: Path,
    extra_space_for_std: float = 2.0,
) -> None:
    fig, ax = plt.subplots()

    ax.set_title(f"{loss_name}")

    xs = np.array([t[0] for t in losses])
    ys = np.array([t[1] for t in losses])
    ystd = np.array([t[2] for t in losses])

    ax.plot(xs, ys, "-")

    ylim = ax.get_ylim()

    ymid = 0.5 * (ylim[1] - ylim[0])
    yrange = ylim[1] - ymid

    ax.fill_between(xs, ys - ystd, ys + ystd, alpha=0.2)

    ax.set_ylim(
        ymin=max(ymid - yrange * extra_space_for_std, ax.get_ylim()[0]),
        ymax=min(ymid + yrange * extra_space_for_std, ax.get_ylim()[1]),
    )

    ax.set_xlabel("Step")
    ax.set_ylabel(loss_metric)

    fig.tight_layout()

    save_figure(fig, file)
