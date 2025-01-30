import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List


@dataclass
class Elbow:
    ks: List[int]
    ds: List[float]
    pred_k: int
    pred_k_d: float
    pred_frac_k: int
    pred_frac_k_d: float


def save_elbow(
    file: Path,
    elbow: Elbow,
) -> None:
    with file.open("w") as f:
        json.dump(asdict(elbow), f)


def load_elbow(
    file: Path,
) -> Elbow:
    with file.open("r") as f:
        return Elbow(**json.load(f))
