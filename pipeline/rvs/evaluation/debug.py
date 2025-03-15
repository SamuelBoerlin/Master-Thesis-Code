from dataclasses import dataclass
from typing import TYPE_CHECKING

from rvs.evaluation.lvis import Category, Uid
from rvs.pipeline.stage import PipelineStage
from rvs.pipeline.state import PipelineState

if TYPE_CHECKING:
    from rvs.evaluation.evaluation import Evaluation


@dataclass
class DebugContext:
    eval: "Evaluation"
    uid: Uid
    category: Category
    stage: PipelineStage
    state: PipelineState
