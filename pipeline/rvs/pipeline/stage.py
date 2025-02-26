from abc import ABC
from enum import Enum
from typing import Dict, Iterable, List, Optional, Set


class PipelineStage(str, Enum):
    SAMPLE_VIEWS = "SAMPLE_VIEWS"
    RENDER_VIEWS = "RENDER_VIEWS"
    TRAIN_FIELD = "TRAIN_FIELD"
    SAMPLE_POSITIONS = "SAMPLE_POSITIONS"
    SAMPLE_EMBEDDINGS = "SAMPLE_EMBEDDINGS"
    CLUSTER_EMBEDDINGS = "CLUSTER_EMBEDDINGS"
    SELECT_VIEWS = "SELECT_VIEWS"
    OUTPUT = "OUTPUT"

    _ignore_ = ["ORDER"]
    ORDER: Dict["PipelineStage", int] = {}

    _ignore_ = ["DEPENDENCIES"]
    DEPENDENCIES: Dict["PipelineStage", Set["PipelineStage"]] = {}

    @property
    def order(self) -> int:
        return PipelineStage.ORDER[self]

    def depends_on(self, stage: "PipelineStage", transitive: bool = True) -> bool:
        if self.order > stage.order:
            if transitive:
                return True
            return stage in PipelineStage.DEPENDENCIES[self]
        return False

    def required_by(self, stages: List["PipelineStage"], transitive: bool = True) -> bool:
        for stage in stages:
            if stage.depends_on(self, transitive=transitive):
                return True
        return False

    def before(self) -> List["PipelineStage"]:
        return [stage for stage in PipelineStage if stage.order <= self.order]

    def after(self) -> List["PipelineStage"]:
        return [stage for stage in PipelineStage if stage.order >= self.order]

    @staticmethod
    def all() -> List["PipelineStage"]:
        return [s for s in PipelineStage]

    @staticmethod
    def between(
        start: Optional["PipelineStage"], end: Optional["PipelineStage"], default: List["PipelineStage"] = []
    ) -> List["PipelineStage"]:
        if start is not None and end is not None:
            return [s for s in start.after() if s in end.before()]
        elif start is not None:
            return start.after()
        elif end is not None:
            return end.before()
        return default


PipelineStage.ORDER = {
    PipelineStage.SAMPLE_VIEWS: 1,
    PipelineStage.RENDER_VIEWS: 2,
    PipelineStage.TRAIN_FIELD: 3,
    PipelineStage.SAMPLE_POSITIONS: 4,
    PipelineStage.SAMPLE_EMBEDDINGS: 5,
    PipelineStage.CLUSTER_EMBEDDINGS: 6,
    PipelineStage.SELECT_VIEWS: 7,
    PipelineStage.OUTPUT: 8,
}

PipelineStage.DEPENDENCIES = {
    PipelineStage.SAMPLE_VIEWS: set(),
    PipelineStage.RENDER_VIEWS: {PipelineStage.SAMPLE_VIEWS},
    PipelineStage.TRAIN_FIELD: {PipelineStage.SAMPLE_VIEWS, PipelineStage.RENDER_VIEWS},
    PipelineStage.SAMPLE_POSITIONS: {PipelineStage.RENDER_VIEWS},  # Normalization
    PipelineStage.SAMPLE_EMBEDDINGS: {PipelineStage.SAMPLE_POSITIONS, PipelineStage.TRAIN_FIELD},
    PipelineStage.CLUSTER_EMBEDDINGS: {PipelineStage.SAMPLE_EMBEDDINGS},
    PipelineStage.SELECT_VIEWS: {PipelineStage.CLUSTER_EMBEDDINGS},
    PipelineStage.OUTPUT: {PipelineStage.SELECT_VIEWS},
}

for stage in PipelineStage:
    assert stage in PipelineStage.ORDER
    assert stage in PipelineStage.DEPENDENCIES


class RequirePipelineStage(ABC):
    __required_stages: Set["PipelineStage"] = True

    @property
    def required_stages(self) -> Set["PipelineStage"]:
        return self.__required_stages

    @required_stages.setter
    def required_stages(self, stages: Iterable["PipelineStage"]) -> None:
        self.__required_stages = set(stages)
