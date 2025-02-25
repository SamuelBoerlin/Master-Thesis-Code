from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from rvs.pipeline.pipeline import Pipeline
    from rvs.pipeline.views import View


class Normalization(NamedTuple):
    scale: NDArray
    offset: NDArray


class PipelineState:
    pipeline: "Pipeline"

    training_views: Optional[List["View"]] = None
    model_normalization: Optional[Normalization] = None
    sample_positions: Optional[NDArray] = None
    sample_embeddings: Optional[NDArray] = None
    sample_embeddings_type: Optional[str] = None
    sample_embeddings_dict: Optional[Dict[str, NDArray]] = None
    sample_cluster_parameters: Optional[NDArray] = None
    sample_cluster_indices: Optional[NDArray[np.intp]] = None
    selected_views: Optional[List["View"]] = None

    scratch_output_dir: Optional[Path] = None

    def __init__(self, pipeline: "Pipeline") -> None:
        self.pipeline = pipeline
