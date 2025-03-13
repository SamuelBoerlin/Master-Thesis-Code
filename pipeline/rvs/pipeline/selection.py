from dataclasses import dataclass, field
from typing import Callable, List, Optional, Type

import numpy as np
import torch
from lerf.lerf import LERFModel
from lerf.lerf_pipeline import LERFPipeline
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.base_datamanager import DataManager
from nerfstudio.utils.rich_utils import CONSOLE
from numpy.typing import NDArray
from torch import Tensor

from rvs.pipeline.renderer import View
from rvs.pipeline.stage import PipelineStage, RequirePipelineStage
from rvs.pipeline.state import PipelineState


@dataclass
class ViewSelectionConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: ViewSelection)


class ViewSelection:
    config: ViewSelectionConfig

    def __init__(self, config: ViewSelectionConfig):
        self.config = config

    def select(
        self,
        num_clusters: int,
        hard_classifier: Callable[[NDArray], NDArray[np.intp]],
        soft_classifier: Callable[[NDArray], NDArray],
        pipeline_state: PipelineState,
    ) -> List[View]:
        pass


@dataclass
class MostSimilarToCentroidTrainingViewSelectionConfig(ViewSelectionConfig):
    _target: Type = field(default_factory=lambda: MostSimilarToCentroidTrainingViewSelection)


class BestTrainingViewSelectionConfig(MostSimilarToCentroidTrainingViewSelectionConfig):
    """Deprecated, only for backwards compatibility"""

    pass


class MostSimilarToCentroidTrainingViewSelection(ViewSelection, RequirePipelineStage):
    config: MostSimilarToCentroidTrainingViewSelectionConfig

    def __init__(self, config: MostSimilarToCentroidTrainingViewSelectionConfig) -> None:
        super().__init__(config)
        self.required_stages = {
            PipelineStage.SAMPLE_VIEWS,
            PipelineStage.TRANSFORM_EMBEDDINGS,
            PipelineStage.TRAIN_FIELD,
        }

    def select(
        self,
        num_clusters: int,
        hard_classifier: Callable[[NDArray], NDArray[np.intp]],
        soft_classifier: Callable[[NDArray], NDArray],
        pipeline_state: PipelineState,
    ) -> List[View]:
        if "centroids" not in pipeline_state.cluster_parameters:
            raise ValueError("Cluster centroids required")

        centroids = pipeline_state.cluster_parameters["centroids"]

        num_clusters = centroids.shape[0]
        if num_clusters <= 0:
            return []

        if not isinstance(pipeline_state.pipeline.field.trainer.pipeline, LERFPipeline):
            raise Exception(f"Field {str(pipeline_state.pipeline.field)} is not based on a LERFPipeline")

        lerf_model: LERFModel = pipeline_state.pipeline.field.trainer.pipeline.model
        lerf_datamanager: DataManager = pipeline_state.pipeline.field.trainer.pipeline.datamanager

        image_embeddings: List[NDArray] = []

        for i in range(len(lerf_datamanager.train_dataset)):
            image: Tensor = lerf_datamanager.train_dataset[i]["image"]
            image = image.to(lerf_model.device)
            image = image[:, :, :3].permute(2, 0, 1).unsqueeze(0)

            embedding: NDArray = None
            with torch.no_grad():
                embedding = lerf_model.image_encoder.encode_image(image).detach().cpu().numpy().reshape((-1,))

            embedding = embedding / np.linalg.norm(embedding)

            image_embeddings.append(embedding)

        transformed_embeddings: NDArray = pipeline_state.pipeline.transform.apply(
            np.array(image_embeddings), pipeline_state.transform_parameters
        )

        assert transformed_embeddings.shape[0] == len(lerf_datamanager.train_dataset)
        assert transformed_embeddings.shape[1] == centroids.shape[1]

        best_sim = [-1] * num_clusters
        best_idx = [0] * num_clusters

        for i in range(transformed_embeddings.shape[0]):
            image_idx = lerf_datamanager.train_dataset[i]["image_idx"]

            embedding = transformed_embeddings[i]

            for j in range(num_clusters):
                sim = np.dot(centroids[j], embedding)

                if sim > best_sim[j]:
                    best_sim[j] = sim
                    best_idx[j] = image_idx

        selected_views: List[View] = []
        for j in range(num_clusters):
            best_view: Optional[View] = None
            for v in pipeline_state.training_views:
                if v.index == best_idx[j]:
                    best_view = v
                    break
            if best_view is not None:
                selected_views.append(best_view)
            else:
                CONSOLE.log(f"[bold yellow]WARNING: No view with index {best_idx[j]} found")

        return selected_views


class BestTrainingViewSelection(MostSimilarToCentroidTrainingViewSelection):
    """Deprecated, only for backwards compatibility"""

    pass
