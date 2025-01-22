from dataclasses import dataclass, field
from typing import List, Type

import numpy as np
import torch
from lerf.lerf import LERFModel
from lerf.lerf_pipeline import LERFPipeline
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.base_datamanager import DataManager
from numpy.typing import NDArray
from torch import Tensor

from rvs.pipeline.renderer import View
from rvs.pipeline.state import PipelineState


@dataclass
class ViewSelectionConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: ViewSelection)


class ViewSelection:
    config: ViewSelectionConfig

    def __init__(self, config: ViewSelectionConfig):
        self.config = config

    def select(self, clusters: NDArray, pipeline_state: PipelineState) -> List[View]:
        pass


@dataclass
class BestTrainingViewSelectionConfig(ViewSelectionConfig):
    _target: Type = field(default_factory=lambda: BestTrainingViewSelection)


class BestTrainingViewSelection(ViewSelection):
    def select(self, clusters: NDArray, pipeline_state: PipelineState) -> List[View]:
        num_clusters = clusters.shape[0]
        if num_clusters <= 0:
            return []

        if not isinstance(pipeline_state.pipeline.field.trainer.pipeline, LERFPipeline):
            raise Exception(f"Field {str(pipeline_state.pipeline.field)} is not based on a LERFPipeline")

        lerf_model: LERFModel = pipeline_state.pipeline.field.trainer.pipeline.model
        lerf_datamanager: DataManager = pipeline_state.pipeline.field.trainer.pipeline.datamanager

        image_embeddings = []

        for i in range(len(lerf_datamanager.train_dataset)):
            image: Tensor = lerf_datamanager.train_dataset[i]["image"]
            image = image.to(lerf_model.device)
            image = image[:, :, :3].permute(2, 0, 1).unsqueeze(0)

            embedding: NDArray = None
            with torch.no_grad():
                embedding = lerf_model.image_encoder.encode_image(image).detach().cpu().numpy().reshape((-1,))

            embedding = embedding / np.linalg.norm(embedding)

            image_embeddings.append(embedding)

        best_sim = [-1] * num_clusters
        best_idx = [0] * num_clusters

        for i in range(len(image_embeddings)):
            image_embedding = image_embeddings[i]
            image_idx = lerf_datamanager.train_dataset[i]["image_idx"]

            for j in range(num_clusters):
                sim = np.dot(clusters[j], image_embedding)

                if sim > best_sim[j]:
                    best_sim[j] = sim
                    best_idx[j] = image_idx

        return [pipeline_state.training_views[best_idx[j]] for j in range(num_clusters)]
