import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Type

import numpy as np
import torch
from lerf.lerf import LERFModel
from lerf.lerf_pipeline import LERFPipeline
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.base_datamanager import DataManager
from nerfstudio.utils.rich_utils import CONSOLE
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN
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
        assert pipeline_state.training_views is not None
        assert pipeline_state.transform_parameters is not None
        assert pipeline_state.cluster_parameters is not None

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


@dataclass
class SpatialViewSelectionConfig(ViewSelectionConfig):
    _target: Type = field(default_factory=lambda: SpatialViewSelection)

    dbscan_eps: Optional[float] = 0.05
    """Maximum distance between two points to be considered neighbors in DBSCAN clustering"""


class SpatialViewSelection(ViewSelection, RequirePipelineStage):
    config: SpatialViewSelectionConfig

    def __init__(self, config: SpatialViewSelectionConfig) -> None:
        super().__init__(config)
        self.required_stages = {
            PipelineStage.SAMPLE_VIEWS,
            PipelineStage.SAMPLE_POSITIONS,
        }

    def select(
        self,
        num_clusters: int,
        hard_classifier: Callable[[NDArray], NDArray[np.intp]],
        soft_classifier: Callable[[NDArray], NDArray],
        pipeline_state: PipelineState,
    ) -> List[View]:
        assert pipeline_state.training_views is not None
        assert pipeline_state.sample_positions is not None
        assert pipeline_state.cluster_indices is not None

        clustered_position_lists: Dict[int, List[NDArray]] = dict()

        for i in range(len(pipeline_state.sample_positions)):
            position = pipeline_state.sample_positions[i]
            cluster_idx = pipeline_state.cluster_indices[i]

            if cluster_idx not in clustered_position_lists:
                clustered_position_lists[cluster_idx] = []

            clustered_position_lists[cluster_idx].append(position)

        clustered_positions: Dict[int, NDArray] = {
            cluster_idx: np.stack(positions_list) for cluster_idx, positions_list in clustered_position_lists.items()
        }

        selected_views: List[View] = []

        dbscan_sample_labels: Dict[int, List[int]] = dict()
        dbscan_view_labels: Dict[int, int] = dict()

        for cluster_idx, positions in clustered_positions.items():
            dbscan = DBSCAN(eps=self.config.dbscan_eps)
            dbscan.fit(positions)

            dbscan_sample_labels[int(cluster_idx)] = [int(label) for label in dbscan.labels_.tolist()]

            max_count = -1
            max_count_label = -1

            label_counts: Dict[int, int] = dict()

            for i in range(positions.shape[0]):
                label = dbscan.labels_[i]

                if label >= 0:
                    if label not in label_counts:
                        label_counts[label] = 0

                    label_counts[label] = label_counts[label] + 1

                    if label_counts[label] > max_count:
                        max_count = label_counts[label]
                        max_count_label = label

            selected_positions: NDArray = None
            other_positions: NDArray = None

            if max_count_label >= 0 and max_count > 0:
                dbscan_view_labels[int(cluster_idx)] = int(max_count_label)

                selected_positions_list: List[NDArray] = []
                other_positions_list: List[NDArray] = []

                for i in range(positions.shape[0]):
                    label = dbscan.labels_[i]

                    if label == max_count_label:
                        selected_positions_list.append(positions[i])
                    else:
                        other_positions_list.append(positions[i])

                assert len(selected_positions_list) > 0

                selected_positions = np.stack(selected_positions_list)

                for i in range(pipeline_state.sample_positions.shape[0]):
                    if pipeline_state.cluster_indices[i] != cluster_idx:
                        other_positions_list.append(pipeline_state.sample_positions[i])

                if len(other_positions_list) > 0:
                    other_positions = np.stack(other_positions_list)
                else:
                    other_positions = np.zeros((1, 3))
            else:
                dbscan_view_labels[int(cluster_idx)] = -1

                selected_positions = positions

                other_positions_list: List[NDArray] = []

                for i in range(pipeline_state.sample_positions.shape[0]):
                    if pipeline_state.cluster_indices[i] != cluster_idx:
                        other_positions_list.append(pipeline_state.sample_positions[i])

                if len(other_positions_list) > 0:
                    other_positions = np.stack(other_positions_list)
                else:
                    other_positions = np.zeros((1, 3))

            assert selected_positions is not None
            assert selected_positions.shape[0] > 0

            assert other_positions is not None
            assert other_positions.shape[0] > 0

            selected_mean_position = np.mean(selected_positions, axis=0)
            other_mean_position = np.mean(other_positions, axis=0)

            dir = selected_mean_position - other_mean_position
            dir /= np.linalg.norm(dir)

            closest_view: View = None
            closest_view_distance = np.nan

            for view in pipeline_state.training_views:
                rel_view_position = view.transform.T[3, :3] - selected_mean_position

                projection = np.dot(rel_view_position, dir)

                if projection < 0:
                    continue

                projected_rel_view_position = selected_mean_position + dir * projection

                rel_view_distance = np.linalg.norm(rel_view_position - projected_rel_view_position)

                if not np.isfinite(closest_view_distance) or rel_view_distance < closest_view_distance:
                    closest_view_distance = rel_view_distance
                    closest_view = view

            assert closest_view is not None

            selected_views.append(closest_view)

        if pipeline_state.scratch_output_dir is not None:
            with (pipeline_state.scratch_output_dir / "dbscan_sample_labels.json").open("w") as f:
                json.dump(
                    [
                        {
                            "cluster": cluster,
                            "labels": labels,
                        }
                        for cluster, labels in dbscan_sample_labels.items()
                    ],
                    f,
                )

            with (pipeline_state.scratch_output_dir / "dbscan_view_labels.json").open("w") as f:
                json.dump(
                    [
                        {
                            "cluster": cluster,
                            "label": label,
                        }
                        for cluster, label in dbscan_view_labels.items()
                    ],
                    f,
                )

        return selected_views
