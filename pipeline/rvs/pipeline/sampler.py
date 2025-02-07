from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import trimesh
from nerfstudio.configs.base_config import InstantiateConfig
from trimesh import triangles
from trimesh.scene import Scene
from trimesh.typed import Integer, NDArray, Number, Optional

from rvs.pipeline.state import PipelineState
from rvs.utils.trimesh import normalize_scene_manual


@dataclass
class PositionSamplerConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: PositionSampler)

    num_samples: int = 10000
    """Desired number of samples"""

    min_distance: float = 0.025
    """Minimum distance between samples"""


class PositionSampler:
    config: PositionSamplerConfig

    def __init__(self, config: PositionSamplerConfig):
        self.config = config

    def sample(self, file: Path, pipeline_state: PipelineState) -> NDArray:
        pass


@dataclass
class TrimeshPositionSamplerConfig(PositionSamplerConfig):
    _target: Type = field(default_factory=lambda: TrimeshPositionSampler)


class TrimeshPositionSampler(PositionSampler):
    def sample(self, file: Path, pipeline_state: PipelineState) -> NDArray:
        if pipeline_state.model_normalization is None:
            raise Exception("Model normalization required")

        obj = trimesh.load(file)

        if not isinstance(obj, Scene):
            raise Exception(f"File {str(file)} is not a scene")

        scene: Scene = obj

        scene = normalize_scene_manual(scene, pipeline_state.model_normalization)

        # NB: This must happen after normalize_scene, otherwise the scales
        # may end up different than what was used to render the views
        # FIXME: Seems not needed after all..
        # self.__apply_coordinate_system_transform(scene)

        tris = scene.triangles

        num_triangles = tris.shape[0]

        vertices = tris.reshape((-1, 3))

        faces = np.zeros((num_triangles, 3), dtype=np.uint32)
        for i in range(num_triangles):
            faces[i][0] = i * 3 + 0
            faces[i][1] = i * 3 + 1
            faces[i][2] = i * 3 + 2

        # Use triangle area as weight
        areas = triangles.area(crosses=triangles.cross(tris))

        positions, _ = self.__sample_surface_even(
            vertices,
            faces,
            self.config.num_samples,
            face_weight=areas,
            radius=self.config.min_distance,
            seed=pipeline_state.pipeline.config.machine.seed,
        )

        return positions

    def __apply_coordinate_system_transform(self, scene: Scene) -> None:
        """Brings the trimesh coordinates system in line with the other coordinate systems"""

        angle_z = 0.0
        rot_z = np.array(
            [
                [np.cos(angle_z), -np.sin(angle_z), 0, 0],
                [np.sin(angle_z), np.cos(angle_z), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        angle_y = 0.0
        rot_y = np.array(
            [
                [np.cos(angle_y), 0, np.sin(angle_y), 0],
                [0, 1, 0, 0],
                [-np.sin(angle_y), 0, np.cos(angle_y), 0],
                [0, 0, 0, 1],
            ]
        )

        angle_x = np.pi * 0.5
        rot_x = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(angle_x), -np.sin(angle_x), 0],
                [0, np.sin(angle_x), np.cos(angle_x), 0],
                [0, 0, 0, 1],
            ]
        )

        transform = np.eye(4)
        transform = np.dot(transform, rot_z)
        transform = np.dot(transform, rot_y)
        transform = np.dot(transform, rot_x)

        scene.apply_transform(transform)

    # Adapted from trimesh to work with arbitrary vertices/faces instead of mesh
    def __sample_surface(
        self,
        vertices: NDArray,
        faces: NDArray,
        count: Integer,
        face_weight: Optional[NDArray] = None,
        seed=None,
    ) -> NDArray:
        if face_weight is None:
            face_weight = np.ones((faces.shape[0]))

        weight_cum = np.cumsum(face_weight)

        if seed is None:
            random = np.random.random
        else:
            random = np.random.default_rng(seed).random

        face_pick = random(count) * weight_cum[-1]
        face_index = np.searchsorted(weight_cum, face_pick)

        tri_origins = vertices[faces[:, 0]]
        tri_vectors = vertices[faces[:, 1:]].copy()
        tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

        tri_origins = tri_origins[face_index]
        tri_vectors = tri_vectors[face_index]

        random_lengths = random((len(tri_vectors), 2, 1))

        random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
        random_lengths[random_test] -= 1.0
        random_lengths = np.abs(random_lengths)

        sample_vector = (tri_vectors * random_lengths).sum(axis=1)

        samples = sample_vector + tri_origins

        return samples, face_index

    # Adapted from trimesh to work with arbitrary vertices/faces instead of mesh
    def __sample_surface_even(
        self,
        vertices: NDArray,
        faces: NDArray,
        count: Integer,
        face_weight: Optional[NDArray] = None,
        radius: Optional[Number] = None,
        seed=None,
    ) -> NDArray:
        from trimesh.points import remove_close

        points, index = self.__sample_surface(
            vertices,
            faces,
            count * 3,
            face_weight=face_weight,
            seed=seed,
        )

        points, mask = remove_close(points, radius)

        if len(points) >= count:
            return points[:count], index[mask][:count]

        return points, index[mask]
