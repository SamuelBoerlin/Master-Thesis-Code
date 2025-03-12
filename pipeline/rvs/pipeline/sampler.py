import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Type

import numpy as np
import trimesh
from nerfstudio.configs.base_config import InstantiateConfig
from trimesh import triangles
from trimesh.scene import Scene
from trimesh.typed import Integer, NDArray, Number, Optional

from rvs.pipeline.pipeline import Normalization
from rvs.pipeline.state import PipelineState
from rvs.utils.trimesh import normalize_scene_manual


@dataclass
class PositionSamplerConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: PositionSampler)


class PositionSampler:
    config: PositionSamplerConfig

    def __init__(self, config: PositionSamplerConfig):
        self.config = config

    def sample(self, file: Path, normalization: Normalization, pipeline_state: PipelineState) -> NDArray:
        pass


class BaseTrimeshPositionSampler(PositionSampler):
    def sample(self, file: Path, normalization: Normalization, pipeline_state: PipelineState) -> NDArray:
        obj = trimesh.load(file)

        if not isinstance(obj, Scene):
            raise Exception(f"File {str(file)} is not a scene")

        scene: Scene = obj

        scene = normalize_scene_manual(scene, normalization)

        tris: NDArray = scene.triangles

        num_triangles = tris.shape[0]

        vertices = tris.reshape((-1, 3))

        faces = np.zeros((num_triangles, 3), dtype=np.uint32)
        for i in range(num_triangles):
            faces[i][0] = i * 3 + 0
            faces[i][1] = i * 3 + 1
            faces[i][2] = i * 3 + 2

        face_weight = self._sample_face_weight(tris, vertices, faces)

        positions = self._sample(
            file,
            pipeline_state,
            tris,
            vertices,
            faces,
            face_weight=face_weight,
            seed=pipeline_state.pipeline.config.machine.seed,
        )

        return positions

    def _sample_face_weight(self, tris: NDArray, vertices: NDArray, faces: NDArray) -> NDArray:
        # Use triangle surface area as weight for uniform distribution
        return self._surface_area(tris)

    def _sample(
        self,
        file: Path,
        pipeline_state: PipelineState,
        tris: NDArray,
        vertices: NDArray,
        faces: NDArray,
        face_weight: Optional[NDArray] = None,
        seed=None,
    ) -> NDArray:
        pass

    def _surface_area(self, tris: NDArray) -> NDArray:
        return triangles.area(crosses=triangles.cross(tris))

    # Adapted from trimesh to work with arbitrary vertices/faces instead of mesh
    def _sample_surface_even(
        self,
        vertices: NDArray,
        faces: NDArray,
        count: Integer,
        face_weight: Optional[NDArray] = None,
        radius: Optional[Number] = None,
        seed=None,
    ) -> Tuple[NDArray, NDArray[np.intp]]:
        from trimesh.points import remove_close

        points, index = self._sample_surface(
            vertices,
            faces,
            count * 3,
            face_weight=face_weight,
            seed=seed,
        )

        if radius is None:
            return points, index

        points, mask = remove_close(points, radius)

        if len(points) >= count:
            return points[:count], index[mask][:count]

        return points, index[mask]

    # Adapted from trimesh to work with arbitrary vertices/faces instead of mesh
    def _sample_surface(
        self,
        vertices: NDArray,
        faces: NDArray,
        count: Integer,
        face_weight: Optional[NDArray] = None,
        seed=None,
    ) -> Tuple[NDArray, NDArray[np.intp]]:
        if face_weight is None:
            face_weight = np.ones((faces.shape[0]))

        weight_cum = np.cumsum(face_weight)

        if seed is None:
            rng = np.random.random
        else:
            rng = np.random.default_rng(seed).random

        face_pick = rng(count) * weight_cum[-1]
        face_index = np.searchsorted(weight_cum, face_pick)

        tri_origins = vertices[faces[:, 0]]
        tri_vectors = vertices[faces[:, 1:]].copy()
        tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

        tri_origins = tri_origins[face_index]
        tri_vectors = tri_vectors[face_index]

        random_lengths = rng((len(tri_vectors), 2, 1))

        random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
        random_lengths[random_test] -= 1.0
        random_lengths = np.abs(random_lengths)

        sample_vector = (tri_vectors * random_lengths).sum(axis=1)

        samples = sample_vector + tri_origins

        return samples, face_index


@dataclass
class MinDistanceTrimeshPositionSamplerConfig(PositionSamplerConfig):
    _target: Type = field(default_factory=lambda: MinDistanceTrimeshPositionSampler)

    num_samples: int = 10000
    """Number of samples before decimation"""

    min_distance: float = 0.025
    """Minimum distance between samples"""


@dataclass
class TrimeshPositionSamplerConfig(MinDistanceTrimeshPositionSamplerConfig):
    """Deprecated, only for backwards compatibility"""

    pass


class MinDistanceTrimeshPositionSampler(BaseTrimeshPositionSampler):
    config: MinDistanceTrimeshPositionSamplerConfig

    def _sample(
        self,
        file: Path,
        pipeline_state: PipelineState,
        tris: NDArray,
        vertices: NDArray,
        faces: NDArray,
        face_weight: Optional[NDArray] = None,
        seed=None,
    ) -> NDArray:
        positions, _ = self._sample_surface_even(
            vertices,
            faces,
            self.config.num_samples,
            face_weight=face_weight,
            radius=self.config.min_distance,
            seed=seed,
        )
        return positions


class TrimeshPositionSampler(MinDistanceTrimeshPositionSampler):
    """Deprecated, only for backwards compatibility"""

    pass


@dataclass
class BinarySearchDensityTrimeshPositonSamplerConfig(PositionSamplerConfig):
    _target: Type = field(default_factory=lambda: BinarySearchDensityTrimeshPositionSampler)

    num_samples: int = 10000
    """Number of samples before decimation"""

    samples_per_unit_area: float = 500
    """Desired number of samples per unit area (after normalization), i.e. density"""

    min_num_samples: int = 250
    """Minimum number of samples"""

    num_iterations: int = 32
    """Number of binary search iterations"""

    lowest_min_distance: float = 0.0
    """Lowest minimum distance between samples"""

    highest_min_distance: float = 0.5
    """Highest minimum distance between samples"""


class BinarySearchDensityTrimeshPositionSampler(BaseTrimeshPositionSampler):
    config: BinarySearchDensityTrimeshPositonSamplerConfig

    def _sample(
        self,
        file: Path,
        pipeline_state: PipelineState,
        tris: NDArray,
        vertices: NDArray,
        faces: NDArray,
        face_weight: Optional[NDArray] = None,
        seed=None,
    ) -> NDArray:
        surface_area = np.sum(self._surface_area(tris))

        target_num_samples = max(
            min(int(math.ceil(self.config.samples_per_unit_area * surface_area)), self.config.num_samples),
            self.config.min_num_samples,
        )

        if target_num_samples <= 0:
            return np.zeros((0, 3))

        lo_min_distance = self.config.lowest_min_distance
        hi_min_distance = self.config.highest_min_distance
        mid_min_distance = (lo_min_distance + hi_min_distance) * 0.5

        positions: NDArray = None

        iterations: List[Tuple[float, float, float, int]] = []

        for i in range(self.config.num_iterations):
            positions, _ = self._sample_surface_even(
                vertices,
                faces,
                self.config.num_samples,
                face_weight=face_weight,
                radius=mid_min_distance,
                seed=seed,
            )

            iterations.append((lo_min_distance, mid_min_distance, hi_min_distance, len(positions)))

            if len(positions) > target_num_samples:
                lo_min_distance = mid_min_distance
                mid_min_distance = (lo_min_distance + hi_min_distance) * 0.5
            elif len(positions) < target_num_samples:
                hi_min_distance = mid_min_distance
                mid_min_distance = (lo_min_distance + hi_min_distance) * 0.5
            else:
                break

        assert positions is not None

        if pipeline_state.scratch_output_dir is not None:
            json_file = pipeline_state.scratch_output_dir / "min_distance_binary_search.json"

            json_obj = [
                {
                    "lo": iteration[0],
                    "mid": iteration[1],
                    "hi": iteration[2],
                    "samples": iteration[3],
                }
                for iteration in iterations
            ]

            with json_file.open("w") as f:
                json.dump(json_obj, f)

        return positions


@dataclass
class FarthestPointSamplingDensityTrimeshPositonSamplerConfig(PositionSamplerConfig):
    _target: Type = field(default_factory=lambda: FarthestPointSamplingDensityTrimeshPositionSampler)

    num_samples: int = 10000
    """Number of samples before decimation"""

    samples_per_unit_area: float = 500
    """Desired number of samples per unit area (after normalization), i.e. density"""

    min_num_samples: int = 250
    """Minimum number of samples"""


# Y. Eldar, M. Lindenbaum, M. Porat and Y. Y. Zeevi (1997), The farthest point strategy for progressive image sampling. https://doi.org/10.1109/83.623193
# Charles R. Qi and Li Yi and Hao Su and Leonidas J. Guibas (2017), PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. https://doi.org/10.48550/arXiv.1706.02413
class FarthestPointSamplingDensityTrimeshPositionSampler(BaseTrimeshPositionSampler):
    config: FarthestPointSamplingDensityTrimeshPositonSamplerConfig

    def _sample(
        self,
        file: Path,
        pipeline_state: PipelineState,
        tris: NDArray,
        vertices: NDArray,
        faces: NDArray,
        face_weight: Optional[NDArray] = None,
        seed=None,
    ) -> NDArray:
        surface_area = np.sum(self._surface_area(tris))

        target_num_samples = max(
            min(int(math.ceil(self.config.samples_per_unit_area * surface_area)), self.config.num_samples),
            self.config.min_num_samples,
        )

        if target_num_samples <= 0:
            return np.zeros((0, 3))

        positions, _ = self._sample_surface_even(
            vertices,
            faces,
            self.config.num_samples,
            face_weight=face_weight,
            radius=None,
            seed=seed,
        )

        if positions.shape[0] == 0 or target_num_samples >= self.config.num_samples:
            return positions

        start_pos_idx = np.random.default_rng(seed).integers(0, positions.shape[0])
        start_pos = positions[start_pos_idx]

        decimated_positions = [start_pos]

        def dst(xs: NDArray, x: NDArray):
            return np.sqrt(np.sum((xs - x) ** 2, axis=1))

        min_distances = dst(positions, start_pos)

        for _ in range(target_num_samples - 1):
            farthest_pos_idx = np.argmax(min_distances)

            assert min_distances[farthest_pos_idx] >= 0

            farthest_pos = positions[farthest_pos_idx]

            decimated_positions.append(farthest_pos)

            min_distances = np.minimum(min_distances, dst(positions, farthest_pos))

            min_distances[farthest_pos_idx] = -1

        return np.array(decimated_positions)
