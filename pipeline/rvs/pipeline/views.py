from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Type

import numpy as np
from nerfstudio.configs.base_config import InstantiateConfig
from numpy.typing import NDArray

from rvs.pipeline.state import PipelineState


@dataclass
class ViewsConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Views)
    """target class to instantiate"""


class View:
    index: int
    transform: NDArray
    path: Optional[Path]

    def __init__(self, index: int, transform: NDArray, path: Optional[Path] = None) -> None:
        self.index = index
        self.transform = transform
        self.path = path


class Views:
    config: ViewsConfig

    def __init__(self, config: ViewsConfig) -> None:
        self.config = config

    def generate(self, pipeline_state: PipelineState) -> List[View]:
        return []


@dataclass
class SphereViewsConfig(ViewsConfig):
    _target: Type = field(default_factory=lambda: SphereViews)
    """target class to instantiate"""

    azimuth_views: int = 16
    """Number of views around azimuthal axis"""

    elevation_views: int = 8
    """Number of views around elevation axis"""

    offset_views: int = 1
    """Number of views along offset axis from center"""

    offset_min_distance: float = 2.0
    """Minimum distance from center"""

    offset_max_distance: float = 2.0
    """Maximum distance from center"""

    exclude_poles: bool = True
    """Whether to exclude the views at the top/bottom poles"""

    # TODO: Possibly also include roll angle?


class SphereViews(Views):
    config: SphereViewsConfig

    def __init__(self, config: SphereViewsConfig) -> None:
        self.config = config

    def generate(self, pipeline_state: PipelineState) -> List[View]:
        views = []

        for i in range(0, self.config.azimuth_views):
            for j in range(0, self.config.elevation_views):
                for k in range(0, self.config.offset_views):
                    azimuth = 2.0 * np.pi / self.config.azimuth_views * i
                    elevation = 0.0
                    if self.config.exclude_poles:
                        elevation = -np.pi * 0.5 + np.pi / (self.config.elevation_views + 1) * (j + 1)
                    else:
                        elevation = -np.pi * 0.5 + np.pi / self.config.elevation_views * j
                    roll = 0.0
                    distance = (
                        self.config.offset_min_distance
                        + (self.config.offset_max_distance - self.config.offset_min_distance)
                        / self.config.offset_views
                        * k
                    )

                    angle_z = roll
                    rot_z = np.array(
                        [
                            [np.cos(angle_z), -np.sin(angle_z), 0, 0],
                            [np.sin(angle_z), np.cos(angle_z), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ]
                    )

                    angle_y = azimuth
                    rot_y = np.array(
                        [
                            [np.cos(angle_y), 0, np.sin(angle_y), 0],
                            [0, 1, 0, 0],
                            [-np.sin(angle_y), 0, np.cos(angle_y), 0],
                            [0, 0, 0, 1],
                        ]
                    )

                    angle_x = elevation
                    rot_x = np.array(
                        [
                            [1, 0, 0, 0],
                            [0, np.cos(angle_x), -np.sin(angle_x), 0],
                            [0, np.sin(angle_x), np.cos(angle_x), 0],
                            [0, 0, 0, 1],
                        ]
                    )

                    rotation = np.eye(4)
                    rotation = np.dot(rotation, rot_z)
                    rotation = np.dot(rotation, rot_y)
                    rotation = np.dot(rotation, rot_x)

                    translation = np.eye(4)
                    translation[0, 3] = 0.0
                    translation[1, 3] = 0.0
                    translation[2, 3] = distance

                    transform = np.dot(rotation, translation)

                    views.append(View(index=len(views), transform=transform))

        return views


@dataclass
class FermatSpiralViewsConfig(ViewsConfig):
    _target: Type = field(default_factory=lambda: FermatSpiralViews)
    """target class to instantiate"""

    n: int = 63
    """Generates n*2+1 views"""

    offset_views: int = 1
    """Number of views along offset axis from center"""

    offset_min_distance: float = 2.0
    """Minimum distance from center"""

    offset_max_distance: float = 2.0
    """Maximum distance from center"""


# González, Á (2009), Measurement of Areas on a Sphere Using Fibonacci and Latitude–Longitude Lattices. http://dx.doi.org/10.1007/s11004-009-9257-x
# Swinbank, R. and James Purser, R. (2006), Fibonacci grids: A novel approach to global modelling. https://doi.org/10.1256/qj.05.227
class FermatSpiralViews(Views):
    config: FermatSpiralViewsConfig

    def __init__(self, config: FermatSpiralViewsConfig) -> None:
        self.config = config

    def generate(self, pipeline_state: PipelineState) -> List[View]:
        views: List[View] = []

        n = self.config.n

        i = np.arange(-n, n + 1)

        phi = (1.0 + np.sqrt(5.0)) / 2.0
        phi_inv = phi - 1.0

        lat = np.arcsin(2.0 * i / (2.0 * n + 1.0))
        lon = 2.0 * np.pi * i * phi_inv

        x = np.cos(lat) * np.cos(lon)
        y = np.sin(lat)
        z = np.cos(lat) * np.sin(lon)

        world_up = np.array([0.0, 1.0, 0.0])

        for k in range(self.config.offset_views):
            distance = (
                self.config.offset_min_distance
                + (self.config.offset_max_distance - self.config.offset_min_distance) / self.config.offset_views * k
            )

            for j in range(i.shape[0]):
                forward = -np.stack([x[j], y[j], z[j]])
                forward /= np.linalg.norm(forward)

                right = np.cross(forward, world_up)
                right /= np.linalg.norm(right)

                up = np.cross(right, forward)
                up /= np.linalg.norm(up)

                pos = -forward * distance

                transform = np.array(
                    [
                        np.append(right, 0.0),
                        np.append(up, 0.0),
                        np.append(-forward, 0.0),
                        np.append(pos, 1.0),
                    ]
                ).T

                views.append(View(index=len(views), transform=transform))

        return views


# TODO: Regular convex polyhedra (platonic solids ~> max. 20 equally spaced views possible)
