from dataclasses import dataclass, field
from typing import List, Type

import numpy as np
from nerfstudio.configs.base_config import InstantiateConfig
from numpy.typing import NDArray


@dataclass
class ViewsConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Views)
    """target class to instantiate"""


class View:
    index: int
    transform: NDArray

    def __init__(self, index: int, transform: NDArray) -> None:
        self.index = index
        self.transform = transform


class Views:
    config: ViewsConfig

    def __init__(self, config: ViewsConfig) -> None:
        self.config = config

    def generate(self) -> List[View]:
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

    def generate(self) -> List[View]:
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
