import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Type

import numpy as np
import pyrender
import trimesh
from nerfstudio.configs.base_config import InstantiateConfig
from numpy.typing import NDArray
from PIL import Image as im
from pyrender.constants import RenderFlags
from scipy.spatial.transform import Rotation as R
from trimesh import Scene
from trimesh.scene import Camera

from rvs.pipeline.views import View
from rvs.utils.trimesh import normalize_scene


@dataclass
class RendererConfig(InstantiateConfig):
    """Configuration for a full RVS pipeline"""

    _target: Type = field(default_factory=lambda: Renderer)
    """target class to instantiate"""

    width: int = 1024
    """Horizontal resolution in pixels"""

    height: int = 1024
    """Vertical resolution in pixels"""

    fov: float = 60
    """Horizontal field of view in degrees"""

    background: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    """Background color: (4,) uint8"""

    @property
    def fov_x(self) -> float:
        return self.fov

    @property
    def fov_y(self) -> float:
        hfov = np.deg2rad(self.fov)
        vfov = 2.0 * np.arctan(np.tan(hfov * 0.5) * self.height / self.width)
        return np.rad2deg(vfov)

    @property
    def focal_length_x(self) -> float:
        return self.width * 0.5 / np.tan(np.deg2rad(self.fov_x) * 0.5)

    @property
    def focal_length_y(self) -> float:
        return self.height * 0.5 / np.tan(np.deg2rad(self.fov_y) * 0.5)


class Renderer:
    config: RendererConfig

    def __init__(self, config: RendererConfig) -> None:
        self.config = config

    def render(self, file: Path, views: List[View], callback: Callable[[View, im.Image], None]) -> None:
        pass


@dataclass
class TrimeshRendererConfig(RendererConfig):
    _target: Type = field(default_factory=lambda: TrimeshRenderer)

    sample_size: float = 0.015
    """Size of sample spheres"""


class TrimeshRenderer(Renderer):
    config: TrimeshRendererConfig

    def __init__(self, config: TrimeshRendererConfig) -> None:
        super().__init__(config)
        self.config = config

    def render(
        self,
        file: Path,
        views: List[View],
        callback: Callable[[View, im.Image], None],
        sample_positions: Optional[NDArray] = None,
        sample_colors: Optional[NDArray] = None,
    ) -> None:
        obj = trimesh.load(file)

        if not isinstance(obj, Scene):
            raise Exception(f"File {str(file)} is not a scene")

        scene: Scene = obj

        scene = normalize_scene(scene)

        if sample_positions is not None:
            self.add_sample_positions(scene, sample_positions, sample_colors=sample_colors)

        camera = Camera(fov=[self.config.fov_x, self.config.fov_y])

        for view in views:
            with self.render_view(scene, camera, view) as image:
                callback(view, image)

    def add_sample_positions(
        self, scene: Scene, sample_positions: NDArray, sample_colors: Optional[NDArray] = None
    ) -> None:
        for i in range(sample_positions.shape[0]):
            position = sample_positions[i]
            color = np.array([1.0, 1.0, 1.0]) if sample_colors is None else sample_colors[i]

            transform = np.eye(4)
            transform[0, 3] = position[0]
            transform[1, 3] = position[1]
            transform[2, 3] = position[2]

            sphere_mesh = trimesh.creation.uv_sphere(count=[3, 3])
            sphere_mesh.vertices *= self.config.sample_size
            sphere_mesh.visual.vertex_colors = sphere_mesh.vertices * 0.0 + color

            scene.add_geometry(sphere_mesh, transform=transform)

    def render_view(self, scene: Scene, camera: Camera, view: View) -> im.Image:
        scene.camera_transform = view.transform

        png_bytes = scene.save_image(
            resolution=[self.config.width, self.config.height], background=self.config.background
        )

        with io.BytesIO(png_bytes) as buf:
            image = im.open(buf)
            image.load()
            self.process_image(image)
            return image

    def process_image(self, image: im.Image) -> None:
        if self.config.background[3] == 0:
            # FIXME: Hack: make background transparent
            # Trimesh *should* natively have proper support for this, but it doesn't seem to work in my setup
            pixels = image.getdata()
            image.putdata(
                [
                    (self.config.background[0], self.config.background[1], self.config.background[2], 0)
                    if p[0] == self.config.background[0]
                    and p[1] == self.config.background[1]
                    and p[2] == self.config.background[2]
                    else p
                    for p in pixels
                ]
            )


@dataclass
class PyrenderRendererConfig(RendererConfig):
    _target: Type = field(default_factory=lambda: PyrenderRenderer)

    render_coordinate_background: bool = True


class PyrenderRenderer(Renderer):
    config: PyrenderRendererConfig

    def __init__(self, config: RendererConfig) -> None:
        super().__init__(self, config)
        self.config = config

    def render(self, file: Path, views: List[View], callback: Callable[[View, im.Image], None]) -> None:
        obj = trimesh.load(file)
        tmesh = list(obj.geometry.values())[0]

        tmesh.vertices -= mesh.centroid
        tmesh.vertices *= 1.0 / np.max(mesh.extents)

        mesh = pyrender.Mesh.from_trimesh(tmesh)

        camera = pyrender.PerspectiveCamera(
            yfov=self.config.fov_y, aspectRatio=self.config.height * 1.0 / self.config.width
        )

        # "Skybox" sphere colored by coordinates
        sphere_size = 10.0
        sphere_tmesh = trimesh.creation.uv_sphere()
        sphere_tmesh.vertices *= sphere_size
        sphere_tmesh.visual.vertex_colors = 0.5 + sphere_mesh.vertices / sphere_size * 0.5
        sphere_mesh = pyrender.Mesh.from_trimesh(sphere_mesh)

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=20.0)

        renderer = pyrender.OffscreenRenderer(self.config.width, self.config.height)

        for view in views:
            scene = pyrender.Scene(bg_color=self.config.background, ambient_light=[1.0, 1.0, 1.0, 1.0])

            scene.add(light, pose=view.transform)
            scene.add(camera, pose=view.transform)
            if self.config.render_coordinate_background:
                scene.add(sphere_mesh)
            scene.add(mesh)

            color, _ = renderer.render(scene, flags=RenderFlags.SKIP_CULL_FACES)

            with im.fromarray(color) as image:
                callback(view, image)
