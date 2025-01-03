import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Type

import numpy as np
import pyglet
import pyrender
import trimesh
from nerfstudio.configs.base_config import InstantiateConfig
from numpy.typing import NDArray
from PIL import Image as im
from pyglet import gl
from pyrender.constants import RenderFlags
from trimesh import Scene
from trimesh.scene import Camera
from trimesh.viewer import SceneViewer

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

        # Need to set alpha_size=8 to enable alpha channel. This is not set by default and it seems
        # in the headless environment it defaults to 0 which of course disables the alpha channel.
        # Other defaults are taken from trimesh SceneViewer.__init__()
        pyglet_conf = gl.Config(sample_buffers=1, samples=4, depth_size=24, double_buffer=True, alpha_size=8)

        viewer = SceneViewer(
            scene,
            start_loop=False,
            visible=False,
            resolution=[self.config.width, self.config.height],
            background=self.config.background,
            window_conf=pyglet_conf,
        )

        try:

            def redraw_viewer() -> None:
                pyglet.clock.tick()
                viewer.switch_to()
                viewer.dispatch_events()
                viewer.dispatch_event("on_draw")
                viewer.flip()

            for _ in range(2):
                redraw_viewer()

            for view in views:
                scene.camera_transform = view.transform

                redraw_viewer()

                with io.BytesIO() as buffer:
                    viewer.save_image(buffer)
                    buffer.seek(0)

                    with im.open(buffer) as image:
                        image.load()
                        self.process_image(image)
                        callback(view, image)
        finally:
            viewer.close()

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

    def process_image(self, image: im.Image) -> None:
        if False and self.config.background[3] == 0:
            # No longer necessary as alpha channel works now
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
