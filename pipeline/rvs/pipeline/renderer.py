import io
import math
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from multiprocessing import Process
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import pyglet
import pyrender
import trimesh
from nerfstudio.configs.base_config import InstantiateConfig
from numpy.typing import NDArray
from PIL import Image as im
from pyglet import gl
from pyrender.constants import RenderFlags
from scipy.spatial.transform import Rotation
from trimesh import Scene
from trimesh.viewer import SceneViewer

from rvs.pipeline.state import Normalization, PipelineState
from rvs.pipeline.views import View
from rvs.utils.blender_renderer import (
    Normalization as BlenderNormalization,
    Render,
    load_normalization_json,
    renderer_worker_func,
    save_normalization_json,
    save_render_json,
)
from rvs.utils.process import ProcessResult, stop_process
from rvs.utils.trimesh import normalize_scene_auto, normalize_scene_manual


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


@dataclass
class RenderOutput:
    path: Optional[Callable[[View], Optional[Path]]]
    callback: Callable[[View, Optional[im.Image]], None]


class Renderer:
    config: RendererConfig

    def __init__(self, config: RendererConfig) -> None:
        self.config = config

    def render(
        self,
        file: Path,
        views: List[View],
        output: RenderOutput,
        pipeline_state: PipelineState,
    ) -> Normalization:
        pass


@dataclass
class TrimeshRendererConfig(RendererConfig):
    _target: Type = field(default_factory=lambda: TrimeshRenderer)

    sample_size: float = 0.015
    """Size of sample spheres"""


class _TrimeshDummyGrid:
    __callback: Callable[[], None]

    def __init__(self, callback: Callable[[], None]) -> None:
        self.__callback = callback

    def delete(self) -> None:
        pass

    def draw(self, mode: int) -> None:
        self.__callback()


class TrimeshRenderer(Renderer):
    config: TrimeshRendererConfig

    def __init__(self, config: TrimeshRendererConfig) -> None:
        super().__init__(config)
        self.config = config

    def render(
        self,
        file: Path,
        views: List[View],
        output: RenderOutput,
        pipeline_state: PipelineState,
        sample_positions: Optional[NDArray] = None,
        sample_colors: Optional[NDArray] = None,
        flat_model_color: Optional[NDArray] = None,
        projection_callback: Optional[Callable[[View, NDArray], None]] = None,
        projection_depth_test: bool = True,
        projection_depth_test_eps: float = 0.0001,
        render_sample_positions: bool = True,
        render_model: bool = True,
    ) -> Normalization:
        obj = trimesh.load(file, skip_materials=flat_model_color is not None)

        if not isinstance(obj, Scene):
            raise Exception(f"File {str(file)} is not a scene")

        scene: Scene = obj

        if not render_model:
            from trimesh import Trimesh, util
            from trimesh.visual.color import ColorVisuals
            from trimesh.visual.material import SimpleMaterial

            for geometry in list(scene.geometry.values()):
                if util.is_instance_named(geometry, "Trimesh"):
                    tm: Trimesh = geometry
                    tm.visual = ColorVisuals(
                        mesh=tm,
                        face_colors=np.zeros((4,), dtype=np.uint8),
                        vertex_colors=np.zeros((4,), dtype=np.uint8),
                    )

        elif flat_model_color is not None:
            from trimesh import Trimesh, util
            from trimesh.visual.material import SimpleMaterial
            from trimesh.visual.texture import TextureVisuals

            mat = SimpleMaterial(diffuse=flat_model_color)
            vis = TextureVisuals(material=mat)

            for geometry in list(scene.geometry.values()):
                if util.is_instance_named(geometry, "Trimesh"):
                    tm: Trimesh = geometry
                    tm.visual = vis
                    # if hasattr(tm.visual, "material"):
                    #    tm.visual.material = mat

        normalization = pipeline_state.model_normalization

        if normalization is not None:
            scene = normalize_scene_manual(scene, normalization)
        else:
            scene, normalization = normalize_scene_auto(scene)

        # Need to set alpha_size=8 to enable alpha channel. This is not set by default and it seems
        # in the headless environment it defaults to 0 which of course disables the alpha channel.
        # Other defaults are taken from trimesh SceneViewer.__init__()
        pyglet_conf = gl.Config(sample_buffers=1, samples=4, depth_size=24, double_buffer=True, alpha_size=8)

        scene.camera.fov = np.array([self.config.fov_x, self.config.fov_y])

        def viewer_callback(scene: Any):
            pass

        viewer = SceneViewer(
            scene,
            start_loop=False,
            visible=False,
            resolution=[self.config.width, self.config.height],
            background=self.config.background,
            window_conf=pyglet_conf,
            flags={
                "cull": False,
            },
            callback=viewer_callback,  # Seems to be required to update buffers after scene modification
        )

        current_rendering_view: int = -1

        if projection_callback is not None and sample_positions is not None:
            depth_maps: Optional[List[NDArray]] = None

            if projection_depth_test:
                mesh: Trimesh = scene.to_mesh()

                depth_maps = []

                for view in views:
                    scene.camera_transform = view.transform

                    # Based on https://github.com/mikedh/trimesh/blob/main/examples/raytrace.py but without normalization
                    origins, vectors, pixels = scene.camera_rays()

                    points, index_ray, _ = mesh.ray.intersects_location(origins, vectors, multiple_hits=False)

                    depth_values = trimesh.util.diagonal_dot(points - origins[0], vectors[index_ray])

                    pixel_ray = pixels[index_ray]

                    depth_map = np.zeros(scene.camera.resolution)

                    depth_map[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_values

                    depth_maps.append(depth_map)

            if depth_maps is not None:
                assert len(depth_maps) == len(views)

            def draw_callback() -> None:
                nonlocal current_rendering_view
                nonlocal sample_positions
                nonlocal depth_maps

                if current_rendering_view >= 0:
                    modelview_matrix = (gl.GLdouble * 16)()
                    gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX, modelview_matrix)

                    projection_matrix = (gl.GLdouble * 16)()
                    gl.glGetDoublev(gl.GL_PROJECTION_MATRIX, projection_matrix)

                    viewport_matrix = (gl.GLint * 4)()
                    gl.glGetIntegerv(gl.GL_VIEWPORT, viewport_matrix)

                    x = (gl.GLdouble)()
                    y = (gl.GLdouble)()
                    z = (gl.GLdouble)()

                    p: List[List[float]] = []

                    for i in range(sample_positions.shape[0]):
                        gl.gluProject(
                            sample_positions[i][0],
                            sample_positions[i][1],
                            sample_positions[i][2],
                            modelview_matrix,
                            projection_matrix,
                            viewport_matrix,
                            x,
                            y,
                            z,
                        )

                        wx = x.value
                        wy = y.value
                        wz = z.value

                        # https://learnopengl.com/Advanced-OpenGL/Depth-testing (Visualizing the depth buffer)
                        z_near = scene.camera.z_near
                        z_far = scene.camera.z_far
                        z_ndc = wz * 2.0 - 1.0
                        z_linear = (2.0 * z_near * z_far) / (z_far + z_near - z_ndc * (z_far - z_near))

                        depth_test = 1

                        if depth_maps is not None and projection_depth_test:
                            ix = math.floor(wx)
                            iy = math.floor(wy)

                            depth_map = depth_maps[current_rendering_view]

                            if ix >= 0 and ix < depth_map.shape[0] and iy >= 0 and iy < depth_map.shape[1]:
                                depth = depth_map[ix, iy]

                                if z_linear > depth + projection_depth_test_eps:
                                    depth_test = 0
                            else:
                                depth_test = 0

                        p.append([wx / self.config.width, 1.0 - wy / self.config.height, z_linear, depth_test])

                    projection_callback(np.array(p))

            dummy_grid = _TrimeshDummyGrid(draw_callback)

            viewer._grid = dummy_grid
            viewer.toggle_grid()

            assert viewer.view["grid"]
            assert viewer._grid == dummy_grid

        if sample_positions is not None and render_sample_positions:
            self.add_sample_positions(scene, sample_positions, sample_colors=sample_colors)

        try:

            def redraw_viewer() -> None:
                pyglet.clock.tick()
                viewer.switch_to()
                viewer.dispatch_events()
                viewer.dispatch_event("on_draw")
                viewer.flip()

            for _ in range(2):
                redraw_viewer()

            for i, view in enumerate(views):
                scene.camera_transform = view.transform

                current_rendering_view = i
                try:
                    redraw_viewer()
                finally:
                    current_rendering_view = -1

                with io.BytesIO() as buffer:
                    viewer.save_image(buffer)
                    buffer.seek(0)

                    with im.open(buffer) as image:
                        image.load()
                        self.process_image(image)
                        output.callback(view, image)
        finally:
            viewer.close()

        return normalization

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

    def render(
        self,
        file: Path,
        views: List[View],
        output: RenderOutput,
        pipeline_state: PipelineState,
    ) -> Normalization:
        obj = trimesh.load(file)
        tmesh = list(obj.geometry.values())[0]

        normalization: Normalization = pipeline_state.model_normalization

        # FIXME Implement normalization
        raise Exception("Not implemented")

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
                output.callback(view, image)

        return normalization


@dataclass
class BlenderRendererConfig(RendererConfig):
    _target: Type = field(default_factory=lambda: BlenderRenderer)

    blender_binary: Path = Path(__file__).parent / ".." / ".." / "blender" / "blender"
    """Path to the blender binary"""


class BlenderRenderer(Renderer):
    config: BlenderRendererConfig

    def __init__(self, config: BlenderRendererConfig) -> None:
        super().__init__(config)
        self.config = config

    def render(
        self,
        file: Path,
        views: List[View],
        output: RenderOutput,
        pipeline_state: PipelineState,
    ) -> Normalization:
        if pipeline_state.scratch_output_dir is None:
            raise Exception("Scratch dir required")

        renders_dir = pipeline_state.scratch_output_dir / "renders"

        if renders_dir.exists():
            shutil.rmtree(renders_dir)

        renders_dir.mkdir(parents=True)

        start_mtime_date = datetime.fromtimestamp(renders_dir.stat().st_mtime, tz=timezone.utc)

        renders = self.create_render_files(views, output, renders_dir)

        normalization = pipeline_state.model_normalization
        normalization_file: Path

        if normalization is not None:
            normalization_file = self.create_normalization_file_manual(normalization, pipeline_state.scratch_output_dir)
        else:
            normalization_file = self.create_normalization_file_auto(file, pipeline_state.scratch_output_dir)
            normalization = self.from_blender_normalization(load_normalization_json(normalization_file))

        self.run_renders(file, list(renders.keys()), normalization_file)

        for render in renders.values():
            output_file = Path(render.output_file)

            if not output_file.exists():
                raise Exception(f"Render output file {render.output_file} does not exist")

            mtime_date = datetime.fromtimestamp(output_file.stat().st_mtime, tz=timezone.utc)
            if mtime_date < start_mtime_date:
                raise Exception(f"Render output file {render.output_file} is outdated")

            for view in views:
                if view.index == render.index:
                    if render.output_file.startswith(str(renders_dir.resolve())):
                        with im.open(render.output_file) as image:
                            image.load()
                            output.callback(view, image)
                    else:
                        output.callback(view, None)

        return normalization

    def create_render_files(self, views: List[View], output: RenderOutput, dir: Path) -> Dict[Path, Render]:
        tol = 0.000001

        focal_length = self.config.focal_length_x

        if np.abs(focal_length - self.config.focal_length_y) > tol:
            raise Exception(
                f"Focal length X {self.config.focal_length_x} and Y {self.config.focal_length_y} are not equal"
            )

        renders: Dict[Path, Render] = dict()

        for view in views:
            position, rotation = self.decompose_transform_for_blender(view.transform)

            output_file: str = None

            if output.path is not None:
                output_file = output.path(view)

            if output_file is None:
                output_file = dir / f"render_{view.index}.png"

            render = Render(
                index=view.index,
                focal_length=focal_length,
                width=self.config.width,
                height=self.config.height,
                position=tuple(position),
                rotation=tuple(rotation),
                output_file=str(output_file.resolve()),
            )

            file = save_render_json(render, dir / f"render_{view.index}.json")

            renders[file] = render

        return renders

    def decompose_transform_for_blender(self, transform: NDArray) -> Tuple[NDArray, NDArray]:
        tol = 0.000001

        transform = self.to_blender_transform(transform)

        if np.linalg.norm(transform[3, :] - np.array([0.0, 0.0, 0.0, 1.0])) > tol:
            raise Exception(
                f"Invalid homogeneous coordinates {transform[3, :].tolist()}, expected [0.0, 0.0, 0.0, 1.0]"
            )

        rotation = transform[:3, :3]

        if np.abs(np.linalg.det(rotation) - 1.0) > tol:
            raise Exception(f"Invalid top left 3x3 matrix determinant {np.linalg.det(rotation)}, expected 1.0")

        if np.sum(rotation.T @ rotation - np.eye(3)) > tol:
            raise Exception("Top left 3x3 matrix is non-orthogonal")

        euler_angles = Rotation.from_matrix(rotation).as_euler("xyz", degrees=False)

        translation = transform[:3, 3]

        return (translation, euler_angles)

    def create_normalization_file_auto(
        self, model_file: Path, dir: Path, gpu: int = 0, disable_stdout: bool = True, disable_stderr=True
    ) -> Path:
        normalization_file = dir / "normalization.json"

        try:
            with ProcessResult() as result:
                process = Process(
                    target=renderer_worker_func,
                    args=(
                        self.config.blender_binary,
                        model_file,
                        [],
                        normalization_file,
                        True,
                        gpu,
                        result,
                        disable_stdout,
                        disable_stderr,
                    ),
                    daemon=True,
                )

                process.start()
                process.join()

                if not result.success:
                    if result.msg is not None:
                        raise Exception(f"Normalization failed due to exception:\n{result.msg}")
                    else:
                        raise Exception("Normalization failed due to unknown reason")
        finally:
            try:
                stop_process(process)
                process.join()
            except Exception:
                pass

            try:
                process.close()
            except Exception:
                pass

        if not normalization_file.exists():
            raise Exception(f"Normalization file {str(normalization_file)} does not exist")

        return normalization_file

    def create_normalization_file_manual(self, normalization: Normalization, dir: Path) -> Path:
        normalization_file = dir / "normalization.json"

        save_normalization_json(self.to_blender_normalization(normalization), normalization_file)

        return normalization_file

    def to_blender_transform(self, transform: NDArray) -> NDArray:
        rot_x = 0.5 * np.pi
        return (
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, np.cos(rot_x), -np.sin(rot_x), 0],
                    [0, np.sin(rot_x), np.cos(rot_x), 0],
                    [0, 0, 0, 1],
                ]
            )
            @ transform
        )

    def to_blender_normalization(self, normalization: Normalization) -> BlenderNormalization:
        return BlenderNormalization(
            scale=(normalization.scale[0], normalization.scale[2], normalization.scale[1]),
            offset=(normalization.offset[0], -normalization.offset[2], normalization.offset[1]),
        )

    def from_blender_normalization(self, normalization: BlenderNormalization) -> Normalization:
        return Normalization(
            np.array([normalization.scale[0], normalization.scale[2], normalization.scale[1]]),
            np.array([normalization.offset[0], normalization.offset[2], -normalization.offset[1]]),
        )

    def run_renders(
        self,
        model_file: Path,
        render_files: List[Path],
        normalization_file: Path,
        processes: int = 4,
        gpu: int = 0,
        disable_stdout: bool = True,
        disable_stderr=True,
    ) -> None:
        if processes < 1:
            raise ValueError(f"processes ({processes}) < 1")

        chunks: List[List[Path]] = [[] for _ in range(processes)]

        for i in range(len(render_files)):
            chunks[i % processes].append(render_files[i])

        processes: List[Tuple[Process, ProcessResult]] = []

        try:
            for chunk in chunks:
                if len(chunk) > 0:
                    result = ProcessResult()

                    process = Process(
                        target=renderer_worker_func,
                        args=(
                            self.config.blender_binary,
                            model_file,
                            chunk,
                            normalization_file,
                            False,
                            gpu,
                            result,
                            disable_stdout,
                            disable_stderr,
                        ),
                        daemon=True,
                    )

                    processes.append((process, result))

                    process.start()

            for process, result in processes:
                process.join()

                if not result.success:
                    if result.msg is not None:
                        raise Exception(f"Renderer failed due to exception:\n{result.msg}")
                    else:
                        raise Exception("Renderer failed due to unknown reason")

        finally:
            for process, result in processes:
                try:
                    result.close()
                except Exception:
                    pass

                try:
                    stop_process(process)
                    process.join()
                except Exception:
                    pass

                try:
                    process.close()
                except Exception:
                    pass
