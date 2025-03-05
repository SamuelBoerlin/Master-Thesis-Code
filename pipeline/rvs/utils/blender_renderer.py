import argparse
import json
import os
import shlex
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import bpy
import mathutils

try:
    from rvs.utils.blender_renderer_objaverse import (
        add_lighting,
        load_object,
        reset_scene,
        scene_bbox,
        scene_root_objects,
        set_scene_render_parameters,
    )
    from rvs.utils.process import ProcessResult
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from blender_renderer_objaverse import (
        add_lighting,
        load_object,
        reset_scene,
        scene_bbox,
        scene_root_objects,
        set_scene_render_parameters,
    )
    from process import ProcessResult


def renderer_worker_func(
    blender_binary: Path,
    model_file: Path,
    render_files: List[Path],
    normalization_file: Optional[Path] = None,
    save_normalization: bool = False,
    gpu: int = 0,
    result: Optional[ProcessResult] = None,
    disable_stdout: bool = True,
    disable_stderr: bool = True,
) -> None:
    try:
        if disable_stdout:
            sys.stdout = open(os.devnull, "w")

        if disable_stderr:
            sys.stderr = open(os.devnull, "w")

        script_file = Path(__file__).resolve().parent / (Path(__file__).stem + ".py")

        if not script_file.exists():
            raise Exception("Unable to locate .py file of renderer")

        if not blender_binary.exists():
            raise Exception(f"Blender binary {str(blender_binary)} does not exist")

        if not model_file.exists():
            raise Exception(f"Model file {str(model_file)} does not exist")

        command = (
            f"export DISPLAY=:0.{gpu} && {shlex.quote(str(blender_binary.absolute()))} -b"
            f" -P {shlex.quote(str(script_file.absolute()))}"
            f" -- --model_file {shlex.quote(str(model_file.absolute()))}"
        )

        for render_file in render_files:
            if not render_file.exists():
                raise Exception(f"Render file {str(render_file)} does not exist")

            command += f" --render_file {shlex.quote(str(render_file.absolute()))}"

        if normalization_file is not None:
            command += f" --normalization_file {shlex.quote(str(normalization_file.absolute()))}"

        if save_normalization:
            command += " --save_normalization"

        subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=open(os.devnull, "w") if disable_stdout else None,
            stderr=open(os.devnull, "w") if disable_stderr else None,
        )

        if result is not None:
            result.success = True
            result.close()
    except BaseException as ex:
        msg = traceback.format_exc()
        print(msg, file=sys.stderr, flush=True)
        if result is not None:
            result.success = False
            result.msg = msg
            result.close()
        raise ex


@dataclass
class Render:
    index: int
    focal_length: float
    width: int
    height: int
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    output_file: str


@dataclass
class Normalization:
    scale: Tuple[float, float, float]
    offset: Tuple[float, float, float]


def save_render_json(render: Render, file: Path) -> Path:
    with file.open("w") as f:
        json.dump(asdict(render), f)
    return file


def load_render_json(file: Path) -> Render:
    with file.open("r") as f:
        return json.load(f, object_hook=lambda d: Render(**d))


def save_normalization_json(normalization: Normalization, file: Path) -> Path:
    with file.open("w") as f:
        json.dump(asdict(normalization), f)
    return file


def load_normalization_json(file: Path) -> Normalization:
    with file.open("r") as f:
        return json.load(f, object_hook=lambda d: Normalization(**d))


def normalize_scene_auto() -> Normalization:
    bbox_min, bbox_max = scene_bbox()

    scale = 1.0 / max(bbox_max - bbox_min)

    for obj in scene_root_objects():
        obj.location = obj.location * scale
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()

    bbox_min, bbox_max = scene_bbox()

    offset = -(bbox_min + bbox_max) / 2.0

    for obj in scene_root_objects():
        obj.matrix_world.translation += offset

    bpy.ops.object.select_all(action="DESELECT")

    return Normalization((scale, scale, scale), (offset.x, offset.y, offset.z))


def normalize_scene_manual(normalization: Normalization) -> None:
    scale = mathutils.Vector(normalization.scale)
    offset = mathutils.Vector(normalization.offset)

    for obj in scene_root_objects():
        obj.location = mathutils.Vector(
            (
                obj.location.x * scale.x,
                obj.location.y * scale.y,
                obj.location.z * scale.z,
            )
        )
        obj.scale = mathutils.Vector(
            (
                obj.scale.x * scale.x,
                obj.scale.y * scale.y,
                obj.scale.z * scale.z,
            )
        )

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()

    for obj in scene_root_objects():
        obj.matrix_world.translation += offset

    bpy.ops.object.select_all(action="DESELECT")


def render(
    model_file: Path,
    renders: List[Render],
    normalization: Optional[Normalization] = None,
    save_normalization_file: Optional[Path] = None,
) -> None:
    set_scene_render_parameters()

    reset_scene()

    load_object(str(model_file.absolute()))

    if save_normalization_file:
        normalization = normalize_scene_auto()
        if save_normalization_file is not None:
            save_normalization_json(normalization, save_normalization_file)
            return

    if normalization is not None:
        normalize_scene_manual(normalization)
    else:
        normalization = normalize_scene_auto()

    add_lighting()

    camera_obj: bpy.types.Object = bpy.context.scene.objects["Camera"]
    camera_data: bpy.types.Camera = camera_obj.data

    for render in renders:
        bpy.context.scene.render.resolution_x = render.width
        bpy.context.scene.render.resolution_y = render.height

        bpy.context.scene.render.filepath = render.output_file

        camera_obj.location = render.position

        camera_obj.rotation_mode = "XYZ"
        camera_obj.rotation_euler = render.rotation

        camera_data.sensor_width = 32

        camera_data.lens_unit = "MILLIMETERS"
        camera_data.lens = render.focal_length / render.width * camera_data.sensor_width

        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_file", type=str)
    parser.add_argument("--render_file", required=False, dest="render_files", action="append", default=[])
    parser.add_argument("--save_normalization", required=False, action="store_true", default=False)
    parser.add_argument("--normalization_file", required=False, type=str, default=None)

    args = parser.parse_args(sys.argv[(sys.argv.index("--") + 1) :])

    model_file: str = args.model_file
    render_files: List[str] = args.render_files
    save_normalization: bool = args.save_normalization
    normalization_file: str = args.normalization_file

    if model_file is None:
        raise Exception("No model file specified")

    model_file: Path = Path(model_file)

    if not model_file.exists():
        raise Exception(f"Model file {model_file} doesn't exist")

    renders: List[Render] = []
    for render_file in render_files:
        render_file: Path = Path(render_file)

        if not render_file.exists():
            raise Exception(f"Render file {render_file} doesn't exist")

        renders.append(load_render_json(render_file))

    if not save_normalization and len(render_files) == 0:
        raise Exception("Missing --render_file")
    elif save_normalization and len(render_files) > 0:
        raise Exception("Cannot use both --save_normalization and --render_file")

    normalization_file: Optional[Path] = Path(normalization_file) if normalization_file is not None else None
    normalization: Optional[Normalization] = None

    if save_normalization:
        if normalization_file is None:
            raise Exception("Missing --normalization_file")
    elif normalization_file is not None:
        if not normalization_file.exists():
            raise Exception(f"Normalization file {normalization_file} doesn't exist")

        normalization = load_normalization_json(normalization_file)

    render(
        model_file,
        renders,
        normalization=normalization,
        save_normalization_file=normalization_file if save_normalization else None,
    )
