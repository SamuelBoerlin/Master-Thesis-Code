import argparse
import json
import shlex
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import bpy

try:
    from rvs.utils.blender_renderer_objaverse import (
        add_lighting,
        load_object,
        normalize_scene,
        reset_scene,
        set_scene_render_parameters,
    )
    from rvs.utils.process import ProcessResult
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from blender_renderer_objaverse import (
        add_lighting,
        load_object,
        normalize_scene,
        reset_scene,
        set_scene_render_parameters,
    )
    from process import ProcessResult


def renderer_worker_func(
    blender_binary: Path,
    model_file: Path,
    render_files: List[Path],
    gpu: int = 0,
    result: Optional[ProcessResult] = None,
) -> None:
    try:
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

        subprocess.run(command, shell=True, check=True)

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


def save_render_json(render: Render, file: Path) -> Path:
    with file.open("w") as f:
        json.dump(asdict(render), f)
    return file


def load_render_json(file: Path) -> Render:
    with file.open("r") as f:
        return json.load(f, object_hook=lambda d: Render(**d))


def render(model_file: Path, renders: List[Render]) -> None:
    set_scene_render_parameters()

    reset_scene()

    load_object(str(model_file.absolute()))

    normalize_scene()

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
    parser.add_argument("--render_file", dest="render_files", action="append", default=[])

    args = parser.parse_args(sys.argv[(sys.argv.index("--") + 1) :])

    model_file: str = args.model_file
    render_files: List[str] = args.render_files

    if model_file is None:
        raise Exception("No model file specified")

    model_file: Path = Path(model_file)

    if not model_file.exists():
        raise Exception(f"Model file {model_file} doesn't exist")

    if len(render_files) == 0:
        raise Exception("No render files specified")

    renders: List[Render] = []
    for render_file in render_files:
        render_file: Path = Path(render_file)

        if not render_file.exists():
            raise Exception(f"Render file {render_file} doesn't exist")

        renders.append(load_render_json(render_file))

    render(model_file, renders)
