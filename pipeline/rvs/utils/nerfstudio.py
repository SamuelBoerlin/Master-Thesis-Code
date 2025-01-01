import json
from pathlib import Path
from typing import List, Optional

from PIL import Image as im

from rvs.pipeline.views import View


def get_frame_name(view: Optional[View], frame_name: Optional[str] = None, frame_index: int = 0) -> str:
    if frame_name is None:
        frame_name = "frame_{}.png"
    return frame_name.format(str((view.index + 1) if view is not None else (frame_index + 1)).zfill(5))


def save_transforms_frame(
    dir: Path,
    view: View,
    image: im.Image,
    frame_dir: Path = Path("images/"),
    frame_name: Optional[str] = None,
    set_path: bool = False,
) -> Path:
    frame_dir = dir / frame_dir
    frame_dir.mkdir(exist_ok=True)
    frame_path = frame_dir / get_frame_name(view, frame_name)
    image.save(frame_path)
    if set_path:
        view.path = frame_path
    return frame_path


def create_transforms_json(
    views: List[View],
    focal_length_x: float,
    focal_length_y: float,
    width: int,
    height: int,
    frame_dir: Path = Path("images/"),
    frame_name: Optional[str] = None,
) -> str:
    transforms_json = {
        "camera_model": "OPENCV",
        "fl_x": focal_length_x,
        "fl_y": focal_length_y,
        "cx": width * 0.5,
        "cy": height * 0.5,
        "w": width,
        "h": height,
        # Lens distortion parameters
        # "k1": 0.0,
        # "k2": 0.0,
        # "k3": 0.0,
        # "k4": 0.0,
        # "p1": 0.0,
        # "p2": 0.0,
        "frames": [],
    }

    for view in views:
        frame_json = {
            "file_path": str(frame_dir / get_frame_name(view, frame_name)),
            "transform_matrix": view.transform.tolist(),
        }
        transforms_json["frames"].append(frame_json)

    return json.dumps(transforms_json, indent=4)


def save_transforms_json(
    dir: Path,
    views: List[View],
    focal_length_x: float,
    focal_length_y: float,
    width: int,
    height: int,
    frame_dir: Path = Path("images/"),
    frame_name: Optional[str] = None,
    transforms_name: Optional[str] = None,
) -> Path:
    if transforms_name is None:
        transforms_name = "transforms.json"
    dir.mkdir(exist_ok=True)
    text = create_transforms_json(views, focal_length_x, focal_length_y, width, height, frame_dir, frame_name)
    path = dir / transforms_name
    with path.open("w") as f:
        f.write(text)
    return path
