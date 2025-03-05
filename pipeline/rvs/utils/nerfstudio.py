import json
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image as im

from rvs.pipeline.views import View


def get_frame_name(view: Optional[View], frame_name: Optional[str] = None, frame_index: int = 0) -> str:
    if frame_name is None:
        frame_name = "frame_{}.png"
    return frame_name.format(str((view.index + 1) if view is not None else (frame_index + 1)).zfill(5))


def get_transforms_frame_path(
    dir: Path,
    view: View,
    frame_dir: Path = Path("images/"),
    frame_name: Optional[str] = None,
) -> Path:
    frame_path = dir / frame_dir / get_frame_name(view, frame_name)
    return frame_path


def save_transforms_frame(
    dir: Path,
    view: View,
    image: im.Image,
    frame_dir: Path = Path("images/"),
    frame_name: Optional[str] = None,
    set_path: bool = False,
) -> Path:
    frame_path = get_transforms_frame_path(
        dir,
        view,
        frame_dir=frame_dir,
        frame_name=frame_name,
    )
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(frame_path)
    if set_path:
        view.path = frame_path
    return frame_path


def create_transforms_dict(
    views: List[View],
    focal_length_x: float,
    focal_length_y: float,
    width: int,
    height: int,
    frame_dir: Path = Path("images/"),
    frame_name: Optional[str] = None,
) -> Dict[str, Any]:
    transforms_dict = {
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
        transforms_dict["frames"].append(frame_json)

    return transforms_dict


def create_transforms_json(
    views: List[View],
    focal_length_x: float,
    focal_length_y: float,
    width: int,
    height: int,
    frame_dir: Path = Path("images/"),
    frame_name: Optional[str] = None,
) -> str:
    transforms_json = create_transforms_dict(
        views, focal_length_x, focal_length_y, width, height, frame_dir=frame_dir, frame_name=frame_name
    )
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


def load_transforms_json(
    dir: Path,
    transforms_name: Optional[str] = None,
    set_view_path: bool = False,
) -> Tuple[Path, List[View], float, float, int, int]:
    if transforms_name is None:
        transforms_name = "transforms.json"
    path = dir / transforms_name

    if not path.exists():
        raise FileNotFoundError(f"Transforms file {str(path)} does not exist")

    if not path.is_file():
        raise FileNotFoundError(f"Transforms file path {str(path)} is not a file")

    text: str = None
    with path.open("r") as f:
        text = f.read()

    transforms_json = json.loads(text)

    focal_length_x = float(transforms_json["fl_x"])
    focal_length_y = float(transforms_json["fl_y"])

    width = int(transforms_json["w"])
    height = int(transforms_json["h"])

    views: List[View] = []
    frames_json = transforms_json["frames"]
    for i, frame_json in enumerate(frames_json):
        file_path = Path(frame_json["file_path"])
        transform_matrix = np.array(frame_json["transform_matrix"])
        view = View(i, transform_matrix)
        if set_view_path:
            view.path = file_path if file_path.is_absolute() else (dir / file_path)
        views.append(view)

    return (path, views, focal_length_x, focal_length_y, width, height)


class ThreadedImageSaver:
    output_dir: Path
    callback: Optional[Callable[[View, im.Image, Path], None]]
    executor: Executor

    def __init__(
        self, output_dir: Path, threads: int = 1, callback: Optional[Callable[[View, im.Image, Path], None]] = None
    ) -> None:
        self.output_dir = output_dir
        self.callback = callback
        self.executor = ThreadPoolExecutor(max_workers=threads)

    def save(
        self,
        view: View,
        image: im.Image,
        frame_dir: Path = Path("images/"),
        frame_name: Optional[str] = None,
        set_path: bool = True,
    ) -> None:
        self.executor.submit(self.__save_image, view, image, frame_dir, frame_name, set_path)

    def __save_image(
        self, view: View, image: im.Image, frame_dir: Path, frame_name: Optional[str], set_path: bool
    ) -> None:
        path = save_transforms_frame(
            self.output_dir, view, image, frame_dir=frame_dir, frame_name=frame_name, set_path=set_path
        )
        if self.callback is not None:
            self.callback(view, image, path)
        return path

    def close(self):
        self.executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
