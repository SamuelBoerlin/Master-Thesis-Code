import json
from pathlib import Path
from typing import Dict, List, Tuple

from rvs.utils.hash import hash_file_sha1


def __create_file_entry(file: Path) -> Dict:
    return {
        "file": file.name,
        "sha1": hash_file_sha1(file),
    }


def save_index(file: Path, images: List[Path], transforms: List[Path]) -> None:
    assert len(images) == len(transforms)

    for image in images:
        if image.parent != file.parent:
            raise Exception("Image not in same directory as index")

    for transform in transforms:
        if transform.parent != file.parent:
            raise Exception("Transform not in same directory as index")

    selected_views = [
        {
            "image": __create_file_entry(images[i]),
            "transform": __create_file_entry(transforms[i]),
        }
        for i in range(len(images))
    ]

    index_json = {"selected_views": selected_views}

    with file.open(mode="w") as f:
        json.dump(index_json, f)


def __load_file_entry(dir: Path, entry: Dict, validate: bool) -> Tuple[Path, Path]:
    file = dir / entry["file"]
    expected_sha1 = entry["sha1"]
    if validate:
        file_sha1 = hash_file_sha1(file)
        if file_sha1 != expected_sha1:
            raise Exception(f"SHA1 of {file} ({file_sha1}) does not match with index ({expected_sha1})")
    return (file, expected_sha1)


def load_index(file: Path, validate: bool = True) -> Tuple[List[Path], List[Path]]:
    index_json = None

    with file.open(mode="r") as f:
        index_json = json.load(f)

    selected_views: List = index_json["selected_views"]

    images: List[Path] = []
    transforms: List[Path] = []

    for entry in selected_views:
        image, _ = __load_file_entry(file.parent, entry["image"], validate)
        images.append(image)

        transform, _ = __load_file_entry(file.parent, entry["transform"], validate)
        transforms.append(transform)

    if len(images) != len(transforms):
        raise Exception(f"Number of images ({len(images)}) and transforms ({len(transforms)}) does not match")

    return (images, transforms)
