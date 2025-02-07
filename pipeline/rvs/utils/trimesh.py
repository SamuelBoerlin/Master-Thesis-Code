from typing import Tuple

import numpy as np
from trimesh.scene import Scene

from rvs.pipeline.state import Normalization


def normalize_scene_auto(scene: Scene) -> Tuple[Scene, Normalization]:
    scale = 1.0 / np.max(scene.extents)

    scene = scene.scaled(scale)

    prev_centroid = scene.centroid

    scene.rezero()

    offset = scene.centroid - prev_centroid

    return (scene, Normalization(scale=scale, offset=offset))


def normalize_scene_manual(scene: Scene, normalization: Normalization) -> Scene:
    scene = scene.scaled(normalization.scale)

    transform = np.eye(4)
    transform[:3, 3] = normalization.offset

    new_base = str(scene.graph.base_frame) + "_N"
    scene.graph.update(frame_from=new_base, frame_to=scene.graph.base_frame, matrix=transform)
    scene.graph.base_frame = new_base

    return scene
