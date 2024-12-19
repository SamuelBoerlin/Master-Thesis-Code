import numpy as np
from trimesh.scene import Scene


def normalize_scene(scene: Scene) -> Scene:
    scene = scene.scaled(1.0 / np.max(scene.extents))
    scene.rezero()
    return scene
