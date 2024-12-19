#!python

import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['PYGLET_HEADLESS'] = '1'

import trimesh
import pyrender
from pyrender.constants import RenderFlags
import numpy as np
from PIL import Image as im
from scipy.spatial.transform import Rotation as R
import json
import argparse
import re
import io

def render_mesh(renderer, camera, mesh, tm_camera, tm_scene, azimuth, elevation, roll, distance):
    scene = pyrender.Scene(bg_color=[0.5, 0.5, 0.5, 0.5], ambient_light=[1.0, 1.0, 1.0, 1.0])

    # "Skybox" sphere
    sphere_size = scale * 10.0
    sphere_mesh = trimesh.creation.uv_sphere()
    sphere_mesh.vertices *= sphere_size
    sphere_mesh.visual.vertex_colors = 0.5 + sphere_mesh.vertices / sphere_size * 0.5
    scene.add(pyrender.Mesh.from_trimesh(sphere_mesh))

    angle_z = roll
    rot_z = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0, 0],
        [np.sin(angle_z), np.cos(angle_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    angle_y = azimuth
    rot_y = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y), 0],
        [0, 1, 0, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y), 0],
        [0, 0, 0, 1]
    ])

    angle_x = np.pi * 0.5 + elevation
    rot_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x), 0],
        [0, np.sin(angle_x), np.cos(angle_x), 0],
        [0, 0, 0, 1]
    ])

    camera_pose = np.eye(4)
    camera_pose = np.dot(camera_pose, rot_z)
    camera_pose = np.dot(camera_pose, rot_y)
    camera_pose = np.dot(camera_pose, rot_x)

    translation = np.eye(4)
    translation[0, 3] = 0.0
    translation[1, 3] = 0.0
    translation[2, 3] = distance

    camera_pose = np.dot(camera_pose, translation)

    tm_scene.camera_transform = camera_pose

    scene.add(camera, pose=camera_pose)

    mesh_pose = np.eye(4)

    scene.add(mesh, pose=mesh_pose)

    # light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi/16.0, outerConeAngle=np.pi/6.0)
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=20.0)

    scene.add(light, pose=camera_pose)

    #color, depth = renderer.render(scene, flags=RenderFlags.SKIP_CULL_FACES)

    #data = im.fromarray(color)

    tm_scene.camera_transform = camera_pose
    png_bytes = tm_scene.save_image(resolution=[1024, 1024])
    data = im.open(io.BytesIO(png_bytes))

    return camera_pose, data

def render_views(renderer, file, output_dir, scale, az_views, el_views, d_views):
    tm = trimesh.load(file)
    tm_scene = tm
    tmm = list(tm.geometry.values())[0]

    #for i in range(1, len(tm.geometry.values())):
    #    tmm += list(tm.geometry.values())[i]
    #tmm = trimesh.util.concatenate(tm.geometry.values())


    #if len(tm.geometry.values()) != 1:
    #    print("Skipping unsupported geometry for now")
    #    return

    # Center and normalize
    #tmm.vertices -= tmm.center_mass
    tmm.vertices -= tmm.centroid
    tmm.vertices *= 1.0 / np.max(tmm.extents) * scale

    # bb = tmm.bounding_box_oriented
    # bb.visual.vertex_colors = trimesh.visual.random_color() #[1.0, 0.0, 0.0, 0.2]

    # Create pyrender mesh
    mesh = pyrender.Mesh.from_trimesh(tmm)

    # Center and normalize
    tm_scene = tm_scene.scaled(1.0 / np.max(tm_scene.extents) * scale)
    tm_scene.rezero()

    fov = np.pi / 3.0

    camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=1.0)

    tm_camera = trimesh.scene.Camera(fov=[fov, fov])

    focal_length = renderer.viewport_width * 0.5 / np.tan(fov * 0.5)

    # Create a transforms.json file for nerfstudio
    transforms_json = {
        "camera_model": "OPENCV",
        "fl_x": focal_length,
        "fl_y": focal_length,
        "cx": renderer.viewport_width * 0.5,
        "cy": renderer.viewport_height * 0.5,
        "w": renderer.viewport_width,
        "h": renderer.viewport_height,
        #"k1": 0.0,
        #"k2": 0.0,
        #"k3": 0.0,
        #"k4": 0.0,
        #"p1": 0.0,
        #"p2": 0.0,
        #"aabb_scale": 16,
        "frames": []
    }

    frame_nr = 1

    for i in range(0, az_views):
        for j in range(0, el_views):
            for k in range(0, d_views):
                azimuth = np.pi / az_views * i
                elevation = 2.0 * np.pi / el_views * j #-np.pi * 0.5 + np.pi / (el_views - 1) * j
                roll = 0.0
                distance = scale * (2.0 + 1.0 / d_views * k)

                #view_str = str(i) + '.' + str(j)
                view_str = 'frame_' + str(frame_nr).zfill(5)
                frame_nr = frame_nr + 1

                print('Rendering view: ' + view_str)

                transform, data = render_mesh(renderer, camera, mesh, tm_camera, tm_scene, azimuth, elevation, roll, distance)

                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)

                image_dir = os.path.join(output_dir, 'images')
                if not os.path.exists(image_dir):
                    os.mkdir(image_dir)

                image_file = os.path.join(image_dir, view_str + '.png')
                data.save(image_file)

                frame_json = {
                    "file_path": 'images/' + view_str + '.png',
                    "transform_matrix": transform.tolist()
                }
                transforms_json['frames'].append(frame_json)

    transforms_json_file = os.path.join(output_dir, 'transforms.json')
    with open(transforms_json_file, 'w') as jf:
        json.dump(transforms_json, jf, indent=4)

parser = argparse.ArgumentParser(
    prog='render',
    description='Renders views of objaverse glb files')

parser.add_argument('--resolution', type=int, help="Resolution of rendered views", default=1024)
parser.add_argument('--src', type=str, help="Source folder containing the GLBs")
parser.add_argument('--dst', type=str, help="Destination folder to save views in")
parser.add_argument('--views', type=int, help="Number of views around Y axis", default=16)
parser.add_argument('--regex', type=str, help="Regex to filter input files")

args = parser.parse_args()

resolution = args.resolution
renderer = pyrender.OffscreenRenderer(resolution, resolution)

source_dir = args.src
output_dir = args.dst

views = args.views

scale = 1.0

regex = args.regex

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for f in os.listdir(source_dir):
    file = os.path.join(source_dir, f)

    if regex and not re.search(regex, file):
        continue

    if os.path.isfile(file) and file.endswith('.glb'):
        views_output_dir = os.path.join(output_dir, f)

        print('Rendering file: ' + file)

        try:
            render_views(renderer, file, views_output_dir, scale=scale, az_views=int(views/2), el_views=views, d_views=1)
        except Exception as e:
            print('Error rendering file: ' + file)
            print(str(e))


