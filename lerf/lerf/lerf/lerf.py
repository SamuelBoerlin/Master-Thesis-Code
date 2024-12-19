from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import open_clip
import torch
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.ray_samplers import PDFSampler
from nerfstudio.model_components.renderers import DepthRenderer
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap
from nerfstudio.viewer.viewer_elements import *
from torch.nn import Parameter

from lerf.encoders.image_encoder import BaseImageEncoder
from lerf.lerf_field import LERFField
from lerf.lerf_fieldheadnames import LERFFieldHeadNames
from lerf.lerf_renderers import CLIPRenderer, MeanRenderer


import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import trimesh
import pyrender
from pyrender.constants import RenderFlags
from PIL import Image as im
import nerfstudio.utils.poses as pose_utils
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.utils.colormaps import apply_pca_colormap
import traceback
from scipy.cluster.vq import kmeans, whiten
from pprint import pprint
from nerfstudio.viewer.viewer_elements import ViewerCheckbox

def create_debug_scene(tm_mesh, clip_sample_positions, clip_sample_colors):
    print("create scene")

    scene = pyrender.Scene(bg_color=[0.5, 0.5, 0.5, 0.5], ambient_light=[1.0, 1.0, 1.0, 1.0])

    # light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi/16.0, outerConeAngle=np.pi/6.0)
    #light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=20.0)
    #scene.add(light, pose=camera_pose)

    # bb = tmm.bounding_box_oriented
    # bb.visual.vertex_colors = trimesh.visual.random_color() #[1.0, 0.0, 0.0, 0.2]
    # Create pyrender mesh

    print("create skybox")

    # "Skybox" sphere
    sphere_size = 1.0 * 10.0
    sphere_mesh = trimesh.creation.uv_sphere()
    sphere_mesh.vertices *= sphere_size
    sphere_mesh.visual.vertex_colors = 0.5 + sphere_mesh.vertices / sphere_size * 0.5
    scene.add(pyrender.Mesh.from_trimesh(sphere_mesh))

    print("add sample spheres")

    for i in range(clip_sample_positions.shape[0]):
        position = clip_sample_positions[i]
        color = clip_sample_colors[i]

        sample_pose = np.eye(4)
        sample_pose[0, 3] = position[0]
        sample_pose[1, 3] = position[1]
        sample_pose[2, 3] = position[2]

        sphere_size = 0.025
        sphere_mesh = trimesh.creation.uv_sphere(count=[3,3])
        sphere_mesh.vertices *= sphere_size
        sphere_mesh.visual.vertex_colors = sphere_mesh.vertices * 0.0 + color

        scene.add(pyrender.Mesh.from_trimesh(sphere_mesh), pose=sample_pose)

    print("convert mesh")

    mesh = pyrender.Mesh.from_trimesh(tm_mesh)
    mesh_pose = np.eye(4)

    print("add mesh")

    #scene.add(mesh, pose=mesh_pose)

    return scene


def render_debug_scene(pyrenderer, camera, camera_pose, scene):
    print("add camera")

    camera_node = scene.add(camera, pose=camera_pose)

    print("render scene")

    color, depth = pyrenderer.render(scene, flags=RenderFlags.SKIP_CULL_FACES)

    print("remove camera")

    scene.remove_node(camera_node)

    return color, depth

def sample(lerf_model, tm_mesh, tm_scene):
    #samples, _ = trimesh.sample.sample_surface_even(tm_mesh, count=128, radius=0.0001, seed=42)
    #samples, _ = trimesh.sample.sample_surface(tm_mesh, count=128, seed=42)
    #samples, _ = trimesh.sample.sample_surface_even(tm_mesh, radius=0.025, count=10000)
    from lerf.sampler import sample_positions
    samples, _ = sample_positions(tm_scene, radius=0.025, count=10000)

    sample_batch = torch.from_numpy(samples).to(lerf_model.device).reshape((1, -1, 3))

    clip_scales = torch.ones((1, sample_batch.shape[1], 1), device=lerf_model.device)
    clip_pass = lerf_model.lerf_field.get_clip_at_positions(sample_batch, clip_scales)

    return samples, clip_pass

def load_mesh(file):
    tm = trimesh.load(file)
    tmm = list(tm.geometry.values())[0]
    # Center and normalize
    # tmm.vertices -= tmm.center_mass
    tmm.vertices -= tmm.centroid
    tmm.vertices *= 1.0 / np.max(tmm.extents) * 1.0
    return tmm

def find_viewpoints(lerf_model, clip_sample_embeddings_kmeans_center):
    #images = [lerf_model.datamanager.train_dataset[i]["image"].permute(2, 0, 1)[None, ...] for i in range(len(lerf_model.datamanager.train_dataset))]
    #images = torch.cat(images)

    #for img in images:
    #    img2 = img.unsqueeze(0)

    if lerf_model.image_embeddings is None:
        lerf_model.image_embeddings = []

        for i in range(len(lerf_model.datamanager.train_dataset)):
            img = lerf_model.datamanager.train_dataset[i]["image"]

            # TODO Hardcoded dimensions
            img = img.reshape(-1, 3, 1024, 1024).to("cuda")

            embedding = None
            with torch.no_grad():
                embedding = lerf_model.image_encoder.encode_image(img).detach().cpu().numpy().reshape((-1,))

            embedding = embedding / np.linalg.norm(embedding)

            lerf_model.image_embeddings.append(embedding)

    best_sim = [-1, -1, -1]
    best_idx = [0, 0, 0]

    for i in range(len(lerf_model.image_embeddings)):
        embedding = lerf_model.image_embeddings[i]
        img_idx = lerf_model.datamanager.train_dataset[i]["image_idx"]

        for j in range(3):
            sim = np.dot(clip_sample_embeddings_kmeans_center[j], embedding)

            if sim > best_sim[j]:
                best_sim[j] = sim
                best_idx[j] = img_idx


    for j in range(3):
        print("Best similarity image")
        print("Cluster: " + str(j))
        print("Similarity: " + str(best_sim[j]))
        print("Image index: " + str(best_idx[j]))


def render_debug_view(lerf_model, ns_camera: Cameras):
    try:
        width = torch.max(ns_camera.width.view(-1)).item()
        height = torch.max(ns_camera.height.view(-1)).item()

        c2w = pose_utils.to4x4(ns_camera.camera_to_worlds[0])
        #c2w[2, :] *= -1
        #c2w = c2w[np.array([0, 2, 1, 3]), :]
        camera_pose = c2w.cpu().data.numpy()

        print("load mesh")

        tm_mesh = lerf_model.debug_mesh
        if tm_mesh is None:
            tm_mesh = load_mesh("/nas/objaverse-data/hf-objaverse-v1/glbs/000-023/5f65399743bb4db1bd9b08e89b8efd41.glb")
            #tm_mesh = load_mesh("/nas/objaverse-data/hf-objaverse-v1/glbs/000-023/astronomer_tower.glb")
            lerf_model.debug_mesh = tm_mesh

        clip_sample_positions = lerf_model.clip_sample_positions
        clip_colors = lerf_model.clip_colors

        if clip_sample_positions is None or clip_colors is None:
            print("sample embeddings")

            #tm_scene = trimesh.load("/nas/objaverse-data/hf-objaverse-v1/glbs/000-023/5f65399743bb4db1bd9b08e89b8efd41.glb")
            tm_scene = trimesh.load("/nas/objaverse-data/hf-objaverse-v1/glbs/000-023/astronomer_tower.glb")
            tm_scene = tm_scene.scaled(1.0 / np.max(tm_scene.extents) * 1.0)
            tm_scene.rezero()
            angle_z = 0.0
            rot_z = np.array([
                [np.cos(angle_z), -np.sin(angle_z), 0, 0],
                [np.sin(angle_z), np.cos(angle_z), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            angle_y = 0.0
            rot_y = np.array([
                [np.cos(angle_y), 0, np.sin(angle_y), 0],
                [0, 1, 0, 0],
                [-np.sin(angle_y), 0, np.cos(angle_y), 0],
                [0, 0, 0, 1]
            ])
            angle_x = np.pi * 0.5
            rot_x = np.array([
                [1, 0, 0, 0],
                [0, np.cos(angle_x), -np.sin(angle_x), 0],
                [0, np.sin(angle_x), np.cos(angle_x), 0],
                [0, 0, 0, 1]
            ])
            tf = np.eye(4)
            tf = np.dot(tf, rot_z)
            tf = np.dot(tf, rot_y)
            tf = np.dot(tf, rot_x)
            tm_scene.apply_transform(tf)

            clip_sample_positions, clip_sample_embeddings = sample(lerf_model, tm_mesh, tm_scene)

            clip_sample_embeddings = clip_sample_embeddings.view(-1, clip_sample_embeddings.shape[-1]).float()

            #clip_sample_embeddings_pca = apply_pca_colormap(clip_sample_embeddings)

            clip_sample_embeddings_centered = whiten(clip_sample_embeddings.cpu().data.numpy())
            clip_sample_embeddings_kmeans_center, clip_sample_embeddings_kmeans_distortion = kmeans(clip_sample_embeddings_centered, 3)
            clip_sample_embeddings_kmeans_colors = np.zeros((clip_sample_embeddings.shape[0], 3))
            for i in range(3):
                clip_sample_embeddings_kmeans_center[i] = clip_sample_embeddings_kmeans_center[i] / np.linalg.norm(clip_sample_embeddings_kmeans_center[i])
            
            print("calculate cluster assignments")

            r_embedding = clip_sample_embeddings_kmeans_center[0]
            g_embedding = clip_sample_embeddings_kmeans_center[1]
            b_embedding = clip_sample_embeddings_kmeans_center[2]

            
            for i in range(clip_sample_embeddings.shape[0]):
                clip_sample_embedding = clip_sample_embeddings_centered[i] / np.linalg.norm(clip_sample_embeddings_centered[i])

                if not lerf_model.hard_assignments:
                    clip_sample_embeddings_kmeans_colors[i, 0] = np.dot(r_embedding, clip_sample_embedding)
                    clip_sample_embeddings_kmeans_colors[i, 1] = np.dot(g_embedding, clip_sample_embedding)
                    clip_sample_embeddings_kmeans_colors[i, 2] = np.dot(b_embedding, clip_sample_embedding)
                else:
                    best = -1.0

                    r_sim = np.dot(r_embedding, clip_sample_embedding)
                    g_sim = np.dot(g_embedding, clip_sample_embedding)
                    b_sim = np.dot(b_embedding, clip_sample_embedding)

                    if r_sim > best:
                        best = r_sim
                        clip_sample_embeddings_kmeans_colors[i, 0] = 1.0
                        clip_sample_embeddings_kmeans_colors[i, 1] = 0.0
                        clip_sample_embeddings_kmeans_colors[i, 2] = 0.0

                    if g_sim > best:
                        best = g_sim
                        clip_sample_embeddings_kmeans_colors[i, 0] = 0.0
                        clip_sample_embeddings_kmeans_colors[i, 1] = 1.0
                        clip_sample_embeddings_kmeans_colors[i, 2] = 0.0

                    if b_sim > best:
                        best = b_sim
                        clip_sample_embeddings_kmeans_colors[i, 0] = 0.0
                        clip_sample_embeddings_kmeans_colors[i, 1] = 0.0
                        clip_sample_embeddings_kmeans_colors[i, 2] = 1.0
                    

            
            #clip_colors = clip_sample_embeddings_pca.cpu().data.numpy()
            clip_colors = clip_sample_embeddings_kmeans_colors

            lerf_model.clip_sample_positions = clip_sample_positions
            lerf_model.clip_colors = clip_colors

            print("find best viewpoints")

            find_viewpoints(lerf_model, clip_sample_embeddings_kmeans_center)

            # Invalidate scene due to changed clustering
            lerf_model.debug_scene = None


        print("setup renderer")

        pyrenderer = pyrender.OffscreenRenderer(width, height)

        print("setup camera")

        camera = pyrender.IntrinsicsCamera(fx=ns_camera.fx[0], fy=ns_camera.fy[0], cx=ns_camera.cx[0], cy=ns_camera.cy[0])

        print("setup scene")

        scene = lerf_model.debug_scene
        if scene is None:
            scene = create_debug_scene(tm_mesh, clip_sample_positions, clip_colors)
            lerf_model.debug_scene = scene

        print("render scene")

        color, depth = render_debug_scene(pyrenderer, camera, camera_pose, scene)

        print("copy to device")

        result = torch.from_numpy(color.copy()).to(ns_camera.device)
        result = result / 255.0
        # TODO: Seems like "rgb" output is flattened (e.g. [64, 3] where 64=8x8 pixels), but if I do the same it errors later
        #result = torch.flatten(result, start_dim=0, end_dim=-2)

        print("delete renderer")

        pyrenderer.delete()

        print("done")

        return result
    except Exception:
        print("==== ERROR ====")
        print(traceback.format_exc())
        print("===============")

    return None


@dataclass
class LERFModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: LERFModel)
    clip_loss_weight: float = 0.1
    n_scales: int = 30
    max_scale: float = 1.5
    """maximum scale used to compute relevancy with"""
    num_lerf_samples: int = 24
    hashgrid_layers: Tuple[int] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[int]] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[int] = (19, 19)


class LERFModel(NerfactoModel):
    config: LERFModelConfig

    clip_sample_positions = None
    clip_colors = None

    def populate_modules(self):
        super().populate_modules()

        self.renderer_clip = CLIPRenderer()
        self.renderer_mean = MeanRenderer()

        self.image_encoder: BaseImageEncoder = self.kwargs["image_encoder"]
        self.lerf_field = LERFField(
            self.config.hashgrid_layers,
            self.config.hashgrid_sizes,
            self.config.hashgrid_resolutions,
            clip_n_dims=self.image_encoder.embedding_dim,
        )

        self.datamanager = self.kwargs["datamanager"]
        self.image_embeddings = None

        self.enable_debug_renderer_checkbox = ViewerCheckbox("Enable debug renderer", False, cb_hook=self.gui_debug_renderer_checkbox_cb)
        self.enable_debug_renderer = False

        self.disable_clip_renderer_checkbox = ViewerCheckbox("Disable clip renderer", False, cb_hook=self.gui_clip_renderer_checkbox_cb)
        self.disable_clip_renderer = False

        self.hard_assignments_checkbox = ViewerCheckbox("Hard cluster assignments", False, cb_hook=self.gui_hard_assignments_checkbox_cb)
        self.hard_assignments = False

        self.debug_mesh = None
        self.debug_scene = None

    def gui_debug_renderer_checkbox_cb(self, element):
        self.enable_debug_renderer = element.value

    def gui_clip_renderer_checkbox_cb(self, element):
        self.disable_clip_renderer = element.value
    
    def gui_hard_assignments_checkbox_cb(self, element):
        self.hard_assignments = element.value

    def get_max_across(self, ray_samples, weights, hashgrid_field, scales_shape, preset_scales=None):
        # TODO smoothen this out
        if preset_scales is not None:
            assert len(preset_scales) == len(self.image_encoder.positives)
            scales_list = torch.tensor(preset_scales)
        else:
            scales_list = torch.linspace(0.0, self.config.max_scale, self.config.n_scales)

        # probably not a good idea bc it's prob going to be a lot of memory
        n_phrases = len(self.image_encoder.positives)
        n_phrases_maxs = [None for _ in range(n_phrases)]
        n_phrases_sims = [None for _ in range(n_phrases)]
        for i, scale in enumerate(scales_list):
            scale = scale.item()
            with torch.no_grad():
                clip_output = self.lerf_field.get_output_from_hashgrid(
                    ray_samples,
                    hashgrid_field,
                    torch.full(scales_shape, scale, device=weights.device, dtype=hashgrid_field.dtype),
                )
            clip_output = self.renderer_clip(embeds=clip_output, weights=weights.detach())

            for j in range(n_phrases):
                if preset_scales is None or j == i:
                    probs = self.image_encoder.get_relevancy(clip_output, j)
                    pos_prob = probs[..., 0:1]
                    if n_phrases_maxs[j] is None or pos_prob.max() > n_phrases_sims[j].max():
                        n_phrases_maxs[j] = scale
                        n_phrases_sims[j] = pos_prob
        return torch.stack(n_phrases_sims), torch.Tensor(n_phrases_maxs)

    def get_outputs(self, ray_bundle: RayBundle):
        if self.training:
            self.clip_sample_positions = None
            self.clip_colors = None

        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        ray_samples_list.append(ray_samples)

        nerfacto_field_outputs, outputs, weights = self._get_outputs_nerfacto(ray_samples)

        if not self.disable_clip_renderer:
            lerf_weights, best_ids = torch.topk(weights, self.config.num_lerf_samples, dim=-2, sorted=False)

            def gather_fn(tens):
                return torch.gather(tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1]))

            dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
            lerf_samples: RaySamples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)

            if self.training:
                with torch.no_grad():
                    clip_scales = ray_bundle.metadata["clip_scales"]
                    clip_scales = clip_scales[..., None]
                    dist = (lerf_samples.frustums.get_positions() - ray_bundle.origins[:, None, :]).norm(
                        dim=-1, keepdim=True
                    )
                clip_scales = clip_scales * ray_bundle.metadata["height"] * (dist / ray_bundle.metadata["fy"])
            else:
                clip_scales = torch.ones_like(lerf_samples.spacing_starts, device=self.device)

            override_scales = (
                None if "override_scales" not in ray_bundle.metadata else ray_bundle.metadata["override_scales"]
            )
            weights_list.append(weights)
            if self.training:
                outputs["weights_list"] = weights_list
                outputs["ray_samples_list"] = ray_samples_list
            for i in range(self.config.num_proposal_iterations):
                outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

            lerf_field_outputs = self.lerf_field.get_outputs(lerf_samples, clip_scales)
            outputs["clip"] = self.renderer_clip(
                embeds=lerf_field_outputs[LERFFieldHeadNames.CLIP], weights=lerf_weights.detach()
            )
            outputs["dino"] = self.renderer_mean(
                embeds=lerf_field_outputs[LERFFieldHeadNames.DINO], weights=lerf_weights.detach()
            )

            if not self.training:
                with torch.no_grad():
                    max_across, best_scales = self.get_max_across(
                        lerf_samples,
                        lerf_weights,
                        lerf_field_outputs[LERFFieldHeadNames.HASHGRID],
                        clip_scales.shape,
                        preset_scales=override_scales,
                    )
                    outputs["raw_relevancy"] = max_across  # N x B x 1
                    outputs["best_scales"] = best_scales.to(self.device)  # N

        else:
            weights_list.append(weights)
            if self.training:
                outputs["weights_list"] = weights_list
                outputs["ray_samples_list"] = ray_samples_list

        return outputs

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Assumes a ray-based model.

        Args:
            camera: generates raybundle
        """
        camera_ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box)

        outputs = {}
        
        if not self.disable_clip_renderer:
            outputs = self.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        else:
            outputs = super().get_outputs_for_camera_ray_bundle(camera_ray_bundle)

        if self.enable_debug_renderer:
            debug_view_output = render_debug_view(self, camera)
            if debug_view_output is not None:
                outputs['debug'] = debug_view_output
        
        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        LERF overrides this from base_model since we need to compute the max_across relevancy in multiple batches,
        which are not independent since they need to use the same scale
        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        # TODO(justin) implement max across behavior
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)  # dict from name:list of outputs (1 per bundle)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle)
            # take the best scale for each query across each ray bundle
            if i == 0:
                best_scales = outputs["best_scales"]
                best_relevancies = [m.max() for m in outputs["raw_relevancy"]]
            else:
                for phrase_i in range(outputs["best_scales"].shape[0]):
                    m = outputs["raw_relevancy"][phrase_i, ...].max()
                    if m > best_relevancies[phrase_i]:
                        best_scales[phrase_i] = outputs["best_scales"][phrase_i]
                        best_relevancies[phrase_i] = m
        # re-render the max_across outputs using the best scales across all batches
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            ray_bundle.metadata["override_scales"] = best_scales
            outputs = self.forward(ray_bundle=ray_bundle)
            # standard nerfstudio concatting
            for output_name, output in outputs.items():  # type: ignore
                if output_name == "best_scales":
                    continue
                if output_name == "raw_relevancy":
                    for r_id in range(output.shape[0]):
                        outputs_lists[f"relevancy_{r_id}"].append(output[r_id, ...])
                else:
                    outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        for i in range(len(self.image_encoder.positives)):
            p_i = torch.clip(outputs[f"relevancy_{i}"] - 0.5, 0, 1)
            outputs[f"composited_{i}"] = apply_colormap(p_i / (p_i.max() + 1e-6), ColormapOptions("turbo"))
            mask = (outputs["relevancy_0"] < 0.5).squeeze()
            outputs[f"composited_{i}"][mask, :] = outputs["rgb"][mask, :]
        return outputs

    def _get_outputs_nerfacto(self, ray_samples: RaySamples):
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        return field_outputs, outputs, weights

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training and not self.disable_clip_renderer:
            unreduced_clip = self.config.clip_loss_weight * torch.nn.functional.huber_loss(
                outputs["clip"], batch["clip"], delta=1.25, reduction="none"
            )
            loss_dict["clip_loss"] = unreduced_clip.sum(dim=-1).nanmean()
            unreduced_dino = torch.nn.functional.mse_loss(outputs["dino"], batch["dino"], reduction="none")
            loss_dict["dino_loss"] = unreduced_dino.sum(dim=-1).nanmean()
        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["lerf"] = list(self.lerf_field.parameters())
        return param_groups
