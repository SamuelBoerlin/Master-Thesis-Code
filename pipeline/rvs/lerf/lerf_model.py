from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

import torch
from lerf.lerf import LERFModel, LERFModelConfig

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import OrientedBox
from rvs.pipeline.training_controller import RuntimeModelParameters


@dataclass
class CustomLERFModelConfig(LERFModelConfig):
    _target: Type = field(default_factory=lambda: CustomLERFModel)


class CustomLERFModel(LERFModel, RuntimeModelParameters):
    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, Any]:
        if self.enable_embeddings:
            return super().get_outputs(ray_bundle)
        else:
            if self.training:
                self.camera_optimizer.apply_to_raybundle(ray_bundle)

            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
                ray_bundle, density_fns=self.density_fns
            )
            ray_samples_list.append(ray_samples)

            _, outputs, weights = self._get_outputs_nerfacto(ray_samples)

            weights_list.append(weights)
            if self.training:
                outputs["weights_list"] = weights_list
                outputs["ray_samples_list"] = ray_samples_list

            return outputs

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        camera_ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box)

        outputs = {}

        if self.enable_embeddings:
            outputs = super().get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        else:
            outputs = super(LERFModel, self).get_outputs_for_camera_ray_bundle(camera_ray_bundle)

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super(LERFModel, self).get_loss_dict(outputs, batch, metrics_dict)
        if self.training and self.enable_embeddings:
            unreduced_clip = self.config.clip_loss_weight * torch.nn.functional.huber_loss(
                outputs["clip"], batch["clip"], delta=1.25, reduction="none"
            )
            loss_dict["clip_loss"] = unreduced_clip.sum(dim=-1).nanmean()
            unreduced_dino = torch.nn.functional.mse_loss(outputs["dino"], batch["dino"], reduction="none")
            loss_dict["dino_loss"] = unreduced_dino.sum(dim=-1).nanmean()
        return loss_dict
