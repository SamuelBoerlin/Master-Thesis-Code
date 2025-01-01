import os.path as osp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type, Union

import torch
from lerf.data.lerf_datamanager import LERFDataManager, LERFDataManagerConfig
from lerf.data.utils.dino_dataloader import DinoDataloader
from lerf.data.utils.pyramid_embedding_dataloader import PyramidEmbeddingDataloader
from lerf.encoders.image_encoder import BaseImageEncoder


@dataclass
class CustomLERFDataManagerConfig(LERFDataManagerConfig):
    _target: Type = field(default_factory=lambda: CustomLERFDataManager)

    cache_dir: Path = Path("outputs/")
    """Directory where data like e.g. CLIP weights can be cached"""


class CustomLERFDataManager(LERFDataManager):
    config: CustomLERFDataManagerConfig

    def __init__(
        self,
        config: CustomLERFDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        # super(VanillaDataManager, self).__init__(config, device, test_mode, world_size, local_rank, **kwargs)
        super(LERFDataManager, self).__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

        self.image_encoder: BaseImageEncoder = kwargs["image_encoder"]
        # TODO: Added [:, :, :3] to "support" images with alpha channel by simply stripping away alpha channel
        images = [
            self.train_dataset[i]["image"][:, :, :3].permute(2, 0, 1)[None, ...] for i in range(len(self.train_dataset))
        ]
        images = torch.cat(images)

        cache_dir = str(self.config.cache_dir)
        clip_cache_path = Path(osp.join(cache_dir, f"clip_{self.image_encoder.name}"))
        dino_cache_path = Path(osp.join(cache_dir, "dino.npy"))
        # NOTE: cache config is sensitive to list vs. tuple, because it checks for dict equality
        self.dino_dataloader = DinoDataloader(
            image_list=images,
            device=self.device,
            cfg={"image_shape": list(images.shape[2:4])},
            cache_path=dino_cache_path,
        )
        torch.cuda.empty_cache()
        self.clip_interpolator = PyramidEmbeddingDataloader(
            image_list=images,
            device=self.device,
            cfg={
                "tile_size_range": [0.05, 0.5],
                "tile_size_res": 7,
                "stride_scaler": 0.5,
                "image_shape": list(images.shape[2:4]),
                "model_name": self.image_encoder.name,
            },
            cache_path=clip_cache_path,
            model=self.image_encoder,
        )
