from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Type

import numpy as np
import torch
from nerfstudio.configs.base_config import InstantiateConfig
from numpy.typing import NDArray
from open_clip import CLIP, create_model_and_transforms, get_tokenizer
from PIL import Image as im
from torch import Tensor
from torchvision.transforms import Compose, Normalize, Resize


@dataclass
class EmbedderConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Embedder)

    open_clip_model: str = "ViT-L-14"
    open_clip_model_pretrained: str = "laion2b_s32b_b82k"

    input_size: int = 224

    device: str = "cuda"

    background_color: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


class Embedder:
    config: EmbedderConfig

    image_preprocessing: Compose
    text_preprocessing: Callable[[str], Tensor]

    clip_model: CLIP

    def __init__(self, config: EmbedderConfig) -> None:
        self.config = config

        self.image_preprocessing = Compose(
            [
                Resize((self.config.input_size, self.config.input_size)),
                Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        self.text_preprocessing = get_tokenizer(self.config.open_clip_model)

        self.clip_model, _, _ = create_model_and_transforms(
            self.config.open_clip_model,
            pretrained=self.config.open_clip_model_pretrained,
        )
        self.clip_model.eval()
        self.clip_model.to(self.config.device)

    def embed_text(self, text: str) -> Tensor:
        """Embeds the given text into a pytorch float32 tensor of shape B=1 x N"""
        embedding = self.clip_model.encode_text(self.text_preprocessing(text).to(self.config.device))
        embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding

    def embed_text_numpy(self, text: str) -> NDArray:
        """Embeds the given text into a numpy float32 array of shape N"""
        with torch.no_grad():
            return self.embed_text(text).detach().squeeze().cpu().numpy()

    def embed_image(self, file: Path) -> Tensor:
        """Embeds the given image into a pytorch float32 tensor of shape B=1 x N"""
        image = self.__load_image_tensor(file)
        image = self.image_preprocessing(image)
        embedding = self.clip_model.encode_image(image)
        embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding

    def embed_image_numpy(self, file: Path) -> NDArray:
        """Embeds the given image into a numpy float32 array of shape N"""
        with torch.no_grad():
            return self.embed_image(file).detach().squeeze().cpu().numpy()

    def __load_image_numpy(self, file: Path) -> NDArray:
        """Loads the image into a numpy uint8 array of shape H x W x C=3"""
        with im.open(file) as image:
            data = np.array(image, dtype="float32") / 255.0
            assert len(data.shape) == 3
            if data.shape[2] == 4:
                rgb = data[:, :, :3]
                alpha = data[:, :, 3:]
                data = rgb * alpha + self.config.background_color * (1.0 - alpha)
            assert data.shape[2] == 3
            return (data * 255.0).clip(0, 255).astype("uint8")

    def __load_image_tensor(self, file: Path) -> Tensor:
        """Loads the image into a pytorch float32 tensor of shape B=1 x C=3 x H x W"""
        image = (
            torch.from_numpy(self.__load_image_numpy(file).astype("float32") / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.config.device)
        )
        return image
