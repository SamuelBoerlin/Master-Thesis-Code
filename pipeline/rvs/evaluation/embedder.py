import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Type, Union

import numpy as np
import torch
from nerfstudio.configs.base_config import InstantiateConfig
from numpy.typing import NDArray
from open_clip import CLIP, create_model_and_transforms, get_tokenizer
from PIL import Image as im
from torch import Tensor
from torchvision.transforms import Compose, Normalize, Resize

from rvs.utils.hash import hash_file_sha1, hash_text_sha1


@dataclass
class EmbedderConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Embedder)

    open_clip_model: str = "ViT-L-14"
    open_clip_model_pretrained: str = "laion2b_s32b_b82k"

    input_size: int = 224

    device: str = "cuda"

    background_color: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


class Embedder:
    __config: EmbedderConfig

    @property
    def config(self) -> EmbedderConfig:
        return self.__config

    __image_preprocessing: Compose
    __text_preprocessing: Callable[[str], Tensor]

    __clip_model: CLIP

    def __init__(self, config: EmbedderConfig) -> None:
        self.__config = config

        self.__image_preprocessing = Compose(
            [
                Resize((self.__config.input_size, self.__config.input_size), antialias=False),  # TODO Anti-aliasing?
                Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        self.__text_preprocessing = get_tokenizer(self.__config.open_clip_model)

        self.__clip_model, _, _ = create_model_and_transforms(
            self.__config.open_clip_model,
            pretrained=self.__config.open_clip_model_pretrained,
        )
        self.__clip_model.eval()
        self.__clip_model.to(self.__config.device)

    def embed_text(self, text: str) -> Tensor:
        """Embeds the given text into a pytorch float32 tensor of shape B=1 x N"""
        with torch.no_grad():
            embedding = self.__clip_model.encode_text(self.__text_preprocessing(text).to(self.__config.device))
            embedding /= embedding.norm(dim=-1, keepdim=True)
            return embedding.detach()

    def embed_text_numpy(self, text: str) -> NDArray:
        """Embeds the given text into a numpy float32 array of shape N"""
        return self.embed_text(text).squeeze().cpu().numpy()

    def embed_image(self, file: Path) -> Tensor:
        """Embeds the given image into a pytorch float32 tensor of shape B=1 x N"""
        with torch.no_grad():
            image = self.__load_image_tensor(file)
            image = self.__image_preprocessing(image)
            embedding = self.__clip_model.encode_image(image)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            return embedding.detach()

    def embed_image_numpy(self, file: Path) -> NDArray:
        """Embeds the given image into a numpy float32 array of shape N"""
        return self.embed_image(file).squeeze().cpu().numpy()

    def __load_image_numpy(self, file: Path) -> NDArray:
        """Loads the image into a numpy uint8 array of shape H x W x C=3"""
        with im.open(file) as image:
            data = np.array(image, dtype=np.float32) / 255.0
            assert len(data.shape) == 3
            if data.shape[2] == 4:
                rgb = data[:, :, :3]
                alpha = data[:, :, 3:]
                data = rgb * alpha + self.__config.background_color * (1.0 - alpha)
            assert data.shape[2] == 3
            return (data * 255.0).clip(0, 255).astype(np.uint8)

    def __load_image_tensor(self, file: Path) -> Tensor:
        """Loads the image into a pytorch float32 tensor of shape B=1 x C=3 x H x W"""
        image = (
            torch.from_numpy(self.__load_image_numpy(file).astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.__config.device)
        )
        return image


class CachedEmbedder(Embedder):
    __parent: Embedder
    __cache_dir: Optional[Path]
    __validate_hash: bool
    __cache_only: bool

    @Embedder.config.getter
    def config(self) -> EmbedderConfig:
        return self.__parent.config

    def __init__(
        self, parent: Embedder, cache_dir: Optional[Path], validate_hash: bool = False, cache_only: bool = False
    ) -> None:
        self.__parent = parent
        self.__cache_dir = cache_dir
        self.__validate_hash = validate_hash
        self.__cache_only = cache_only

    @staticmethod
    def create_text_cache_file(file: Path, text: str, config: EmbedderConfig, value: Union[Tensor, NDArray]) -> Path:
        return CachedEmbedder.__create_cache_file(file, hash_text_sha1(text), config, value)

    @staticmethod
    def create_image_cache_file(file: Path, image: Path, config: EmbedderConfig, value: Union[Tensor, NDArray]) -> Path:
        return CachedEmbedder.__create_cache_file(file, hash_file_sha1(image), config, value)

    @staticmethod
    def __create_cache_file(
        file: Path, object_hash: str, config: EmbedderConfig, value: Union[Tensor, NDArray]
    ) -> Path:
        config_hash = CachedEmbedder.__hash_config_sha1(config)

        np_value: NDArray
        if torch.is_tensor(value):
            np_value = value.squeeze().cpu().numpy()
        else:
            np_value = value

        data_file = file.parent / (file.stem + ".npy")

        np.save(data_file, np_value)

        data_hash = hash_file_sha1(data_file)

        with file.open("w") as f:
            json.dump(
                {
                    "config_hash": config_hash,
                    "object_hash": object_hash,
                    "data_hash": data_hash,
                },
                f,
            )

        return file

    @staticmethod
    def __hash_config_sha1(config: EmbedderConfig) -> str:
        config_dict = asdict(config)
        del config_dict["_target"]
        del config_dict["device"]
        json_str = json.dumps(config_dict)
        return hash_text_sha1(json_str)

    def embed_text(self, text: str, cache_key: Optional[str] = None) -> Tensor:
        cached = self.__load_cached_tensor(cache_key, lambda: hash_text_sha1(text))
        if cached is not None:
            return cached

        if self.__cache_only:
            raise ValueError(f"{cache_key} not cached")

        return self.__parent.embed_text(text)

    def embed_text_numpy(self, text: str, cache_key: Optional[str] = None) -> NDArray:
        cached = self.__load_cached_ndarray(cache_key, lambda: hash_text_sha1(text))
        if cached is not None:
            return cached

        if self.__cache_only:
            raise ValueError(f"{cache_key} not cached")

        return self.__parent.embed_text_numpy(text)

    def embed_image(self, file: Path, cache_key: Optional[str] = None) -> Tensor:
        cached = self.__load_cached_tensor(cache_key, lambda: hash_file_sha1(file))
        if cached is not None:
            return cached

        if self.__cache_only:
            raise ValueError(f"{cache_key} not cached")

        return self.__parent.embed_image(file)

    def embed_image_numpy(self, file: Path, cache_key: Optional[str] = None) -> NDArray:
        cached = self.__load_cached_ndarray(cache_key, lambda: hash_file_sha1(file))
        if cached is not None:
            return cached

        if self.__cache_only:
            raise ValueError(f"{cache_key} not cached")

        return self.__parent.embed_image_numpy(file)

    def __load_cached_ndarray(self, cache_key: Optional[str], hash_func: Callable[[], str]) -> Optional[NDArray]:
        if self.__cache_dir is None or cache_key is None:
            return None

        cache_file = self.__cache_dir / (cache_key + ".json")
        data_file = self.__cache_dir / (cache_key + ".npy")

        if cache_file.exists() and cache_file.is_file() and data_file.exists() and data_file.is_file():
            with cache_file.open("r") as f:
                json_obj = json.load(f)

                if self.__validate_hash:
                    if "config_hash" not in json_obj or "object_hash" not in json_obj or "data_hash" not in json_obj:
                        return None

                    if (
                        json_obj["config_hash"] != CachedEmbedder.__hash_config_sha1(self.config)
                        or json_obj["object_hash"] != hash_func()
                        or json_obj["data_hash"] != hash_file_sha1(data_file)
                    ):
                        return None

                return np.load(data_file)

        return None

    def __load_cached_tensor(self, cache_key: Optional[str], hash_func: Callable[[], str]) -> Optional[Tensor]:
        if cache_key is None:
            return None

        value = self.__load_cached_ndarray(cache_key, hash_func)
        if value is None:
            return None

        return torch.from_numpy(value).unsqueeze(0).to(self.config.device)
