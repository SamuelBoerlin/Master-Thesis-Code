from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class DefaultEmbeddingTypes(str, Enum):
    CLIP = "CLIP"
    DINO = "DINO"


@dataclass
class EmbeddingConfig:
    @property
    def type(self) -> str:
        pass


@dataclass
class ClipEmbeddingConfig(EmbeddingConfig):
    @property
    def type(self) -> str:
        return DefaultEmbeddingTypes.CLIP


@dataclass
class ClipAtScaleEmbeddingConfig(ClipEmbeddingConfig):
    scale: float = 1.0
    """LERF scale parameter"""


@dataclass
class DinoEmbeddingConfig(EmbeddingConfig):
    @property
    def type(self) -> str:
        return DefaultEmbeddingTypes.DINO


def get_embedding_config(tuple: Tuple[EmbeddingConfig], type: str) -> Optional[EmbeddingConfig]:
    for config in tuple:
        if config.type == type:
            return config
    return None
