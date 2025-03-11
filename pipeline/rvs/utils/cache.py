from pathlib import Path

from rvs.utils.hash import hash_text_sha1


def get_pipeline_render_embedding_cache_key(model_file: Path, image_file: Path) -> str:
    return f"{model_file.stem}_{image_file.stem}"


def get_evaluation_prompt_embedding_cache_key(text: str) -> str:
    return f"prompt_{hash_text_sha1(text)}"
