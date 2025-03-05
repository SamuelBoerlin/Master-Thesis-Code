from pathlib import Path


def get_pipeline_render_embedding_cache_key(model_file: Path, image_file: Path) -> str:
    return f"{model_file.stem}_{image_file.stem}"
