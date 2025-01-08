from pathlib import Path


def file_link(path: Path):
    if path.name is None:
        raise ValueError("Path is not a file")
    return f"[link file://{str(path.absolute())}]{path.name}[/link file://{str(path.absolute())}]"
