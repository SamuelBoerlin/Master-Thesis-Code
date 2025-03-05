import hashlib
from pathlib import Path


def hash_file_sha1(file: Path) -> str:
    digest = hashlib.sha1()
    with file.open(mode="rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            digest.update(data)
    return str(digest.hexdigest())


def hash_text_sha1(text: str) -> str:
    digest = hashlib.sha1()
    digest.update(text.encode())
    return str(digest.hexdigest())
