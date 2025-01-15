import os
from pathlib import Path

from rvs.evaluation.index import save_index

# Applied 14.01.2025 15:39


def apply():
    base_path = Path("/nas/eval/outputs/intermediate/results")

    for result_path in base_path.iterdir():
        if result_path.is_dir() and str(result_path).endswith(".glb"):
            image_files = []
            transforms_files = []

            for image_file in result_path.iterdir():
                if image_file.is_file() and image_file.name.endswith(".png"):
                    transforms_file = result_path / (image_file.name + ".transforms.json")

                    if not transforms_file.exists() or not transforms_file.is_file():
                        raise Exception("Missing .transforms.json for " + str(image_file))

                    image_files.append(image_file)
                    transforms_files.append(transforms_file)

            index_json_path = result_path / "index.json"

            if len(image_files) > 0:
                save_index(index_json_path, image_files, transforms_files)
            elif index_json_path.exists():
                os.remove(index_json_path)


if __name__ == "__main__":
    apply()
