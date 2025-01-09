import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

from nerfstudio.utils.rich_utils import CONSOLE
from objaverse import load_lvis_annotations, load_objects

from rvs.utils.console import file_link


class LVISDataset:
    # Cache relevant settings
    categories: Optional[Set[str]]
    uids: Optional[Set[str]]

    # Cache irrelevant settings
    download_processes: int

    # Dataset
    dataset: Dict[str, List[str]] = dict()
    """Mapping of LVIS category to list of objaverse 1.0 uids"""

    uid_to_file: Dict[str, str] = dict()
    """Mapping of LVIS objaverse 1.0 uid to local file path"""

    uid_to_category: Dict[str, str] = dict()
    """Mapping of LVIS objaverse 1.0 uid to LVIS category"""

    @property
    def cache_key(self) -> str:
        digest = hashlib.sha1()
        if self.categories is not None:
            for category in sorted(self.categories):
                digest.update(str.encode(category))
        digest.update(str.encode("\0"))
        if self.uids is not None:
            for uids in sorted(self.uids):
                digest.update(str.encode(uids))
        return str(digest.hexdigest())

    def __init__(
        self, lvis_categories: Optional[Set[str]], lvis_uids: Optional[Set[str]], lvis_download_processes: int
    ) -> None:
        self.categories = lvis_categories
        self.uids = lvis_uids
        self.download_processes = lvis_download_processes

    def load(self) -> None:
        CONSOLE.log("Loading LVIS dataset...")
        self.dataset = self.fetch_lvis_dataset(self.categories, self.uids)

        CONSOLE.rule("Loading LVIS files...")
        self.uid_to_file = self.get_lvis_files()
        self.__update_uid_to_category_mapping()
        CONSOLE.rule()

    def fetch_lvis_dataset(self, categories: Optional[Set[str]], uids: Optional[Set[str]]) -> Dict[str, List[str]]:
        dataset = load_lvis_annotations()
        if categories is not None:
            for k in list(dataset.keys()):
                if k not in categories:
                    del dataset[k]
        if uids is not None:
            for k in list(dataset.keys()):
                filtered = [u for u in dataset[k] if u in uids]
                if len(filtered) > 0:
                    dataset[k] = filtered
                else:
                    del dataset[k]
        return dataset

    def get_lvis_files(self) -> Dict[str, str]:
        files = {}
        for k in self.dataset.keys():
            CONSOLE.log(f"Category: {k}")
            category_files = load_objects(self.dataset[k], download_processes=self.download_processes)
            CONSOLE.log(f"Files: {len(category_files)}")
            files.update(category_files)
        return files

    def save_cache(self, dir: Path) -> Path:
        cache_file = dir / (self.cache_key + ".json")

        CONSOLE.log(f"Saving LVIS dataset to cache ({file_link(cache_file)})...")

        cache_json = {
            "dataset": self.dataset,
            "uid_to_file": self.uid_to_file,
        }
        if self.categories is not None:
            cache_json["categories"] = list(self.categories)
        if self.uids is not None:
            cache_json["uids"] = list(self.uids)

        text = json.dumps(cache_json)
        with cache_file.open("w") as f:
            f.write(text)

        return cache_file

    def load_cache(self, dir: Path) -> Optional[Path]:
        cache_file = dir / (self.cache_key + ".json")

        CONSOLE.log(f"Loading LVIS dataset from cache ({file_link(cache_file)})...")

        if cache_file.exists() and cache_file.is_file():
            text: str = None
            with cache_file.open("r") as f:
                text = f.read()

            cache_json = json.loads(text)

            if (
                ("categories" not in cache_json and self.categories is None)
                or ("categories" in cache_json and set(cache_json["categories"]) == self.categories)
            ) and (
                ("uids" not in cache_json and self.uids is None)
                or ("uids" in cache_json and set(cache_json["uids"]) == self.uids)
            ):
                self.dataset = cache_json["dataset"]
                self.uid_to_file = cache_json["uid_to_file"]
                self.__update_uid_to_category_mapping()

                return cache_file

        CONSOLE.log("Failed loading LVIS dataset from cache")

        return None

    def __update_uid_to_category_mapping(self) -> None:
        self.uid_to_category = dict()
        for category in self.dataset.keys():
            for uid in self.dataset[category]:
                self.uid_to_category[uid] = category
