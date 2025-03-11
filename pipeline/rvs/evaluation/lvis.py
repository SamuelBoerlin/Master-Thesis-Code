import hashlib
import json
from pathlib import Path
from typing import Dict, List, NewType, Optional, Set

from nerfstudio.utils.rich_utils import CONSOLE
from objaverse import load_lvis_annotations, load_objects

from rvs.utils.console import file_link

Uid = NewType("Uid", str)
Category = NewType("Category", str)


class LVISDataset:
    # Cache relevant settings
    categories: Optional[Set[Category]]
    uids: Optional[Set[Uid]]
    per_category_limit: Optional[int]

    # Cache irrelevant settings
    download_processes: int
    __category_names: Optional[Dict[Category, str]]

    # Dataset
    dataset: Dict[Category, List[Uid]] = dict()
    """Mapping of LVIS category to list of objaverse 1.0 uids"""

    uid_to_file: Dict[Uid, str] = dict()
    """Mapping of LVIS objaverse 1.0 uid to local file path"""

    uid_to_category: Dict[Uid, Category] = dict()
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
        digest.update(str.encode("\0"))
        if self.per_category_limit is not None:
            digest.update(str.encode(str(self.per_category_limit)))
        return str(digest.hexdigest())

    def __init__(
        self,
        lvis_categories: Optional[Set[str]],
        lvis_uids: Optional[Set[str]],
        lvis_download_processes: int = 4,
        per_category_limit: Optional[int] = None,
        category_names: Optional[Dict[str, str]] = None,
    ) -> None:
        self.categories = lvis_categories
        self.uids = lvis_uids
        self.download_processes = int(lvis_download_processes)
        self.per_category_limit = int(per_category_limit) if per_category_limit is not None else None
        self.__category_names = (
            {key: str(value) for key, value in category_names.items()} if category_names is not None else None
        )

    def load(self) -> None:
        CONSOLE.log("Loading LVIS dataset...")
        self.dataset = self.fetch_lvis_dataset()

        CONSOLE.rule("Loading LVIS files...")
        self.uid_to_file = self.get_lvis_files()
        self.__update_uid_to_category_mapping()
        CONSOLE.rule()

    def fetch_lvis_dataset(self) -> Dict[str, List[str]]:
        dataset = load_lvis_annotations()
        if self.categories is not None:
            for k in list(dataset.keys()):
                if k not in self.categories:
                    del dataset[k]
        if self.uids is not None:
            for k in list(dataset.keys()):
                filtered = [u for u in dataset[k] if u in self.uids]
                if len(filtered) > 0:
                    dataset[k] = filtered
                else:
                    del dataset[k]
        if self.per_category_limit is not None:
            for category in dataset.keys():
                dataset[category] = dataset[category][: self.per_category_limit]
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
        if self.per_category_limit is not None:
            cache_json["per_category_limit"] = self.per_category_limit

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

            cache_json: Dict = json.loads(text)

            if (
                (
                    ("categories" not in cache_json and self.categories is None)
                    or ("categories" in cache_json and set(cache_json["categories"]) == self.categories)
                )
                and (
                    ("uids" not in cache_json and self.uids is None)
                    or ("uids" in cache_json and set(cache_json["uids"]) == self.uids)
                )
                and cache_json.get("per_category_limit", None) == self.per_category_limit
            ):
                self.dataset = cache_json["dataset"]
                self.uid_to_file = cache_json["uid_to_file"]
                self.per_category_limit = cache_json.get("per_category_limit", None)
                self.__update_uid_to_category_mapping()

                return cache_file

        CONSOLE.log("Failed loading LVIS dataset from cache")

        return None

    def __update_uid_to_category_mapping(self) -> None:
        self.uid_to_category = dict()
        for category in self.dataset.keys():
            for uid in self.dataset[category]:
                self.uid_to_category[uid] = category

    def get_category_name(self, category: Category) -> str:
        if self.__category_names is not None and category in self.__category_names:
            return self.__category_names[category]
        return category


def create_dataset(
    lvis_categories: Optional[Set[str]] = None,
    lvis_categories_file: Optional[Path] = None,
    lvis_uids: Optional[Set[str]] = None,
    lvis_uids_file: Optional[Path] = None,
    lvis_download_processes: int = 4,
    lvis_per_category_limit: Optional[int] = None,
    lvis_category_names: Optional[Dict[str, str]] = None,
    lvis_category_names_file: Optional[Path] = None,
) -> LVISDataset:
    if lvis_categories is not None:
        lvis_categories = set(lvis_categories)

    if lvis_uids is not None:
        lvis_uids = set(lvis_uids)

    if lvis_category_names is not None:
        lvis_category_names = dict(lvis_category_names)

    if lvis_categories_file is not None:
        if lvis_categories is None:
            lvis_categories = set()
        with lvis_categories_file.open("r") as f:
            lvis_categories = lvis_categories.union(set(json.load(f)))

    if lvis_uids_file is not None:
        if lvis_uids is None:
            lvis_uids = set()
        with lvis_uids_file.open("r") as f:
            lvis_uids = lvis_uids.union(set(json.load(f)))

    if lvis_category_names_file is not None:
        if lvis_category_names is None:
            lvis_category_names = dict()
        with lvis_category_names_file.open("r") as f:
            lvis_category_names.update(json.load(f))

    return LVISDataset(
        lvis_categories,
        lvis_uids,
        lvis_download_processes=lvis_download_processes,
        per_category_limit=lvis_per_category_limit,
        category_names=lvis_category_names,
    )
