import json
from dataclasses import dataclass
from pathlib import Path
from typing import Set

import numpy as np
import tyro
from git import List, Optional
from nerfstudio.configs.config_utils import convert_markup_to_ansi

from rvs.evaluation.lvis import LVISDataset


@dataclass
class Command:
    def run(self) -> None:
        pass


@dataclass
class FindCategoriesOfSize(Command):
    min_size: int
    """Minimum size of category"""

    categories_json_file: Optional[Path] = None
    """Path where to save categories array .json file"""

    def run(self) -> None:
        lvis = LVISDataset(lvis_categories=None, lvis_uids=None)

        dataset = lvis.fetch_lvis_dataset()

        categories: List[str] = []

        for category, category_uids in dataset.items():
            if len(category_uids) >= self.min_size:
                categories.append(category)

        categories = sorted(categories)

        for category in categories:
            print(category)

        if self.categories_json_file is not None:
            with self.categories_json_file.open("w") as f:
                json.dump(categories, f)


@dataclass
class SelectUidsInCategories(Command):
    categories_json_file: Path
    """Path of categories array .json file"""

    count: int
    """Number of uids"""

    seed: int = 42
    """Seed for random selection"""

    excluded_uids_json_file: Optional[Path] = None
    """Path of uids array .json file which should be excluded from selection"""

    uids_json_file: Optional[Path] = None
    """Path where to save uids array .json file"""

    def run(self) -> None:
        categories: List[str]
        with self.categories_json_file.open("r") as f:
            categories = sorted(list(json.load(f)))

        excluded_uids = set()
        if self.excluded_uids_json_file is not None:
            with self.excluded_uids_json_file.open("r") as f:
                excluded_uids = set(json.load(f))

        lvis = LVISDataset(lvis_categories=set(categories), lvis_uids=None)

        dataset = lvis.fetch_lvis_dataset()

        uids: List[str] = []

        main_rng = np.random.default_rng(seed=self.seed)

        for category in categories:
            category_uids = sorted(dataset[category])

            filtered_category_uids: List[str] = [uid for uid in category_uids if uid not in excluded_uids]

            selection_rng = np.random.default_rng(seed=main_rng)

            for _ in range(min(len(filtered_category_uids), self.count)):
                idx = selection_rng.integers(low=0, high=len(filtered_category_uids))
                uids.append(filtered_category_uids[idx])
                del filtered_category_uids[idx]

        uids = sorted(uids)

        for uid in uids:
            print(uid)

        if self.uids_json_file is not None:
            with self.uids_json_file.open("w") as f:
                json.dump(uids, f)


commands = {
    "find_categories_of_size": FindCategoriesOfSize(
        min_size=tyro.MISSING,
    ),
    "select_uids_in_categories": SelectUidsInCategories(
        categories_json_file=tyro.MISSING,
        count=tyro.MISSING,
    ),
}

SubcommandTypeUnion = tyro.conf.SuppressFixed[
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(
            defaults=commands, descriptions={key: type(command).__doc__ for key, command in commands.items()}
        )
    ]
]


def main(cmd: Command):
    cmd.run()


def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            SubcommandTypeUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    entrypoint()
