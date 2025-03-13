from typing import Any, Dict, List, NewType, Optional, Set, Tuple, TypeVar

from rvs.evaluation.lvis import Category, LVISDataset, Uid
from rvs.utils.map import get_keys_of_nested_maps, rename_dict_keys

Method = NewType("Method", str)


def get_categories_of_uids(lvis: LVISDataset, uids: List[Uid]) -> Set[Uid]:
    categories = set()
    for uid in uids:
        categories.add(lvis.uid_to_category[uid])
    return categories


def get_categories_tuple(lvis: LVISDataset, map: Dict[Method, Dict[Uid, Any]]) -> Tuple[Category, ...]:
    return tuple(sorted(list(get_categories_of_uids(lvis, get_keys_of_nested_maps(map)))))


def rename_categories_tuple(
    categories: Tuple[Category, ...],
    category_names: Optional[Dict[Category, str]],
) -> Tuple[Category, ...]:
    if category_names is None:
        return categories
    return tuple([category_names[category] if category in category_names else category for category in categories])


T = TypeVar("T")


def rename_methods_dict(
    map: Dict[Method, T],
    method_names: Optional[Dict[Category, str]],
) -> Dict[str, T]:
    return rename_dict_keys(map, method_names)


def rename_categories_dict(
    map: Dict[Category, T],
    category_names: Optional[Dict[Category, str]],
) -> Dict[str, T]:
    return rename_dict_keys(map, category_names)


def count_category_items(map: Dict[Uid, Category], uids: List[Uid], category: Category) -> int:
    count = 0
    for uid in uids:
        item_category = map[uid]
        if item_category == category:
            count += 1
    return count
