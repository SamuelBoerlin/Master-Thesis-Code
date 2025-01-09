from typing import Any, Dict, List, Set, Tuple, TypeVar

T = TypeVar("T")


def convert_nested_maps_to_tuples(map: Dict[str, Dict[str, T]], key_order: List[str]) -> Dict[str, Tuple[T, ...]]:
    ret: Dict[str, Tuple[T, ...]] = dict()
    for key in map:
        d = map[key]
        values = []
        for order_key in key_order:
            if order_key in d:
                values.append(d[order_key])
        ret[key] = tuple(values)
    return ret


def get_keys_of_nested_maps(map: Dict[str, Dict[str, Any]]) -> Set[str]:
    uids = set()
    for key in map:
        for uid in map[key]:
            uids.add(uid)
    return uids
