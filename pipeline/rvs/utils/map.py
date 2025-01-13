from typing import Dict, List, Set, Tuple, TypeVar

K1 = TypeVar("K1")
K2 = TypeVar("K2")
V = TypeVar("V")


def convert_map_to_tuple(map: Dict[K1, V], key_order: List[K1]) -> Tuple[V, ...]:
    values: List[V] = []
    for order_key in key_order:
        if order_key in map:
            values.append(map[order_key])
    return tuple(values)


def convert_nested_maps_to_tuples(map: Dict[K1, Dict[K2, V]], key_order: List[K2]) -> Dict[K1, Tuple[V, ...]]:
    return {key: convert_map_to_tuple(map[key], key_order) for key in map.keys()}


def get_keys_of_nested_maps(map: Dict[K1, Dict[K2, V]]) -> Set[K2]:
    uids: Set[K2] = set()
    for key in map:
        for uid in map[key]:
            uids.add(uid)
    return uids


def extract_nested_maps(map: Dict[K1, Dict[K2, V]], key: K2) -> Dict[K1, V]:
    ret: Dict[K1, V] = dict()
    for k, inner in map.items():
        ret[k] = inner[key]
    return ret
