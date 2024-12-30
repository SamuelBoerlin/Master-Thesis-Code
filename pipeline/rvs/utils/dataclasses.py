import dataclasses
from typing import Any, Type, TypeVar

T = TypeVar("T")


def extend_dataclass_obj(obj: Any, cls: Type[T]) -> T:
    fields = [f.name for f in dataclasses.fields(obj)]
    fields.remove("_target")
    return cls(**{f: getattr(obj, f) for f in fields})
