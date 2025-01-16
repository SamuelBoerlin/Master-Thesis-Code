import dataclasses
from inspect import getattr_static
from pathlib import Path
from typing import Any, Callable, Optional, Type, TypeVar

import yaml

MISSING_FIELD = object()
AUTO_DEFAULTS = object()

T = TypeVar("T")


def load_config(
    file: Path,
    cls: Type[T],
    defaults: Optional[Callable[[], T]] = AUTO_DEFAULTS,
    on_default_applied: Optional[Callable[[str, Any], None]] = None,
    on_default_detected: Optional[Callable[[str, Any], None]] = None,
) -> T:
    config: T

    try:
        config = yaml.load(file.read_text(encoding="utf8"), Loader=yaml.Loader)
    except FileNotFoundError as ex:
        raise FileNotFoundError(f'Config "{file}" does not exist') from ex

    if not isinstance(config, cls):
        raise ValueError(f"Config type is incorrect, expected {cls} but got {type(config)}")

    if defaults == AUTO_DEFAULTS:
        defaults = cls

    if defaults is not None:
        __apply_missing_defaults(
            defaults(),
            config,
            on_default_applied=on_default_applied,
            on_default_detected=on_default_detected,
        )

    return config


def find_config_working_dir(file: Path, saved_dir: Path) -> Optional[Path]:
    if not file.exists():
        raise FileNotFoundError(f'Config file "{file}" does not exist')

    if not file.is_file():
        raise FileExistsError(f'Config path "{file}" is not a file')

    if saved_dir.is_absolute():
        return None

    config_dir = file.parent.resolve()

    common_dir = __find_common_base_dir(config_dir, saved_dir)

    if common_dir is None:
        raise Exception(
            f'Unable to determine working directory from saved path "{str(saved_dir)}" and config directory "{str(config_dir)}"'
        )

    return common_dir.parent


def __apply_missing_defaults(
    defaults: Any,
    data: Any,
    path: str = "config",
    on_default_applied: Optional[Callable[[str, Any], None]] = None,
    on_default_detected: Optional[Callable[[str, Any], None]] = None,
) -> None:
    cls = type(defaults)

    if not dataclasses.is_dataclass(cls):
        return

    fields = dataclasses.fields(cls)

    for f in fields:
        fname = f.name
        fpath = path + "." + fname

        if hasattr(defaults, fname):
            default = getattr(defaults, fname)

            if not hasattr(data, fname):
                setattr(data, fname, default)
                if on_default_applied is not None:
                    on_default_applied(fpath, default)
            elif getattr_static(data, fname, default=MISSING_FIELD) == MISSING_FIELD:
                if on_default_detected:
                    on_default_detected(fpath, getattr(data, fname))
            else:
                __apply_missing_defaults(
                    default,
                    getattr(data, fname),
                    path=fpath,
                    on_default_applied=on_default_applied,
                    on_default_detected=on_default_detected,
                )


def __find_common_base_dir(full: Path, part: Path) -> Optional[Path]:
    full = full.resolve()

    while True:
        match = __match_dirs(full, part)
        if match is not None:
            return match

        next_full = full.parent
        if next_full == full:
            return None
        full = next_full


def __match_dirs(full: Path, part: Path) -> Optional[Path]:
    while True:
        if part.is_absolute():
            raise ValueError(f'Partial path "{str(part)}" is absolute path')

        if part.name != full.name:
            return None

        next_part = part.parent
        if next_part == next_part.parent:
            return full
        part = next_part

        next_full = full.parent
        if next_full == full:
            return None
        full = next_full
