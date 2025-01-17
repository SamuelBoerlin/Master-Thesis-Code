import dataclasses
from pathlib import Path
from typing import Any, Callable, Optional, Set, Type, TypeVar

import yaml

MISSING_FIELD = object()
AUTO_DEFAULTS = object()

T = TypeVar("T")


def load_config(
    file: Path,
    cls: Type[T],
    default_config_factory: Optional[Callable[[], T]] = AUTO_DEFAULTS,
    on_default_applied: Optional[Callable[[str, Any, str, Any], None]] = None,
    path: str = "config",
) -> T:
    config: T

    try:
        config = yaml.load(file.read_text(encoding="utf8"), Loader=yaml.Loader)
    except FileNotFoundError as ex:
        raise FileNotFoundError(f'Config "{file}" does not exist') from ex

    if not isinstance(config, cls):
        raise ValueError(f"Config type is incorrect, expected {cls} but got {type(config)}")

    if default_config_factory == AUTO_DEFAULTS:
        default_config_factory = cls

    if default_config_factory is not None:
        __apply_missing_defaults(
            default_config_factory(),
            config,
            path=path,
            on_default_applied=on_default_applied,
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


def apply_missing_config_defaults(
    config: T,
    cls: Type[T],
    default_config_factory: Optional[Callable[[], T]] = AUTO_DEFAULTS,
    on_default_applied: Optional[Callable[[str, Any, str, Any], None]] = None,
    path: str = "config",
) -> None:
    if default_config_factory == AUTO_DEFAULTS:
        default_config_factory = cls

    __apply_missing_defaults(
        default_config_factory(),
        config,
        path=path,
        on_default_applied=on_default_applied,
    )


def __apply_missing_defaults(
    default_data: Any,
    data: Any,
    path: str = "config",
    on_default_applied: Optional[Callable[[str, Any, str, Any], None]] = None,
) -> None:
    def apply_default(fpath: str, actual_data: Any, fname: str, expected_value: Any) -> None:
        setattr(actual_data, fname, expected_value)
        if on_default_applied is not None:
            on_default_applied(fpath, actual_data, fname, expected_value)

    __find_changed_fields(
        default_data,
        data,
        path=path,
        on_missing_field=apply_default,
    )


def find_changed_config_fields(
    expected_config: T,
    config: T,
    on_missing_field: Optional[Callable[[str, Any, str, Any], None]],
    on_changed_field: Optional[Callable[[str, Any, str, Any, Any], None]],
    on_unknown_field: Optional[Callable[[str, Any, str, Any], None]],
    path: str = "config",
    bidirectional: bool = True,
) -> None:
    suppressed: Set[str] = set()

    def handle_missing_field(fpath: str, actual_data: Any, fname: str, expected_value: Any) -> None:
        suppressed.add(fpath)
        on_missing_field(fpath, actual_data, fname, expected_value)

    def handle_changed_field(
        fpath: str, expected_data: Any, actual_data: Any, fname: str, expected_value: Any, actual_value: Any
    ) -> None:
        suppressed.add(fpath)
        on_changed_field(fpath, actual_data, fname, expected_value, actual_value)

    __find_changed_fields(
        expected_data=expected_config,
        actual_data=config,
        on_missing_field=handle_missing_field,
        on_changed_field=handle_changed_field,
        path=path,
    )

    if bidirectional:

        def handle_missing_field_rev(fpath: str, data: Any, fname: str, actual_value: Any) -> None:
            on_unknown_field(fpath, data, fname, actual_value)

        def handle_changed_field_rev(
            fpath: str, actual_data: Any, expected_data: Any, fname: str, actual_value: Any, expected_value: Any
        ) -> None:
            on_changed_field(fpath, actual_data, fname, expected_value, actual_value)

        __find_changed_fields(
            expected_data=config,
            actual_data=expected_config,
            on_missing_field=handle_missing_field_rev,
            on_changed_field=handle_changed_field_rev,
            path=path,
            suppressed=suppressed,
        )


def __find_changed_fields(
    expected_data: Any,
    actual_data: Any,
    on_missing_field: Optional[Callable[[str, Any, str, Any], None]] = None,
    on_changed_field: Optional[Callable[[str, Any, Any, str, Any, Any], None]] = None,
    path: str = "config",
    suppressed: Optional[Set[str]] = None,
) -> None:
    cls = type(expected_data)

    if not dataclasses.is_dataclass(cls):
        return

    def is_suppressed(fpath: str) -> bool:
        return suppressed is not None and fpath in suppressed

    fields = dataclasses.fields(cls)

    for f in fields:
        fname = f.name
        fpath = path + "." + fname

        if hasattr(expected_data, fname):
            expected_value = getattr(expected_data, fname)

            if not hasattr(actual_data, fname):
                if on_missing_field is not None and not is_suppressed(fpath):
                    on_missing_field(fpath, actual_data, fname, expected_value)
            else:
                actual_value = getattr(actual_data, fname)

                if dataclasses.is_dataclass(type(expected_value)):
                    __find_changed_fields(
                        expected_value,
                        actual_value,
                        on_missing_field=on_missing_field,
                        on_changed_field=on_changed_field,
                        path=fpath,
                        suppressed=suppressed,
                    )
                elif on_changed_field is not None and not is_suppressed(fpath) and actual_value != expected_value:
                    on_changed_field(fpath, expected_data, actual_data, fname, expected_value, actual_value)


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
