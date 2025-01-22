from pathlib import Path
from typing import Callable, List, Optional, Set, Type, TypeVar

T = TypeVar("T")


class PipelineIO:
    output_dir: Path
    input_dirs: List[Path]

    def __init__(
        self,
        output_dir: Path,
        input_dirs: Optional[List[Path]] = None,
    ) -> None:
        self.output_dir = output_dir
        if input_dirs is None:
            self.input_dirs = [self.output_dir]
        else:
            self.input_dirs = list(input_dirs)
            self.input_dirs.append(self.output_dir)

    def get_output_path(self, path: Path) -> Path:
        return self.output_dir / path

    def mk_output_path(self, path: Path) -> Path:
        path = self.get_output_path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_input_path(
        self,
        path: Path,
        condition: Optional[Callable[[Path], bool]] = None,
    ) -> Path:
        for input_dir in self.input_dirs:
            input_path = input_dir / path
            if input_path.exists() and (condition is None or condition(input_path)):
                return input_path
        return self.get_output_path(path)

    def load_input(
        self,
        path: Path,
        loader: Callable[[Path], T],
        expected_errors: Set[Type] = {FileNotFoundError},
    ) -> T:
        for input_dir in self.input_dirs:
            input_path = input_dir / path
            try:
                return loader(input_path)
            except Exception as ex:
                if type(ex) in expected_errors:
                    continue
                raise ex
        return loader(self.get_output_path(path))
