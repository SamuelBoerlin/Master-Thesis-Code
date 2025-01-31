from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Set, Type, TypeVar, Union

if TYPE_CHECKING:
    from rvs.pipeline.pipeline import PipelineStage

T = TypeVar("T")


class PipelineIO:
    output_dir: Path
    input_dirs: List[Path]

    __stages: Optional[Set["PipelineStage"]]

    def __init__(
        self,
        output_dir: Path,
        stages: Optional[Union[List["PipelineStage"], Set["PipelineStage"]]],
        input_dirs: Optional[List[Path]] = None,
    ) -> None:
        self.output_dir = output_dir

        self.__stages = None if stages is None else set(stages)

        if input_dirs is None:
            self.input_dirs = [self.output_dir]
        else:
            self.input_dirs = list(input_dirs)
            self.input_dirs.append(self.output_dir)

    def get_path(
        self,
        stage: "PipelineStage",
        path: Path,
        condition: Optional[Callable[[Path], bool]] = None,
    ) -> Path:
        if self.__stages is None or stage in self.__stages:
            return self.get_output_path(path)
        else:
            return self.get_input_path(path, condition=condition)

    def load(
        self,
        stage: "PipelineStage",
        path: Path,
        loader: Callable[[Path], T],
        expected_errors: Set[Type] = {FileNotFoundError},
    ) -> T:
        if self.__stages is None or stage in self.__stages:
            return self.load_output(path, loader)
        else:
            return self.load_input(path, loader, expected_errors=expected_errors)

    def get_output_path(self, path: Path) -> Path:
        return self.output_dir / path

    def mk_output_path(self, path: Path) -> Path:
        path = self.get_output_path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load_output(
        self,
        path: Path,
        loader: Callable[[Path], T],
    ) -> T:
        return loader(self.get_output_path(path))

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
