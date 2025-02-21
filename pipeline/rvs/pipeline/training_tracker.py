import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from nerfstudio.configs.base_config import LocalWriterConfig
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.writer import EVENT_STORAGE, EventType, LocalWriter
from torch import Tensor

from rvs.utils.console import file_link


@dataclass
class TrainingTrackerConfig:
    output_dir: Optional[Path] = None

    tracked_scalars: Dict[str, str] = field(
        default_factory=lambda: {
            "Train Loss": "train_loss",
        }
    )

    tracked_dicts: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: {
            "Train Loss Dict": {
                "rgb_loss": "rgb_loss",
                "interlevel_loss": "interlevel_loss",
                "distortion_loss": "distortion_loss",
                "clip_loss": "clip_loss",
                "dino_loss": "dino_loss",
            },
            "Train Metrics Dict": {
                "psnr": "image_psnr",
            },
        }
    )


@dataclass
class LocalWriterShimConfig(LocalWriterConfig):
    _target: Type = field(default_factory=lambda: TrainingTracker)

    parent: Optional[LocalWriterConfig] = None

    config: TrainingTrackerConfig = field(default_factory=TrainingTrackerConfig)

    def setup(self, **kwargs) -> Any:
        parent_writer: Optional[LocalWriter] = None
        if self.parent is not None:
            parent_writer = self.parent.setup(**kwargs)
        return self._target(self.config, parent_writer=parent_writer, **kwargs)


class TrainingTracker(LocalWriter):
    __parent_writer: Optional[LocalWriter]

    __tracked_scalars: Dict[str, str]
    __tracked_dicts: Dict[str, Dict[str, str]]

    __output_dir: Path

    __outputs: Dict[str, List[Tuple[int, float]]] = dict()

    def __init__(
        self,
        config: TrainingTrackerConfig,
        parent_writer: Optional[LocalWriter] = None,
        **kwargs,
    ) -> None:
        self.__parent_writer = parent_writer
        self.__tracked_scalars = config.tracked_scalars
        self.__tracked_dicts = config.tracked_dicts
        if config.output_dir is None:
            raise ValueError("output_dir is required")
        self.__output_dir = config.output_dir

    def write_stats_log(self, step: int) -> None:
        self._write_data()

        if self.__parent_writer is not None:
            self.__parent_writer.write_stats_log(step)

    def write_config(self, name: str, config_dict: Dict[str, Any], step: int):
        if self.__parent_writer is not None:
            self.__parent_writer.write_config(name, config_dict, step)

    def _write_data(self) -> None:
        if self._process_events(EVENT_STORAGE):
            self._save_outputs()

    def _process_events(self, events: List[Dict[str, Any]]) -> bool:
        changed = False

        for event in events:
            event_name = event["name"]
            event_type = event["write_type"]
            event_step = event["step"]
            event_value = event["event"]

            if event_type in (EventType.SCALAR, EventType.SCALAR.value):
                try:
                    scalar = self.__get_scalar(event_value)

                    if event_name in self.__tracked_scalars:
                        self._add_output_value(self.__tracked_scalars[event_name], event_step, scalar)
                        changed = True
                except ValueError:
                    pass
            elif event_type in (EventType.DICT, EventType.DICT.value):
                event_dict: Dict[str, Any] = event_value

                if event_name in self.__tracked_dicts:
                    tracked_dict_scalars = self.__tracked_dicts[event_name]

                    for key, value in event_dict.items():
                        try:
                            scalar = self.__get_scalar(value)

                            if key in tracked_dict_scalars:
                                self._add_output_value(tracked_dict_scalars[key], event_step, scalar)
                                changed = True
                        except ValueError:
                            pass

        return changed

    def __get_scalar(self, value: Any) -> float:
        if torch.is_tensor(value):
            try:
                value = value.float().data[0]
            except (KeyError, IndexError):
                pass
        return float(value)

    def _add_output_value(self, name: str, step: int, value: float) -> None:
        output_values: List[float]

        if name in self.__outputs:
            output_values = self.__outputs[name]
        else:
            output_values = []
            self.__outputs[name] = output_values

        output_values.append((step, value))

    def _save_outputs(self) -> None:
        if self.__output_dir.exists() and self.__output_dir.is_dir():
            for output_name, output_values in self.__outputs.items():
                output_file = self.__output_dir / (output_name + ".json")

                outputs = []
                for step, value in output_values:
                    outputs.append({"step": step, "value": value})

                try:
                    with output_file.open("w") as f:
                        json.dump(outputs, f)
                except Exception as e:
                    CONSOLE.log(
                        f"[bold red]ERROR: Failed saving training tracking output {output_name} to {file_link(output_file)}:\n{str(e)}"
                    )
