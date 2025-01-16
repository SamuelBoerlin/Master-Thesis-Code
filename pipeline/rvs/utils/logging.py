from logging import FileHandler, Formatter, Handler, Logger
from pathlib import Path
from typing import List


class LoggerHandle:
    logger: Logger

    __handlers: List[Handler]

    def __init__(
        self,
        name: str,
        formatter: Formatter,
    ) -> None:
        self.logger = Logger(name)
        self.logger_format = formatter
        self.__handlers = []

    def add_handler(
        self,
        handler: Handler,
        set_formatter: bool = True,
    ) -> "LoggerHandle":
        if set_formatter:
            handler.setFormatter(self.logger_format)
        self.logger.addHandler(handler)
        return self

    def close(self) -> None:
        for handler in self.__handlers:
            handler.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def create_logger(
    name: str,
    formatter: Formatter = Formatter(fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"),
    files: List[Path] = [],
) -> LoggerHandle:
    handle = LoggerHandle(name, formatter)
    for file in files:
        handle.add_handler(FileHandler(file))
    return handle
