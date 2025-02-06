import os
import signal
from queue import Empty, Full
from time import sleep, time
from typing import Any, Iterable, Optional

import torch.multiprocessing
from torch.multiprocessing import Process, Queue, set_start_method


class ProcessResult:
    __success_queue: Queue
    __msg_queue: Queue

    __cached_success: bool = True
    __cached_msg: str = None

    def __init__(self) -> None:
        self.__success_queue = torch.multiprocessing.Queue(maxsize=1)
        self.__msg_queue = torch.multiprocessing.Queue(maxsize=1)

    @property
    def success(self) -> bool:
        try:
            self.__cached_success = self.__success_queue.get_nowait()
        except Empty:
            pass
        return self.__cached_success

    @success.setter
    def success(self, value: bool) -> None:
        ProcessResult.__set_queue_value(self.__success_queue, value)
        self.__cached_success = value

    @property
    def msg(self) -> Optional[str]:
        try:
            self.__cached_msg = self.__msg_queue.get_nowait()
        except Empty:
            pass
        return self.__cached_msg

    @msg.setter
    def msg(self, value: str) -> None:
        ProcessResult.__set_queue_value(self.__msg_queue, value)
        self.__cached_msg = value

    @staticmethod
    def __set_queue_value(queue: Queue, value: Any) -> None:
        while True:
            try:
                while queue.get_nowait() is not None:
                    pass  # Not yet empty
            except Empty:
                pass
            try:
                queue.put_nowait(value)
            except Full:
                continue  # Retry
            break

    def close(self) -> None:
        self.__success_queue.close()
        self.__success_queue.join_thread()
        self.__msg_queue.close()
        self.__msg_queue.join_thread()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def start_process(target: Any, args: Iterable[Any]) -> Process:
    set_start_method("spawn", force=True)  # Required for CUDA: https://pytorch.org/docs/main/notes/multiprocessing.html
    process = Process(
        target=target,
        args=args,
        daemon=True,
    )
    process.start()
    return process


def stop_process(process: Optional[Process]) -> None:
    if process is not None:
        stop_time = time()
        while process.is_alive():
            elapsed_time = time() - stop_time
            if elapsed_time > 1.0:
                process.kill()
            elif elapsed_time > 0.1:
                os.kill(process.pid, signal.SIGINT)
            sleep(0.01)
