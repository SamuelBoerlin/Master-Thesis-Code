import sys
import traceback
from pathlib import Path
from typing import Optional, Set

from rvs.evaluation.pipeline import PipelineEvaluationInstance
from rvs.pipeline.stage import PipelineStage
from rvs.utils.process import ProcessResult


def pipeline_worker_func(
    instance: PipelineEvaluationInstance,
    file: Path,
    stages_filter: Optional[Set[PipelineStage]] = None,
    result: Optional[ProcessResult] = None,
    stdout_file: Optional[str] = None,
    stderr_file: Optional[str] = None,
) -> None:
    try:
        if stdout_file is not None:
            stdout_file_path = Path(stdout_file)
            if not stdout_file_path.is_absolute():
                raise ValueError("stdout_file path must be absolute")
            sys.stdout = stdout_file_path.open(mode="a", buffering=1, encoding="utf-8")

        if stderr_file is not None:
            stderr_file_path = Path(stdout_file)
            if not stderr_file_path.is_absolute():
                raise ValueError("stderr_file path must be absolute")
            sys.stderr = stderr_file_path.open(mode="a", buffering=1, encoding="utf-8")

        instance.run(file, stages_filter=stages_filter)

        if result is not None:
            result.success = True
            result.close()
    except BaseException as ex:
        msg = traceback.format_exc()
        print(msg, file=sys.stderr, flush=True)
        if result is not None:
            result.success = False
            result.msg = msg
            result.close()
        raise ex
