import traceback
from pathlib import Path
from typing import List, Optional

from rvs.evaluation.pipeline import PipelineEvaluationInstance
from rvs.evaluation.process import ProcessResult
from rvs.pipeline.pipeline import Pipeline


def pipeline_worker_func(
    instance: PipelineEvaluationInstance,
    file: Path,
    stages: Optional[List[Pipeline.Stage]] = None,
    result: Optional[ProcessResult] = None,
) -> None:
    try:
        instance.run(file, stages)
        if result is not None:
            result.success = True
            result.close()
    except BaseException as ex:
        if result is not None:
            result.success = False
            result.msg = traceback.format_exc()
            result.close()
        raise ex
