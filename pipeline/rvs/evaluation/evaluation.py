import json
import threading
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
from logging import Logger
from pathlib import Path
from threading import Semaphore
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import yaml
from git import Repo
from git.exc import InvalidGitRepositoryError
from nerfstudio.configs.base_config import InstantiateConfig, PrintableConfig
from nerfstudio.utils.rich_utils import CONSOLE
from torch.multiprocessing import Process

from rvs.evaluation.embedder import CachedEmbedder, EmbedderConfig
from rvs.evaluation.evaluation_method import evaluate_results
from rvs.evaluation.index import load_index
from rvs.evaluation.lvis import LVISDataset, Uid
from rvs.evaluation.pipeline import PipelineEvaluationInstance
from rvs.evaluation.worker import pipeline_worker_func
from rvs.pipeline.pipeline import PipelineConfig
from rvs.pipeline.stage import PipelineStage
from rvs.utils.config import find_changed_config_fields, find_config_working_dir, load_config
from rvs.utils.console import file_link
from rvs.utils.hash import hash_file_sha1
from rvs.utils.logging import create_logger
from rvs.utils.process import ProcessResult, start_process, stop_process


@dataclass
class RuntimeSettings(PrintableConfig):
    metadata: Optional[Dict[str, str]] = None
    """Additional metadata to be saved in the config"""

    stage_by_stage: Optional[bool] = None
    """Whether to process the objects stage-by-stage (i.e. first SAMPLE_VIEWS for all objects, then RENDER_VIEWS, etc.) instead of all stages object-by-object"""

    from_stage: Optional[PipelineStage] = None
    """If configured the pipeline is only run from this stage and no earlier"""

    to_stage: Optional[PipelineStage] = None
    """If configured the pipeline is only run up to this stage and no further"""

    run_limit: Optional[int] = None
    """Maximum number of pipeline runs before it aborts"""

    skip_finished: bool = False
    """Whether to skip re-running pipelines if final output is already available"""

    override_existing: bool = False
    """Whether existing evaluation config can be overwritten"""

    results_only: bool = False
    """Run results part only"""

    partial_results: bool = False
    """Run results part even if not all objects have been processed yet"""

    set_read_only: Optional[bool] = None
    """Sets the data to be read-only and exists immediately after if flag set"""

    threads: int = 1
    """Number of threads for running the pipelines or pipeline stages"""


@dataclass
class EvaluationConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Evaluation)

    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    """Configuration of the pipeline to use for the evaluation"""

    lvis_categories: Optional[Set[str]] = None
    """List of LVIS categories used in the evaluation (unconfigured = all)"""

    lvis_categories_file: Optional[Path] = None
    """Same as lvis_categories but loaded from the array in the specified .json file"""

    lvis_uids: Optional[Set[str]] = None
    """List of LVIS uids used in the evaluation (unconfigured = all)"""

    lvis_uids_file: Optional[Path] = None
    """Same as lvis_uids but loaded from the array in the specified .json file"""

    lvis_download_processes: int = 8
    """Number of processes to use for downloading the 3D model files"""

    lvis_per_category_limit: Optional[int] = None
    """Per category limit, useful for testing"""

    output_dir: Path = Path("outputs")
    """Relative or absolute output directory to save all output data"""

    inputs: Optional[List[Path]] = None
    """Inputs (paths to evaluation configs) to be used to load data from other evaluations for stages that haven't been processed in this evaluation. Sources later in the list take precendece over sources earlier in the list."""

    seed: int = 42
    """Seed used for random operations"""

    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    """Configuration of the CLIP embedder used for the precision/recall/accuracy evaluation"""

    embedder_cache_dir: Optional[Path] = None
    """Cache directory for embeddings"""

    embedder_cache_validation: bool = True
    """Whether to validate the cache by hash"""

    embedder_image_cache_required: bool = False
    """Whether to load images only from cache and abort if not cached"""

    embedder_text_cache_required: bool = False
    """Whether to load text only from cache and abort if not cached"""

    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    """Runtime settings that do not affect the results"""


class ConsoleLogger:
    __semaphore: Semaphore

    def __init__(self) -> None:
        self.__semaphore = Semaphore()

    def log(self, text: str) -> None:
        self.__semaphore.acquire()
        try:
            thread_name = threading.current_thread().getName()
            if thread_name.startswith("__"):
                thread_name = thread_name[2:]
            CONSOLE.log(f"[{thread_name}] {text}")
        finally:
            self.__semaphore.release()


@dataclass
class EvaluationRun:
    instance: PipelineEvaluationInstance
    progress_logger: Logger
    console_logger: ConsoleLogger


@dataclass
class PipelineRun:
    parent: EvaluationRun
    file: Path
    stages_filter: Optional[Set[PipelineStage]]


class Evaluation:
    config: EvaluationConfig
    config_base: EvaluationConfig

    lvis: LVISDataset
    """Objaverse 1.0 LVIS dataset"""

    lvis_cache_dir: Path
    """Directory for caching lvis dataset"""

    intermediate_dir: Path
    """Output directory for intermediate results"""

    results_dir: Path
    """Output directory for final results"""

    runs_dir: Path
    """Output directory for eval configs and logs"""

    embedder: Optional[CachedEmbedder] = None

    input_pipelines: Optional[List[PipelineEvaluationInstance]]

    def __init__(self, config: EvaluationConfig, overrides: Callable[[EvaluationConfig], EvaluationConfig] = None):
        self.config_base = config

        self.config = replace(self.config_base)
        if overrides is not None:
            self.config = replace(overrides(self.config))

        self.__set_nested_metadata_key(
            self.config,
            "validated",
            "tracking",
            self.__populate_tracking_metadata(
                self.__get_nested_metadata_key(self.config, "validated", "tracking", dict())
            ),
        )

        # Always persist metadata
        self.config_base.runtime.metadata = dict(self.config.runtime.metadata)

    def init(self) -> None:
        CONSOLE.rule("Initializing evaluation...")

        CONSOLE.log("Creating directories...")
        self.intermediate_dir = self.config.output_dir / "intermediate"
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = self.config.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.runs_dir = self.config.output_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self.lvis_cache_dir = self.config.output_dir / "lvis"
        self.lvis_cache_dir.mkdir(parents=True, exist_ok=True)

        CONSOLE.log("Setting up inputs...")
        self.input_pipelines = self.__setup_inputs()

        if self.config.runtime.partial_results or PipelineStage.OUTPUT in PipelineStage.between(
            self.config.runtime.from_stage, self.config.runtime.to_stage, default=PipelineStage.all()
        ):
            CONSOLE.log("Setting up embedder...")
            self.embedder = CachedEmbedder(
                self.config.embedder.setup(),
                self.config.embedder_cache_dir,
                validate_hash=self.config.embedder_cache_validation,
                image_cache_required=self.config.embedder_image_cache_required,
                text_cache_required=self.config.embedder_text_cache_required,
            )
        else:
            CONSOLE.log("Setting up embedder skipped...")

        CONSOLE.log("Setting up dataset...")

        lvis_categories = self.config.lvis_categories
        if self.config.lvis_categories_file is not None:
            if lvis_categories is None:
                lvis_categories = set()
            with self.config.lvis_categories_file.open("r") as f:
                lvis_categories = lvis_categories.union(set(json.load(f)))

        lvis_uids = self.config.lvis_uids
        if self.config.lvis_uids_file is not None:
            if lvis_uids is None:
                lvis_uids = set()
            with self.config.lvis_uids_file.open("r") as f:
                lvis_uids = lvis_uids.union(set(json.load(f)))

        self.lvis = LVISDataset(
            lvis_categories,
            lvis_uids,
            self.config.lvis_download_processes,
            self.config.lvis_per_category_limit,
        )

        if self.lvis.load_cache(self.lvis_cache_dir) is None:
            self.lvis.load()
            self.lvis.save_cache(self.lvis_cache_dir)

    def __populate_tracking_metadata(self, old_metadata: Dict[str, str]) -> Dict[str, str]:
        new_metadata: Dict[str, str] = dict()

        try:
            source_path = Path(__file__).resolve()
            with Repo(path=source_path.parent, search_parent_directories=True) as repo:
                hash_list: List[str] = None
                if "git_commit_hash" in old_metadata:
                    hash_list = old_metadata["git_commit_hash"]
                if not isinstance(hash_list, List):
                    hash_list = []
                hash = repo.head.object.hexsha
                if len(hash_list) == 0 or hash_list[-1] != hash:
                    hash_list.append(hash)
                new_metadata["git_commit_hash"] = hash_list
        except InvalidGitRepositoryError:
            pass

        if self.config.lvis_categories_file is not None:
            try:
                new_metadata["lvis_categories_file_hash"] = hash_file_sha1(self.config.lvis_categories_file)
            except FileNotFoundError:
                pass

        if self.config.lvis_uids_file is not None:
            try:
                new_metadata["lvis_uids_file_hash"] = hash_file_sha1(self.config.lvis_uids_file)
            except FileNotFoundError:
                pass

        return new_metadata

    def __setup_inputs(self) -> Optional[List[PipelineEvaluationInstance]]:
        if self.config.inputs is not None and len(self.config.inputs) > 0:
            input_pipelines: List[PipelineEvaluationInstance] = []

            for input_config_path in self.config.inputs:
                try:
                    input_pipelines.append(self.__setup_input(input_config_path))
                except Exception as ex:
                    err_msg = f'Invalid input "{input_config_path}": {str(ex)}'
                    CONSOLE.log(f"[bold red]ERROR: {err_msg}")
                    raise Exception(err_msg)

            return list(reversed(input_pipelines))
        else:
            return None

    def __setup_input(self, input_config_path: Path) -> Path:
        input_config = load_config(input_config_path, EvaluationConfig)
        working_dir = find_config_working_dir(input_config_path, input_config.output_dir)

        evaluation_dir = input_config.output_dir
        if working_dir is not None:
            evaluation_dir = working_dir / evaluation_dir

        evaluation_dir = evaluation_dir.resolve()

        if not evaluation_dir.is_absolute():
            raise Exception(f'Input evaluation path "{evaluation_dir}" is not absolute')

        if not evaluation_dir.exists():
            raise Exception(f'Input evaluation path "{evaluation_dir}" does not exist')

        intermediate_dir = evaluation_dir / "intermediate"

        if not intermediate_dir.exists():
            raise Exception(f'Input evaluation intermediate path "{intermediate_dir}" does not exist')

        input_eval: Evaluation = input_config.setup()

        instance = PipelineEvaluationInstance(
            Evaluation.__configure_pipeline(input_eval.config),
            intermediate_dir,
        )

        if not instance.pipeline_dir.exists():
            raise Exception(f'Input evaluation pipeline path "{instance.pipeline_dir}" does not exist')

        return instance

    def run(self) -> bool:
        CONSOLE.rule("Running evaluation...")

        saved_config = replace(self.config_base)

        self.__set_nested_metadata_key(saved_config, "run", "timestamp", datetime.now().strftime("%Y-%m-%d_%H%M%S"))

        if self.config.runtime.set_read_only is not None:
            self.__set_nested_metadata_key(saved_config, "mode", "read_only", self.config.runtime.set_read_only)

        CONSOLE.log("Validating config...")
        self.__validate_eval_config(saved_config, self.config.output_dir / "config.yaml")

        run_dir = self.__create_run_dir()

        CONSOLE.log("Saving config...")
        self.__save_eval_config(saved_config, [self.config.output_dir / "config.yaml", run_dir / "config.yaml"])

        if self.config.runtime.set_read_only is not None:
            CONSOLE.log(f"Data read-only mode set to {str(self.config.runtime.set_read_only)}...")
            return False

        is_read_only = self.config.runtime.set_read_only or self.__get_nested_metadata_key(
            self.config, "mode", "read_only", False
        )

        if is_read_only and not self.config.runtime.results_only:
            CONSOLE.log(
                "[bold red]ERROR: Data is read-only. Disable with runtime.set_read_only=False or use runtime.results_only=True."
            )
            return False

        CONSOLE.log("Starting runs...")
        with create_logger(
            __name__, files=[self.config.output_dir / "progress.log", run_dir / "progress.log"]
        ) as progress_logger_handle:
            stages = PipelineStage.between(
                self.config.runtime.from_stage, self.config.runtime.to_stage, default=PipelineStage.all()
            )

            run = EvaluationRun(
                instance=PipelineEvaluationInstance(
                    Evaluation.__configure_pipeline(self.config),
                    self.intermediate_dir,
                    stages=stages,
                    input_pipelines=self.input_pipelines,
                ),
                progress_logger=progress_logger_handle.logger,
                console_logger=ConsoleLogger(),
            )

            run.instance.init()

            aborted = False

            if not self.config.runtime.results_only:
                assert not is_read_only
                if self.config.runtime.stage_by_stage:
                    aborted = not self.__run_stage_by_stage(run)
                else:
                    aborted = not self.__run_object_by_object(run)

            if (not aborted or self.config.runtime.partial_results) and self.embedder is not None:
                evaluate_results(
                    self.lvis,
                    self.embedder,
                    run.instance,
                    self.config.seed,
                    self.results_dir,
                )

            return not aborted

    def __create_run_dir(self) -> Path:
        base_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = self.runs_dir / base_name
        if run_dir.exists():
            i = 2
            while True:
                run_dir = self.runs_dir / (base_name + "_" + str(i))
                if not run_dir.exists():
                    break
                i += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def __save_eval_config(self, config: EvaluationConfig, files: Union[Path, List[Path]]) -> None:
        if isinstance(files, Path):
            files = [files]
        cfg = yaml.dump(config)
        for file in files:
            file.parent.mkdir(parents=True, exist_ok=True)
            CONSOLE.log(f"Saving evaluation config to: {file_link(file)}")
            file.write_text(cfg, "utf8")

    def __validate_eval_config(self, config: EvaluationConfig, file: Path) -> None:
        fail_on_difference = not self.config.runtime.override_existing

        prefix = "[bold yellow]WARNING: "
        if fail_on_difference:
            prefix = "[bold red]ERROR: "

        if file.exists():
            new_config, metadata_removed = self.__strip_unvalidated_metadata(config)

            if new_config.runtime.metadata is None:
                new_config.runtime.metadata = dict()
            else:
                new_config.runtime.metadata = dict(new_config.runtime.metadata)

            if metadata_removed:
                new_config.runtime.metadata["<...>"] = "<...>"

            matches = True

            existing_config: EvaluationConfig = None
            try:
                existing_config = load_config(file, EvaluationConfig, default_config_factory=None)
                # Runtime data should be ignored in comparison except for validated metadata
                # so replace with new metadata and keep old validated metadata if it exists
                existing_validated_metadata = self.__get_nested_metadata(existing_config, "validated")
                existing_config.runtime = replace(new_config.runtime)
                existing_config.runtime.metadata = dict(new_config.runtime.metadata)
                self.__set_nested_metadata(existing_config, "validated", existing_validated_metadata)
            except Exception as ex:
                matches = False
                err_msg = (
                    f'Unable to validate existing config "{file}" Run with runtime.override_existing=True to override.'
                )
                CONSOLE.log(f"{prefix}{err_msg}")
                if fail_on_difference:
                    raise Exception(err_msg) from ex

            if existing_config is not None:

                def on_missing_field(fpath, obj, fname, expected_value):
                    CONSOLE.log(f"{prefix}Found missing field {fpath}, old value: {expected_value}")
                    nonlocal matches
                    matches = False

                def on_changed_field(fpath, obj, fname, expected_value, actual_value):
                    CONSOLE.log(
                        f"{prefix}Found changed field {fpath}, old value: {expected_value}, new value: {actual_value}"
                    )
                    nonlocal matches
                    matches = False

                def on_unknown_field(fpath, obj, fname, actual_value):
                    CONSOLE.log(f"{prefix}Found unexpected field {fpath}, old value: N/A, new value: {actual_value}")
                    nonlocal matches
                    matches = False

                find_changed_config_fields(
                    expected_config=existing_config,
                    config=new_config,
                    on_missing_field=on_missing_field,
                    on_changed_field=on_changed_field,
                    on_unknown_field=on_unknown_field,
                )

            if matches:
                try:
                    if existing_config != new_config:
                        matches = False
                except (AttributeError, KeyError):
                    matches = False

            if not matches:
                err_msg = f'Existing config "{file}" does not match.'
                if fail_on_difference:
                    err_msg += " Run with runtime.override_existing=True to override."
                CONSOLE.log(f"{prefix}{err_msg}")
                if fail_on_difference:
                    raise Exception(err_msg)

    def __strip_unvalidated_metadata(self, config: EvaluationConfig) -> Tuple[EvaluationConfig, bool]:
        runtime = replace(config.runtime)
        config = replace(config, runtime=runtime)

        removed = False

        if config.runtime.metadata is not None:
            stripped_metadata = dict(config.runtime.metadata)

            for key in list(stripped_metadata.keys()):
                if key != "validated":
                    del stripped_metadata[key]
                    removed = True

            config.runtime.metadata = stripped_metadata

        return (config, removed)

    def __get_nested_metadata(self, config: EvaluationConfig, group: str) -> Dict[str, str]:
        if config.runtime.metadata is not None and group in config.runtime.metadata:
            nested_metadata = config.runtime.metadata[group]
            if isinstance(nested_metadata, Dict):
                return nested_metadata
        return dict()

    T = TypeVar("T")

    def __get_nested_metadata_key(self, config: EvaluationConfig, group: str, key: str, default: T) -> T:
        metadata = self.__get_nested_metadata(config, group)
        if key in metadata:
            value = metadata[key]
            if isinstance(value, type(default)):
                return value
        return default

    def __set_nested_metadata(self, config: EvaluationConfig, group: str, metadata: Any) -> None:
        if config.runtime.metadata is None:
            config.runtime.metadata = dict()
        config.runtime.metadata[group] = metadata

    def __set_nested_metadata_key(self, config: EvaluationConfig, group: str, key: str, metadata: Any) -> None:
        nested_metadata = self.__get_nested_metadata(config, group)
        nested_metadata[key] = metadata
        if config.runtime.metadata is None:
            config.runtime.metadata = dict()
        config.runtime.metadata[group] = nested_metadata

    def __run_stage_by_stage(self, run: EvaluationRun) -> bool:
        num_queued_runs = 0

        num_runs = 0
        num_runs_semaphore = Semaphore()

        def incr_num_runs() -> int:
            nonlocal run
            nonlocal num_runs_semaphore
            nonlocal num_runs
            nonlocal total_runs

            local_num_runs = 0

            num_runs_semaphore.acquire()
            try:
                num_runs += 1
                local_num_runs = num_runs
            finally:
                num_runs_semaphore.release()

            run.progress_logger.info(
                f"Progress: {local_num_runs} / {total_runs} ({'{:.2f}'.format(float(local_num_runs) / total_runs * 100.0)}%)"
            )

        total_runs = 0
        for stage in run.instance.stages:
            for category in self.lvis.dataset.keys():
                total_runs += len(self.lvis.dataset[category])

        for stage in run.instance.stages:
            CONSOLE.log(f"Processing stage {str(stage)}...")

            for category in self.lvis.dataset.keys():
                CONSOLE.log(f"Processing category {category}...")

                is_partial = False
                uids: List[Uid] = []
                for uid in self.lvis.dataset[category]:
                    if self.config.runtime.run_limit is not None and num_queued_runs >= self.config.runtime.run_limit:
                        is_partial = True
                        break
                    uids.append(uid)
                    num_queued_runs += 1

                def process_uid(uid: Path) -> None:
                    file = Path(self.lvis.uid_to_file[uid])

                    run.console_logger.log(f"Processing uid {uid} ({file_link(file)})...")

                    self.__run_pipeline(
                        PipelineRun(
                            parent=run,
                            file=file,
                            stages_filter={stage},
                        )
                    )

                    incr_num_runs()

                with ThreadPoolExecutor(max_workers=self.config.runtime.threads, thread_name_prefix="_") as executor:
                    deque(executor.map(process_uid, uids), maxlen=0)  # Consume generator

                if is_partial:
                    return False

        return True

    def __run_object_by_object(self, run: EvaluationRun) -> bool:
        num_queued_runs = 0

        num_runs = 0
        num_runs_semaphore = Semaphore()

        def incr_num_runs() -> int:
            nonlocal run
            nonlocal num_runs_semaphore
            nonlocal num_runs
            nonlocal total_runs

            local_num_runs = 0

            num_runs_semaphore.acquire()
            try:
                num_runs += 1
                local_num_runs = num_runs
            finally:
                num_runs_semaphore.release()

            run.progress_logger.info(
                f"Progress: {local_num_runs} / {total_runs} ({'{:.2f}'.format(float(local_num_runs) / total_runs * 100.0)}%)"
            )

        total_runs = 0
        for category in self.lvis.dataset.keys():
            total_runs += len(self.lvis.dataset[category])

        for category in self.lvis.dataset.keys():
            CONSOLE.log(f"Processing category {category}...")

            is_partial = False
            uids: List[Uid] = []
            for uid in self.lvis.dataset[category]:
                if self.config.runtime.run_limit is not None and num_queued_runs >= self.config.runtime.run_limit:
                    is_partial = True
                    break
                uids.append(uid)
                num_queued_runs += 1

            def process_uid(uid: Path) -> None:
                file = Path(self.lvis.uid_to_file[uid])

                run.console_logger.log(f"Processing uid {uid} ({file_link(file)})...")

                self.__run_pipeline(
                    PipelineRun(
                        parent=run,
                        file=file,
                        stages_filter=None,
                    )
                )

                incr_num_runs()

            with ThreadPoolExecutor(max_workers=self.config.runtime.threads, thread_name_prefix="_") as executor:
                deque(executor.map(process_uid, uids), maxlen=0)  # Consume generator

            if is_partial:
                return False

        return True

    def __run_pipeline(
        self,
        run: PipelineRun,
        handle_errors: bool = True,
    ) -> bool:
        run_stages = run.parent.instance.get_pipeline_stages(run.stages_filter)
        stages_str = "..." if run_stages is None else ", ".join([stage.name for stage in run_stages])
        pipeline_str = f'"{self.__safe_resolve(run.file)}" -> [{stages_str}] -> "{self.__safe_resolve(run.parent.instance.get_pipeline_dir(run.file))}" -> "{self.__safe_resolve(run.parent.instance.get_results_dir(run.file))}"'

        skip = False

        if self.config.runtime.skip_finished:
            try:
                index_file = run.parent.instance.get_index_file(run.file)
                if index_file.exists():
                    load_index(index_file, validate=True)
                    skip = True  # If index exists and is valid then pipeline has finished
            except Exception:
                run.parent.progress_logger.warning(
                    "Index of pipeline %s is invalid:\n%s", pipeline_str, traceback.format_exc()
                )

        if skip:
            run.parent.progress_logger.info("Skipping pipeline %s", pipeline_str)
            return False

        with ProcessResult() as result:
            run.parent.progress_logger.info("Starting pipeline %s", pipeline_str)

            process: Process = None
            try:
                stdout_file: Optional[Path] = None
                stderr_file: Optional[Path] = None

                if self.config.runtime.threads > 1:
                    pipeline_dir = run.parent.instance.get_pipeline_dir(run.file)
                    pipeline_dir.mkdir(parents=True, exist_ok=True)

                    stdout_file = str((pipeline_dir / "stdout.log").resolve())
                    stderr_file = str((pipeline_dir / "stderr.log").resolve())

                process = start_process(
                    target=pipeline_worker_func,
                    args=(
                        run.parent.instance,
                        run.file,
                        run.stages_filter,
                        result,
                        stdout_file,
                        stderr_file,
                    ),
                )
                process.join()
            except Exception as ex:
                if handle_errors:
                    result.success = False
                    result.msg = traceback.format_exc()
                else:
                    raise ex
            finally:
                try:
                    stop_process(process)
                    process.join()
                except Exception:
                    pass

                try:
                    process.close()
                except Exception:
                    pass

                if result.success:
                    run.parent.progress_logger.info("Finished pipeline %s", pipeline_str)
                elif result.msg is not None:
                    run.parent.progress_logger.error(
                        "Failed pipeline %s due to exception:\n%s", pipeline_str, result.msg
                    )
                else:
                    run.parent.progress_logger.error("Failed pipeline %s due to unknown reason", pipeline_str)

            return result.success

    def __safe_resolve(self, file: Path) -> str:
        try:
            return str(file.resolve())
        except Exception:
            return str(file)

    @staticmethod
    def __configure_pipeline(eval_config: EvaluationConfig) -> PipelineConfig:
        pipeline_config = replace(eval_config.pipeline)
        pipeline_config.project_name = "evaluation"
        pipeline_config.experiment_name = "evaluation"
        pipeline_config.method_name = "evaluation"
        pipeline_config.timestamp = "evaluation"
        pipeline_config.machine.seed = eval_config.seed
        return pipeline_config


@dataclass
class EvaluationResumeConfig(PrintableConfig):
    config: Path
    """Path to config file to resume"""

    runtime: RuntimeSettings = field(
        default_factory=lambda: RuntimeSettings(
            # Skip finished pipelines by default since we're resuming
            skip_finished=True,
        )
    )
    """Runtime settings that do not affect the results"""

    def load(self) -> Tuple[EvaluationConfig, Callable[[EvaluationConfig], EvaluationConfig], Optional[Path]]:
        if self.config is None:
            raise ValueError("No config path specified")

        if not self.config.exists() or not self.config.is_file():
            raise ValueError(f'Config file "{self.config}" does not exist')

        config = load_config(
            self.config,
            EvaluationConfig,
            on_default_applied=lambda fpath, obj, fname, value: CONSOLE.log(
                f"[bold yellow]WARNING: Applied default value to missing field {fpath}: {value}"
            ),
        )

        working_dir = find_config_working_dir(self.config, config.output_dir)

        return (config, self.__apply_config_overrides, working_dir)

    def __apply_config_overrides(self, config: EvaluationConfig) -> EvaluationConfig:
        runtime = replace(self.runtime)

        # Keep these settings from saved config as defaults

        if runtime.metadata is None:
            runtime.metadata = config.runtime.metadata

        if runtime.stage_by_stage is None:
            runtime.stage_by_stage = config.runtime.stage_by_stage

        if runtime.from_stage is None:
            runtime.from_stage = config.runtime.from_stage

        if runtime.to_stage is None:
            runtime.to_stage = config.runtime.to_stage

        return replace(config, runtime=runtime)
