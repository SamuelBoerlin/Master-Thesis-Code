import os
import re
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Semaphore
from typing import List

import tyro
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.utils.rich_utils import CONSOLE

from rvs.evaluation.embedder import CachedEmbedder, Embedder, EmbedderConfig
from rvs.evaluation.evaluation import EvaluationConfig
from rvs.evaluation.evaluation_method import category_name_to_embedding_prompt
from rvs.evaluation.lvis import create_dataset
from rvs.pipeline.pipeline import PipelineConfig
from rvs.utils.cache import get_evaluation_prompt_embedding_cache_key, get_pipeline_render_embedding_cache_key
from rvs.utils.config import find_config_working_dir, load_config
from rvs.utils.console import file_link


@dataclass
class Command:
    embedder: EmbedderConfig = field(default_factory=lambda: EmbedderConfig())

    def run(self) -> None:
        embedder = self.embedder.setup()
        self._run(embedder)

    def _run(self, embedder: Embedder) -> None:
        pass


@dataclass
class TextCommand(Command):
    text: str = tyro.MISSING

    output_file: Path = tyro.MISSING

    def _run(self, embedder: Embedder) -> None:
        CachedEmbedder.create_text_cache_file(
            self.output_file, self.text, self.embedder, embedder.embed_text_numpy(self.text)
        )


@dataclass
class ImageCommand(Command):
    file: Path = tyro.MISSING

    output_file: Path = tyro.MISSING

    def _run(self, embedder: Embedder) -> None:
        CachedEmbedder.create_image_cache_file(
            self.output_file, self.file, self.embedder, embedder.embed_image_numpy(self.file)
        )


@dataclass
class PipelineRendersCommand(Command):
    config: Path = tyro.MISSING

    image_name_pattern: str = "^frame_\d{5}.png$"

    threads: int = 4

    output_dir: Path = tyro.MISSING

    def _run(self, embedder: Embedder) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        PipelineRendersCommand.embed_pipeline_renders(
            self.config,
            self.image_name_pattern,
            embedder,
            self.output_dir,
            self.threads,
        )

    @staticmethod
    def embed_pipeline_renders(
        config_file: Path,
        image_name_pattern: str,
        embedder: Embedder,
        output_dir: Path,
        threads: int,
    ) -> None:
        pipeline_config = load_config(config_file, PipelineConfig)

        files: List[Path] = []

        renders_dir = (
            PipelineRendersCommand.__get_pipeline_base_dir(config_file, pipeline_config) / "renderer" / "images"
        )

        if renders_dir.exists() and renders_dir.is_dir():
            for file in renders_dir.iterdir():
                if file.is_file() and re.match(image_name_pattern, file.name):
                    files.append(file)

        console_semaphore = Semaphore()

        def log(text: str) -> None:
            console_semaphore.acquire()
            try:
                thread_name = threading.current_thread().getName()
                if thread_name.startswith("__"):
                    thread_name = thread_name[2:]
                CONSOLE.log(f"[{thread_name}] {text}")
            finally:
                console_semaphore.release()

        def embed(file: Path) -> None:
            cache_object_file = (
                output_dir / f"{get_pipeline_render_embedding_cache_key(pipeline_config.model_file, file)}.json"
            )

            CachedEmbedder.create_image_cache_file(
                cache_object_file, file, embedder.config, embedder.embed_image_numpy(file)
            )

            log(f"Embedded {file_link(file)} -> {file_link(cache_object_file)}")

        with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="_") as executor:
            deque(executor.map(embed, files), maxlen=0)  # Consume generator

    @staticmethod
    def __get_pipeline_base_dir(config_file: Path, config: PipelineConfig) -> Path:
        working_dir = os.getcwd()

        try:
            pipeline_working_dir = find_config_working_dir(config_file, config.output_dir)

            if pipeline_working_dir is not None:
                os.chdir(pipeline_working_dir)

            return config.get_base_dir().resolve()
        finally:
            os.chdir(working_dir)


@dataclass
class EvaluationRendersCommand(Command):
    config: Path = tyro.MISSING

    image_name_pattern: str = "^frame_\d{5}.png$"

    threads: int = 4

    output_dir: Path = tyro.MISSING

    def _run(self, embedder: Embedder) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        eval_config = load_config(self.config, EvaluationConfig)

        pipelines_dir = self.__get_evaluation_base_dir(eval_config) / "intermediate" / "pipeline"

        for dir in pipelines_dir.iterdir():
            if dir.is_dir():
                pipeline_config = dir / "evaluation" / "evaluation" / "evaluation" / "config.yml"

                CONSOLE.rule(dir.name)

                PipelineRendersCommand.embed_pipeline_renders(
                    pipeline_config,
                    self.image_name_pattern,
                    embedder,
                    self.output_dir,
                    self.threads,
                )

    def __get_evaluation_base_dir(self, config: EvaluationConfig) -> Path:
        working_dir = os.getcwd()

        try:
            eval_working_dir = find_config_working_dir(self.config, config.output_dir)

            if eval_working_dir is not None:
                os.chdir(eval_working_dir)

            return config.output_dir.resolve()
        finally:
            os.chdir(working_dir)


@dataclass
class EvaluationCategoryPromptsCommand(Command):
    config: Path = tyro.MISSING

    output_dir: Path = tyro.MISSING

    def _run(self, embedder: Embedder) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        eval_config = load_config(self.config, EvaluationConfig)

        lvis = create_dataset(
            lvis_categories=eval_config.lvis_categories,
            lvis_categories_file=eval_config.lvis_categories_file,
            lvis_uids=eval_config.lvis_uids,
            lvis_uids_file=eval_config.lvis_uids_file,
            lvis_download_processes=eval_config.lvis_download_processes,
            lvis_per_category_limit=eval_config.lvis_per_category_limit,
            lvis_category_names=eval_config.lvis_category_names,
            lvis_category_names_file=eval_config.lvis_category_names_file,
        )

        for category in lvis.categories:
            prompt = category_name_to_embedding_prompt(category, lvis)

            cache_object_file = self.output_dir / f"{get_evaluation_prompt_embedding_cache_key(prompt)}.json"

            CachedEmbedder.create_text_cache_file(
                cache_object_file, prompt, embedder.config, embedder.embed_text_numpy(prompt)
            )

            CONSOLE.log(f"Embedded '{prompt}' (category: {category}) -> {file_link(cache_object_file)}")


commands = {
    "text": TextCommand(),
    "image": ImageCommand(),
    "pipeline_renders": PipelineRendersCommand(),
    "evaluation_renders": EvaluationRendersCommand(),
    "evaluation_category_prompts": EvaluationCategoryPromptsCommand(),
}

SubcommandTypeUnion = tyro.conf.SuppressFixed[
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(
            defaults=commands, descriptions={key: type(command).__doc__ for key, command in commands.items()}
        )
    ]
]


def main(cmd: Command):
    cmd.run()


def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            SubcommandTypeUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    entrypoint()
