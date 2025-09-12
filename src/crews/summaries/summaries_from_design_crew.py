from __future__ import annotations
from crewai import Crew, Process, Task
from crewai.project import crew, task
from crewai.project import crew
from .crew import BaseSummariesCrew
from .output_format.summaries import SummariesOutput


class SummariesFromDesignCrew(BaseSummariesCrew):
    """
    Crew that generates high-signal Markdown summaries for files and packages
    based on project design and the resulting source code of a chunk.

    The output is a compact JSON string describing summaries for files
    (excluding `__init__.py`) and packages. The flow consumes it and writes
    Markdown files under the repository's digests directory.
    """

    original_tasks_config_path = "config/from_design_tasks.yaml"

    @task
    def summarize_chunk(self) -> Task:
        return Task(
            config=self.tasks_config["summarize_chunk"],
            output_json=SummariesOutput,
        )

    @task
    def summarize_modules(self) -> Task:
        return Task(
            config=self.tasks_config["summarize_modules"],
            output_json=SummariesOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.summarizer(), self.module_summarizer()],
            tasks=[self.summarize_chunk(), self.summarize_modules()],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )
