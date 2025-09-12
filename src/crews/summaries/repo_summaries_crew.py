from __future__ import annotations
from crewai import Crew, Process, Task
from crewai.project import crew, task
from crewai.project import crew
from .crew import BaseSummariesCrew
from .output_format.summaries import SummariesOutput


class RepoSummariesCrew(BaseSummariesCrew):
    """
    Crew that generates Markdown summaries for existing repositories without a prior
    design spec. It reads code chunks and produces:
      - Per-file summaries (excluding __init__.py)
      - Per-module overviews per folder
    """

    original_tasks_config_path = "config/from_repo_tasks.yaml"

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
