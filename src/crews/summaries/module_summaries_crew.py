from __future__ import annotations
from crewai import Crew, Process, Task
from crewai.project import crew, task
from .crew import BaseSummariesCrew
from .output_format.summaries import ModuleSummariesOutput


class ModuleSummariesCrew(BaseSummariesCrew):
    """
    Specialized crew that only generates per-module summaries.
    Expects the input chunk to contain per-file summaries as content.
    """

    original_tasks_config_path = "config/module_task.yaml"

    @task
    def summarize_modules(self) -> Task:
        return Task(
            config=self.tasks_config["summarize_modules"],
            output_json=ModuleSummariesOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.module_summarizer()],
            tasks=[self.summarize_modules()],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )
