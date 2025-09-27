from __future__ import annotations
from crewai import Crew, Process, Task
from crewai.project import crew, task
from .crew import BaseSummariesCrew
from .output_format.summaries import FileSummariesOutput


class FileSummariesCrew(BaseSummariesCrew):
    """
    Specialized crew that only generates per-file summaries.
    Uses the same agents and task definitions but executes only the file task.
    """

    original_tasks_config_path = "config/file_summaries_task.yaml"

    @task
    def summarize_chunk(self) -> Task:
        return Task(
            config=self.tasks_config["summarize_chunk"],
            output_json=FileSummariesOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.summarizer()],
            tasks=[self.summarize_chunk()],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )
