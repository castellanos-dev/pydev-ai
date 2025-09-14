from __future__ import annotations
from crewai import Task, Crew, Process
from crewai.project import CrewBase, crew, task
from .output_format.summaries_dir import SummariesDirOutput
from .crew import BaseSummariesCrew


class SummariesDirCrew(BaseSummariesCrew):
    """
    Crew that decides the summaries directory given src, docs and tests.
    """

    original_tasks_config_path = "config/summaries_dir_tasks.yaml"

    @task
    def locate_summaries_dir(self) -> Task:
        return Task(
            config=self.tasks_config["locate_summaries_dir"],
            output_json=SummariesDirOutput,
        )

    @crew
    def crew(self) -> Crew:
        # Single-task crew; task configured in tasks YAML
        return Crew(
            agents=[self.locator()],
            tasks=[self.locate_summaries_dir()],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )
