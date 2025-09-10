from __future__ import annotations
from crewai import Task, Crew, Process
from crewai.project import crew, task
from .crew import BaseDebugCrew
from .output_format.analyze_involved_files import AnalyzeInvolvedFilesOutput


class AnalyzeInvolvedFilesCrew(BaseDebugCrew):
    """
    Crew focused on analyzing the involved files.
    """
    original_tasks_config_path = "config/tasks_analyze_involved_files.yaml"

    @task
    def analyze_involved_files(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_involved_files"],  # type: ignore[index]
            output_json=AnalyzeInvolvedFilesOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.analyst()],
            tasks=[
                self.analyze_involved_files(),
            ],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )
