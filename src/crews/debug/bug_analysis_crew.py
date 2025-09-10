from __future__ import annotations
from crewai import Task, Crew, Process
from crewai.project import crew, task
from .crew import BaseDebugCrew
from .output_format.bug_analysis import AnalyzeTestFailuresOutput


class BugAnalysisCrew(BaseDebugCrew):
    """
    Crew focused on analyzing bugs in the code.
    """
    original_tasks_config_path = "config/tasks_bug_analysis.yaml"

    @task
    def analyze_test_failures(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_test_failures"],  # type: ignore[index]
            output_json=AnalyzeTestFailuresOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.analyst()],
            tasks=[
                self.analyze_test_failures(),
            ],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )
