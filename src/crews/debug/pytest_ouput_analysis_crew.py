from __future__ import annotations
from crewai import Task, Crew, Process
from ...flows.utils import is_something_to_fix
from crewai.tasks.conditional_task import ConditionalTask
from crewai.project import crew, task
from .crew import BaseDebugCrew
from .output_format.pytest_output import GroupFailuresByRootCause


class PytestOutputAnalysisCrew(BaseDebugCrew):
    """
    Crew focused on analyzing the pytest output.
    """
    original_tasks_config_path = "config/tasks_pytest_output.yaml"

    # Tasks
    @task
    def parse_pytest_output(self) -> Task:
        return Task(config=self.tasks_config["parse_pytest_output"])

    @task
    def group_failures_by_root_cause(self) -> Task:
        return ConditionalTask(
            config=self.tasks_config["group_failures_by_root_cause"],
            condition=is_something_to_fix,
            output_json=GroupFailuresByRootCause,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.reporter(), self.grouper()],
            tasks=[
                self.parse_pytest_output(),
                self.group_failures_by_root_cause(),
            ],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )
