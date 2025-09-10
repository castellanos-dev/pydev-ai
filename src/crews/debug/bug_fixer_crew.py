from __future__ import annotations
from crewai import Task, Crew, Process
from crewai.project import crew, task
from .crew import BaseDebugCrew
from .output_format.bug_fixes import ImplementBugFixesOutput


class BaseBugFixerCrew(BaseDebugCrew):
    """
    Crew responsible for implementing minimal fixes for a specific bug.
    Input is a single bug from the bug analysis and the output is a file map with patches.
    """
    original_tasks_config_path = "config/tasks_bug_fixer.yaml"

    @task
    def implement_bug_fixes(self) -> Task:
        return Task(
            config=self.tasks_config["implement_bug_fixes"],  # type: ignore[index]
            output_json=ImplementBugFixesOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.bug_fixer()],
            tasks=[self.implement_bug_fixes()],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )


class JuniorBugFixerCrew(BaseBugFixerCrew):
    def __init__(self):
        super().__init__()
        self.llm_bug_fixer = self.llm_light


class SeniorBugFixerCrew(BaseBugFixerCrew):
    def __init__(self):
        super().__init__()
        self.llm_bug_fixer = self.llm_medium


class LeadBugFixerCrew(BaseBugFixerCrew):
    def __init__(self):
        super().__init__()
        self.llm_bug_fixer = self.llm_reasoning


def bug_fixer_for_points(points: int) -> BaseBugFixerCrew:
    """Select crew according to the bug's points.

    - 1 point: JuniorBugFixerCrew
    - 2 points: SeniorBugFixerCrew
    - 3 points or more: LeadBugFixerCrew
    """
    if points <= 1:
        return JuniorBugFixerCrew()
    if points == 2:
        return SeniorBugFixerCrew()
    return LeadBugFixerCrew()
