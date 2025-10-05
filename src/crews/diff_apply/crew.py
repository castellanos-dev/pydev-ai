from __future__ import annotations
from typing import List
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from ...utils.routing import llms
from .output_format.full_file import FullFileOutput


@CrewBase
class DiffApplyCrew:
    """
    LLM fallback to convert a unified diff and original content into final file content.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self) -> None:
        self.llm_light = llms()["light"]

    @agent
    def diff_apply_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["diff_apply_engineer"],  # type: ignore[index]
            llm=self.llm_light,
            verbose=True,
        )

    @task
    def apply_unified_diff(self) -> Task:
        return Task(
            config=self.tasks_config["apply_unified_diff"],  # type: ignore[index]
            output_json=FullFileOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.diff_apply_engineer()],
            tasks=[self.apply_unified_diff()],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )
