from __future__ import annotations
from typing import List
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from ...utils.routing import llms
from .output_format.doc_unified_diff import DocUnifiedDiffOutput


@CrewBase
class DocsDiffCrew:
    """
    Generate unified diffs for documentation files that need updates, given the
    selected doc paths, current doc contents, user prompt and action plan.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self) -> None:
        self.llm_light = llms()["light"]

    @agent
    def docs_diff_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["docs_diff_engineer"],  # type: ignore[index]
            llm=self.llm_light,
            verbose=True,
        )

    @task
    def generate_docs_diffs(self) -> Task:
        return Task(
            config=self.tasks_config["generate_docs_diffs"],  # type: ignore[index]
            output_json=DocUnifiedDiffOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.docs_diff_engineer()],
            tasks=[self.generate_docs_diffs()],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )
