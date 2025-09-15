from __future__ import annotations

from typing import List
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from ...utils.routing import llms
from .output_format.rename_map import RenameMapOutput


@CrewBase
class RenameMappingCrew:
    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self) -> None:
        self.llm_light = llms()["light"]

    @agent
    def mapper(self) -> Agent:
        return Agent(
            config=self.agents_config["mapper"],
            llm=self.llm_light,
            verbose=True,
        )

    @task
    def produce_rename_map(self) -> Task:
        return Task(
            config=self.tasks_config["produce_rename_map"],
            output_json=RenameMapOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.mapper()],
            tasks=[self.produce_rename_map()],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )
