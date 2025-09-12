from __future__ import annotations
from typing import List
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from ...utils.routing import llms
from .output_format.project_structure import ProjectStructureOutput


@CrewBase
class ProjectStructureCrew:
    """
    Crew that identifies code, docs, and tests locations from a list of files.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        self.llm_light = llms()["light"]

    @agent
    def structure_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["structure_analyst"],
            llm=self.llm_light,
            verbose=True,
        )

    @task
    def analyze_project_structure(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_project_structure"],
            output_json=ProjectStructureOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.structure_analyst()],
            tasks=[self.analyze_project_structure()],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )
