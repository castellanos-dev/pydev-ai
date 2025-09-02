from __future__ import annotations
from typing import List
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from ...utils.routing import llms


@CrewBase
class FixIntegratorCrew:
    """
    Crew responsible for integrating multiple partial fixes into full file contents.
    It takes the generated code files and the list of file-level fixes, and outputs
    a final list of files with integrated contents.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        self.llm_light = llms()["light"]
        self.llm_reasoning = llms()["reasoning"]

    @agent
    def fix_integrator(self) -> Agent:
        return Agent(
            config=self.agents_config["fix_integrator"],  # type: ignore[index]
            llm=self.llm_light,
            verbose=True,
        )

    @task
    def integrate_fixes(self) -> Task:
        return Task(config=self.tasks_config["integrate_fixes"])  # type: ignore[index]

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.fix_integrator()],
            tasks=[self.integrate_fixes()],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )
