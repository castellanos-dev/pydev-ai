from __future__ import annotations
from typing import List
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from ...utils.routing import llms


@CrewBase
class JSONFixerCrew:
    """
    Crew that receives a broken JSON string and context, and returns a fixed JSON string
    that strictly adheres to the expected schema for downstream parsers.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        self.llm_light = llms()["light"]
        self.llm_reasoning = llms()["reasoning"]

    @agent
    def json_doctor(self) -> Agent:
        return Agent(
            config=self.agents_config["json_doctor"],  # type: ignore[index]
            llm=self.llm_light,
            verbose=True,
        )

    @task
    def fix_json(self) -> Task:
        return Task(config=self.tasks_config["fix_json"])  # type: ignore[index]

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.json_doctor()],
            tasks=[self.fix_json()],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )
