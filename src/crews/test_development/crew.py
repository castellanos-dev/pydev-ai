from __future__ import annotations
from typing import List
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from ...utils.routing import llms
from .output_format.generate_tests import GenerateTestsOutput


@CrewBase
class JuniorTestDevelopmentCrew:
    """
    Crew for generating unit tests based on the project design and code.
    Does not perform writes: the flow adds deterministic steps to write.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        self.llm_light = llms()["light"]
        self.llm_medium = llms()["medium"]
        self.llm_reasoning = llms()["reasoning"]

        self.llm_test_writer = llms()["light"]

    # Agents
    @agent
    def test_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["test_generator"],  # type: ignore[index]
            llm=self.llm_test_writer,
            verbose=True,
        )

    # Tasks
    @task
    def generate_tests(self) -> Task:
        return Task(
            config=self.tasks_config["generate_tests"],  # type: ignore[index]
            output_json=GenerateTestsOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=[
                self.generate_tests(),
            ],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )


class SeniorTestDevelopmentCrew(JuniorTestDevelopmentCrew):
    def __init__(self):
        super().__init__()
        self.llm_test_writer = llms()["medium"]


class LeadTestDevelopmentCrew(JuniorTestDevelopmentCrew):
    def __init__(self):
        super().__init__()
        self.llm_test_writer = llms()["reasoning"]
