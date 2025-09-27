from __future__ import annotations
from typing import List
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from ...utils.routing import llms


@CrewBase
class TestsIntegratorCrew:
    """
    Advanced crew to integrate newly generated test snippets into existing test files.

    Responsibilities:
    - Merge new tests into existing classes/modules when appropriate
    - Reuse existing fixtures from conftest.py or local fixtures
    - Keep framework/style consistent (pytest/unittest)
    - Output the full, final content of the target test file
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        # Use a stronger model for code integration
        self.llm_medium = llms()["medium"]

    @agent
    def tests_integrator(self) -> Agent:
        return Agent(
            config=self.agents_config["tests_integrator"],  # type: ignore[index]
            llm=self.llm_medium,
            verbose=True,
        )

    @task
    def integrate_tests(self) -> Task:
        return Task(config=self.tasks_config["integrate_tests"])  # type: ignore[index]

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.tests_integrator()],
            tasks=[self.integrate_tests()],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )
