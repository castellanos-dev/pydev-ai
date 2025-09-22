from __future__ import annotations
from typing import List
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from ...utils.routing import llms
from .output_format.implement_tests import ImplementTestsOutput


@CrewBase
class JuniorTestsImplementationCrew:
    """
    Crew that, given a plan of tests grouped by src_file and project context,
    writes or updates the concrete test files under tests/.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        self.llm_light = llms()["light"]
        self.llm_medium = llms()["medium"]
        self.llm_reasoning = llms()["reasoning"]

        self.llm_test_implementer = llms()["light"]

    # Agents
    @agent
    def test_implementer(self) -> Agent:
        return Agent(
            config=self.agents_config["test_implementer"],  # type: ignore[index]
            llm=self.llm_test_implementer,
            verbose=True,
        )

    # Tasks
    @task
    def implement_tests(self) -> Task:
        return Task(
            config=self.tasks_config["implement_tests"],  # type: ignore[index]
            output_json=ImplementTestsOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=[
                self.implement_tests(),
            ],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )


class SeniorTestsImplementationCrew(JuniorTestsImplementationCrew):
    def __init__(self):
        super().__init__()
        self.llm_test_implementer = llms()["medium"]


class LeadTestsImplementationCrew(JuniorTestsImplementationCrew):
    def __init__(self):
        super().__init__()
        self.llm_test_implementer = llms()["reasoning"]
