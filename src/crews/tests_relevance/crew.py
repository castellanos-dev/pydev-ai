from __future__ import annotations
from typing import List
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from ...utils.routing import llms
from .output_format.relevant_tests import RelevantTestsOutput


@CrewBase
class TestsRelevanceCrew:
    """
    Crew that, given the list of test files, an action-plan step detail, and
    a list of modified code files, returns the subset of relevant test files
    that should be updated due to the code changes.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self) -> None:
        self.llm_light = llms()["light"]

    @agent
    def relevance_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["relevance_analyst"],  # type: ignore[index]
            llm=self.llm_light,
            verbose=True,
        )

    @task
    def select_relevant_tests(self) -> Task:
        return Task(
            config=self.tasks_config["select_relevant_tests"],  # type: ignore[index]
            output_json=RelevantTestsOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.relevance_analyst()],
            tasks=[self.select_relevant_tests()],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )
