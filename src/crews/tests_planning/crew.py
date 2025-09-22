from __future__ import annotations
from typing import List
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from ...utils.routing import llms
from .output_format.test_plan import GenerateTestPlanOutput


@CrewBase
class JuniorTestsPlanningCrew:
    """
    Crew that, given the test setup and recent code changes, proposes a
    prioritized list of unit test descriptions to be implemented.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        self.llm_light = llms()["light"]
        self.llm_medium = llms()["medium"]
        self.llm_reasoning = llms()["reasoning"]

        self.llm_planner = llms()["light"]

    # Agents
    @agent
    def test_planner(self) -> Agent:
        return Agent(
            config=self.agents_config["test_planner"],  # type: ignore[index]
            llm=self.llm_planner,
            verbose=True,
        )

    # Tasks
    @task
    def plan_tests(self) -> Task:
        return Task(
            config=self.tasks_config["plan_tests"],  # type: ignore[index]
            output_json=GenerateTestPlanOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=[
                self.plan_tests(),
            ],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )


class SeniorTestsPlanningCrew(JuniorTestsPlanningCrew):
    def __init__(self):
        super().__init__()
        self.llm_planner = llms()["medium"]


class LeadTestsPlanningCrew(JuniorTestsPlanningCrew):
    def __init__(self):
        super().__init__()
        self.llm_planner = llms()["reasoning"]
