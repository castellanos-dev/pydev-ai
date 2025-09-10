from __future__ import annotations
from typing import List
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from ...utils.routing import llms
from .output_format.task_assignment import TaskAssignmentOutput


@CrewBase
class ProjectDesignCrew:
    """
    Phase 1: requirements → architecture → plan → code generation (without writing).
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        self.llm_light = llms()["light"]
        self.llm_medium = llms()["medium"]
        self.llm_reasoning = llms()["reasoning"]

    # Agents
    @agent
    def requirements_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["requirements_analyst"],  # type: ignore[index]
            llm=self.llm_medium,
            verbose=True,
        )

    @agent
    def software_architect(self) -> Agent:
        return Agent(
            config=self.agents_config["software_architect"],  # type: ignore[index]
            llm=self.llm_reasoning,
            verbose=True,
        )

    @agent
    def project_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["project_manager"],  # type: ignore[index]
            llm=self.llm_medium,
            verbose=True,
        )

    # Tasks
    @task
    def gather_requirements(self) -> Task:
        return Task(config=self.tasks_config["gather_requirements"])  # type: ignore[index]

    @task
    def design_architecture(self) -> Task:
        return Task(config=self.tasks_config["design_architecture"])  # type: ignore[index]

    @task
    def detailed_architecture(self) -> Task:
        return Task(config=self.tasks_config["detailed_architecture"])  # type: ignore[index]

    @task
    def task_assignment(self) -> Task:
        return Task(
            config=self.tasks_config["task_assignment"],
            output_json=TaskAssignmentOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=[
                self.gather_requirements(),
                self.design_architecture(),
                self.detailed_architecture(),
                self.task_assignment(),
            ],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )
