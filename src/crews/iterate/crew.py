from __future__ import annotations
from typing import List
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from ...utils.routing import llms
from crewai_tools import FileReadTool, FileWriterTool, DirectoryReadTool
from ...tools.test_runner import RunPytestTool
from ...tools.rag_tools import RAGSearchTool, RAGIndexTool


@CrewBase
class IterateCrew:
    """
    Iterate crew using standard CrewAI structure with decorators.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        # Initialize tools
        self.read = FileReadTool()
        self.write = FileWriterTool()
        self.ls = DirectoryReadTool()
        self.pytest = RunPytestTool()
        self.rag_search = RAGSearchTool()
        self.rag_index = RAGIndexTool()

        # Initialize LLMs
        self.llm_light = llms()["light"]
        self.llm_reasoning = llms()["reasoning"]

    # Agents
    @agent
    def state_assessor(self) -> Agent:
        return Agent(
            config=self.agents_config["state_assessor"],  # type: ignore[index]
            tools=[self.read, self.ls, self.rag_search],
            llm=self.llm_light,
            verbose=True,
        )

    @agent
    def change_planner(self) -> Agent:
        return Agent(
            config=self.agents_config["change_planner"],  # type: ignore[index]
            tools=[self.read],
            llm=self.llm_reasoning,
            verbose=True,
        )

    @agent
    def code_modder(self) -> Agent:
        return Agent(
            config=self.agents_config["code_modder"],  # type: ignore[index]
            tools=[self.read, self.write],
            llm=self.llm_reasoning,
            verbose=True,
        )

    @agent
    def quality_guard(self) -> Agent:
        return Agent(
            config=self.agents_config["quality_guard"],  # type: ignore[index]
            tools=[self.pytest],
            llm=self.llm_light,
            verbose=True,
        )

    @agent
    def debugger(self) -> Agent:
        return Agent(
            config=self.agents_config["debugger"],  # type: ignore[index]
            tools=[self.pytest, self.read, self.write],
            llm=self.llm_reasoning,
            verbose=True,
        )

    @agent
    def summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config["summarizer"],  # type: ignore[index]
            tools=[self.write, self.rag_index],
            llm=self.llm_light,
            verbose=True,
        )

    # Tasks
    @task
    def assess_state(self) -> Task:
        return Task(config=self.tasks_config["assess_state"])  # type: ignore[index]

    @task
    def plan_changes(self) -> Task:
        return Task(config=self.tasks_config["plan_changes"])  # type: ignore[index]

    @task
    def apply_changes(self) -> Task:
        return Task(config=self.tasks_config["apply_changes"])  # type: ignore[index]

    @task
    def quality_and_tests(self) -> Task:
        return Task(
            config=self.tasks_config["quality_and_tests"]  # type: ignore[index]
        )

    @task
    def debug_if_needed(self) -> Task:
        return Task(config=self.tasks_config["debug_if_needed"])  # type: ignore[index]

    @task
    def update_knowledge(self) -> Task:
        return Task(config=self.tasks_config["update_knowledge"])  # type: ignore[index]

    @crew
    def crew(self) -> Crew:
        """Creates the iterate crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
