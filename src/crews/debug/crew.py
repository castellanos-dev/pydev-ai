from __future__ import annotations
from typing import List
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import FileReadTool, DirectoryReadTool
from ...utils.routing import llms


@CrewBase
class BaseDebugCrew:
    """
    Crew focused on debugging the code.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        self.read = FileReadTool()
        self.ls = DirectoryReadTool()

        self.llm_light = llms()["light"]
        self.llm_reasoning = llms()["reasoning"]

    # Agents
    @agent
    def reporter(self) -> Agent:
        return Agent(
            config=self.agents_config["reporter"],  # type: ignore[index]
            llm=self.llm_light,
            verbose=True,
        )

    @agent
    def grouper(self) -> Agent:
        return Agent(
            config=self.agents_config["grouper"],  # type: ignore[index]
            llm=self.llm_light,
            verbose=True,
        )

    # Agents
    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["analyst"],  # type: ignore[index]
            llm=self.llm_light,
            verbose=True,
        )

    # Agents
    @agent
    def bug_fixer(self) -> Agent:
        return Agent(
            config=self.agents_config["bug_fixer"],  # type: ignore[index]
            llm=self.llm_reasoning,
            verbose=True,
        )
