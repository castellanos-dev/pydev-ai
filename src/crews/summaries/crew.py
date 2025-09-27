from __future__ import annotations
from typing import List
from crewai import Agent, Task
from crewai.project import CrewBase, agent
from crewai.agents.agent_builder.base_agent import BaseAgent
from ...utils.routing import llms


@CrewBase
class BaseSummariesCrew:
    """
    Base crew for generating Markdown summaries for files and modules.
    Subclasses must set `original_tasks_config_path` to select the YAML file.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        self.llm_light = llms()["light"]
        self.llm_medium = llms()["medium"]
        self.llm_reasoning = llms()["reasoning"]

    @agent
    def summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config["summarizer"],
            llm=self.llm_light,
            verbose=True,
        )

    @agent
    def module_summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config["module_summarizer"],
            llm=self.llm_light,
            verbose=True,
        )
