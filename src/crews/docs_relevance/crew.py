from __future__ import annotations
from typing import List
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from ...utils.routing import llms
from .output_format.relevant_docs import RelevantDocsOutput


@CrewBase
class DocsRelevanceCrew:
    """
    Given a mapping of document paths to content, the action plan text, and the
    user's prompt, select the subset of document paths that truly need
    updating.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self) -> None:
        self.llm_reasoning = llms()["reasoning"]

    @agent
    def relevance_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["relevance_analyst"],  # type: ignore[index]
            llm=self.llm_reasoning,
            verbose=True,
        )

    @task
    def select_relevant_docs(self) -> Task:
        return Task(
            config=self.tasks_config["select_relevant_docs"],  # type: ignore[index]
            output_json=RelevantDocsOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.relevance_analyst()],
            tasks=[self.select_relevant_docs()],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )
