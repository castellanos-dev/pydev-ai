from __future__ import annotations
from typing import List
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from ...utils.routing import llms


@CrewBase
class SummariesCrew:
    """
    Crew that generates high-signal Markdown summaries for files and packages
    based on project design and the resulting source code of a chunk.

    The output is a compact JSON string describing summaries for files
    (excluding `__init__.py`) and packages. The flow consumes it and writes
    Markdown files under the repository's digests directory.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        self.llm_light = llms()["light"]
        self.llm_reasoning = llms()["reasoning"]

    @agent
    def summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config["summarizer"],  # type: ignore[index]
            llm=self.llm_reasoning,
            verbose=True,
        )

    @agent
    def module_summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config["module_summarizer"],  # type: ignore[index]
            llm=self.llm_reasoning,
            verbose=True,
        )

    @task
    def summarize_chunk(self) -> Task:
        return Task(config=self.tasks_config["summarize_chunk"])  # type: ignore[index]

    @task
    def summarize_modules(self) -> Task:
        return Task(config=self.tasks_config["summarize_modules"])  # type: ignore[index]

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.summarizer(), self.module_summarizer()],
            tasks=[self.summarize_chunk(), self.summarize_modules()],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )
