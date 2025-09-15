from __future__ import annotations
from typing import List
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from ...utils.routing import llms
from .output_format.relevant_files import RelevantFilesOutput
from .output_format.file_detail import FileDetailOutput
from .output_format.action_plan import ActionPlanOutput


@CrewBase
class BaseCrew:
    """
    Decide which per-file summaries are relevant given the user's prompt and
    the available per-module summaries.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        self.llm_light = llms()["light"]
        self.llm_medium = llms()["medium"]
        self.llm_reasoning = llms()["reasoning"]

    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["analyst"],
            llm=self.llm_reasoning,
            verbose=True,
        )

    @agent
    def classifier(self) -> Agent:
        return Agent(
            config=self.agents_config["classifier"],  # type: ignore[index]
            llm=self.llm_reasoning,
            verbose=True,
        )

    @agent
    def planner(self) -> Agent:
        return Agent(
            config=self.agents_config["planner"],  # type: ignore[index]
            llm=self.llm_reasoning,
            verbose=True,
        )


class RelevanceCrew(BaseCrew):
    """
    Decide which per-file summaries are relevant given the user's prompt and
    the available per-module summaries.
    """

    @task
    def select_relevant_files(self) -> Task:
        return Task(
            config=self.tasks_config["select_relevant_files"],
            output_json=RelevantFilesOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.analyst()],
            tasks=[self.select_relevant_files()],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )


class FileDetailCrew(BaseCrew):
    """
    Classify relevant files into two sets: those for which summaries are
    sufficient to plan, and those for which code must be read to plan.
    """

    @task
    def classify_file_detail(self) -> Task:
        return Task(
            config=self.tasks_config["classify_file_detail"],  # type: ignore[index]
            output_json=FileDetailOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.classifier()],
            tasks=[self.classify_file_detail()],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )


class ActionPlanCrew(BaseCrew):
    """
    Generate a detailed action plan (a step-by-step recipe) to address the
    user's prompt using provided summaries and, where necessary, source code.
    """

    @task
    def produce_action_plan(self) -> Task:
        return Task(
            config=self.tasks_config["produce_action_plan"],  # type: ignore[index]
            output_json=ActionPlanOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.planner()],
            tasks=[self.produce_action_plan()],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )
