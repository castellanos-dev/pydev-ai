from __future__ import annotations
from typing import List
from crewai import Agent, Task, Crew, Process, TaskOutput
from crewai.tasks.conditional_task import ConditionalTask
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from ...utils.routing import llms
from .output_format.generate_code import GenerateCodeOutput
from .output_format.debug_if_needed import DebugIfNeededOutput


def is_something_to_fix(output: TaskOutput) -> bool:
    # "[]" is not a valid output
    output = str(output)
    return len(output) > 2  and '{' in output and '}' in output and 'fix' in output


@CrewBase
class JuniorDevelopmentCrew:
    """
    Phase 2: quality, test generation, debugging, documentation and summaries.
    Does not perform writes: the flow adds deterministic steps to write.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        self.llm_light = llms()["light"]
        self.llm_medium = llms()["medium"]
        self.llm_reasoning = llms()["reasoning"]

        self.llm_developer = llms()["light"]

    @agent
    def code_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["code_generator"],  # type: ignore[index]
            llm=self.llm_developer,
            verbose=True,
        )

    # Agents
    @agent
    def reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config["reviewer"],  # type: ignore[index]
            llm=self.llm_medium,
            verbose=True,
        )

    @agent
    def debugger(self) -> Agent:
        return Agent(
            config=self.agents_config["debugger"],  # type: ignore[index]
            llm=self.llm_medium,
            verbose=True,
        )

    @task
    def generate_code(self) -> Task:
        return Task(
            config=self.tasks_config["generate_code"],  # type: ignore[index]
            output_json=GenerateCodeOutput,
        )

    @task
    def code_review(self) -> Task:
        return Task(config=self.tasks_config["code_review"])  # type: ignore[index]

    @task
    def debug_if_needed(self) -> Task:
        return ConditionalTask(
            config=self.tasks_config["debug_if_needed"],
            condition=is_something_to_fix,
            output_json=DebugIfNeededOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=[
                self.generate_code(),
                self.code_review(),
                self.debug_if_needed(),
            ],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )


class SeniorDevelopmentCrew(JuniorDevelopmentCrew):
    """
    Phase 2: quality, test generation, debugging, documentation and summaries.
    Does not perform writes: the flow adds deterministic steps to write.
    """

    def __init__(self):
        super().__init__()
        self.llm_developer = llms()["medium"]


class LeadDevelopmentCrew(JuniorDevelopmentCrew):
    """
    Phase 2: quality, test generation, debugging, documentation and summaries.
    Does not perform writes: the flow adds deterministic steps to write.
    """

    def __init__(self):
        super().__init__()
        self.llm_developer = llms()["reasoning"]
