from __future__ import annotations
from typing import List
from crewai import Agent, Task, Crew, Process, TaskOutput
from crewai.tasks.conditional_task import ConditionalTask
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from ...utils.routing import llms
from .output_format.generate_diffs import GenerateDiffsOutput


def has_diffs(output: TaskOutput) -> bool:
    output_str = str(output)
    return len(output_str) > 2 and '{' in output_str and '}' in output_str and 'content_diff' in output_str


@CrewBase
class JuniorDevelopmentDiffCrew:
    """
    Equivalent to development crew but emits diffs per file instead of full contents.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        self.llm_light = llms()["light"]
        self.llm_medium = llms()["medium"]
        self.llm_reasoning = llms()["reasoning"]

        self.llm_developer = llms()["light"]

    @agent
    def code_diff_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["code_diff_generator"],
            llm=self.llm_developer,
            verbose=True,
        )

    @task
    def generate_diffs(self) -> Task:
        return Task(
            config=self.tasks_config["generate_diffs"],
            output_json=GenerateDiffsOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=[
                self.generate_diffs(),
            ],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )


class SeniorDevelopmentDiffCrew(JuniorDevelopmentDiffCrew):
    def __init__(self):
        super().__init__()
        self.llm_developer = llms()["medium"]


class LeadDevelopmentDiffCrew(JuniorDevelopmentDiffCrew):
    def __init__(self):
        super().__init__()
        self.llm_developer = llms()["reasoning"]
