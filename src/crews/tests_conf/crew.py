from __future__ import annotations
from typing import List
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from ...utils.routing import llms
from .output_format.tests_conf import TestsConfOutput


@CrewBase
class TestsConfCrew:
    """
    Infer unit test framework, execution command, and provide a brief description.
    Input context will include repository hints like src_dir, test_dirs, and file list.
    Output is a structured JSON that can be serialized to tests_conf.yaml.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        self.llm_reasoning = llms()["reasoning"]

    @agent
    def test_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["test_analyst"],
            llm=self.llm_reasoning,
            verbose=True,
        )

    @task
    def determine_tests_conf(self) -> Task:
        return Task(
            config=self.tasks_config["determine_tests_conf"],
            output_json=TestsConfOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.test_analyst()],
            tasks=[self.determine_tests_conf()],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )
