from __future__ import annotations

from typing import List
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from ...utils.routing import llms
from ..docs_diff.output_format.doc_unified_diff import DocUnifiedDiffOutput


@CrewBase
class ReleaseNotesUpdateCrew:
    """
    Produce the updated full contents of a release notes/changelog file given
    the current content, target version/date, format (md/rst) and change items.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self) -> None:
        self.llm_light = llms()["light"]

    @agent
    def release_notes_editor(self) -> Agent:
        return Agent(
            config=self.agents_config["release_notes_editor"],  # type: ignore[index]
            llm=self.llm_light,
            verbose=True,
        )

    @task
    def update_release_notes(self) -> Task:
        return Task(
            config=self.tasks_config["update_release_notes"],  # type: ignore[index]
            output_json=DocUnifiedDiffOutput,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.release_notes_editor()],
            tasks=[self.update_release_notes()],
            process=Process.sequential,
            output_log_file=True,
            verbose=True,
        )
