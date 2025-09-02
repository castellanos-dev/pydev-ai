from __future__ import annotations
import pathlib
from typing import Dict, Any
from crewai.flow import Flow, start, listen
from .utils import ensure_knowledge
from ..crews.iterate.crew import IterateCrew

# Flow-level guardrails for iteration tasks
MAX_DEBUG_LOOPS = 2
MAX_TOKENS_PER_RESPONSE = 2000
TOKEN_CAP_HINT = (
    f"Keep responses under {MAX_TOKENS_PER_RESPONSE} tokens unless strictly necessary."
)


class IterateFlow(Flow):
    """
    CrewAI Flow for iterating on existing projects.

    Steps:
    1. Bootstrap knowledge if needed (digests + RAG)
    2. Execute IterateCrew with flow-level limits and guardrails
    """

    @start()
    def bootstrap_knowledge(self) -> Dict[str, Any]:
        """Initialize knowledge directory if needed and prepare inputs for the crew."""
        user_prompt = self.state["user_prompt"]
        repo = self.state["repo"]
        ensure_knowledge(repo)
        return {
            "user_prompt": user_prompt,
            "repo": repo,
            "flow_limits": {
                "max_debug_loops": MAX_DEBUG_LOOPS,
                "token_cap_hint": TOKEN_CAP_HINT,
                "max_tokens_per_response": MAX_TOKENS_PER_RESPONSE,
            },
        }

    @listen(bootstrap_knowledge)
    def execute_iterate_crew(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the IterateCrew with the prepared inputs."""
        crew = IterateCrew().crew()
        result = crew.kickoff(inputs=inputs)

        return {
            "result": str(result),
            "repo": inputs["repo"],
            "success": True,
            "flow_type": "iterate",
        }

    def run(self, user_prompt: str, repo: str) -> Dict[str, Any]:
        """Convenience method for CLI integration."""
        return self.bootstrap_knowledge(user_prompt=user_prompt, repo=repo)


def run_iterate(user_prompt: str, repo: str) -> None:
    """
    Execute the iterate flow using CrewAI Flows.

    This function maintains backward compatibility with the CLI
    while using the new Flow-based architecture internally.
    """
    repo_path = pathlib.Path(repo)
    assert repo_path.exists(), f"Repo not found: {repo}"

    flow = IterateFlow()
    result = flow.kickoff(inputs={"user_prompt": user_prompt, "repo": str(repo_path)})

    print(f"Iterate flow completed: {result}")
    return result
