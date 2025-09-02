from __future__ import annotations
import pathlib
import subprocess
import sys
from typing import Dict, Any
from crewai.flow import Flow, start, listen
from .utils import (
    ensure_repo, parse_file_map_from_text, write_file_map, parse_code_output,
    parse_code_fixes_output, parse_code_design_output, parse_summaries_output
)
from ..crews.design.project_design_crew import ProjectDesignCrew
from ..crews.development.crew import JuniorDevelopmentCrew, SeniorDevelopmentCrew, LeadDevelopmentCrew
from ..crews.summaries.crew import SummariesCrew
from ..crews.fix_integrator.crew import FixIntegratorCrew
from ..crews.build.phase2_crew import BuildCrewPhase2

# Flow-level guardrails for consistency and cost control
MAX_DEBUG_LOOPS = 2
MAX_TOKENS_PER_RESPONSE = 2000
TOKEN_CAP_HINT = (
    f"Keep responses under {MAX_TOKENS_PER_RESPONSE} tokens unless strictly necessary."
)

DEVELOPERS = {
    1: JuniorDevelopmentCrew,
    2: SeniorDevelopmentCrew,
    3: LeadDevelopmentCrew,
}


class NewProjectFlow(Flow):
    """
    Two-phase CrewAI Flow with deterministic write steps in-between.
    """

    @start()
    def process_inputs(self) -> Dict[str, Any]:
        user_prompt = self.state["user_prompt"]
        out_dir = self.state["out_dir"]
        self.out_dir = ensure_repo(out_dir)
        return {
            "user_prompt": user_prompt,
        }

    @listen(process_inputs)
    def project_design(self, user_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run project design."""
        result = ProjectDesignCrew().crew().kickoff(
            inputs={
                "new_project_prompt": user_inputs["user_prompt"],
            }
        )
        return {
            "project_design_result": str(result),
        }

    @listen(project_design)
    def code_development(self, design_result: Dict[str, Any]) -> Dict[str, Any]:
        """Run code development."""
        design_result = parse_code_design_output(str(design_result["project_design_result"]))

        code = {}
        summaries = {}
        for development_task in design_result:
            result = DEVELOPERS[development_task["developer"]]().crew().kickoff(
                inputs={
                    "project_design": development_task["set_of_files"],
                }
            )
            code_output = parse_code_output(str(result.tasks_output[0]))
            code_fixes_output = parse_code_fixes_output(str(result.tasks_output[2]))
            for file in code_output:
                file_fixes = [
                    {k: v for k, v in fix.items() if k != "file_path"}
                    for fix in code_fixes_output if fix["file_path"] == file["path"]
                ]
                if file_fixes:
                    fix_result = FixIntegratorCrew().crew().kickoff(
                        inputs={
                            "original_code": file["content"],
                            "code_fixes": file_fixes,
                        }
                    )
                    code[file["path"]] = str(fix_result.tasks_output[0])
                else:
                    code[file["path"]] = file["content"]
            # Generate summaries for this chunk (design + resulting code for this design)
            result_summ = SummariesCrew().crew().kickoff(
                inputs={
                    "project_design": development_task["set_of_files"],
                    "code_chunk": code_output,
                }
            )
            file_summaries = parse_summaries_output(str(result_summ.tasks_output[0]))
            module_summaries = parse_summaries_output(str(result_summ.tasks_output[1]))
            for summary in file_summaries:
                summaries[summary["path"]] = summary["content"]
            for summary in module_summaries:
                summaries[summary["path"]] = summary["content"]
        return {
            "code": code,
            "summaries": summaries,

        }

    @listen(code_development)
    def write_generated_code(self, code_result: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministically write the codebase."""
        code_logs = write_file_map(code_result["code"], self.out_dir, 'src')  # TODO: review the logs
        summaries_logs = write_file_map(code_result["summaries"], self.out_dir, 'summaries')  # TODO: review the logs
        return {}

    @listen(write_generated_code)
    def lint_and_format(self, _) -> Dict[str, Any]:
        """Lint and format the codebase."""
        src_dir = pathlib.Path(self.out_dir) / "src"

        if not src_dir.exists():
            return {
                "linting_skipped": True,
                "reason": f"Source directory not found: {src_dir}",
                "out_dir": str(self.out_dir),
            }

        def run_cmd(cmd: list[str]) -> Dict[str, Any]:
            completed = subprocess.run(
                cmd,
                cwd=str(src_dir),
                capture_output=True,
                text=True,
                check=False,
            )
            return {
                "cmd": " ".join(cmd),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "returncode": completed.returncode,
            }

        # 1) Ruff auto-fix
        run_cmd([sys.executable, "-m", "ruff", "check", "--fix", "."])

        # 2) Black format
        black_fmt = run_cmd([sys.executable, "-m", "black", "."])

        # 3) Ruff final check (capture remaining issues)
        ruff_check = run_cmd([sys.executable, "-m", "ruff", "check", "."])

        return {
            "linting_skipped": False,
            "black_format": black_fmt,
            "ruff_report": ruff_check,
        }

    # @listen(write_generated_code)
    # def phase2_generate_tests(self, state: Dict[str, Any]) -> Dict[str, Any]:
    #     """Run Phase 2 (part 1): format/lint and generate tests (no writes)."""
    #     result = BuildCrewPhase2().crew().kickoff(inputs={
    #         "out_dir": state["out_dir"]
    #     })
    #     return {
    #         **state,
    #         "phase2_tests_result": str(result),
    #     }

    # @listen(phase2_generate_tests)
    # def write_tests(self, state: Dict[str, Any]) -> Dict[str, Any]:
    #     """Deterministically write tests before running pytest."""
    #     files = parse_file_map_from_text(state["phase2_tests_result"])
    #     if files:
    #         log = write_file_map(files, state["out_dir"])
    #     else:
    #         log = []
    #     return {
    #         **state,
    #         "write_tests_log": log,
    #     }

    # @listen(write_tests)
    # def phase2_debug_and_docs(self, state: Dict[str, Any]) -> Dict[str, Any]:
    #     """Run Phase 2 (part 2): run tests/debug and produce docs (no writes)."""
    #     result = (
    #         BuildCrewPhase2()
    #         .debug_and_finalize_docs()
    #         .kickoff(inputs={"out_dir": state["out_dir"]})
    #     )
    #     return {
    #         **state,
    #         "phase2_docs_result": str(result),
    #     }

    # @listen(phase2_debug_and_docs)
    # def write_docs(self, state: Dict[str, Any]) -> Dict[str, Any]:
    #     """Deterministically write documentation files."""
    #     files = parse_file_map_from_text(state["phase2_docs_result"])
    #     log = write_file_map(files, state["out_dir"]) if files else []
    #     return {
    #         **state,
    #         "write_docs_log": log,
    #     }

    # @listen(write_docs)
    # def summarize_and_index(self, state: Dict[str, Any]) -> Dict[str, Any]:
    #     """Run summarization/indexing after docs are on disk."""
    #     result = (
    #         BuildCrewPhase2()
    #         .summarize_and_index()
    #         .kickoff(inputs={"out_dir": state["out_dir"]})
    #     )
    #     return {
    #         **state,
    #         "result": str(result),
    #         "success": True,
    #         "flow_type": "new_project",
    #     }

    def run(self, user_prompt: str, out_dir: str) -> Dict[str, Any]:
        """Convenience method for CLI integration."""
        return self.kickoff(inputs={"user_prompt": user_prompt, "out_dir": out_dir})


def run_new_project(user_prompt: str, out_dir: str) -> None:
    """
    Execute the new project flow using CrewAI Flows.
    """
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    flow = NewProjectFlow()
    result = flow.kickoff(inputs={"user_prompt": user_prompt, "out_dir": str(out)})

    print(f"New project flow completed: {result}")
    return result
