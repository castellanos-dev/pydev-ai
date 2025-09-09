from __future__ import annotations
import pathlib
import subprocess
import sys
import os
from typing import Dict, Any, List, Tuple
import glob
from crewai.flow import Flow, start, listen
from .utils import (
    ensure_repo, parse_file_map_from_text, write_file_map, parse_code_output,
    parse_code_fixes_output, parse_code_design_output, parse_summaries_output,
    parse_test_output, sanitize_generated_content,
    parse_pytest_groups_output,
    parse_involved_files_output,
    parse_bug_analysis_output,
)
from .. import settings
from ..crews.design.project_design_crew import ProjectDesignCrew
from ..crews.development.crew import JuniorDevelopmentCrew, SeniorDevelopmentCrew, LeadDevelopmentCrew
from ..crews.summaries.crew import SummariesCrew
from ..crews.test_development.crew import (
    JuniorTestDevelopmentCrew,
    SeniorTestDevelopmentCrew,
    LeadTestDevelopmentCrew,
)
from ..crews.fix_integrator.crew import FixIntegratorCrew
from ..crews.build.phase2_crew import BuildCrewPhase2
from ..crews.debug import (
    BugAnalysisCrew,
    PytestOutputAnalysisCrew,
    AnalyzeInvolvedFilesCrew,
    bug_fixer_for_points,
)

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

TEST_DEVELOPERS = {
    1: JuniorTestDevelopmentCrew,
    2: SeniorTestDevelopmentCrew,
    3: LeadTestDevelopmentCrew,
}

def _run_cmd(cmd: List[str], cwd: pathlib.Path) -> Dict[str, Any]:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
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


def lint_and_format(code_dir: pathlib.Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Lint and format the codebase."""
    # 1) Ruff auto-fix
    _run_cmd([sys.executable, "-m", "ruff", "check", "--fix", "."], cwd=code_dir)

    # 2) Black format
    black_fmt = _run_cmd([sys.executable, "-m", "black", "."], cwd=code_dir)

    # 3) Ruff final check (capture remaining issues)
    ruff_check = _run_cmd([sys.executable, "-m", "ruff", "check", "."], cwd=code_dir)

    return black_fmt, ruff_check


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
                    code[file["path"]] = sanitize_generated_content(str(fix_result.tasks_output[0]))
                else:
                    code[file["path"]] = sanitize_generated_content(file["content"])
            # TODO: uncomment
            # # Generate summaries for this chunk (design + resulting code for this design)
            # result_summ = SummariesCrew().crew().kickoff(
            #     inputs={
            #         "project_design": development_task["set_of_files"],
            #         "code_chunk": code_output,
            #     }
            # )
            # file_summaries = parse_summaries_output(str(result_summ.tasks_output[0]))
            # module_summaries = parse_summaries_output(str(result_summ.tasks_output[1]))
            # for summary in file_summaries:
            #     summaries[summary["path"]] = sanitize_generated_content(summary["content"])
            # for summary in module_summaries:
            #     summaries[summary["path"]] = sanitize_generated_content(summary["content"])
        return {
            "code": code,
            "summaries": summaries,
            "project_design": design_result,
        }

    @listen(code_development)
    def write_generated_code(self, code_result: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministically write the codebase."""
        code_logs = write_file_map(code_result["code"], self.out_dir, 'src')  # TODO: review the logs
        summaries_logs = write_file_map(code_result["summaries"], self.out_dir, 'summaries')  # TODO: review the logs
        return {
            "project_design": code_result["project_design"],
        }

    @listen(write_generated_code)
    def apply_linting(self, code_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply linting to the codebase."""
        src_dir = pathlib.Path(self.out_dir) / "src"
        if not src_dir.exists():
            return {
                "linting_skipped": True,
                "reason": f"Source directory not found: {src_dir}",
                "out_dir": str(self.out_dir),
                "project_design": code_result["project_design"],
            }

        black_fmt, ruff_check = lint_and_format(src_dir)
        return {
            "linting_skipped": False,
            "black_format": black_fmt,
            "ruff_report": ruff_check,
            "project_design": code_result["project_design"],
        }

    @listen(apply_linting)
    def test_development(self, lint_result: Dict[str, Any]) -> Dict[str, Any]:
        """Unit tests generation"""
        project_info = lint_result["project_design"].copy()
        # Enrich the design with the file contents to support better tests
        src_dir = pathlib.Path(self.out_dir) / "src"
        tests_to_write: Dict[str, str] = {}
        for test_task in project_info:
            for rel_path, spec in test_task["set_of_files"].items():
                file_path = src_dir / rel_path
                try:
                    content = file_path.read_text(encoding="utf-8")
                except Exception:
                    content = ""
                test_task[rel_path] = {
                    **spec,
                    "file_content": content,
                }
            result = TEST_DEVELOPERS[test_task["developer"]]().crew().kickoff(
                inputs={
                    "project": test_task,
                }
            )
            generated_tests = parse_test_output(str(result.tasks_output[0]))
            for f in generated_tests:
                tests_to_write[f["path"]] = sanitize_generated_content(f["content"])
        if tests_to_write:
            write_file_map(tests_to_write, self.out_dir, "tests")
        return {
            "project_design": project_info,
            "tests_written": list(tests_to_write.keys()),
        }

    @listen(test_development)
    def project_debugging(self, test_dev_result: Dict[str, Any]) -> Dict[str, Any]:
        for _ in range(settings.MAX_TEST_RUN_ATTEMPTS):
            test_result = self._lint_and_execute_tests(test_dev_result)
            if test_result.get("returncode") != 0:
                return {
                    "project_design": test_dev_result.get("project_design"),
                    "bug_analysis": [],
                }
            pytests_output_analysis = self._pytest_output_analysis(test_result)
            involved_files = self._analyze_involved_files(pytests_output_analysis)
            bug_analysis = self._bug_analysis(involved_files)
        return {
            "project_design": test_dev_result.get("project_design"),
            "bug_analysis": bug_analysis,
        }

    def _lint_and_execute_tests(self, test_dev_result: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-fix lint in tests/ and execute pytest, returning structured results."""
        repo_dir = pathlib.Path(self.out_dir)
        src_dir = repo_dir / "src"
        tests_dir = repo_dir / "tests"

        tests_black_fmt: Dict[str, Any] | None = None
        tests_ruff_report: Dict[str, Any] | None = None

        if tests_dir.exists():
            # Ruff auto-fix and Black formatting scoped to tests/
            tests_black_fmt, tests_ruff_report = lint_and_format(tests_dir)
        else:
            return {
                "tests_dir": str(tests_dir),
                "tests_linting_skipped": True,
                "tests_linting_reason": f"Tests directory not found: {tests_dir}",
                "project_design": test_dev_result.get("project_design"),
                "tests_black_format": None,
                "tests_ruff_report": None,
                "pytest_output": None,
            }

        # Execute pytest from repo root; ensure PYTHONPATH includes src/
        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH", "")
        path_sep = ":"  # POSIX path separator
        if str(src_dir) not in existing_pp.split(path_sep) if existing_pp else True:
            env["PYTHONPATH"] = (str(src_dir) + (path_sep + existing_pp if existing_pp else ""))
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                check=False,
                timeout=settings.PYTEST_TIMEOUT,
                env=env,
            )
            pytest_result = {
                "cmd": "python -m pytest -q",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "returncode": completed.returncode,
            }
        except subprocess.TimeoutExpired:
            pytest_result = {
                "cmd": "python -m pytest -q",
                "stdout": "",
                "stderr": f"Timeout after {settings.PYTEST_TIMEOUT} seconds",
                "returncode": -2,  # TODO: handle this
            }
        except Exception as e:
            pytest_result = {
                "cmd": "python -m pytest -q",
                "stdout": "",
                "stderr": f"Error: {e}",
                "returncode": -1,  # TODO: handle this
            }

        return {
            "tests_dir": str(tests_dir),
            "tests_linting_skipped": False,
            "tests_black_format": tests_black_fmt,
            "tests_ruff_report": tests_ruff_report,
            "pytest_output": pytest_result,
            "project_design": test_dev_result.get("project_design"),
        }

    def _pytest_output_analysis(self, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the pytest output."""
        pytest_output_analysis = PytestOutputAnalysisCrew().crew().kickoff(
            inputs={
                "pytest_output": test_result.get("pytest_output", {}),
            }
        )

        try:
            groups = parse_pytest_groups_output(str(pytest_output_analysis.tasks_output[1]))
        except Exception:
            groups = []

        if len(groups) == 0:
            return {
                "complete_output_analysis": groups,
                "flat_output_analysis": [],
            }

        flat_groups = [
            {
                "file_path": group["file_path"][0]
                if isinstance(group.get("file_path", ""), list) else group.get("file_path", ""),
                "affected_callable": group["affected_callable"][0]
                if isinstance(group.get("affected_callable", ""), list) else group.get("affected_callable", ""),
                "error": group["error"][0]
                if isinstance(group.get("error", ""), list) else group.get("error", ""),
                "traceback": group["traceback"][0]
                if isinstance(group.get("traceback", ""), list) else group.get("traceback", ""),
            } for group in groups
        ]

        return {
            "complete_output_analysis": groups,
            "flat_output_analysis": flat_groups,
        }

    def _analyze_involved_files(self, pytests_output_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the involved files."""

        if len(pytests_output_analysis.get("flat_output_analysis", [])) == 0:
            return {
                "debug_info": [],
            }

        # Collect project and test Python files using glob
        repo_dir = pathlib.Path(self.out_dir)
        src_dir = repo_dir / "src"
        tests_dir = repo_dir / "tests"

        code_files = sorted([
            str(pathlib.Path(p).relative_to(repo_dir))
            for p in glob.glob(str(src_dir / "**" / "*.py"), recursive=True)
        ])
        test_files = sorted([
            str(pathlib.Path(p).relative_to(repo_dir))
            for p in glob.glob(str(tests_dir / "**" / "*.py"), recursive=True)
        ])

        involved_files = AnalyzeInvolvedFilesCrew().crew().kickoff(
            inputs={
                "code_files": code_files,
                "test_files": test_files,
                "flat_output_analysis": pytests_output_analysis.get("flat_output_analysis", []),
            }
        )
        involved_files = parse_involved_files_output(str(involved_files.tasks_output[0]))

        if not isinstance(involved_files, list) or len(involved_files) == 0:
            return {
                "debug_info": [],
            }

        enriched: List[Dict[str, Any]] = []
        for entry in involved_files:
            try:
                file_list = entry.get("involved_files", [])
            except AttributeError:
                file_list = []
            code_blobs: List[Dict[str, str]] = []
            for p in file_list:
                try:
                    content = pathlib.Path(p).read_text(encoding="utf-8")
                except Exception:
                    content = ""
                code_blobs.append({"path": str(p), "content": content})
            new_entry = dict(entry)
            new_entry["involved_files_code"] = code_blobs
            enriched.append(new_entry)

        return {
            "debug_info": enriched
        }

    def _bug_analysis(self, involved_files_result: Dict[str, Any]) -> Dict[str, Any]:
        """Debug the project."""

        # If pytest succeeded, skip debugging
        debug_info = involved_files_result.get("debug_info", {})
        if not isinstance(debug_info, list) or len(debug_info) == 0:
            return {
                "debugging_skipped": True,
                "reason": "No debug info",
                **involved_files_result,
            }

        bug_analysis = BugAnalysisCrew().crew().kickoff(
            inputs={
                "debug_info": debug_info,
            }
        )
        bug_analysis = parse_bug_analysis_output(str(bug_analysis.tasks_output[0]))
        if not isinstance(bug_analysis, list) or len(bug_analysis) == 0:
            return {
                "debugging_skipped": True,
                "reason": "No bugs found",
                **involved_files_result,
            }

        # Apply fixes per bug using appropriate BugFixerCrew based on points
        files_to_write: Dict[str, str] = {}
        for bug in bug_analysis:
            points = int(bug.get("points", 1) or 1)
            fixer = bug_fixer_for_points(points)
            result = fixer.crew().kickoff(inputs={
                "bug": bug,
                "debug_info": involved_files_result.get("debug_info", []),
            })
            file_map = parse_file_map_from_text(str(result.tasks_output[0]))
            for f in file_map:
                path = f.get("path")
                content = sanitize_generated_content(f.get("content", ""))
                if path:
                    files_to_write[path] = content

        if files_to_write:
            write_file_map(files_to_write, self.out_dir)

        return {
            "debugging_skipped": False,
            "reason": "Bugs fixed",
            "fixes_applied": list(files_to_write.keys()),
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
