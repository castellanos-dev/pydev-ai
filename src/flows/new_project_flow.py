from __future__ import annotations
import pathlib
import subprocess
import sys
import os
from typing import Dict, Any, List, Tuple
import glob
from crewai.flow import Flow, start, listen
from .utils import (
    ensure_repo, is_something_to_fix, write_file_map, sanitize_generated_content,
    load_json_output, process_path,
)
from .common import (
    generate_file_summaries_from_chunk,
    generate_module_summaries_from_file_summaries,
)
from .. import settings
from ..crews.design.crew import ProjectDesignCrew
from ..crews.development.crew import JuniorDevelopmentCrew, SeniorDevelopmentCrew, LeadDevelopmentCrew
from ..crews.test_development.crew import (
    JuniorTestDevelopmentCrew,
    SeniorTestDevelopmentCrew,
    LeadTestDevelopmentCrew,
)
from ..crews.fix_integrator.crew import FixIntegratorCrew
from ..crews.debug import (
    BugAnalysisCrew,
    PytestOutputAnalysisCrew,
    AnalyzeInvolvedFilesCrew,
    bug_fixer_for_points,
)
from ..crews.design.output_format.task_assignment import TASK_ASSIGNMENT_SCHEMA
from ..crews.development.output_format.generate_code import GENERATE_CODE_SCHEMA
from ..crews.development.output_format.debug_if_needed import DEBUG_IF_NEEDED_SCHEMA
from ..crews.summaries.output_format.summaries import SUMMARIES_SCHEMA
from ..crews.test_development.output_format.generate_tests import GENERATE_TESTS_SCHEMA
from ..crews.debug.output_format.pytest_output import PYTEST_OUTPUT_ANALYSIS_SCHEMA
from ..crews.debug.output_format.analyze_involved_files import INVOLVED_FILES_SCHEMA
from ..crews.debug.output_format.bug_analysis import BUG_ANALYSIS_SCHEMA
from ..crews.debug.output_format.bug_fixes import BUG_FIXES_SCHEMA


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

    def _process_file_summaries_chunk(self, chunk: List[Dict[str, str]]) -> Dict[str, str]:
        return generate_file_summaries_from_chunk(chunk)

    def _process_module_summaries_from_file_summaries(self, file_summaries: Dict[str, str]) -> Dict[str, str]:
        return generate_module_summaries_from_file_summaries(file_summaries)

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
        return load_json_output(result, TASK_ASSIGNMENT_SCHEMA)

    @listen(project_design)
    def code_development(self, design_result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run code development."""
        code = {}
        summaries = {}
        for development_task in design_result:
            result = DEVELOPERS[development_task["developer"]]().crew().kickoff(
                inputs={
                    "project_design": development_task["set_of_files"],
                }
            )
            code_output = load_json_output(result, GENERATE_CODE_SCHEMA, 0)
            if len(code_output) == 0:
                continue
            code_fixes_output = load_json_output(result, DEBUG_IF_NEEDED_SCHEMA, 2)

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

            # Generate summaries iteratively to avoid LLM output limits
            # 1) Per-file summaries (iterate item-by-item)
            file_summaries_map = self._process_file_summaries_chunk(code_output)

            # 2) Per-module summaries built from file summaries only
            # Group file summaries by their parent folder
            folders = set(pathlib.Path(p).parent for p in file_summaries_map.keys())
            module_summaries_map: Dict[str, str] = {}
            for folder in folders:
                per_module_input = {
                    path: content
                    for path, content in file_summaries_map.items()
                    if pathlib.Path(path).parent == folder
                }
                if not per_module_input:
                    continue
                generated = self._process_module_summaries_from_file_summaries(per_module_input)
                module_summaries_map.update(generated)

            # 3) Accumulate
            summaries.update(file_summaries_map)
            summaries.update(module_summaries_map)
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
        return code_result["project_design"]

    @listen(write_generated_code)
    def apply_linting(self, project_design: Dict[str, Any]) -> Dict[str, Any]:
        """Apply linting to the codebase."""
        src_dir = pathlib.Path(self.out_dir) / "src"
        if src_dir.exists():
            black_fmt, ruff_check = lint_and_format(src_dir)  # TODO: review the logs
        return project_design

    @listen(apply_linting)
    def test_development(self, project_design: Dict[str, Any]) -> Dict[str, Any]:
        """Unit tests generation"""
        project_info = project_design.copy()
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
            generated_tests = load_json_output(result, GENERATE_TESTS_SCHEMA, 0)
            for f in generated_tests:
                tests_to_write[f["path"]] = sanitize_generated_content(f["content"])
        if tests_to_write:
            write_file_map(tests_to_write, self.out_dir, "tests")
        return project_info

    @listen(test_development)
    def project_debugging(self, project_info: Dict[str, Any]) -> Dict[str, Any]:
        for _ in range(settings.MAX_TEST_RUN_ATTEMPTS):
            test_result = self._lint_and_execute_tests(project_info)
            if (
                test_result.get("pytest_output") is None or
                len(test_result.get("pytest_output", {})) == 0 or
                test_result.get("pytest_output", {}).get("returncode") == 0
            ):
                return test_result
            pytests_output_analysis = self._pytest_output_analysis(test_result)
            involved_files = self._analyze_involved_files(pytests_output_analysis)
            self._bug_analysis(involved_files)
        test_result = self._lint_and_execute_tests(project_info)
        return test_result

    def _lint_and_execute_tests(self, project_info: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-fix lint in tests/ and execute pytest, returning structured results."""
        repo_dir = pathlib.Path(self.out_dir)
        src_dir = repo_dir / "src"
        tests_dir = repo_dir / "tests"

        if tests_dir.exists():
            # Ruff auto-fix and Black formatting scoped to tests/
            tests_black_fmt, tests_ruff_report = lint_and_format(tests_dir)  # TODO: review the logs
        else:
            return {
                "project_design": project_info,
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
            "pytest_output": pytest_result,
            "project_design": project_info,
        }

    def _pytest_output_analysis(self, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the pytest output."""
        pytest_output_analysis = PytestOutputAnalysisCrew().crew().kickoff(
            inputs={
                "pytest_output": test_result.get("pytest_output", {}),
            }
        )
        if not is_something_to_fix(pytest_output_analysis.tasks_output[0]):
            return {
                "complete_output_analysis": [],
                "flat_output_analysis": [],
            }

        groups = load_json_output(pytest_output_analysis, PYTEST_OUTPUT_ANALYSIS_SCHEMA, 1)

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
                "id": i,
            } for i, group in enumerate(groups)
        ]

        return flat_groups

    def _analyze_involved_files(self, pytests_output_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the involved files."""

        if len(pytests_output_analysis) == 0:
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
                "flat_output_analysis": pytests_output_analysis,
            }
        )
        involved_files = load_json_output(involved_files, INVOLVED_FILES_SCHEMA, 0)

        file_contents: Dict[str, str] = {}
        for entry in involved_files:
            try:
                file_list = entry.get("involved_files", [])
            except AttributeError:
                file_list = []
            for p in file_list:
                if str(p) not in file_contents:
                    try:
                        content = process_path(repo_dir, p, "src").read_text(encoding="utf-8")
                        file_contents[str(p)] = content
                    except Exception:
                        pass

        return {
            "debug_info": involved_files,
            "file_contents": file_contents,
            "code_files": code_files,
            "test_files": test_files,
        }

    def _bug_analysis(self, debug_info: Dict[str, Any]) -> Dict[str, Any]:
        """Debug the project."""
        # If pytest succeeded, skip debugging
        if len(debug_info.get("debug_info", [])) == 0:
            return {}

        bug_analysis = BugAnalysisCrew().crew().kickoff(
            inputs={
                "file_contents": debug_info.get("file_contents", {}),
                "code_files": debug_info.get("code_files", []),
                "test_files": debug_info.get("test_files", []),
                "debug_info": debug_info.get("debug_info", []),
            }
        )
        bug_analysis = load_json_output(bug_analysis, BUG_ANALYSIS_SCHEMA, 0)
        if len(bug_analysis) == 0:
            return {}

        # Apply fixes per bug using appropriate BugFixerCrew based on points
        files_to_write: Dict[str, str] = {}
        test_files_to_write: Dict[str, str] = {}
        changes_by_file: Dict[str, List[Dict[str, str]]] = {}
        for bug in bug_analysis:
            bug = bug.copy()
            bug["file_contents"] = []
            file_contents = debug_info.get("file_contents", {})
            for file_path in bug.get("file_paths", []):
                if file_path in file_contents:
                    bug["file_contents"].append({"path": file_path, "content": file_contents[file_path]})

            points = int(bug.get("points", 1) or 1)
            fixer = bug_fixer_for_points(points)
            result = fixer.crew().kickoff(inputs={
                "bug": bug,
                "debug_info": debug_info,
            })
            file_changes = load_json_output(result, BUG_FIXES_SCHEMA, 0)

            for file_change in file_changes:
                path = file_change.get("path")
                content_diff = file_change.get("content_diff", "")
                if path not in changes_by_file:
                    changes_by_file[path] = []
                changes_by_file[path].append(content_diff)

        for path, changes in changes_by_file.items():
            try:
                original_code = file_contents[path]
            except KeyError:
                try:
                    original_code = file_contents[f'src/{path}']
                except KeyError:
                    original_code = "-"
            fix_result = FixIntegratorCrew().crew().kickoff(
                inputs={
                    "original_code": original_code,
                    "code_fixes": changes,
                }
            )
            file_result = sanitize_generated_content(str(fix_result.tasks_output[0]))
            if path.startswith('tests/'):
                test_files_to_write[path] = file_result
            else:
                files_to_write[path] = file_result

        if files_to_write:
            write_file_map(files_to_write, self.out_dir, 'src')
        if test_files_to_write:
            write_file_map(test_files_to_write, self.out_dir, 'tests')

        return list(files_to_write.keys()) + list(test_files_to_write.keys())

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
