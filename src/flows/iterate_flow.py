from __future__ import annotations
from pathlib import Path
import os
from typing import Dict, Any, List
import shutil
import yaml
from crewai.flow import Flow, start, listen
from .utils import ensure_repo, load_json_output, load_json_list, load_json_object
from ..crews.project_structure.crew import ProjectStructureCrew
from ..crews.project_structure.output_format.project_structure import PROJECT_STRUCTURE_SCHEMA
from .utils import to_yaml_file_map, write_file
from .common import (
    generate_file_summaries_from_chunk,
    generate_module_summaries_from_file_summaries,
)
from ..crews.planning import RelevanceCrew, FileDetailCrew, ActionPlanCrew
from ..crews.planning.output_format.relevant_files import RELEVANT_FILES_SCHEMA
from ..crews.planning.output_format.file_detail import FILE_DETAIL_SCHEMA
from ..crews.planning.output_format.action_plan import ACTION_PLAN_SCHEMA
from ..crews.rename_mapping.crew import RenameMappingCrew
from ..crews.rename_mapping.output_format.rename_map import RENAME_MAP_SCHEMA
from ..crews.move_mapping.crew import MoveMappingCrew
from ..crews.move_mapping.output_format.move_map import MOVE_MAP_SCHEMA
from ..crews.copy_mapping.crew import CopyMappingCrew
from ..crews.copy_mapping.output_format.copy_map import COPY_MAP_SCHEMA
from ..crews.development_diff.crew import (
    JuniorDevelopmentDiffCrew,
    SeniorDevelopmentDiffCrew,
    LeadDevelopmentDiffCrew,
)
from ..crews.development_diff.output_format.generate_diffs import GENERATE_DIFFS_SCHEMA
from ..crews.fix_integrator.crew import FixIntegratorCrew
from ..crews.tests_integrator.crew import TestsIntegratorCrew
from .utils import sanitize_generated_content
from ..crews.tests_conf.crew import TestsConfCrew
from ..crews.tests_conf.output_format.tests_conf import TESTS_CONF_SCHEMA
from ..crews.tests_planning.crew import (
    JuniorTestsPlanningCrew,
    SeniorTestsPlanningCrew,
    LeadTestsPlanningCrew,
)
from ..crews.tests_relevance.crew import TestsRelevanceCrew
from .utils import load_json_list, load_json_output
from ..crews.tests_relevance.output_format.relevant_tests import RELEVANT_TESTS_SCHEMA
from ..crews.tests_planning.output_format.test_plan import TEST_PLAN_SCHEMA
from ..crews.tests_implementation.crew import (
    JuniorTestsImplementationCrew,
    SeniorTestsImplementationCrew,
    LeadTestsImplementationCrew,
)
from ..crews.tests_implementation.output_format.implement_tests import IMPLEMENT_TESTS_SCHEMA
from ..tools.file_system import (
    write_empty_file,
    delete_file,
    create_directory,
    delete_directory,
    rename_file,
    move_file,
    copy_file,
)


class IterateFlow(Flow):
    """
    CrewAI Flow for iterating on existing projects.

    Steps:
    1. Bootstrap knowledge if needed (digests + RAG)
    2. Execute IterateCrew with flow-level limits and guardrails
    """

    def _process_file_summaries_chunk(self, chunk: List[Dict[str, str]]) -> Dict[str, str]:
        return generate_file_summaries_from_chunk(chunk)

    def _process_module_summaries_from_file_summaries(self, file_summaries: Dict[str, str]) -> Dict[str, str]:
        return generate_module_summaries_from_file_summaries(file_summaries)

    def _collect_module_file_summaries_from_py_paths(self, module_dir: Path, py_paths: List[Path]) -> Dict[str, str]:
        """
        Build a mapping of repo-relative Python paths -> file summary (YAML content)
        for all files under the provided module directory, using existing per-file
        summaries located under self.summaries_dir.
        """
        chunk: Dict[str, str] = {}
        if not self.summaries_dir:
            return chunk
        module_files = [p for p in py_paths if p.parent == module_dir]
        for py_file in module_files:
            rel_py = py_file.relative_to(self.src_dir)
            yaml_file = (self.summaries_dir / rel_py).with_suffix(".yaml")
            if not yaml_file.exists():
                continue
            try:
                yaml_content = yaml_file.read_text(encoding="utf-8")
            except Exception:
                continue
            chunk[str(rel_py)] = yaml_content
        return chunk

    def _collect_module_file_summaries(self, yaml_dir: Path) -> Dict[str, str]:
        """
        Build a mapping of repo-relative Python paths -> file summary (YAML content)
        by scanning a summaries module directory directly (excluding _module.yaml).
        """
        chunk: Dict[str, str] = {}
        if not self.summaries_dir or not yaml_dir.exists() or not yaml_dir.is_dir():
            return chunk
        for yaml_file in yaml_dir.glob("*.yaml"):
            if yaml_file.name == "_module.yaml":
                continue
            try:
                yaml_content = yaml_file.read_text(encoding="utf-8")
            except Exception:
                continue
            rel_yaml = yaml_file.relative_to(self.summaries_dir)
            rel_py = str(Path(rel_yaml).with_suffix(".py"))
            chunk[rel_py] = yaml_content
        return chunk

    def _write_pydev_snapshot(self) -> None:
        """
        Write a full snapshot of the current project state to pydev.yaml.
        Fields may be None or empty when not yet determined.
        """
        try:
            data: Dict[str, Any] = {
                "src_dir": self._to_repo_relative(self.src_dir),
                "docs_dir": self._to_repo_relative(self.docs_dir),
                "test_dirs": [self._to_repo_relative(p) for p in (self.test_dirs or [])],
                "summaries_dir": self._to_repo_relative(self.summaries_dir),
                "test": {
                    "framework": self.test_framework,
                    "command": self.test_command if self.test_command is not None else "",
                    "description": self.test_description if self.test_description is not None else "",
                    "examples": getattr(self, "test_examples", []) or [],
                },
            }
            with self.pydev_yaml_path.open("w", encoding="utf-8") as fp:
                yaml.safe_dump(data, fp, allow_unicode=True, sort_keys=False)
        except Exception:
            pass

    def _resolve_repo_path(self, value: str | None) -> Path | None:
        if not value:
            return None
        try:
            path = Path(value)
            if not path.is_absolute():
                path = (self.repo_dir / path)
            return path.resolve()
        except Exception:
            return None

    def _to_repo_relative(self, path: Path | None) -> str | None:
        if not path:
            return None
        try:
            return os.path.relpath(str(Path(path).resolve()), start=str(self.repo_dir))
        except Exception:
            return None

    def _load_pydev_snapshot(self) -> None:
        """
        Load initial configuration from pydev.yaml if present, resolving paths
        relative to repo_dir when necessary.
        """
        try:
            pydev_path = getattr(self, "pydev_yaml_path", None)
            if not pydev_path:
                return
            p = Path(pydev_path)
            if not p.exists() or not p.is_file():
                return
            with p.open("r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}

            # Directories
            for key in ("src_dir", "docs_dir"):
                resolved = self._resolve_repo_path(data.get(key))
                if key == "src_dir":
                    self.src_dir = resolved or self.src_dir
                elif key == "docs_dir":
                    self.docs_dir = resolved or self.docs_dir

            self.summaries_dir = (self.pydev_dir / "summaries").resolve()

            tds = [self._resolve_repo_path(td) for td in (data.get("test_dirs") or [])]
            self.test_dirs = [td for td in tds if td] or self.test_dirs

            # Test configuration
            test_cfg = data.get("test") or {}
            self.test_framework = test_cfg.get("framework", self.test_framework)
            self.test_command = test_cfg.get("command", self.test_command)
            self.test_description = test_cfg.get("description", self.test_description)
            self.test_examples = test_cfg.get("examples", getattr(self, "test_examples", [])) or []

        except Exception:
            # Best-effort loader; ignore errors
            pass

    def _regenerate_single_file_summary(self, code_path: Path, new_file_content: str) -> None:
        """
        Delete and regenerate the per-file summary for a given repo-relative Python file
        path using the provided latest file content.
        """
        if not self.summaries_dir:
            return
        summary_path = self.summaries_dir / code_path.relative_to(self.src_dir).with_suffix(".yaml")
        regenerated = self._process_file_summaries_chunk(new_file_content)
        if regenerated:
            if summary_path.exists():
                try:
                    summary_path.unlink()
                except Exception:
                    pass
            write_file(to_yaml_file_map(regenerated), summary_path)

    def _get_test_inputs_payload(self) -> Dict[str, Any]:
        payload = {}
        if self.test_dirs and len(self.test_dirs) > 0:
            self.test_file_paths = sorted(str(p) for test_dir in self.test_dirs for p in test_dir.glob("**/*.py"))
            payload = {
                "framework": self.test_framework or "",
                "command": self.test_command or "",
                "description": self.test_description or "",
                "test_files": self.test_file_paths,
                "examples": getattr(self, "test_examples", []) or [],
            }
        return payload

    def _should_test_be_modified(self) -> bool:
        return self.test_dirs is not None

    def _select_relevant_test_file(
        self,
        src_file: str,
        action_step_detail: str,
        test_file_paths: List[str],
    ) -> str:
        """
        Given all available test_files, a textual detail of one action-plan step,
        and the list of modified code files (repo-relative), return the single
        relevant test file that likely needs modifications due to code changes.
        This uses the TestsRelevance crew for selection.
        """
        if not src_file:
            return None
        result = TestsRelevanceCrew().crew().kickoff(inputs={
            "src_file": str(src_file),
            "action_step_detail": action_step_detail or "",
            "test_file_paths": test_file_paths,
        })
        decided = load_json_output(result, RELEVANT_TESTS_SCHEMA, 0)
        if isinstance(decided, str) and decided:
            return decided
        return None

    @start()
    def process_inputs(self) -> Dict[str, Any]:
        user_prompt = self.state["user_prompt"]
        repo = self.state["repo"]
        self.repo_dir = Path(ensure_repo(repo, check_empty=True)).resolve()
        self.summaries_dir = None
        self.test_dirs = None
        self.src_dir = None
        self.docs_dir = None
        self.test_framework = None
        self.test_command = None
        self.test_description = None
        self.pydev_dir = (self.repo_dir / ".pydev").resolve()
        self.pydev_dir.mkdir(parents=True, exist_ok=True)
        self.pydev_yaml_path = (self.pydev_dir / "pydev.yaml").resolve()
        if self.pydev_yaml_path.exists():
            self._load_pydev_snapshot()
        return {
            "user_prompt": user_prompt,
        }

    @listen(process_inputs)
    def identify_project_structure(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

        # Collect relevant files using glob: .py, .md, .rst
        patterns = ["**/*.md", "**/*.rst"]
        repo_py_files_list = sorted(str(p) for p in self.repo_dir.glob("**/*.py"))
        file_list = sorted(str(p) for pat in patterns for p in self.repo_dir.glob(pat))
        file_list.extend(repo_py_files_list)

        # If pydev.yaml already provided structure, skip detection
        if self.src_dir and self.src_dir.exists() and self.test_dirs:
            py_paths = [p for p in self.src_dir.rglob("*.py") if p.name != "__init__.py"]
            return {**inputs, "file_list": file_list, "py_paths": py_paths, "repo_py_files_list": repo_py_files_list}

        # 2) Fallback: Run the ProjectStructure crew
        result = ProjectStructureCrew().crew().kickoff(
            inputs={
                "files": file_list,
            }
        )

        structure = load_json_output(result, PROJECT_STRUCTURE_SCHEMA, 0)
        self.src_dir = Path(structure["code_dir"]).resolve()
        self.docs_dir = Path(structure["docs_dir"]).resolve() if structure["docs_dir"] else None
        self.test_dirs = [Path(test_dir).resolve() for test_dir in structure["test_dirs"]]
        self.summaries_dir = (self.pydev_dir / "summaries").resolve()
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        # Snapshot after discovering structure and enforcing dirs
        self._write_pydev_snapshot()
        # Collect Python files excluding __init__.py
        py_paths = [p for p in self.src_dir.rglob("*.py") if p.name != "__init__.py"]
        return {**inputs, "file_list": file_list, "py_paths": py_paths, "repo_py_files_list": repo_py_files_list}

    @listen(identify_project_structure)
    def generate_tests_conf(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine test framework, command, and description. If pydev.yaml already
        provides non-null test configuration, skip detection.
        """
        try:
            # If test configuration already loaded from pydev.yaml, skip
            if all([
                self.test_framework is not None,
                self.test_command is not None,
                self.test_description is not None,
            ]):
                return inputs

            # Build a small set of representative test-related files to help detection
            sample_paths: list[Path] = []
            for td in self.test_dirs:
                # conftest.py is highly indicative of pytest
                conf = td / "conftest.py"
                if conf.exists() and conf.is_file():
                    sample_paths.append(conf)
                # collect a few test_*.py files
                for f in td.glob("**/test_*.py"):
                    sample_paths.append(f)
                    if len(sample_paths) >= 10:
                        break
                if len(sample_paths) >= 10:
                    break

            # Also include common test-related configuration files from repo root
            config_names = [
                "pyproject.toml",
                "pytest.ini",
                "tox.ini",
                "setup.cfg",
                "noxfile.py",
                ".coveragerc",
                "Makefile",
            ]
            for name in config_names:
                p = self.repo_dir / name
                if p.exists() and p.is_file():
                    sample_paths.append(p)

            # Read and truncate samples
            def read_truncated(path: Path, limit: int = 2000) -> str:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    return text[:limit]
                except Exception:
                    return ""

            samples_lines: list[str] = []
            seen: set[str] = set()
            for sp in sample_paths[:10]:
                rel = str(sp.resolve())
                if rel in seen:
                    continue
                seen.add(rel)
                samples_lines.append(f"=== {rel} ===")
                samples_lines.append(read_truncated(sp))
            samples_str = "\n\n".join(samples_lines)

            tests_conf_result = TestsConfCrew().crew().kickoff(inputs={
                "src_dir": str(self.src_dir),
                "test_dirs": [str(p) for p in self.test_dirs],
                "file_list": inputs.get("repo_py_files_list", []),
                "samples": samples_str,
            })
            decided = load_json_object(tests_conf_result, TESTS_CONF_SCHEMA)
            self.test_framework = decided.get("framework")
            self.test_command = decided.get("command", "")
            self.test_description = decided.get("description", "")
            # Accept examples provided by TestsConfCrew
            self.test_examples = decided.get("examples", []) or []
            # Snapshot after deciding tests
            self._write_pydev_snapshot()
        except Exception:
            # best-effort; do not stop the flow
            pass
        return inputs

    @listen(generate_tests_conf)
    def generate_summaries_if_needed(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure every Python file has a corresponding summary and every module (folder)
        has a module summary. Missing file summaries are generated from code. Missing
        module summaries are generated using the per-file summaries as context (not code).
        """

        py_paths = inputs["py_paths"]

        # 1) Check and generate missing FILE summaries
        missing_file_rel_paths: list[str] = []
        for py_path in py_paths:
            rel = py_path.relative_to(self.src_dir)
            expected_md = (self.summaries_dir / rel).with_suffix(".yaml")
            if not expected_md.exists():
                missing_file_rel_paths.append(str(rel))

        new_file_summaries: Dict[str, Any] = {}
        if missing_file_rel_paths:
            # Agrupar por módulo (carpeta) y hacer una única llamada por módulo
            py_paths = inputs["py_paths"]
            module_dirs = sorted({p.parent for p in py_paths})
            missing_set = set(missing_file_rel_paths)
            for module_dir in module_dirs:
                # Archivos faltantes dentro de este módulo
                rel_missing_in_module: list[str] = [
                    str(p.relative_to(self.src_dir))
                    for p in py_paths
                    if p.parent == module_dir and str(p.relative_to(self.src_dir)) in missing_set
                ]
                if not rel_missing_in_module:
                    continue
                for rel_str in rel_missing_in_module:
                    code_path = (self.src_dir / rel_str).resolve()
                    try:
                        content = code_path.read_text(encoding="utf-8")
                    except Exception:
                        continue
                    generated = self._process_file_summaries_chunk(content)
                    if generated:
                        yaml_dir = (self.summaries_dir / rel_str).with_suffix(".yaml")
                        write_file(to_yaml_file_map(generated), yaml_dir)
                        new_file_summaries.update(generated)

        # 2) Check and generate missing MODULE summaries using existing file summaries
        # Determine module folders (parents of Python files)
        module_dirs = sorted({p.parent for p in py_paths})
        missing_module_dirs: list[Path] = []
        for folder in module_dirs:
            rel_dir = folder.relative_to(self.src_dir)
            expected_module_yaml = (self.summaries_dir / rel_dir / "_module.yaml").resolve()
            if not expected_module_yaml.exists():
                missing_module_dirs.append(expected_module_yaml)

        new_module_summaries: Dict[str, Any] = {}
        if missing_module_dirs:
            for module_dir_yaml in missing_module_dirs:
                # Build input using only the file summaries within this module directory
                chunk: Dict[str, str] = self._collect_module_file_summaries_from_py_paths(module_dir, py_paths)
                if not chunk:
                    continue
                generated = self._process_module_summaries_from_file_summaries(chunk)
                if generated:
                    # Persist each module summary immediately (intermediate save)
                    write_file(to_yaml_file_map(generated), module_dir_yaml)
                    new_module_summaries.update(generated)

        return {
            "user_prompt": inputs["user_prompt"],
        }

    @listen(generate_summaries_if_needed)
    def action_plan(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Phase 1: deterministically load all module summaries
        module_summaries: Dict[str, str] = {}
        if not self.summaries_dir:
            return inputs
        for yaml_path in self.summaries_dir.rglob("_module.yaml"):
            try:
                rel = str(yaml_path.relative_to(self.summaries_dir))
                module_summaries[rel] = yaml_path.read_text(encoding="utf-8")
            except Exception:
                continue

        # Use RelevanceCrew to select relevant file summary paths
        user_prompt = inputs["user_prompt"]
        relevance_result = RelevanceCrew().crew().kickoff(inputs={
            "src_dir": str(self.src_dir),
            "user_prompt": user_prompt,
            "module_summaries": module_summaries,
        })
        relevant_paths: List[str] = load_json_list(relevance_result, RELEVANT_FILES_SCHEMA)

        # Phase 2: load relevant file summaries content deterministically
        relevant_map: Dict[str, str] = {}
        for rel_py in relevant_paths:
            yaml_file = (self.summaries_dir / Path(rel_py).relative_to(self.src_dir)).with_suffix(".yaml").resolve()
            try:
                if yaml_file.exists():
                    rel_md = str(yaml_file.relative_to(self.summaries_dir))
                    relevant_map[rel_md] = yaml_file.read_text(encoding="utf-8")
            except Exception:
                continue

        if relevant_map:
            # Classify into summaries_only and need_code
            detail_result = FileDetailCrew().crew().kickoff(inputs={
                "user_prompt": user_prompt,
                "relevant_file_summaries": relevant_map,
            })
            detail = load_json_object(detail_result, FILE_DETAIL_SCHEMA)
            summaries_only: List[str] = detail.get("summaries_only", [])
            need_code: List[str] = detail.get("need_code", [])
        else:
            summaries_only = []
            need_code = []

        # Deterministically read code for the need_code set (map file.yaml -> {code,path})
        code_map: Dict[str, str] = {}
        for rel_yaml in need_code:
            # Convert summaries path like "pkg/mod/file.yaml" -> source file path under src_dir
            src_rel = rel_yaml[:-5] if rel_yaml.endswith(".yaml") else rel_yaml
            code_file = (self.src_dir / src_rel).with_suffix(".py").resolve()
            try:
                if code_file.exists():
                    code_map[str(code_file.relative_to(self.src_dir))] = code_file.read_text(encoding="utf-8")
            except Exception:
                continue

        file_list = sorted(str(p) for p in self.src_dir.glob("**/*.py"))

        # Phase 3: generate action plan
        plan_result = ActionPlanCrew().crew().kickoff(inputs={
            "src_dir": str(self.src_dir),
            "file_list": file_list,
            "user_prompt": user_prompt,
            "summaries": {k: relevant_map.get(k, "") for k in summaries_only},
            "code": code_map,
        })
        action_plan = load_json_list(plan_result, ACTION_PLAN_SCHEMA)

        return {**inputs, "action_plan": action_plan}

    @listen(action_plan)
    def execute_action_plan(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the file-system related steps from the action plan.

        Supported types -> function mapping:
          - "Create new file" -> write_empty_files
          - "Delete file" -> delete_files
          - "Create new directory" -> create_directories
          - "Delete directory" -> delete_directories
          - "Rename file" -> rename_files
          - "Move file" -> move_files
          - "Copy file" -> copy_files

        "Modify code" steps are recorded but left unimplemented for now.
        """

        plan = inputs.get("action_plan", [])

        def resolve_path(p: str) -> str:
            path = Path(p).expanduser()
            if path.is_absolute():
                return str(path)
            return str((self.repo_dir / p).resolve())

        # --- Helpers to mirror operations into summaries directory ---
        def _is_py_file_under_src(abs_path: str) -> bool:
            try:
                p = Path(abs_path).resolve()
                # Must be under src_dir
                _ = p.relative_to(self.src_dir)
            except Exception:
                return False
            return p.suffix == ".py" and p.name != "__init__.py"

        def _summary_path_for_code(abs_path: str) -> Path | None:
            if not self.summaries_dir:
                return None
            p = Path(abs_path).resolve()
            try:
                rel = p.relative_to(self.src_dir)
            except Exception:
                return None
            return (self.summaries_dir / rel).with_suffix(".yaml")

        def _mirror_delete_files(file_paths: List[str]) -> None:
            if not self.summaries_dir:
                return
            for fp in file_paths:
                if not _is_py_file_under_src(fp):
                    continue
                sp = _summary_path_for_code(fp)
                if sp and sp.exists():
                    try:
                        sp.unlink()
                    except Exception:
                        pass

        def _mirror_create_dirs(dir_paths: List[str]) -> None:
            if not self.summaries_dir:
                return
            for dp in dir_paths:
                try:
                    p = Path(dp).resolve()
                    rel = p.relative_to(self.src_dir)
                except Exception:
                    continue
                target_dir = (self.summaries_dir / rel).resolve()
                target_dir.mkdir(parents=True, exist_ok=True)

        def _mirror_delete_dirs(dir_paths: List[str]) -> None:
            if not self.summaries_dir:
                return
            for dp in dir_paths:
                try:
                    p = Path(dp).resolve()
                    rel = p.relative_to(self.src_dir)
                except Exception:
                    continue
                target_dir = (self.summaries_dir / rel).resolve()
                if target_dir.exists():
                    try:
                        shutil.rmtree(target_dir)
                    except Exception:
                        pass

        def _mirror_move_file(src: str, dst: str, op: str) -> None:
            if not self.summaries_dir:
                return
            # Only mirror for Python source files under src_dir
            if not _is_py_file_under_src(src) and not _is_py_file_under_src(dst):
                return
            sp_src = _summary_path_for_code(src)
            sp_dst = _summary_path_for_code(dst)
            if not sp_dst:
                return
            sp_dst.parent.mkdir(parents=True, exist_ok=True)
            if sp_src and sp_src.exists():
                try:
                    if op in {"rename", "move"}:
                        sp_src.rename(sp_dst)
                    elif op == "copy":
                        shutil.copy2(str(sp_src), str(sp_dst))
                except Exception:
                    # As a fallback, ensure destination exists
                    if not sp_dst.exists():
                        sp_dst.touch()
            else:
                # If source summary doesn't exist, create an empty destination summary
                if not sp_dst.exists():
                    sp_dst.touch()

        # --- Helpers to mirror operations into tests directories ---
        def _mirror_tests_delete_files(file_paths: List[str]) -> None:
            if not self._should_test_be_modified():
                return
            for fp in file_paths:
                if not _is_py_file_under_src(fp):
                    continue
                # Prefer path suggested by mapping helper if it exists
                try:
                    src_rel = str(Path(fp).resolve().relative_to(self.src_dir))
                    mapped = _map_src_path_to_test_path(src_rel, [])  # type: ignore[arg-type]
                    if mapped:
                        tp = (self.repo_dir / mapped).resolve() if not mapped.is_absolute() else mapped
                        if tp.exists():
                            try:
                                tp.unlink()
                            except Exception:
                                pass
                            continue
                except Exception:
                    pass

        def _mirror_tests_create_dirs(dir_paths: List[str]) -> None:
            if not self._should_test_be_modified():
                return
            for dp in dir_paths:
                try:
                    rel = Path(dp).resolve().relative_to(self.src_dir)
                except Exception:
                    continue
                # TODO: añadir en el pydev.yaml un flag de si los tests estan generados como un mirror de los source files
                # TODO: hace falta una crew que infiera la ruta

        def _mirror_tests_delete_dirs(dir_paths: List[str], step_plan: List[dict]) -> None:
            if not self._should_test_be_modified():
                return
            for dp in dir_paths:
                target_dir = _get_test_path(dp, step_plan)
                if target_dir.exists():
                    try:
                        shutil.rmtree(target_dir)
                    except Exception:
                        pass

        def _mirror_tests_move_file(src: str, dst: str, step_plan: List[dict]) -> None:
            if not self._should_test_be_modified():
                return
            if not _is_py_file_under_src(src) and not _is_py_file_under_src(dst):
                return
            target_dir = _get_test_path(src, step_plan)
            # TODO: hacer un mirror para la estructura destino y moverlo

        self.test_file_paths = None
        test_inputs_payload = self._get_test_inputs_payload()

        created: list[str] = []
        deleted_files: list[str] = []
        created_dirs: list[str] = []
        deleted_dirs: list[str] = []
        renamed: list[str] = []
        moved: list[str] = []
        copied: list[str] = []
        errors: list[dict] = []
        modifications: list[dict] = []
        generated_tests_code: Dict[str, Dict[str, List[str]]] = {}
        modules_to_refresh = set()

        for step in plan:
            step_type = (step.get("type") or "").strip()
            path_str: str = (step.get("path") or "").strip()
            if not path_str:
                continue
            try:
                if step_type == "Create new file":
                    created.append(write_empty_file(path_str))
                elif step_type == "Delete file":
                    # Mirror summaries for deleted source files
                    _mirror_delete_files(path_str)
                    # Mirror tests for deleted source files
                    _mirror_tests_delete_files(path_str)
                    deleted_files.append(delete_file(path_str))
                    # Mark affected modules for refresh
                    if _is_py_file_under_src(path_str):
                        try:
                            rel = Path(path_str).resolve().relative_to(self.src_dir)
                            modules_to_refresh.add(rel.parent)
                        except Exception:
                            pass
                elif step_type == "Create new directory":
                    # Mirror summaries directory structure for created source directories
                    _mirror_create_dirs(path_str)
                    # Mirror tests directory structure for created source directories
                    _mirror_tests_create_dirs(path_str)
                    created_dirs.append(create_directory(path_str))
                elif step_type == "Delete directory":
                    # Mirror summaries directory deletion for source directories
                    _mirror_delete_dirs(path_str)
                    # Mirror tests directory deletion for source directories
                    _mirror_tests_delete_dirs(path_str)
                    deleted_dirs.append(delete_directory(path_str))
                elif step_type == "Rename file":
                    rm_result = RenameMappingCrew().crew().kickoff(inputs={
                        "input": {"input_path": path_str},
                    })
                    rename_map = load_json_object(rm_result, RENAME_MAP_SCHEMA)
                    if not rename_map:
                        continue
                    src = list(rename_map.keys())[0]
                    dst = rename_map[src]
                    # Mirror summaries rename
                    _mirror_move_file(src, dst, op="rename")
                    # Mirror tests rename
                    _mirror_tests_move_file(src, dst, step_plan=step)
                    # Perform rename and track
                    renamed.append(rename_file(src, dst))
                    if _is_py_file_under_src(src):
                        try:
                            rel = Path(src).resolve().relative_to(self.src_dir)
                            modules_to_refresh.add(rel.parent)
                        except Exception:
                            pass
                    if _is_py_file_under_src(dst):
                        try:
                            rel = Path(dst).resolve().relative_to(self.src_dir)
                            modules_to_refresh.add(rel.parent)
                        except Exception:
                            pass
                elif step_type == "Move file":
                    mm_result = MoveMappingCrew().crew().kickoff(inputs={
                        "input": {"input_path": path_str},
                    })
                    move_map = load_json_object(mm_result, MOVE_MAP_SCHEMA)
                    if not move_map:
                        continue
                    src = list(move_map.keys())[0]
                    dst = move_map[src]
                    # Mirror summaries move
                    _mirror_move_file(src, dst, op="move")
                    # Mirror tests move
                    _mirror_tests_move_file(src, dst, step_plan=step)
                    moved.append(move_file(src, dst))
                    # Mark both source and destination modules for refresh
                    if _is_py_file_under_src(src):
                        try:
                            rel = Path(src).resolve().relative_to(self.src_dir)
                            modules_to_refresh.add(rel.parent)
                        except Exception:
                            pass
                    if _is_py_file_under_src(dst):
                        try:
                            rel = Path(dst).resolve().relative_to(self.src_dir)
                            modules_to_refresh.add(rel.parent)
                        except Exception:
                            pass
                elif step_type == "Copy file":
                    cm_result = CopyMappingCrew().crew().kickoff(inputs={
                        "input": {"input_path": path_str},
                    })
                    copy_map = load_json_object(cm_result, COPY_MAP_SCHEMA)
                    if not copy_map:
                        continue
                    src = list(copy_map.keys())[0]
                    dst = copy_map[src]
                    # Mirror summaries copy
                    _mirror_move_file(src, dst, op="copy")
                    # Mirror tests copy
                    _mirror_tests_move_file(src, dst, step_plan=step)
                    copied.append(copy_file(src, dst))
                    # Mark destination modules for refresh
                    if _is_py_file_under_src(dst):
                        try:
                            rel = Path(dst).resolve().relative_to(self.src_dir)
                            modules_to_refresh.add(rel.parent)
                        except Exception:
                            pass
                elif step_type == "Modify code":
                    # Choose diff-based development crew by points (1=junior, 2=senior, 3=lead)
                    points = int(step.get("points", 1) or 1)
                    if points <= 1:
                        dev_crew = JuniorDevelopmentDiffCrew()
                        test_planner_crew = JuniorTestsPlanningCrew()
                        test_implementer_crew = JuniorTestsImplementationCrew()
                    elif points == 2:
                        dev_crew = SeniorDevelopmentDiffCrew()
                        test_planner_crew = SeniorTestsPlanningCrew()
                        test_implementer_crew = SeniorTestsImplementationCrew()
                    else:
                        dev_crew = LeadDevelopmentDiffCrew()
                        test_planner_crew = LeadTestsPlanningCrew()
                        test_implementer_crew = LeadTestsImplementationCrew()

                    # Build inputs: map repo-relative path -> current file content
                    file_code: Dict[str, str] = {}
                    try:
                        if path_str:
                            abs_path = resolve_path(path_str)
                            path = Path(abs_path).resolve()
                            content = path.read_text(encoding="utf-8")
                            file_code[str(path.relative_to(self.src_dir))] = content
                    except Exception:
                        content = ""

                    dev_result = dev_crew.crew().kickoff(inputs={
                        "instructions": step,
                        "file_code": file_code,
                        "src_dir": str(self.src_dir),
                    })

                    # Parse diffs and integrate like bug resolution flow
                    file_changes = load_json_output(dev_result, GENERATE_DIFFS_SCHEMA)
                    modifications.extend(file_changes)

                    # Step 2: update test file
                    if len(test_inputs_payload) > 0 and self._should_test_be_modified():
                        # Build inputs for the crew
                        inputs_payload = {**test_inputs_payload, **{
                            "src_dir": str(self.src_dir),
                            "action_plan": step.get("step"),
                            "modified_files": file_changes,
                        }}
                        crew_result = test_planner_crew.crew().kickoff(inputs=inputs_payload)
                        tests_plan = load_json_output(crew_result, TEST_PLAN_SCHEMA)
                        # Aggregate planned tests by original source file (src_file)
                        for test_item in tests_plan:
                            try:
                                src_file = (test_item or {}).get("src_file")
                            except AttributeError:
                                continue
                            if not src_file:
                                continue

                            if src_file not in generated_tests_code:
                                generated_tests_code[src_file] = {'code': [], 'test_plan': []}
                            generated_tests_code[src_file]['test_plan'].append(test_item)

                        # Implement tests code (return only code snippets, no paths)
                        impl_inputs = {
                            "framework": self.test_framework or "",
                            "test_context": self.test_description or "",
                            "examples": "\n\n".join(getattr(self, "test_examples", []) or []),
                            "test_plan": tests_plan,
                            "src_code": content,
                            "file_changes": file_changes,
                        }
                        impl_result = test_implementer_crew.crew().kickoff(inputs=impl_inputs)
                        impl_code_list = load_json_output(impl_result, IMPLEMENT_TESTS_SCHEMA)
                        for item in impl_code_list or []:
                            try:
                                code_str = (item or {}).get("code")
                            except AttributeError:
                                continue
                            if code_str:
                                generated_tests_code[src_file]['code'].append(code_str)

                else:
                    errors.append({
                        "step": step.get("step"),
                        "type": step_type,
                        "error": f"Unsupported step type: {step_type}",
                    })
            except Exception as exc:
                errors.append({
                    "step": step.get("step"),
                    "type": step_type,
                    "error": str(exc),
                })

        changes_by_file: Dict[str, List[str]] = {}
        for modification in modifications:
            path = modification.get("path")
            content_diff = modification.get("content_diff")
            if path not in changes_by_file:
                changes_by_file[path] = []
            changes_by_file[path].append(content_diff)

        modules_to_refresh = set()
        for path, changes in changes_by_file.items():
            # Retrieve original code from disk
            try:
                code_path = (self.src_dir / path).resolve()
                original_code = code_path.read_text(encoding="utf-8")
            except Exception:
                code_path = None
                original_code = "-"
            fix_result = FixIntegratorCrew().crew().kickoff(
                inputs={
                    "original_code": original_code,
                    "code_fixes": changes,
                }
            )
            file_result = sanitize_generated_content(str(fix_result.tasks_output[0]))
            # Accumulate to write; currently tracking only
            write_file(file_result, code_path)

            # For modified files: delete original summary and regenerate a new one
            try:
                if code_path is not None:
                    # Track module directory for later module summary regeneration
                    try:
                        modules_to_refresh.add(Path(path).parent)
                    except Exception:
                        pass
                    self._regenerate_single_file_summary(code_path, file_result)
            except Exception:
                # Best-effort; do not fail on summary regeneration issues
                pass

        # --- Integrate generated unit tests into mirrored test files ---
        def _map_src_path_to_test_path(src_path: str, test_plan: List[dict]) -> Path | None:
            """
            Given a repo-relative source path (under src_dir), return a tuple of
            (chosen_test_root, repo-relative test file path under that root) using
            a mirror strategy: tests/<same_dir>/test_<module>.py
            Prefer an existing file if found.
            """
            try:
                src_path = Path(src_path).relative_to(self.src_dir)
                test_path_dir = src_path.parent
                test_file_name = f"test_{src_path.stem}.py"
                for test_dir in self.test_dirs:
                    candidate_path = test_dir / test_path_dir / test_file_name
                    if candidate_path.exists():
                        return candidate_path
            except Exception:
                pass
            return _get_test_path(src_path, test_plan)

        def _get_test_path(src_file: str, test_plan: List[dict]) -> Path | None:
            test_file = self._select_relevant_test_file(
                src_file=src_file,
                action_step_detail=test_plan,
                test_file_paths=self.test_file_paths,
            )
            if Path(test_file).exists():
                return Path(test_file)
            return None

        def _collect_nearby_fixtures(test_file: Path) -> str:
            """
            Collect fixture definitions from closest conftest.py upwards to test_root.
            """
            current = test_file
            walked = set()
            collected: list[str] = []
            while True:
                try:
                    if current in walked:
                        break
                    walked.add(current)
                    conf = (current / "conftest.py").resolve()
                    if conf.exists() and conf.is_file():
                        try:
                            collected.append(conf.read_text(encoding="utf-8"))
                        except Exception:
                            pass
                    if current in self.test_dirs:
                        break
                    current = current.parent
                except Exception:
                    break
            return "\n\n".join(collected)

        if generated_tests_code:
            for src_rel, snippets in generated_tests_code.items():
                if 'code' not in snippets or not snippets['code']:
                    continue
                if 'test_plan' not in snippets or not snippets['test_plan']:
                    continue
                # Determine test file location by mirroring src path
                test_file = _map_src_path_to_test_path(src_rel, snippets['test_plan'])
                if not test_file:
                    continue
                try:
                    original_test_code = test_file.read_text(encoding="utf-8")
                except Exception:
                    continue
                # Collect nearest fixtures
                fixtures_text = _collect_nearby_fixtures(test_file)

                # Integrate using advanced tests integrator crew
                ti_result = TestsIntegratorCrew().crew().kickoff(inputs={
                    "framework": self.test_framework or "",
                    "original_test_code": original_test_code,
                    "generated_tests_code": snippets,
                    "available_fixtures": fixtures_text,
                })
                final_test_code = sanitize_generated_content(str(ti_result.tasks_output[0]))
                # Write via deterministic writer relative to test root
                write_file(final_test_code, test_file)

        # Regenerate module summaries (_module.yaml) for affected modules
        for module_rel in sorted(modules_to_refresh, key=lambda p: str(p)):
            try:
                module_yaml_path = (self.summaries_dir / module_rel / "_module.yaml").resolve()
                module_yaml_path.parent.mkdir(parents=True, exist_ok=True)
                # Build input chunk using only per-file summaries in this module directory (exclude _module.yaml)
                chunk: Dict[str, str] = self._collect_module_file_summaries(module_yaml_path.parent)
                if not chunk:
                    continue
                generated = self._process_module_summaries_from_file_summaries(chunk)
                if generated:
                    # Persist each module summary immediately (intermediate save)
                    write_file(to_yaml_file_map(generated), module_yaml_path)
            except Exception:
                # Best-effort; do not fail the flow if module regen fails
                continue

        execution_summary: Dict[str, Any] = {
            "created_files": created,
            "deleted_files": deleted_files,
            "created_directories": created_dirs,
            "deleted_directories": deleted_dirs,
            "renamed_files": renamed,
            "moved_files": moved,
            "copied_files": copied,
            "modified_files": modifications,
            "errors": errors,
        }

        return {**inputs, "execution_summary": execution_summary}

    def run(self, user_prompt: str, repo: str) -> Dict[str, Any]:
        """Convenience method for CLI integration."""
        return self.kickoff(inputs={"user_prompt": user_prompt, "repo": repo})


def run_iterate(user_prompt: str, repo: str) -> None:
    """
    Execute the iterate flow using CrewAI Flows.

    This function maintains backward compatibility with the CLI
    while using the new Flow-based architecture internally.
    """

    flow = IterateFlow()
    result = flow.kickoff(inputs={"user_prompt": user_prompt, "repo": repo})

    print(f"Iterate flow completed: {result}")
    return result
