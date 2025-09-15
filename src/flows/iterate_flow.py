from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import shutil
from crewai.flow import Flow, start, listen
from .utils import ensure_repo, load_json_output, load_json_list, load_json_object
from ..crews.project_structure.crew import ProjectStructureCrew
from ..crews.project_structure.output_format.project_structure import PROJECT_STRUCTURE_SCHEMA
from ..crews.summaries.output_format.summaries_dir import SUMMARIES_DIR_SCHEMA
from ..crews.summaries.summaries_dir_crew import SummariesDirCrew
from .utils import write_file_map
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
from .utils import sanitize_generated_content
from ..tools.file_system import (
    write_empty_files,
    delete_files,
    create_directories,
    delete_directories,
    rename_files,
    move_files,
    copy_files,
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
        Build a mapping of repo-relative Python paths -> file summary (markdown content)
        for all files under the provided module directory, using existing per-file
        summaries located under self.summaries_dir.
        """
        chunk: Dict[str, str] = {}
        if not self.summaries_dir:
            return chunk
        module_files = [p for p in py_paths if p.parent == module_dir]
        for py_file in module_files:
            rel_py = py_file.relative_to(self.src_dir)
            md_file = (self.summaries_dir / rel_py).with_suffix(".md")
            if not md_file.exists():
                continue
            try:
                md_content = md_file.read_text(encoding="utf-8")
            except Exception:
                continue
            chunk[str(rel_py)] = md_content
        return chunk

    def _collect_module_file_summaries_from_md_dir(self, md_dir: Path) -> Dict[str, str]:
        """
        Build a mapping of repo-relative Python paths -> file summary (markdown content)
        by scanning a summaries module directory directly (excluding _module.md).
        """
        chunk: Dict[str, str] = {}
        if not self.summaries_dir or not md_dir.exists() or not md_dir.is_dir():
            return chunk
        for md_file in md_dir.glob("*.md"):
            if md_file.name == "_module.md":
                continue
            try:
                md_content = md_file.read_text(encoding="utf-8")
            except Exception:
                continue
            rel_md = md_file.relative_to(self.summaries_dir)
            rel_py = str(Path(rel_md).with_suffix(".py"))
            chunk[rel_py] = md_content
        return chunk

    def _regenerate_single_file_summary(self, rel_path: str, new_file_content: str) -> None:
        """
        Delete and regenerate the per-file summary for a given repo-relative Python file
        path using the provided latest file content.
        """
        if not self.summaries_dir:
            return
        try:
            code_path = (self.src_dir / rel_path).resolve()
        except Exception:
            return
        if code_path.suffix != ".py" or code_path.name == "__init__.py":
            return
        summary_path = (self.summaries_dir / Path(rel_path)).with_suffix(".md").resolve()
        if summary_path.exists():
            try:
                summary_path.unlink()
            except Exception:
                pass
        regenerated = self._process_file_summaries_chunk([
            {"path": rel_path, "content": new_file_content}
        ])
        if regenerated:
            write_file_map(regenerated, str(self.summaries_dir))

    @start()
    def process_inputs(self) -> Dict[str, Any]:
        user_prompt = self.state["user_prompt"]
        repo = self.state["repo"]
        self.repo_dir = Path(ensure_repo(repo, check_empty=True)).resolve()
        return {
            "user_prompt": user_prompt,
        }

    @listen(process_inputs)
    def identify_project_structure(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

        # Collect relevant files using glob: .py, .md, .rst
        patterns = ["**/*.py", "**/*.md", "**/*.rst"]
        file_list = []
        for pattern in patterns:
            file_list.extend(sorted(str(p) for p in self.repo_dir.glob(pattern)))

        # Run the ProjectStructure crew
        result = ProjectStructureCrew().crew().kickoff(
            inputs={
                "files": file_list,
            }
        )

        structure = load_json_output(result, PROJECT_STRUCTURE_SCHEMA, 0)
        self.src_dir = Path(structure["code_dir"]).resolve()
        self.docs_dir = Path(structure["docs_dir"]).resolve() if structure["docs_dir"] else None
        self.test_dirs = [Path(test_dir).resolve() for test_dir in structure["test_dirs"]]
        self.summaries_dir = Path(structure["summaries_dir"]).resolve() if structure.get("summaries_dir") else None
        # Collect Python files excluding __init__.py
        py_paths = [p for p in self.src_dir.rglob("*.py") if p.name != "__init__.py"]
        return {**inputs, "file_list": file_list, "py_paths": py_paths}

    @listen(identify_project_structure)
    def generate_summaries_if_needed(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # If summaries already exist, do nothing
        if self.summaries_dir and self.summaries_dir.exists() and any(self.summaries_dir.iterdir()):
            return inputs

        # Use crew to decide summaries_dir based on src_dir, docs_dir, and test_dirs
        result = SummariesDirCrew().crew().kickoff(inputs={
            "src_dir": str(self.src_dir),
            "docs_dir": str(self.docs_dir) if self.docs_dir else None,
            "test_dirs": [str(p) for p in self.test_dirs],
        })
        decided = load_json_output(result, SUMMARIES_DIR_SCHEMA, 0)
        # decided is dict-like from schema root
        summaries_dir_str = decided.get("summaries_dir") if isinstance(decided, dict) else decided[0]["summaries_dir"]
        try:
            self.summaries_dir = Path(summaries_dir_str).resolve()
        except Exception:
            self.summaries_dir = self.repo_dir / "summaries"
        self.summaries_dir.mkdir(parents=True, exist_ok=True)

        py_paths = inputs["py_paths"]
        folders = set(file_path.parent for file_path in py_paths)

        modules: Dict[str, list[str]] = {}
        for folder in folders:
            modules[folder] = [p for p in py_paths if p.parent == folder]

        # One crew call per module (folder)
        for script_paths in modules.values():
            chunk: List[str, str] = []
            for path in script_paths:
                rel_path = str(path.relative_to(self.src_dir))
                try:
                    chunk.append({"path": rel_path, "content": path.read_text(encoding="utf-8")})
                except Exception:
                    continue
            if not chunk:
                continue
            file_summaries = self._process_file_summaries_chunk(chunk)
            if not file_summaries:
                continue
            module_summaries = self._process_module_summaries_from_file_summaries(file_summaries)
            summaries: Dict[str, str] = {}
            summaries.update(file_summaries)
            summaries.update(module_summaries)
            if summaries:
                write_file_map(summaries, str(self.summaries_dir))

        return inputs

    @listen(generate_summaries_if_needed)
    def verify_and_fill_missing_summaries(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
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
            expected_md = (self.summaries_dir / rel).with_suffix(".md")
            if not expected_md.exists():
                missing_file_rel_paths.append(str(rel))

        new_file_summaries: Dict[str, str] = {}
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
                chunk: List[Dict[str, str]] = []
                for rel_str in rel_missing_in_module:
                    code_path = (self.src_dir / rel_str).resolve()
                    try:
                        content = code_path.read_text(encoding="utf-8")
                    except Exception:
                        continue
                    chunk.append({"path": rel_str, "content": content})
                if not chunk:
                    continue
                generated = self._process_file_summaries_chunk(chunk)
                new_file_summaries.update(generated)

        if new_file_summaries:
            write_file_map(new_file_summaries, str(self.summaries_dir))

        # 2) Check and generate missing MODULE summaries using existing file summaries
        # Determine module folders (parents of Python files)
        module_dirs = sorted({p.parent for p in py_paths})
        missing_module_dirs: list[Path] = []
        for folder in module_dirs:
            rel_dir = folder.relative_to(self.src_dir)
            expected_module_md = (self.summaries_dir / rel_dir / "_module.md").resolve()
            if not expected_module_md.exists():
                missing_module_dirs.append(folder)

        new_module_summaries: Dict[str, str] = {}
        if missing_module_dirs:
            for module_dir in missing_module_dirs:
                # Build input using only the file summaries within this module directory
                chunk: Dict[str, str] = self._collect_module_file_summaries_from_py_paths(module_dir, py_paths)
                if not chunk:
                    continue
                generated = self._process_module_summaries_from_file_summaries(chunk)
                new_module_summaries.update(generated)

        if new_module_summaries:
            write_file_map(new_module_summaries, str(self.summaries_dir))
        return {
            "user_prompt": inputs["user_prompt"],
        }

    @listen(verify_and_fill_missing_summaries)
    def action_plan(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Phase 1: deterministically load all module summaries
        module_summaries: Dict[str, str] = {}
        if not self.summaries_dir:
            return inputs
        for md_path in self.summaries_dir.rglob("_module.md"):
            try:
                rel = str(md_path.relative_to(self.summaries_dir))
                module_summaries[rel] = md_path.read_text(encoding="utf-8")
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
            md_file = (self.summaries_dir / Path(rel_py).relative_to(self.src_dir)).with_suffix(".md").resolve()
            try:
                if md_file.exists():
                    rel_md = str(md_file.relative_to(self.summaries_dir))
                    relevant_map[rel_md] = md_file.read_text(encoding="utf-8")
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

        # Deterministically read code for the need_code set (map file.md -> {code,path})
        code_map: Dict[str, str] = {}
        for rel_md in need_code:
            # Convert summaries path like "pkg/mod/file.md" -> source file path under src_dir
            src_rel = rel_md[:-3] if rel_md.endswith(".md") else rel_md
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
          - "Create new files" -> write_empty_files
          - "Delete files" -> delete_files
          - "Create new directories" -> create_directories
          - "Delete directories" -> delete_directories
          - "Rename files" -> rename_files
          - "Move files" -> move_files
          - "Copy files" -> copy_files

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
            return (self.summaries_dir / rel).with_suffix(".md")

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

        def _mirror_pair_files(pairs: List[tuple[str, str]], op: str) -> None:
            if not self.summaries_dir:
                return
            for src, dst in pairs:
                # Only mirror for Python source files under src_dir
                if not _is_py_file_under_src(src) and not _is_py_file_under_src(dst):
                    continue
                sp_src = _summary_path_for_code(src)
                sp_dst = _summary_path_for_code(dst)
                if not sp_dst:
                    continue
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

        created: list[str] = []
        deleted_files: list[str] = []
        created_dirs: list[str] = []
        deleted_dirs: list[str] = []
        renamed: list[str] = []
        moved: list[str] = []
        copied: list[str] = []
        code_todos: list[dict] = []
        errors: list[dict] = []
        modifications: list[dict] = []
        modules_to_refresh = set()

        for step in plan:
            step_type = (step.get("type") or "").strip()
            artifacts: list[str] = step.get("artifacts", []) or []
            try:
                if step_type == "Create new files":
                    files = [resolve_path(a) for a in artifacts]
                    created.extend(write_empty_files(files))
                elif step_type == "Delete files":
                    files = [resolve_path(a) for a in artifacts]
                    # Mirror summaries for deleted source files
                    _mirror_delete_files(files)
                    deleted_files.extend(delete_files(files))
                    # Mark affected modules for refresh
                    for fp in files:
                        if _is_py_file_under_src(fp):
                            try:
                                rel = Path(fp).resolve().relative_to(self.src_dir)
                                modules_to_refresh.add(rel.parent)
                            except Exception:
                                pass
                elif step_type == "Create new directories":
                    dirs = [resolve_path(a) for a in artifacts]
                    # Mirror summaries directory structure for created source directories
                    _mirror_create_dirs(dirs)
                    created_dirs.extend(create_directories(dirs))
                elif step_type == "Delete directories":
                    dirs = [resolve_path(a) for a in artifacts]
                    # Mirror summaries directory deletion for source directories
                    _mirror_delete_dirs(dirs)
                    deleted_dirs.extend(delete_directories(dirs))
                elif step_type == "Rename files":
                    rm_result = RenameMappingCrew().crew().kickoff(inputs={
                        "input": {"artifacts": artifacts},
                    })
                    rename_map = load_json_object(rm_result, RENAME_MAP_SCHEMA)
                    pairs = []
                    if isinstance(rename_map, dict) and rename_map:
                        pairs = [(resolve_path(k), resolve_path(v)) for k, v in rename_map.items() if k and v]
                    # Mirror summaries rename
                    _mirror_pair_files(pairs, op="rename")
                    # Perform rename and track
                    renamed.extend(rename_files(pairs))
                    # Mark both source and destination modules for refresh
                    for src_p, dst_p in pairs:
                        if _is_py_file_under_src(src_p):
                            try:
                                rel = Path(src_p).resolve().relative_to(self.src_dir)
                                modules_to_refresh.add(rel.parent)
                            except Exception:
                                pass
                        if _is_py_file_under_src(dst_p):
                            try:
                                rel = Path(dst_p).resolve().relative_to(self.src_dir)
                                modules_to_refresh.add(rel.parent)
                            except Exception:
                                pass
                elif step_type == "Move files":
                    mm_result = MoveMappingCrew().crew().kickoff(inputs={
                        "input": {"artifacts": artifacts},
                    })
                    move_map = load_json_object(mm_result, MOVE_MAP_SCHEMA)
                    pairs = []
                    if isinstance(move_map, dict) and move_map:
                        pairs = [(resolve_path(k), resolve_path(v)) for k, v in move_map.items() if k and v]
                    # Mirror summaries move
                    _mirror_pair_files(pairs, op="move")
                    moved.extend(move_files(pairs))
                    # Mark both source and destination modules for refresh
                    for src_p, dst_p in pairs:
                        if _is_py_file_under_src(src_p):
                            try:
                                rel = Path(src_p).resolve().relative_to(self.src_dir)
                                modules_to_refresh.add(rel.parent)
                            except Exception:
                                pass
                        if _is_py_file_under_src(dst_p):
                            try:
                                rel = Path(dst_p).resolve().relative_to(self.src_dir)
                                modules_to_refresh.add(rel.parent)
                            except Exception:
                                pass
                elif step_type == "Copy files":
                    cm_result = CopyMappingCrew().crew().kickoff(inputs={
                        "input": {"artifacts": artifacts},
                    })
                    copy_map = load_json_object(cm_result, COPY_MAP_SCHEMA)
                    pairs = []
                    if isinstance(copy_map, dict) and copy_map:
                        pairs = [(resolve_path(k), resolve_path(v)) for k, v in copy_map.items() if k and v]
                    # Mirror summaries copy
                    _mirror_pair_files(pairs, op="copy")
                    copied.extend(copy_files(pairs))
                    # Mark destination modules for refresh
                    for _src_p, dst_p in pairs:
                        if _is_py_file_under_src(dst_p):
                            try:
                                rel = Path(dst_p).resolve().relative_to(self.src_dir)
                                modules_to_refresh.add(rel.parent)
                            except Exception:
                                pass
                elif step_type == "Modify code":
                    # Choose diff-based development crew by points (1=junior, 2=senior, 3=lead)
                    points = int(step.get("points", 1) or 1)
                    if points <= 1:
                        dev_crew = JuniorDevelopmentDiffCrew()
                    elif points == 2:
                        dev_crew = SeniorDevelopmentDiffCrew()
                    else:
                        dev_crew = LeadDevelopmentDiffCrew()

                    # Build inputs: map repo-relative path -> current file content
                    file_code: Dict[str, str] = {}
                    for abs_path in artifacts:
                        try:
                            path = Path(abs_path).resolve()
                            content = path.read_text(encoding="utf-8")
                            file_code[str(path.relative_to(self.src_dir))] = content
                        except Exception:
                            continue

                    dev_result = dev_crew.crew().kickoff(inputs={
                        "instructions": step,
                        "file_code": file_code,
                    })

                    # Parse diffs and integrate like bug resolution flow
                    file_changes = load_json_output(dev_result, GENERATE_DIFFS_SCHEMA, 0)
                    modifications.extend(file_changes)
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
                original_code = (self.src_dir / path).read_text(encoding="utf-8")
            except Exception:
                original_code = "-"
            fix_result = FixIntegratorCrew().crew().kickoff(
                inputs={
                    "original_code": original_code,
                    "code_fixes": changes,
                }
            )
            file_result = sanitize_generated_content(str(fix_result.tasks_output[0]))
            # Accumulate to write; currently tracking only
            write_file_map({path: file_result}, str(self.src_dir))

            # For modified files: delete original summary and regenerate a new one
            try:
                code_path = (self.src_dir / path).resolve()
                if code_path.suffix == ".py" and code_path.name != "__init__.py":
                    # Track module directory for later module summary regeneration
                    try:
                        modules_to_refresh.add(Path(path).parent)
                    except Exception:
                        pass
                    self._regenerate_single_file_summary(path, file_result)
            except Exception:
                # Best-effort; do not fail on summary regeneration issues
                pass

        # Regenerate module summaries (_module.md) for affected modules
        if self.summaries_dir and modules_to_refresh:
            for module_rel in sorted(modules_to_refresh, key=lambda p: str(p)):
                try:
                    md_dir = (self.summaries_dir / module_rel).resolve()
                    if not md_dir.exists() or not md_dir.is_dir():
                        continue
                    # Build input chunk using only per-file summaries in this module directory (exclude _module.md)
                    chunk: Dict[str, str] = self._collect_module_file_summaries_from_md_dir(md_dir)
                    if not chunk:
                        continue
                    new_module_summaries = self._process_module_summaries_from_file_summaries(chunk)
                    # Remove existing module summary if present
                    module_md_path = (self.summaries_dir / module_rel / "_module.md").resolve()
                    if module_md_path.exists():
                        try:
                            module_md_path.unlink()
                        except Exception:
                            pass
                    if new_module_summaries:
                        write_file_map(new_module_summaries, str(self.summaries_dir))
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
            "code_modification_todos": code_todos,
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
