from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
from crewai.flow import Flow, start, listen
from .utils import ensure_repo, load_json_output
from ..crews.project_structure.crew import ProjectStructureCrew
from ..crews.project_structure.output_format.project_structure import PROJECT_STRUCTURE_SCHEMA
from ..crews.summaries.output_format.summaries_dir import SUMMARIES_DIR_SCHEMA
from ..crews.summaries.summaries_dir_crew import SummariesDirCrew
from .utils import write_file_map
from .common import (
    generate_file_summaries_from_chunk,
    generate_module_summaries_from_file_summaries,
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
                chunk: Dict[str, str] = {}
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
                if not chunk:
                    continue
                generated = self._process_module_summaries_from_file_summaries(chunk)
                new_module_summaries.update(generated)

        if new_module_summaries:
            write_file_map(new_module_summaries, str(self.summaries_dir))
        return inputs

    @listen(verify_and_fill_missing_summaries)
    def action_plan(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: implement action plan
        return inputs

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
