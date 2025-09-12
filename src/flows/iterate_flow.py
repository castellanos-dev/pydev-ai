from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
from crewai.flow import Flow, start, listen
from .utils import ensure_repo, load_json_output
from ..crews.project_structure.crew import ProjectStructureCrew
from ..crews.project_structure.output_format.project_structure import PROJECT_STRUCTURE_SCHEMA
from ..crews.summaries.output_format.summaries import SUMMARIES_SCHEMA
from ..crews.summaries.repo_summaries_crew import RepoSummariesCrew
from ..crews.summaries.file_summaries_crew import FileSummariesCrew
from ..crews.summaries.module_summaries_crew import ModuleSummariesCrew
from .utils import write_file_map, sanitize_generated_content
from .. import settings


class IterateFlow(Flow):
    """
    CrewAI Flow for iterating on existing projects.

    Steps:
    1. Bootstrap knowledge if needed (digests + RAG)
    2. Execute IterateCrew with flow-level limits and guardrails
    """

    def _process_summaries_chunk(self, chunk: Dict[str, str]) -> Dict[str, str]:
        """
        Process a chunk of code files and generate summaries.

        Args:
            chunk: Dictionary mapping file paths to their content

        Returns:
            Dictionary mapping file paths to their summaries
        """
        result = RepoSummariesCrew().crew().kickoff(inputs={
            "code_chunk": chunk,
        })
        file_summaries = load_json_output(result, SUMMARIES_SCHEMA, 0)
        module_summaries = load_json_output(result, SUMMARIES_SCHEMA, 1)

        summaries = {}
        for s in file_summaries + module_summaries:
            summaries[s["path"]] = sanitize_generated_content(s["content"])

        return summaries

    def _process_file_summaries_chunk(self, chunk: Dict[str, str]) -> Dict[str, str]:
        """
        Generate per-file summaries for the given code chunk.

        Returns a dict mapping expected summary paths (relative to summaries root)
        to sanitized Markdown contents.
        """
        result = FileSummariesCrew().crew().kickoff(inputs={
            "code_chunk": chunk,
        })
        file_summaries = load_json_output(result, SUMMARIES_SCHEMA, 0)

        summaries: Dict[str, str] = {}
        for s in file_summaries:
            summaries[s["path"]] = sanitize_generated_content(s["content"])
        return summaries

    def _process_module_summaries_from_file_summaries(self, chunk: Dict[str, str]) -> Dict[str, str]:
        """
        Generate per-module summaries using ONLY the per-file summaries as context.

        The provided chunk must map source file relative paths (e.g., "pkg/mod.py")
        to the Markdown content of their corresponding file summaries. No real code
        is included, satisfying the requirement to base module summaries on summaries.
        """
        result = ModuleSummariesCrew().crew().kickoff(inputs={
            "invidual_summaries": chunk,
        })
        module_summaries = load_json_output(result, SUMMARIES_SCHEMA, 0)

        summaries: Dict[str, str] = {}
        for s in module_summaries:
            summaries[s["path"]] = sanitize_generated_content(s["content"])
        return summaries

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
        if self.summaries_dir and self.summaries_dir.exists():
            return inputs

        self.summaries_dir = self.repo_dir / "summaries"
        self.summaries_dir.mkdir(parents=True, exist_ok=True)

        py_paths = inputs["py_paths"]
        folders = set(file_path.parent for file_path in py_paths)

        modules: Dict[str, list[str]] = {}
        for folder in folders:
            modules[folder] = [p for p in py_paths if p.parent == folder]

        summaries: Dict[str, str] = {}
        chunk: Dict[str, str] = {}
        acc = 0
        # Process each group in manageable chunks by total characters
        for script_paths in modules.items().values():
            for path in script_paths:
                path_str = str(path.relative_to(self.src_dir))
                try:
                    chunk[path_str] = path.read_text(encoding="utf-8")
                except Exception:
                    continue
                acc += len(chunk[path_str])
            if acc > settings.MAX_CHARS and chunk:
                # run crew for current chunk
                chunk_summaries = self._process_summaries_chunk(chunk)
                summaries.update(chunk_summaries)
                # reset
                chunk = {}
                acc = 0

        if chunk:
            chunk_summaries = self._process_summaries_chunk(chunk)
            summaries.update(chunk_summaries)

        if summaries:
            write_file_map(summaries, str(self.repo_dir), "summaries")
            self.summaries_dir = self.repo_dir / "summaries"

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
            chunk: Dict[str, str] = {}
            acc = 0
            for rel_str in missing_file_rel_paths:
                code_path = (self.src_dir / rel_str).resolve()
                try:
                    content = code_path.read_text(encoding="utf-8")
                except Exception:
                    continue
                chunk[rel_str] = content
                acc += len(content)
                if acc > settings.MAX_CHARS:
                    generated = self._process_file_summaries_chunk(chunk)
                    new_file_summaries.update(generated)
                    chunk = {}
                    acc = 0
            if chunk:
                generated = self._process_file_summaries_chunk(chunk)
                new_file_summaries.update(generated)

        if new_file_summaries:
            write_file_map(new_file_summaries, str(self.repo_dir), "summaries")

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
            write_file_map(new_module_summaries, str(self.repo_dir), "summaries")
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
