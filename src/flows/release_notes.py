from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import os
import re
import configparser
from datetime import datetime, timezone

from ..tools.rag_tools import DocsRAG
from ..crews.release_notes_update.crew import ReleaseNotesUpdateCrew
from ..flows.utils import load_json_output
from ..crews.docs_diff.output_format.doc_unified_diff import DOC_UNIFIED_DIFF_SCHEMA
from .utils import apply_combined_unified_diffs


def _find_release_notes_file(
    repo_dir: Path,
    docs_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Locate a release notes/changelog file in the repository.
    1) Exact filename scan across repo/docs
    2) Fallback via docs RAG by semantic search
    """
    _DOC_SUFFIXES = {".md", ".rst", ".txt", ".mdx"}
    _RELEASE_NOTES_NAME_RE = re.compile(r"(?i)\b(?:changelog|change(?:s|log)?|release[-_ ]?notes?|history|whats[-_ ]?new|news)\b")

    def scan_for_candidates(root: Path) -> Optional[Path]:
        try:
            # Walk and prune build directories
            for dirpath, dirs, files in os.walk(str(root)):
                # exclude 'build' directories from recursion
                dirs[:] = [d for d in dirs if d != "build"]
                for filename in files:
                    try:
                        p = Path(dirpath) / filename
                        if not p.is_file():
                            continue
                        name_lower = p.name.lower()
                        # suffix must be a documentation-like suffix
                        if not any(name_lower.endswith(suf) for suf in _DOC_SUFFIXES):
                            continue
                        # conservative regex match against the stem (exclude extension)
                        if _RELEASE_NOTES_NAME_RE.search(p.stem):
                            return p.resolve()
                    except Exception:
                        continue
        except Exception:
            return None
        return None

    def scan_upwards(start: Path, stop: Path) -> Optional[Path]:
        """Scan at start and each parent directory up to and including stop."""
        try:
            start = start.resolve()
            stop = stop.resolve()
        except Exception:
            return None
        visited: set[Path] = set()
        current = start
        while True:
            try:
                if current in visited:
                    break
                visited.add(current)
                found = scan_for_candidates(current)
                if found:
                    return found
                if current == stop or current.parent == current:
                    break
                current = current.parent
            except Exception:
                break
        return None

    # Prefer scanning from docs_dir upwards to repo_dir when docs_dir is provided
    if docs_dir:
        p = scan_upwards(docs_dir, repo_dir) or scan_for_candidates(repo_dir)
    else:
        p = scan_for_candidates(repo_dir)
    if p:
        return p

    docs_rag = DocsRAG(repo_dir=repo_dir, docs_dir=docs_dir) if docs_dir else None
    if docs_rag is not None:
        docs_rag.index()
        queries = ["release notes"]
        paths: list[str] = []
        for q in queries:
            res = docs_rag.search(query=q, top_k_files=3)
            paths.extend(res.get("paths", []) or [])

        def score(path: str) -> int:
            base = os.path.basename(path).lower()
            s = 0
            for tok in ("change", "changelog", "release", "releasenote", "history", "news"):
                if tok in base:
                    s += 1
            if base.endswith((".md", ".rst", ".txt")):
                s += 1
            return s

        paths = sorted(set(paths), key=lambda x: (-score(x), len(x)))
        # TODO: comprobar si es un archivo de release notes con una crew
        # TODO: gestionar la ruta de las release notes en .pydev/pydev.yaml
        for rel in paths:
            try:
                abs_p = (repo_dir / rel).resolve()
                if abs_p.exists() and abs_p.is_file():
                    return abs_p
            except Exception:
                continue

    return None


def _detect_current_version(repo_dir: Path, src_dir: Optional[Path]) -> Optional[str]:
    """Best-effort detection of current project version."""
    # 1) pyproject.toml
    try:
        pyproject = (repo_dir / "pyproject.toml").resolve()
        if pyproject.exists():
            text = pyproject.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"^version\s*=\s*['\"]([^'\"]+)['\"]", text, flags=re.MULTILINE)
            if m:
                return m.group(1).strip()
    except Exception:
        pass
    # 2) setup.cfg
    try:
        setup_cfg = (repo_dir / "setup.cfg").resolve()
        if setup_cfg.exists():
            parser = configparser.ConfigParser()
            parser.read([str(setup_cfg)], encoding="utf-8")
            if parser.has_section("metadata") and parser.has_option("metadata", "version"):
                ver = parser.get("metadata", "version", fallback="").strip()
                if ver:
                    return ver
    except Exception:
        pass
    # 3) setup.py
    try:
        setup_py = (repo_dir / "setup.py").resolve()
        if setup_py.exists():
            text = setup_py.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", text)
            if m:
                return m.group(1).strip()
    except Exception:
        pass
    # 4) __version__ in package __init__.py under src_dir
    try:
        if src_dir and src_dir.exists():
            for init_path in src_dir.rglob("__init__.py"):
                try:
                    t = init_path.read_text(encoding="utf-8", errors="ignore")
                    m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", t)
                    if m:
                        return m.group(1).strip()
                except Exception:
                    continue
    except Exception:
        pass
    # 5) git tag (most recent)
    try:
        import subprocess
        res = subprocess.run(["git", "describe", "--tags", "--abbrev=0"], cwd=str(repo_dir), check=False, capture_output=True, text=True)
        cand = (res.stdout or "").strip()
        if cand:
            cand = cand.lstrip("vV")
            if re.match(r"^[0-9]+\.[0-9]+(\.[0-9]+)?", cand):
                return cand
            return cand
    except Exception:
        pass
    return None


def _generate_release_notes_diff(original: str, version: str, today: str, doc_path: str, user_prompt: str = "", action_plan: str = "") -> str:
    """
    Delegate release notes content update to a Crew for consistency.
    """
    crew = ReleaseNotesUpdateCrew().crew()
    result = crew.kickoff(inputs={
        "user_prompt": user_prompt or "",
        "action_plan": action_plan or "",
        "current_content": original,
        "version": version,
        "date": today,
        "doc_path": doc_path,
    })
    try:
        text = load_json_output(result, DOC_UNIFIED_DIFF_SCHEMA)
        if isinstance(text, list):
            text = "\n".join([str(x) for x in text])
        return str(text)
    except Exception:
        return ""


def update_release_notes(
    repo_dir: Path,
    docs_dir: Optional[Path],
    src_dir: Optional[Path],
    user_prompt: str = "",
    action_plan: List[dict] | str = "",
) -> None:
    """Best-effort release notes update (idempotent)."""
    try:
        rn_path = _find_release_notes_file(repo_dir=repo_dir, docs_dir=docs_dir)
        print('**********************************************************')
        print('rn_path', rn_path)
        print('**********************************************************')
        if not rn_path:
            return

        try:
            original_text = rn_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            original_text = ""

        version = _detect_current_version(repo_dir=repo_dir, src_dir=src_dir) or "Unreleased"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Prepare text context for the crew: serialize action_plan succinctly if it's a list
        if isinstance(action_plan, list):
            try:
                plan_text = "\n".join(str((step or {}).get("description", "")).strip() for step in action_plan[:20])
            except Exception:
                plan_text = ""
        else:
            plan_text = str(action_plan or "")

        # Compute repo-relative doc path for headers (e.g., docs/CHANGELOG.md)
        try:
            doc_path_rel = str(rn_path.relative_to(repo_dir))
        except Exception:
            doc_path_rel = rn_path.name

        unified_diff = _generate_release_notes_diff(
            original=original_text,
            version=version,
            today=today,
            doc_path=doc_path_rel,
            user_prompt=user_prompt,
            action_plan=plan_text,
        )
        print('**********************************************************')
        print(unified_diff)
        print('**********************************************************')
        if unified_diff:
            apply_combined_unified_diffs(repo_dir, repo_dir, {str(rn_path): [unified_diff]})
    except Exception:
        # Best-effort; ignore failures
        pass
