import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from ..summaries.storage import digests_root
from ..summaries.summarizer import bootstrap_digest
from ..crews.json_fixer import JSONFixerCrew


def ensure_repo(repo: str) -> Union[None, str]:
    """
    Initialize repository directory if needed.

    Args:
        repo: Repository path (absolute or relative)

    Returns:
        None if repo is absolute path, absolute path string if repo was relative

    Raises:
        ValueError: If repo is not a valid path or absolute path doesn't exist
    """
    # Initial path validation
    if not repo or not isinstance(repo, str):
        raise ValueError("Repository path must be a non-empty string")

    # Check if it's an absolute or relative path
    repo_path = Path(repo)

    if repo_path.is_absolute():
        # If it's an absolute path, verify it exists and is not empty
        if not repo_path.exists():
            raise ValueError(f"Absolute repository path does not exist: {repo}")
        if not repo_path.is_dir():
            raise ValueError(f"Repository path is not a directory: {repo}")
        if not any(repo_path.iterdir()):
            raise ValueError(f"Repository directory is empty: {repo}")

    else:
        # If it's a relative path, create the path if it doesn't exist
        repo_path = repo_path.resolve()
        if not repo_path.exists():
            repo_path.mkdir(parents=True, exist_ok=True)

    return str(repo_path)


def ensure_knowledge(repo: str) -> Union[None, str]:
    """
    Initialize knowledge directory if needed.

    Args:
        repo: Repository path (absolute or relative)

    Returns:
        None if repo is absolute path, absolute path string if repo was relative

    Raises:
        ValueError: If repo is not a valid path or absolute path doesn't exist
    """
    # Initial path validation
    if not repo or not isinstance(repo, str):
        raise ValueError("Repository path must be a non-empty string")

    # Check if it's an absolute or relative path
    # Ensure digests directory exists and bootstrap overview if missing.
    digests_dir = digests_root(repo)
    digests_dir.mkdir(parents=True, exist_ok=True)
    overview = digests_dir / "00_overview.md"
    if not overview.exists():
        bootstrap_digest(str(digests_dir.parent))
    return str(digests_dir)

def parse_summaries_output(text: str) -> List[Dict[str, str]]:
    """
    Parse the summaries output from the SummariesCrew.
    Expected shape (new):
      [{"path": str, "content": str}]
    Backward compatible: if "packages" appears, it is ignored by callers.
    """
    expected_kind = "summaries_output"
    expected_schema = "List of objects: {\"path\": str, \"content\": str}"
    obj: Any
    if len(text) <= 2:
        return []
    try:
        obj = json.loads(text)
    except Exception:
        fixed = _fix_json_text(text, expected_kind, expected_schema)
        obj = json.loads(fixed)
    if not isinstance(obj, list) or not all(
        isinstance(item, dict) and "path" in item and "content" in item for item in obj
    ):
        fixed = _fix_json_text(text, expected_kind, expected_schema)
        obj = json.loads(fixed)
        if not isinstance(obj, list) or not all(
            isinstance(item, dict) and "path" in item and "content" in item for item in obj
        ):
            raise ValueError(f"Invalid JSON after fix attempt: {text}")
    return obj


def normalize_file_map(obj: Any) -> List[Dict[str, str]]:
    """
    Normalize different shapes:
      - {"path": str, "content": str}
      - {"files": [{"path":..., "content":...}, ...]}
      - [{"path":..., "content":...}, ...]
    Returns a flat list of {"path","content"} dicts.
    """
    if isinstance(obj, dict) and "path" in obj and "content" in obj:
        return [{"path": str(obj["path"]), "content": str(obj["content"])}]
    if isinstance(obj, dict) and "files" in obj and isinstance(obj["files"], list):
        return [{"path": str(f["path"]), "content": str(f["content"])} for f in obj["files"] if isinstance(f, dict) and "path" in f and "content" in f]
    if isinstance(obj, list):
        return [{"path": str(f["path"]), "content": str(f["content"])} for f in obj if isinstance(f, dict) and "path" in f and "content" in f]
    return []


def parse_file_map_from_text(text: str) -> List[Dict[str, str]]:
    """
    Parse any file-map-like JSONs embedded in text and return merged list.
    Later objects override earlier duplicates by path.
    """
    expected_kind = "file_map"
    expected_schema = "List of objects: {\"path\": str, \"content\": str}"
    obj: Any
    if len(text) <= 2:
        return []
    try:
        obj = json.loads(text)
    except Exception:
        fixed = _fix_json_text(text, expected_kind, expected_schema)
        obj = json.loads(fixed)
    if not isinstance(obj, list) or not all(
        isinstance(item, dict) and "path" in item and "content" in item for item in obj
    ):
        fixed = _fix_json_text(text, expected_kind, expected_schema)
        obj = json.loads(fixed)
        if not isinstance(obj, list) or not all(
            isinstance(item, dict) and "path" in item and "content" in item for item in obj
        ):
            raise ValueError(f"Invalid JSON after fix attempt: {text}")
    return obj


def parse_code_design_output(text: str) -> List[Dict[str, Any]]:
    """
    Parse the code design output from the ProjectDesignCrew result.

    Expected shape (new):
      [{"developer": int, "set_of_files": {"file_path": { ... }}}]
    """
    expected_kind = "code_design"
    expected_schema = (
        "List of objects: {\"developer\": int [1, 2, 3], \"set_of_files\": {\"file_path\": {"
        "\"project_dependencies\": [str], \"classes\": [ ... ], \"functions\": [ ... ]}}}"
    )
    obj: Any
    if len(text) <= 2:
        return []
    try:
        obj = json.loads(text)
    except Exception:
        fixed = _fix_json_text(text, expected_kind, expected_schema)
        obj = json.loads(fixed)
    if not (
        isinstance(obj, list)
        and all(
            isinstance(it, dict) and
            "set_of_files" in it and
            isinstance(it.get("set_of_files"), dict) and
            "developer" in it and
            it.get("developer") in [1, 2, 3]
            for it in obj
        )
    ):
        fixed = _fix_json_text(text, expected_kind, expected_schema)
        obj = json.loads(fixed)
        if not (
            isinstance(obj, list)
            and all(
                isinstance(it, dict) and
                "set_of_files" in it and
                isinstance(it.get("set_of_files"), dict) and
                "developer" in it and
                it.get("developer") in [1, 2, 3]
                for it in obj
            )
        ):
            raise ValueError(f"Invalid JSON after fix attempt: {text}")
    return obj


def parse_code_output(text: str) -> List[Dict[str, str]]:
    """
    Parse the code output from the text.
    """
    expected_kind = "code_output"
    expected_schema = "List of objects: {\"path\": str, \"content\": str}"
    obj: Any
    if len(text) <= 2:
        return []
    try:
        obj = json.loads(text)
    except Exception:
        fixed = _fix_json_text(text, expected_kind, expected_schema)
        obj = json.loads(fixed)
    if not isinstance(obj, list) or not all(
        isinstance(item, dict) and "path" in item and "content" in item for item in obj
    ):
        fixed = _fix_json_text(text, expected_kind, expected_schema)
        obj = json.loads(fixed)
        if not isinstance(obj, list) or not all(
            isinstance(item, dict) and "path" in item and "content" in item for item in obj
        ):
            raise ValueError(f"Invalid JSON after fix attempt: {text}")
    return obj


def parse_code_fixes_output(text: str) -> List[Dict[str, str]]:
    """
    Parse the code fixes output from the text.
    """
    expected_kind = "code_fixes_output"
    expected_schema = (
        "List of objects: {\"file_path\": str, \"affected_callable\": str, \"fix\": str}"
    )
    obj: Any
    if len(text) <= 2:
        return []
    try:
        obj = json.loads(text)
    except Exception:
        fixed = _fix_json_text(text, expected_kind, expected_schema)
        obj = json.loads(fixed)
    if not isinstance(obj, list) or not all(
        isinstance(item, dict)
        and "file_path" in item
        and "affected_callable" in item
        and "fix" in item
        for item in obj
    ):
        fixed = _fix_json_text(text, expected_kind, expected_schema)
        obj = json.loads(fixed)
        if not isinstance(obj, list) or not all(
            isinstance(item, dict)
            and "file_path" in item
            and "affected_callable" in item
            and "fix" in item
            for item in obj
        ):
            raise ValueError(f"Invalid JSON after fix attempt: {text}")
    return obj


def _fix_json_text(original_text: str, expected_kind: str, expected_schema: str) -> str:
    """
    Use the JSONFixerCrew to repair malformed JSON according to an expected schema.
    Returns a JSON string that should be loadable by json.loads.
    """
    result = JSONFixerCrew().crew().kickoff(
        inputs={
            "expected_kind": expected_kind,
            "expected_schema": expected_schema,
            "original_text": original_text,
        }
    )
    # Single-task crew; take first task output string
    try:
        fixed_text = str(result.tasks_output[0])  # type: ignore[attr-defined]
    except Exception:
        fixed_text = str(result)
    return fixed_text


def write_file_map(files: Dict[str, str], out_dir: str, sub_dir: str = "") -> List[Tuple[str, int]]:
    """
    Deterministically write files under out_dir with path traversal protection.
    Returns a log [ (relative_path, bytes_written) ].
    """
    if sub_dir:
        sub_dir = Path(sub_dir)
        base = (Path(out_dir) / sub_dir).resolve()
    else:
        base = Path(out_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)
    log: List[Tuple[str, int]] = []
    for path, content in files.items():
        rel = Path(path)
        target = (base / rel).resolve()
        # prevent escaping base
        if base != target and base not in target.parents:
            raise ValueError(f"Illegal path outside base: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        written = target.write_text(content, encoding="utf-8")
        log.append((str(rel), written))
    return log
