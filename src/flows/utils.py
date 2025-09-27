from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Set
import tempfile
import subprocess
from crewai import TaskOutput
import json
import yaml

from ..summaries.storage import digests_root
from ..summaries.summarizer import bootstrap_digest

from ..crews.json_fixer import JSONFixerCrew


def is_something_to_fix(output: TaskOutput) -> bool:
    # "[]" is not a valid output
    output = str(output)
    return len(output) > 2  and '{' in output and '}' in output and 'error' in output


def sanitize_generated_content(text: str) -> str:
    """
    Normalize LLM-generated file contents before writing to disk.

    - Strip surrounding code fences if present (```...``` with optional language)
    - Convert visible escape sequences (e.g., "\n", "\t") into real characters
    - Ensure trailing newline at EOF (common convention for linters)
    """
    if not isinstance(text, str):
        text = str(text)

    s = text.strip()

    # Strip triple backtick fences if present
    if s.startswith("```"):
        # remove opening fence line
        lines = s.splitlines()
        if lines:
            # drop first line (``` or ```lang)
            lines = lines[1:]
        # remove closing fence line if present at end
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines)

    # Unescape common sequences if they are literal
    # Only apply if string contains backslash-n or backslash-t patterns
    if "\\n" in s or "\\t" in s or "\\r" in s:
        s = s.encode("utf-8").decode("unicode_escape")

    # Ensure single trailing newline
    if not s.endswith("\n"):
        s = s + "\n"

    return s

def ensure_repo(repo: str, check_empty: bool = False) -> Union[None, str]:
    """
    Initialize repository directory if needed.

    Args:
        repo: Repository path (absolute or relative)
        check_empty: Whether to check if the repository is empty

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
        if check_empty and not any(repo_path.iterdir()):
            raise ValueError(f"Repository directory is empty: {repo}")

    else:
        # If it's a relative path, create the path if it doesn't exist
        repo_path = repo_path.resolve()
        if not repo_path.exists():
            repo_path.mkdir(parents=True, exist_ok=True)
        if check_empty and not any(repo_path.iterdir()):
            raise ValueError(f"Repository directory is empty: {repo}")

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
    overview = digests_dir / "00_overview.yaml"
    if not overview.exists():
        bootstrap_digest(str(digests_dir.parent))
    return str(digests_dir)


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
        return [
            {"path": str(f["path"]), "content": str(f["content"])}
            for f in obj["files"] if isinstance(f, dict) and "path" in f and "content" in f
        ]
    if isinstance(obj, list):
        return [
            {"path": str(f["path"]), "content": str(f["content"])}
            for f in obj if isinstance(f, dict) and "path" in f and "content" in f
        ]
    return []


def load_json_output(result: TaskOutput, schema: str, task: int = -1) -> List[Dict[str, Any]]:
    """
    Parse the JSON output from a given schema.
    """
    if result.tasks_output[task].json_dict is not None:
        if "root" in result.tasks_output[task].json_dict:
            return result.tasks_output[task].json_dict["root"]
        else:
            return result.tasks_output[task].json_dict
    text = str(result.tasks_output[task])
    obj: Any
    if len(text) <= 2:
        return []
    try:
        obj = json.loads(text)
    except Exception:
        fixed = _fix_json_text(text, schema)
        obj = json.loads(fixed)
    if "root" in obj:
        return obj["root"]
    return obj


def load_json_list(result: TaskOutput, schema: str, task: int = -1) -> List[Any]:
    """
    Convenience wrapper to parse a list-shaped JSON output.
    Returns an empty list if the parsed data is not a list.
    """
    data = load_json_output(result, schema, task)
    if not isinstance(data, list):
        return []
    return data


def load_json_object(result: TaskOutput, schema: str, task: int = -1) -> Dict[str, Any]:
    """
    Convenience wrapper to parse an object-shaped JSON output.
    Handles the common case where the model returns a single-item list containing the object.
    Returns an empty dict if parsing fails to produce an object.
    """
    data = load_json_output(result, schema, task)
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return {}


def _fix_json_text(original_text: str, expected_schema: str) -> str:
    """
    Use the JSONFixerCrew to repair malformed JSON according to an expected schema.
    Returns a JSON string that should be loadable by json.loads.
    """
    result = JSONFixerCrew().crew().kickoff(
        inputs={
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


def process_path(base_path: str, path: str, sub_dir: str = "") -> Path:
    """
    Process a path to ensure it is a valid path.
    """

    # If the provided path is absolute, return it directly as a resolved Path
    p = Path(path)
    if p.is_absolute():
        return p.resolve()

    if sub_dir and path.startswith(f'{sub_dir}/'):
        path = path[len(sub_dir) + 1:]
    return ((Path(base_path) / sub_dir / path) if sub_dir else Path(base_path) / path).resolve()


def write_file_map(files: Dict[str, str], out_dir: str, sub_dir: str = "") -> List[Tuple[str, int]]:
    """
    Deterministically write files under out_dir with path traversal protection.
    Returns a log [ (relative_path, bytes_written) ].
    """
    base = Path(out_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)
    log: List[Tuple[str, int]] = []
    for path, content in files.items():
        target = process_path(out_dir, path, sub_dir)
        # prevent escaping base
        if base != target and base not in target.parents:
            raise ValueError(f"Illegal path outside base: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        written = target.write_text(str(content), encoding="utf-8")
        log.append((str(path), written))
    return log


def write_file(file_content: str, path: Path) -> int:
    """
    Deterministically write files under out_dir with path traversal protection.
    Returns the number of bytes written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = path.write_text(str(file_content), encoding="utf-8")
    return written


def to_yaml_file_map(content: Dict[str, Any]) -> str:
    """
    Convert a JSON-serializable object into a YAML string ready to be written.
    Ensures stable key order and trailing newline.
    """
    try:
        text = yaml.safe_dump(content, allow_unicode=True, sort_keys=False)
        if not text.endswith("\n"):
            text = text + "\n"
    except Exception:
        return ''
    return text


def apply_combined_unified_diffs(
    repo_dir: Union[str, Path],
    src_dir: Union[str, Path],
    diffs_by_file: Dict[str, List[str]],
) -> Tuple[bool, str]:
    """
    Apply a combined unified diff to a repository using `git apply`.

    Parameters
    ----------
    repo_dir: base repository directory where .git lives
    src_dir: root of source tree (used as fallback cwd)
    diffs_by_file: mapping file path -> list of unified diff strings

    Returns
    -------
    (success, error)
      - success: True if git apply succeeded in any attempt
      - error: stderr text if failed, empty string on success
    """
    if not diffs_by_file:
        return True, ""

    repo_dir = Path(repo_dir).resolve()
    src_dir = Path(src_dir).resolve()

    try:
        with tempfile.TemporaryDirectory() as td:
            patch_path = Path(td) / "changes.diff"
            combined: List[str] = []
            for _path, diffs in diffs_by_file.items():
                for d in diffs:
                    if not isinstance(d, str):
                        continue
                    combined.append(d.rstrip("\n") + "\n")
            patch_content = "\n".join(combined)
            patch_path.write_text(patch_content, encoding="utf-8")

            def _run_git_apply(cwd: Path, args: List[str]) -> subprocess.CompletedProcess:
                return subprocess.run(
                    ["git", "apply", *args, str(patch_path)],
                    cwd=str(cwd),
                    check=False,
                    capture_output=True,
                    text=True,
                )

            result = _run_git_apply(repo_dir, ["--whitespace=fix", "--index"])
            if result.returncode == 0:
                return True, ""
            result = _run_git_apply(repo_dir, ["--whitespace=fix"])
            if result.returncode == 0:
                return True, ""
            result = _run_git_apply(src_dir, ["--whitespace=fix"])
            if result.returncode == 0:
                return True, ""
            return False, (result.stderr or "git apply failed")
    except Exception as exc:
        return False, str(exc)


def extract_diffs_by_file(items: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Normalize a list of diff items into a mapping path -> list[content_diff].
    Ignores entries without a valid path or string diff.
    """
    grouped: Dict[str, List[str]] = {}
    for item in items or []:
        try:
            path = str(item.get("path", "")).strip()
            diff = item.get("content_diff")
        except Exception:
            continue
        if not path or not isinstance(diff, str):
            continue
        grouped.setdefault(path, []).append(diff)
    return grouped


def collect_module_dirs_from_diffs_map(diffs_by_file: Dict[str, List[str]]) -> Set[Path]:
    """
    Given a diffs map path -> list[diff], return the set of parent directories
    (as Path objects) that should have their module summaries refreshed.
    """
    modules: Set[Path] = set()
    for path in (diffs_by_file or {}).keys():
        try:
            modules.add(Path(path).parent)
        except Exception:
            continue
    return modules
