from __future__ import annotations
from typing import Dict, List
from .storage import write_digest


def default_digest_header(repo: str) -> str:
    return f"""# Project Knowledge Digest

This digest captures high-signal context for LLM agents. Keep it short & semantic.

- Repo: {repo}
- Updated: (auto)
- Sections: Overview, Modules, Public APIs, Known Constraints, TODOs
"""


def bootstrap_digest(repo: str) -> str:
    content = default_digest_header(repo) + "\n\n## Overview\nTBD.\n"
    return write_digest(repo, "00_overview", content)


def write_summaries(repo: str, summaries: Dict[str, str]) -> List[str]:
    """
    Persist a set of summaries as individual markdown files under the digests
    directory. Keys are logical names (safe for filenames) and values are
    markdown contents. Returns list of written paths.
    """
    written: List[str] = []
    for name, md in summaries.items():
        path = write_digest(repo, name, md)
        written.append(path)
    return written
