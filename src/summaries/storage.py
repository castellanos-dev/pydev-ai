from __future__ import annotations
import pathlib
from typing import List
from .. import settings


def digests_root(repo_root: str | pathlib.Path) -> pathlib.Path:
    base = pathlib.Path(repo_root)
    # Store digests under configured subdirectory inside the repository
    return base / settings.DIGESTS_DIRNAME


def write_digest(repo_root: str | pathlib.Path, name: str, content: str) -> str:
    root = digests_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    # Preserve nested directories under digests; always write .yaml extension
    p = root / f"{name}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return str(p)


def list_digests(repo_root: str | pathlib.Path) -> List[str]:
    root = digests_root(repo_root)
    if not root.exists():
        return []
    return [str(p) for p in root.rglob("*.yaml")]


def read_all_digests(repo_root: str | pathlib.Path) -> str:
    paths = list_digests(repo_root)
    if not paths:
        return ""
    texts = []
    for path in paths:
        try:
            texts.append(pathlib.Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
    return "\n\n".join(texts)
