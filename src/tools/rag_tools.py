from __future__ import annotations
from typing import Iterable, List, Dict, Any, Optional, Union
from pathlib import Path
import os
import shutil
import json
from .. import settings
import chromadb
from chromadb import Collection
from openai import OpenAI


# -----------------------------
# Docs-only RAG (persistent)
# -----------------------------

_DOC_SUFFIXES = {".md", ".rst", ".txt", ".mdx"}
_EXCLUDED_DOCS_NORMALIZED = {"releasenotes"}


def _iter_doc_files(doc_root: Path) -> Iterable[Path]:
    if not doc_root.exists() or not doc_root.is_dir():
        return []
    seen: set[Path] = set()
    # Efficient walk with pruning: exclude only `build` directory
    try:
        for root, dirs, files in os.walk(str(doc_root)):
            # prune `build` directories in-place to avoid descending
            dirs[:] = [d for d in dirs if d != "build"]
            for filename in files:
                # suffix match against allowed doc suffixes
                if any(filename.endswith(suf) for suf in _DOC_SUFFIXES):
                    p = Path(root) / filename
                    stem_normalized = "".join(ch for ch in p.stem.lower() if ch.isalnum())
                    if stem_normalized in _EXCLUDED_DOCS_NORMALIZED:
                        continue
                    if p.is_file() and p not in seen:
                        seen.add(p)
                        yield p
    except Exception:
        return []


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _split_heading_chunks(text: str, max_chars: int = 1200, overlap: int = 150) -> List[Dict[str, Any]]:
    """
    Split documentation text into chunks, preferring to start chunks at markdown-like headings.
    Returns a list of {"heading": str, "text": str}.
    """
    if not text:
        return []
    lines = text.splitlines()
    chunks: List[Dict[str, Any]] = []

    def flush(buffer: List[str], heading: str) -> None:
        if not buffer:
            return
        buf_text = "\n".join(buffer).strip()
        if buf_text:
            chunks.append({"heading": heading.strip()[:200], "text": buf_text})

    current: List[str] = []
    current_len = 0
    current_heading = ""

    for ln in lines:
        is_heading = ln.lstrip().startswith("#") and ln.strip().startswith("#")
        if is_heading:
            # Start a new chunk at heading boundary
            if current:
                flush(current, current_heading)
            current = []
            current_len = 0
            # Extract heading title after leading #'s
            title = ln.lstrip().lstrip('#').strip()
            current_heading = title or current_heading
            continue
        # Append line; if exceeds max, flush with overlap
        current.append(ln)
        current_len += len(ln) + 1
        if current_len >= max_chars:
            flush(current, current_heading)
            if overlap > 0 and len(current) > 1:
                # Keep tail lines as overlap
                tail_text = "\n".join(current).strip()
                if len(tail_text) > overlap:
                    # heuristic: keep last ~overlap chars by lines
                    tail = []
                    acc = 0
                    for l in reversed(current):
                        acc += len(l) + 1
                        tail.append(l)
                        if acc >= overlap:
                            break
                    current = list(reversed(tail))
                else:
                    current = current[-max(1, int(len(current) * 0.25)) :]
            else:
                current = []
            current_len = sum(len(l) + 1 for l in current)

    flush(current, current_heading)
    if not chunks:
        return [{"heading": "", "text": text}]
    return chunks


class DocsRAG:
    """
    Lightweight, persistent RAG specifically for project documentation under a given docs directory.

    - Uses ChromaDB persistence under `<repo>/.pydev/rag_docs/`
    - Indexes only documentation files: .md, .mdx, .rst, .txt
    - Stores a simple manifest to detect changes; on change, rebuilds the index for determinism
    """

    def __init__(self, repo_dir: Path, docs_dir: Path, collection_name: str = "docs"):
        self.repo_dir = Path(str(repo_dir)).resolve()
        self.docs_dir = Path(str(docs_dir)).resolve()
        self.vectors_dir = (self.repo_dir / ".pydev" / "rag_vectors").resolve()
        self.manifest_path = (self.vectors_dir / "manifest.json").resolve()
        self.collection_name = collection_name
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        # Chroma persistent client + collection
        self._client: chromadb.PersistentClient = chromadb.PersistentClient(path=str(self.vectors_dir))
        self._collection: Collection = self._client.get_or_create_collection(name=self.collection_name)
        # OpenAI embeddings client
        self._openai = OpenAI()

    # -------- Manifest helpers --------
    def _reset_store(self) -> None:
        # Remove all persisted vectors and manifest
        try:
            if self.vectors_dir.exists():
                shutil.rmtree(self.vectors_dir)
        except Exception:
            pass
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        try:
            if self.manifest_path.exists():
                self.manifest_path.unlink()
        except Exception:
            pass
        # Re-create Chroma client and collection fresh
        self._client = chromadb.PersistentClient(path=str(self.vectors_dir))
        self._collection = self._client.get_or_create_collection(name=self.collection_name)

    def _load_manifest(self) -> Dict[str, Dict[str, int]]:
        if not self.manifest_path.exists():
            return {}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): {"size": int(v.get("size", 0)), "mtime": int(v.get("mtime", 0))} for k, v in data.items()}
        except Exception:
            pass
        return {}

    def _save_manifest(self, manifest: Dict[str, Dict[str, int]]) -> None:
        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    # -------- Indexing --------
    def _iter_docs(self) -> List[Path]:
        return list(_iter_doc_files(self.docs_dir))

    def _to_rel(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.repo_dir))

    def index(self, paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Incrementally index docs. If paths is provided, update only those files.
        """
        # Collect current state from disk
        all_docs = self._iter_docs()
        manifest_prev = self._load_manifest()

        # If targeted update requested, restrict to those paths that are docs
        target_docs: List[Path]
        if paths:
            target_docs = []
            for p in paths:
                try:
                    pp = Path(p)
                    if not pp.is_absolute():
                        pp = (self.repo_dir / p)
                    pp = pp.resolve()
                    if pp.exists() and pp.is_file() and any(str(pp).endswith(s) for s in _DOC_SUFFIXES):
                        target_docs.append(pp)
                except Exception:
                    continue
            # Ensure collection exists even if nothing to do
            _ = self._collection.name
        else:
            target_docs = all_docs

        # Determine changes
        current_map: Dict[str, Dict[str, int]] = {}
        for p in all_docs:
            try:
                st = p.stat()
                current_map[self._to_rel(p)] = {"size": int(st.st_size), "mtime": int(st.st_mtime)}
            except Exception:
                continue

        added: List[str] = []
        modified: List[str] = []
        removed: List[str] = []

        if paths:
            # Only consider targeted docs for add/modify
            for p in target_docs:
                rel = self._to_rel(p)
                cur = current_map.get(rel)
                prev = manifest_prev.get(rel)
                if prev is None:
                    added.append(rel)
                elif prev != cur:
                    modified.append(rel)
            # No removal detection in targeted mode
        else:
            prev_keys = set(manifest_prev.keys())
            cur_keys = set(current_map.keys())
            added = sorted(list(cur_keys - prev_keys))
            removed = sorted(list(prev_keys - cur_keys))
            for rel in (cur_keys & prev_keys):
                if manifest_prev.get(rel) != current_map.get(rel):
                    modified.append(rel)

        # Deletes for removed
        deleted_count = 0
        if removed:
            try:
                for rel in removed:
                    self._collection.delete(where={"path": rel})
                deleted_count = len(removed)
            except Exception:
                pass

        # Upserts for added+modified
        upsert_list = added + modified
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        ids: List[str] = []
        for rel in upsert_list:
            # Remove existing chunks for this file before re-upserting
            try:
                self._collection.delete(where={"path": rel})
            except Exception:
                pass
            p = (self.repo_dir / rel).resolve()
            text = _read_text(p)
            if not text:
                continue
            parts = _split_heading_chunks(text, max_chars=1200, overlap=150)
            for idx, part in enumerate(parts):
                ids.append(f"{rel}::chunk::{idx}")
                texts.append(part["text"])
                metas.append({"path": rel, "heading": part.get("heading", "")})

        batch_size = 32
        for i in range(0, len(texts), batch_size):
            docs_batch = texts[i:i + batch_size]
            ids_batch = ids[i:i + batch_size]
            metas_batch = metas[i:i + batch_size]
            try:
                emb_resp = self._openai.embeddings.create(model=settings.EMBEDDING_MODEL, input=docs_batch)
                vectors = [d.embedding for d in emb_resp.data]
            except Exception:
                continue
            try:
                self._collection.upsert(documents=docs_batch, metadatas=metas_batch, ids=ids_batch, embeddings=vectors)
            except Exception:
                continue

        # Persist manifest with full current_map if not targeted; else patch just updated entries
        if not paths:
            self._save_manifest(current_map)
        else:
            merged = dict(manifest_prev)
            for rel in upsert_list:
                if rel in current_map:
                    merged[rel] = current_map[rel]
            self._save_manifest(merged)

        return {
            "status": "incremental",
            "added": len(added),
            "modified": len(modified),
            "removed": len(removed),
            "upserted": len(upsert_list),
            "deleted": deleted_count,
        }

    # -------- Search --------
    def search(self, query: str, top_k_files: Union[int, None] = None, top_k_chunks: int = 30) -> List[str]:
        """
        Execute a semantic search over docs and return repo-relative paths.
        """
        try:
            emb = self._openai.embeddings.create(model=settings.EMBEDDING_MODEL, input=[query]).data[0].embedding
        except Exception:
            return []
        n_results = max(top_k_files * 5, top_k_chunks) if top_k_files else top_k_chunks
        try:
            res = self._collection.query(query_embeddings=[emb], n_results=n_results)
        except Exception:
            return []
        metas = (res or {}).get("metadatas") or []
        metas = metas[0] if metas else []
        result: Dict[str, List[str]] = {
            'paths': [],
        }
        seen: set[str] = set()
        for m in metas:
            try:
                rel = str(m.get("path", "")).strip()
            except Exception:
                continue
            if top_k_files is None:
                result['paths'].append(rel)
                continue
            elif rel and rel not in seen:
                seen.add(rel)
                result['paths'].append(rel)
                if len(result['paths']) >= top_k_files:
                    break
        result['distances'] = res.get("distances", None)[0][:len(result['paths'])]
        result['documents'] = res.get("documents", None)[0][:len(result['paths'])]
        return result
