from __future__ import annotations
from typing import Type, Iterable, List
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from crewai_tools import RagTool
from pathlib import Path
from .. import settings


def _iter_repo_files(root: str | Path, globs: List[str]) -> Iterable[Path]:
    root_path = Path(root)
    seen: set[Path] = set()
    for g in globs:
        for p in root_path.glob(g):
            if p.is_file() and p.suffix.lower() in {".py", ".md", ".yaml", ".yml", ".txt", ".rst"}:
                if p not in seen:
                    seen.add(p)
                    yield p


class RAGIndexArgs(BaseModel):
    repo: str = Field(..., description="Repository root to index")
    glob: str = Field(
        "**/*.py,**/*.md,**/*.yaml,**/*.yml,**/*.txt,**/*.rst", description="Comma-separated globs to include"
    )


class RAGIndexTool(BaseTool):
    """
    Tool for indexing repository contents into a local vector store.

    This tool scans a repository for files matching specified patterns and
    indexes them into a persistent vector database for semantic search.
    """

    name: str = "rag_index_repo"
    description: str = (
        "Index repository contents into a persistent local vector store (RagTool)."
    )
    args_schema: Type[BaseModel] = RAGIndexArgs

    def _run(self, repo: str, glob: str = "**/*.py,**/*.md,**/*.yaml,**/*.yml,**/*.txt,**/*.rst") -> str:
        """
        Index files from a repository into the vector store.

        Args:
            repo: Path to the repository directory
            glob: Comma-separated file patterns to include (default: Python, Markdown, text files)

        Returns:
            Status message indicating how many files were indexed
        """
        rag = RAGSearchTool()
        include_globs = [g.strip() for g in glob.split(",")]
        paths = list(_iter_repo_files(repo, include_globs))
        if not paths:
            return "[rag_index_repo] No files to index."
        indexed = 0
        for p in paths:
            try:
                rag.add(data_type="file", path=str(p))
                indexed += 1
            except Exception:
                # Skip problematic files quietly to keep tool deterministic
                continue
        return f"[rag_index_repo] Indexed {indexed} files from {repo}."


class RAGSearchArgs(BaseModel):
    query: str = Field(..., description="Natural language query")


class RAGSearchTool(RagTool):
    def __init__(self):
        config = {
            "vectordb": {
                "provider": "chromadb",
                "config": {"persist_directory": settings.VECTORS_DIR},
            },
            "embedder": {
                "provider": "openai",
                "config": {"model": settings.EMBEDDING_MODEL},
            },
        }
        super().__init__(
            config=config,
            name="rag_search",
            description="Semantic search over repository knowledge base. Returns compact context.",
        )
