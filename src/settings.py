from __future__ import annotations
import os
from dotenv import load_dotenv


load_dotenv()


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# LLM models
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_LIGHT = os.getenv("MODEL_LIGHT", "gpt-5-nano")
MODEL_REASONING = os.getenv("MODEL_REASONING", "gpt-5-nano")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Knowledge paths
DIGESTS_DIRNAME = "digests"
VECTORS_DIR = "data/knowledge/vectors"  # persistent local vector store (Chroma)
DEFAULT_KNOWLEDGE_ROOT = "data/knowledge"

# Tool timeouts (in seconds)
PYTEST_TIMEOUT = int(os.getenv("PYTEST_TIMEOUT", "1800"))  # 30 minutes default
