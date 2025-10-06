from __future__ import annotations
from typing import Dict, Any
from crewai import LLM
from litellm import num_retries
from .. import settings


def llms(**kwargs: Any) -> Dict[str, LLM]:
    """Return CrewAI LLM instances."""
    return {
        "light": LLM(model=settings.MODEL_LIGHT, max_tokens=8000, temperature=0.0, num_retries=3, **kwargs),
        "medium": LLM(model=settings.MODEL_MEDIUM, max_tokens=8000, temperature=0.0, num_retries=3, **kwargs),
        "reasoning": LLM(model=settings.MODEL_REASONING, temperature=0.0, num_retries=3, **kwargs),
    }
