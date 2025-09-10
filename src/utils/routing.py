from __future__ import annotations
from typing import Dict, Any
from crewai import LLM
from .. import settings


def llms(**kwargs: Any) -> Dict[str, LLM]:
    """Return CrewAI LLM instances."""
    return {
        "light": LLM(model=settings.MODEL_LIGHT, **kwargs),
        "medium": LLM(model=settings.MODEL_MEDIUM, **kwargs),
        "reasoning": LLM(model=settings.MODEL_REASONING, **kwargs),
    }
