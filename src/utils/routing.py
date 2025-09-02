from __future__ import annotations
from typing import Dict, Any
from crewai import LLM
from .. import settings


def llms(**kwargs: Any) -> Dict[str, LLM]:
    """Return CrewAI LLM instances."""
    return {
        "light": LLM(model=settings.OPENAI_MODEL_LIGHT, **kwargs),
        "reasoning": LLM(model=settings.OPENAI_MODEL_REASONING, **kwargs),
    }
