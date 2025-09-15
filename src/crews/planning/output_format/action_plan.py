from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field, RootModel


ACTION_PLAN_SCHEMA = '''
[
  {
    "step": int,
    "title": str,
    "description": str,
    "artifacts": [str],
    "type": str,
    "points": int
  }
]
'''


class ActionStep(BaseModel):
    step: int
    title: str
    description: str
    artifacts: List[str] = Field(default_factory=list)
    type: str
    points: int = Field(default=1)

class ActionPlanOutput(RootModel[List[ActionStep]]):
    pass
