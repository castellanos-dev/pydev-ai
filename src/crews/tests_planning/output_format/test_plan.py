from __future__ import annotations
from typing import List, Literal
from pydantic import BaseModel, RootModel


TEST_PLAN_SCHEMA = '''
[
  {
    "title": str,
    "description": str,
    "targets": [str],
    "src_file": str,
    "reason": str,
    "test_type": "unit|integration"
  }
]
'''


class TestPlanItem(BaseModel):
    title: str
    description: str
    targets: List[str]
    src_file: str
    reason: str
    test_type: Literal["unit", "integration"]


class GenerateTestPlanOutput(RootModel[List[TestPlanItem]]):
    pass
