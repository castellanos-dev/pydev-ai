from __future__ import annotations
from typing import List
from pydantic import BaseModel, RootModel


IMPLEMENT_TESTS_SCHEMA = '''
[
  {"code": str}
]
'''


class TestCode(BaseModel):
    code: str


class ImplementTestsOutput(RootModel[List[TestCode]]):
    pass
