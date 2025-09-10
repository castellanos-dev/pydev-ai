from __future__ import annotations
from typing import List
from pydantic import BaseModel, RootModel


GENERATE_TESTS_SCHEMA = '''
[{
    "path": str,
    "content": str
}]
'''


class TestFile(BaseModel):
    path: str
    content: str


class GenerateTestsOutput(RootModel[List[TestFile]]):
    pass
