from __future__ import annotations
from typing import List
from pydantic import BaseModel, RootModel


SUMMARIES_SCHEMA = '''
[{
    "path": str,
    "content": str
}]
'''


class SummaryFile(BaseModel):
    path: str
    content: str


class SummariesOutput(RootModel[List[SummaryFile]]):
    pass
