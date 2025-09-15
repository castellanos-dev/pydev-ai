from __future__ import annotations
from typing import List
from pydantic import BaseModel, RootModel


GENERATE_DIFFS_SCHEMA = '''
[
  {"path": str, "content_diff": str}
]
'''


class DiffFile(BaseModel):
    path: str
    content_diff: str


class GenerateDiffsOutput(RootModel[List[DiffFile]]):
    pass
