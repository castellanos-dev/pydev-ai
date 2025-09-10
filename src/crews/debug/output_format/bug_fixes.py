from __future__ import annotations
from typing import List
from pydantic import BaseModel, RootModel


BUG_FIXES_SCHEMA = '''
[{
    "path": str,
    "content_diff": str
}]
'''


class BugFixFile(BaseModel):
    path: str
    content_diff: str


class ImplementBugFixesOutput(RootModel[List[BugFixFile]]):
    pass
