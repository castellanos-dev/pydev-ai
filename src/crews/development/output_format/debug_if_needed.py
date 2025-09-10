from __future__ import annotations
from typing import List
from pydantic import BaseModel, RootModel


DEBUG_IF_NEEDED_SCHEMA = '''
[{
    "file_path": str,
    "affected_callable": str,
    "fix": str
}]
'''


class DebugFix(BaseModel):
    file_path: str
    affected_callable: str
    fix: str


class DebugIfNeededOutput(RootModel[List[DebugFix]]):
    pass
