from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, RootModel


PYTEST_OUTPUT_ANALYSIS_SCHEMA = '''
[{
    "file_path": str|null,
    "affected_callable": str|null,
    "error": [str],
    "traceback": [str]
}]
'''


class GroupedFailure(BaseModel):
    file_path: Optional[str] = None
    affected_callable: Optional[str] = None
    error: List[str]
    traceback: List[str]


class GroupFailuresByRootCause(RootModel[List[GroupedFailure]]):
    pass
