from __future__ import annotations
from typing import List
from pydantic import BaseModel, RootModel


BUG_ANALYSIS_SCHEMA = '''
[{
    "file_paths": [str],
    "affected_callables": [str],
    "points": int,
    "description": str,
    "fix": str,
    "id": int
}]
'''


class BugAnalysisItem(BaseModel):
    file_paths: List[str]
    affected_callables: List[str]
    points: int
    description: str
    fix: str
    id: int

class AnalyzeTestFailuresOutput(RootModel[List[BugAnalysisItem]]):
    pass
