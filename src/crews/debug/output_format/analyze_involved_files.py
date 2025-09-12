from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, RootModel


INVOLVED_FILES_SCHEMA = '''
[{
    "file_path": [str|null],
    "affected_callable": [str|null],
    "error": [str],
    "traceback": [str],
    "involved_files": [str]
    "id": int
}]
'''


class InvolvedFilesItem(BaseModel):
    file_path: List[Optional[str]]
    affected_callable: List[Optional[str]]
    error: List[str]
    traceback: List[str]
    involved_files: List[str]
    id: int

class AnalyzeInvolvedFilesOutput(RootModel[List[InvolvedFilesItem]]):
    pass
