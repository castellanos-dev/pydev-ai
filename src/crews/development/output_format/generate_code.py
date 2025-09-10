from __future__ import annotations
from typing import List
from pydantic import BaseModel, RootModel


GENERATE_CODE_SCHEMA = '[{"path": str, "content": str}]'


class CodeFile(BaseModel):
    path: str
    content: str


class GenerateCodeOutput(RootModel[List[CodeFile]]):
    pass
