from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field


FILE_DETAIL_SCHEMA = '''
{
  "summaries_only": [str],
  "need_code": [str]
}
'''


class FileDetailOutput(BaseModel):
    summaries_only: List[str] = Field(default_factory=list)
    need_code: List[str] = Field(default_factory=list)
