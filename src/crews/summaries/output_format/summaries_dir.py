from __future__ import annotations
from pydantic import BaseModel, RootModel


SUMMARIES_DIR_SCHEMA = '''
{
  "summaries_dir": str
}
'''


class SummariesDir(BaseModel):
    summaries_dir: str


class SummariesDirOutput(RootModel[SummariesDir]):
    pass
