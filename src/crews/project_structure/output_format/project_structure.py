from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


PROJECT_STRUCTURE_SCHEMA = '''
{
  "code_dir": str,
  "docs_dir": str|null,
  "test_dirs": [str],
  "summaries_dir": str|null
}
'''


class ProjectStructureOutput(BaseModel):
    code_dir: str
    docs_dir: Optional[str]
    test_dirs: List[str]
    summaries_dir: Optional[str]
