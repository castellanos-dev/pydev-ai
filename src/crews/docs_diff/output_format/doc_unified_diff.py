from __future__ import annotations
from pydantic import RootModel


DOC_UNIFIED_DIFF_SCHEMA = '''
str
'''


class DocUnifiedDiffOutput(RootModel[str]):
    pass
