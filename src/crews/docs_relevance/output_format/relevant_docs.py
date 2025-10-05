from __future__ import annotations
from typing import List
from pydantic import RootModel


RELEVANT_DOCS_SCHEMA = '''
[str]
'''


class RelevantDocsOutput(RootModel[List[str]]):
    pass

