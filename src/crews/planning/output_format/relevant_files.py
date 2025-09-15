from __future__ import annotations
from typing import List
from pydantic import RootModel


RELEVANT_FILES_SCHEMA = '''
[str]
'''


class RelevantFilesOutput(RootModel[List[str]]):
    pass
