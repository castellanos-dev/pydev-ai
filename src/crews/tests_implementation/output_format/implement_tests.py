from __future__ import annotations
from typing import List
from pydantic import BaseModel, RootModel


IMPLEMENT_TESTS_SCHEMA = '''
str
'''


class ImplementTestsOutput(RootModel[str]):
    pass
