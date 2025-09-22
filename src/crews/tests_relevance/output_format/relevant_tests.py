from __future__ import annotations
from pydantic import RootModel


RELEVANT_TESTS_SCHEMA = '''
str
'''


class RelevantTestsOutput(RootModel[str]):
    pass
