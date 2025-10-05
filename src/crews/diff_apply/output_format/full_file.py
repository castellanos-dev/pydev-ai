from __future__ import annotations
from pydantic import RootModel


FULL_FILE_SCHEMA = '''
str
'''


class FullFileOutput(RootModel[str]):
    pass
