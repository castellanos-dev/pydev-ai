from __future__ import annotations
from typing import Dict
from pydantic import RootModel


COPY_MAP_SCHEMA = '{"old_path": "new_path"}'


class CopyMapOutput(RootModel[Dict[str, str]]):
    pass
