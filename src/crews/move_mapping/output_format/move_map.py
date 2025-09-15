from __future__ import annotations
from typing import Dict
from pydantic import RootModel


MOVE_MAP_SCHEMA = '{"old_path": "new_path"}'


class MoveMapOutput(RootModel[Dict[str, str]]):
    pass
