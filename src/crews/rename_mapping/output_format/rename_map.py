from __future__ import annotations
from typing import Dict
from pydantic import RootModel


RENAME_MAP_SCHEMA = '{"old_path": "new_path"}'


class RenameMapOutput(RootModel[Dict[str, str]]):
    pass
