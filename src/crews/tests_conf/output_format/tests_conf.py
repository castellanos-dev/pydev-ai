from __future__ import annotations
from pydantic import BaseModel


TESTS_CONF_SCHEMA = '''
{
  "framework": "pytest|unittest|django|script|null",
  "command": str,
  "description": str
}
'''


class TestsConf(BaseModel):
    framework: str
    command: str
    description: str


class TestsConfOutput(BaseModel):
    framework: str
    command: str
    description: str
