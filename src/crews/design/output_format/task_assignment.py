from __future__ import annotations
from typing import List, Dict
from pydantic import BaseModel, Field, RootModel


TASK_ASSIGNMENT_SCHEMA = '''
    [{
      "developer": int,
      "set_of_files": {
        "file_path": {
          "project_dependencies": [str],
          "classes": [{
            "name": str,
            "methods": [{
              "name": str,
              "parameters": [{"name": str, "type": str}],
              "functionalities": [str],
              "points": int,
            }],
            "parameters": [{"name": str, "type": str}],
            "functionalities": [str],
            "points": int,
          }],
          "functions": [{
            "name": str,
            "parameters": [{"name": str, "type": str}],
            "functionalities": [str],
            "points": int,
          }]
        },
      }
    }]
'''

class Parameter(BaseModel):
    name: str
    type: str


class Method(BaseModel):
    name: str
    parameters: List[Parameter] = Field(default_factory=list)
    functionalities: List[str] = Field(default_factory=list)
    points: int


class ClassSpec(BaseModel):
    name: str
    methods: List[Method] = Field(default_factory=list)
    parameters: List[Parameter] = Field(default_factory=list)
    functionalities: List[str] = Field(default_factory=list)
    points: int


class FunctionSpec(BaseModel):
    name: str
    parameters: List[Parameter] = Field(default_factory=list)
    functionalities: List[str] = Field(default_factory=list)
    points: int


class FileSpec(BaseModel):
    project_dependencies: List[str] = Field(default_factory=list)
    classes: List[ClassSpec] = Field(default_factory=list)
    functions: List[FunctionSpec] = Field(default_factory=list)


class Assignment(BaseModel):
    developer: int
    set_of_files: Dict[str, FileSpec]


class TaskAssignmentOutput(RootModel[List[Assignment]]):
    pass
