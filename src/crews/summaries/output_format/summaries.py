from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


# ---------------------------
# File summaries schema and models
# ---------------------------
FILE_SUMMARIES_SCHEMA = '''
{
  "location": str,
  "purpose": str,
  "dependencies": {
    "internal": [{"path": str, "reason": str}],
    "external": [{"name": str, "purpose": str}]
  },
  "structure": {
    "classes": [{
      "name": str,
      "responsibility": str?,
      "attributes": [{"name": str, "type": str?}],
      "methods": [{
        "name": str,
        "signature": str?,
        "description": str?,
        "parameters": [{"name": str, "type": str?}]?,
        "returns": {"type": str?, "description": str?}?,
        "raises": [{"exception": str, "description": str?}]?
      }]
    }],
    "functions": [{
      "name": str,
      "signature": str?,
      "purpose": str?,
      "parameters": [{"name": str, "type": str?}]?,
      "returns": {"type": str?, "description": str?}?,
      "raises": [{"exception": str, "description": str?}]?
    }],
    "globals": [{"name": str, "value": str?, "purpose": str?}]
  },
  "examples": [{"language": str?, "code": str?}]
}
'''


class FileInternalDependency(BaseModel):
    path: str
    reason: Optional[str] = None


class FileExternalDependency(BaseModel):
    name: str
    purpose: Optional[str] = None


class FileMethodParameter(BaseModel):
    name: str
    type: Optional[str] = None


class FileMethodReturn(BaseModel):
    type: Optional[str] = None
    description: Optional[str] = None


class FileMethodRaise(BaseModel):
    exception: str
    description: Optional[str] = None


class FileMethod(BaseModel):
    name: str
    signature: Optional[str] = None
    description: Optional[str] = None
    parameters: List[FileMethodParameter] = []
    returns: Optional[FileMethodReturn] = None
    raises: List[FileMethodRaise] = []


class FileAttribute(BaseModel):
    name: str
    type: Optional[str] = None


class FileClass(BaseModel):
    name: str
    responsibility: Optional[str] = None
    attributes: List[FileAttribute] = []
    methods: List[FileMethod] = []


class FileFunction(BaseModel):
    name: str
    signature: Optional[str] = None
    purpose: Optional[str] = None
    parameters: List[FileMethodParameter] = []
    returns: Optional[FileMethodReturn] = None
    raises: List[FileMethodRaise] = []


class FileGlobal(BaseModel):
    name: str
    value: Optional[str] = None
    purpose: Optional[str] = None


class FileStructure(BaseModel):
    classes: List[FileClass] = []
    functions: List[FileFunction] = []
    globals: List[FileGlobal] = []


class FileDependencies(BaseModel):
    internal: List[FileInternalDependency] = []
    external: List[FileExternalDependency] = []


class FileExample(BaseModel):
    language: Optional[str] = None
    code: Optional[str] = None


class FileSummariesOutput(BaseModel):
    location: Optional[str] = None
    purpose: Optional[str] = None
    dependencies: Optional[FileDependencies] = None
    structure: Optional[FileStructure] = None
    examples: List[FileExample] = []


# ---------------------------
# Module summaries schema and models
# ---------------------------
MODULE_SUMMARIES_SCHEMA = '''
{
  "location": str,
  "purpose": str,
  "structure": {"tree": [str]},
  "relationships": {
    "dependencies": {
      "internal": [{"path": str, "reason": str}],
      "external": [{"name": str, "purpose": str}]
    }
  },
  "interfaces": {
    "classes": [{
      "name": str,
      "description": str,
      "methods": [{"name": str, "signature": str, "description": str}]
    }],
    "functions": [{"name": str, "signature": str, "description": str}]
  },
  "workflows": [str],
  "notes": str
}
'''


class ModuleInternalDependency(BaseModel):
    path: str
    reason: Optional[str] = None


class ModuleExternalDependency(BaseModel):
    name: str
    purpose: Optional[str] = None


class Dependencies(BaseModel):
    internal: List[ModuleInternalDependency] = []
    external: List[ModuleExternalDependency] = []


class ModuleRelationships(BaseModel):
    dependencies: Dependencies = Dependencies()


class ModuleMethod(BaseModel):
    name: str
    signature: Optional[str] = None
    description: Optional[str] = None


class ModuleClass(BaseModel):
    name: str
    description: Optional[str] = None
    methods: List[ModuleMethod] = []


class ModuleFunction(BaseModel):
    name: str
    signature: Optional[str] = None
    description: Optional[str] = None


class ModuleInterfaces(BaseModel):
    classes: List[ModuleClass] = []
    functions: List[ModuleFunction] = []


class ModuleStructure(BaseModel):
    tree: List[str] = []


class ModuleSummariesOutput(BaseModel):
    location: Optional[str] = None
    purpose: Optional[str] = None
    structure: Optional[ModuleStructure] = None
    relationships: Optional[ModuleRelationships] = None
    interfaces: Optional[ModuleInterfaces] = None
    workflows: List[str] = []
    notes: Optional[str] = None
