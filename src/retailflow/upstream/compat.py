"""Minimal replacements for tau2's generic base classes, not retail business rules."""

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel


class DB(BaseModel):
    @classmethod
    def load(cls, path: Path):
        return cls.model_validate(json.loads(Path(path).read_text()))


class ToolType(str, Enum):
    READ = "read"
    WRITE = "write"
    GENERIC = "generic"


def is_tool(tool_type: ToolType):
    def mark(function):
        function.__tool_type__ = tool_type
        return function
    return mark


class ToolKitBase:
    def __init__(self, db):
        self.db = db
