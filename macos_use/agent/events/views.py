"""Agent event dataclasses."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EventType(str, Enum):
    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATE = "state"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentEvent:
    """Generic agent event with type discriminator and data dict."""

    type: EventType
    data: dict[str, Any]
