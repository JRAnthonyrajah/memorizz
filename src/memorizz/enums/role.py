from enum import Enum

class Role(Enum):
    """Enum for different roles in a conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    DEVELOPER = "developer"
    TOOL = "tool" 