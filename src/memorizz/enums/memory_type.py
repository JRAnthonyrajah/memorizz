from enum import Enum

class MemoryType(Enum):
    """Enum for different types of memory stores."""
    PERSONAS = "personas"
    TOOLBOX = "toolbox"
    SHORT_TERM_MEMORY = "short_term_memory"
    LONG_TERM_MEMORY = "long_term_memory"
    CONVERSATION_MEMORY = "conversation_memory"
    WORKFLOW_MEMORY = "workflow_memory"
    MEMAGENT = "agents"
    SHARED_MEMORY = "shared_memory"
    SUMMARIES = "summaries"
    SEMANTIC_CACHE = "semantic_cache" 