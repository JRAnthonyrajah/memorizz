from enum import Enum

class SemanticCacheScope(Enum):
    """Scope for semantic cache searches."""
    LOCAL = "local"   # Search only this agent's cache entries (filtered by agent_id)
    GLOBAL = "global" # Search across all cache entries (no agent_id filter)
