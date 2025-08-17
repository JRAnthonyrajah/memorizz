"""
Semantic Cache Entry for MemAgent

Represents a cached query-response pair with metadata for semantic similarity matching.
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class SemanticCacheEntry(BaseModel):
    """
    Represents a cached query-response pair with metadata.
    
    This memory unit stores semantic cache entries that enable fast retrieval
    of similar queries through vector similarity matching.
    """
    
    query: str
    response: str
    embedding: List[float]
    timestamp: float
    session_id: Optional[str] = None
    memory_id: Optional[str] = None
    agent_id: Optional[str] = None
    usage_count: int = 0
    last_accessed: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    cache_key: Optional[str] = None
    
    def model_post_init(self, __context) -> None:
        """Initialize last_accessed if not provided."""
        if self.last_accessed is None:
            self.last_accessed = self.timestamp
