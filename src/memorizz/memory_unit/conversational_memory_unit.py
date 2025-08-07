from pydantic import BaseModel
from typing import Optional

class ConversationMemoryUnit(BaseModel):
    role: str
    content: str
    timestamp: str
    memory_id: str
    conversation_id: str
    embedding: list[float]
    recall_recency: Optional[float] = None
    associated_conversation_ids: Optional[list[str]] = None

