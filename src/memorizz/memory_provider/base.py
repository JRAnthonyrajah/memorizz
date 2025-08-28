from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, TYPE_CHECKING

# Use TYPE_CHECKING for forward references to avoid circular imports
if TYPE_CHECKING:
    from memorizz.memagent import MemAgent

class MemoryProvider(ABC):
    """Abstract base class for memory providers."""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """Initialize the memory provider with configuration settings."""
        pass

    @abstractmethod
    def store(self, data: Dict[str, Any], memory_store_type: str) -> str:
        """Store data in the memory provider."""
        pass

    @abstractmethod
    def retrieve_by_query(self, query: Dict[str, Any], memory_store_type: str, limit: int = 1, **kwargs) -> Optional[Dict[str, Any]]:
        """Retrieve a document from the memory provider."""
        pass

    @abstractmethod
    def retrieve_by_id(self, id: str, memory_store_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document from the memory provider by id."""
        pass

    @abstractmethod
    def retrieve_by_name(self, name: str, memory_store_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document from the memory provider by name."""
        pass

    @abstractmethod
    def delete_by_id(self, id: str, memory_store_type: str) -> bool:
        """Delete a document from the memory provider by id."""
        pass

    @abstractmethod
    def delete_by_name(self, name: str, memory_store_type: str) -> bool:
        """Delete a document from the memory provider by name."""
        pass

    @abstractmethod
    def delete_all(self, memory_store_type: str) -> bool:
        """Delete all documents within a memory store type in the memory provider."""
        pass

    @abstractmethod
    def list_all(self, memory_store_type: str) -> List[Dict[str, Any]]:
        """List all documents within a memory store type in the memory provider."""
        pass

    @abstractmethod
    def retrieve_conversation_history_ordered_by_timestamp(self, memory_id: str) -> List[Dict[str, Any]]:
        """Retrieve the conversation history ordered by timestamp."""
        pass

    @abstractmethod
    def update_by_id(self, id: str, data: Dict[str, Any], memory_store_type: str) -> bool:
        """Update a document in a memory store type in the memory provider by id."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the connection to the memory provider."""
        pass 

    @abstractmethod
    def store_memagent(self, memagent: "MemAgent") -> str:
        """Store a memagent in the memory provider."""
        pass
    
    @abstractmethod
    def delete_memagent(self, agent_id: str, cascade: bool = False) -> bool:
        """Delete a memagent from the memory provider."""
        pass

    @abstractmethod
    def update_memagent_memory_ids(self, agent_id: str, memory_ids: List[str]) -> bool:
        """Update the memory_ids of a memagent in the memory provider."""
        pass
        
    @abstractmethod
    def delete_memagent_memory_ids(self, agent_id: str) -> bool:
        """Delete the memory_ids of a memagent in the memory provider."""
        pass

    @abstractmethod
    def list_memagents(self) -> List[Dict[str, Any]]:
        """List all memagents in the memory provider."""
        pass