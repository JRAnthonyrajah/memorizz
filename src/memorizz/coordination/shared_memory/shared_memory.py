from typing import Dict, Any, List, Optional
from datetime import datetime
from ...memory_provider import MemoryProvider
from ...enums.memory_type import MemoryType
from bson import ObjectId
import json
import logging

logger = logging.getLogger(__name__)

class BlackboardEntry:
    """Individual entry in the shared memory blackboard."""
    
    def __init__(self, 
                 agent_id: str, 
                 content: Any, 
                 entry_type: str, 
                 created_at: datetime = None):
        self.agent_id = agent_id
        self.content = content
        self.entry_type = entry_type  # "tool_call", "conversation", "task_assignment", "result"
        self.created_at = created_at or datetime.now()
        self.memory_id = str(ObjectId())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "memory_id": self.memory_id,
            "agent_id": self.agent_id,
            "content": self.content,
            "entry_type": self.entry_type,
            "created_at": self.created_at.isoformat()
        }

class SharedMemory:
    """Shared memory system for multi-agent coordination."""
    
    def __init__(self, memory_provider: MemoryProvider):
        self.memory_provider = memory_provider
        
    def create_shared_session(self, 
                             root_agent_id: str, 
                             delegate_agent_ids: List[str] = None) -> str:
        """
        Create a new shared memory session for multi-agent coordination.
        
        Parameters:
            root_agent_id (str): The ID of the root/orchestrating agent
            delegate_agent_ids (List[str]): List of delegate agent IDs
            
        Returns:
            str: The memory ID for the shared session
        """
        shared_session = {
            "memory_id": str(ObjectId()),
            "root_agent_id": root_agent_id,
            "delegate_agent_ids": delegate_agent_ids or [],
            "sub_agent_ids": [],  # Will be populated recursively
            "blackboard": [],
            "created_at": datetime.now().isoformat(),
            "status": "active"  # active, completed, failed
        }
        
        # Store in memory provider
        memory_id = self.memory_provider.store(shared_session, MemoryType.SHARED_MEMORY)
        return str(memory_id)
    
    def add_blackboard_entry(self, 
                           memory_id: str, 
                           agent_id: str, 
                           content: Any, 
                           entry_type: str) -> bool:
        """
        Add an entry to the shared blackboard.
        
        Parameters:
            memory_id (str): The shared memory ID
            agent_id (str): The ID of the agent adding the entry
            content (Any): The content to add
            entry_type (str): Type of entry (tool_call, conversation, etc.)
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Adding blackboard entry - memory_id: {memory_id}, agent_id: {agent_id}, entry_type: {entry_type}")
            
            # Get the shared session
            session = self.memory_provider.retrieve_by_id(memory_id, MemoryType.SHARED_MEMORY)
            if not session:
                logger.error(f"Session not found: {memory_id}")
                return False
            
            logger.info(f"Retrieved session with {len(session.get('blackboard', []))} existing entries")
            
            # Create blackboard entry
            entry = BlackboardEntry(agent_id, content, entry_type)
            logger.info(f"Created blackboard entry with memory_id: {entry.memory_id}")
            
            # Add to blackboard
            session["blackboard"].append(entry.to_dict())
            logger.info(f"Added entry to session blackboard, now has {len(session['blackboard'])} entries")
            
            # Update in storage
            update_result = self.memory_provider.update_by_id(memory_id, session, MemoryType.SHARED_MEMORY)
            logger.info(f"Memory provider update result: {update_result}")
            
            return update_result
            
        except Exception as e:
            logger.error(f"Error adding blackboard entry: {e}", exc_info=True)
            return False
    
    def get_blackboard_entries(self, 
                              memory_id: str, 
                              agent_id: str = None, 
                              entry_type: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve blackboard entries with optional filtering.
        
        Parameters:
            memory_id (str): The shared memory ID
            agent_id (str, optional): Filter by agent ID
            entry_type (str, optional): Filter by entry type
            
        Returns:
            List[Dict[str, Any]]: List of blackboard entries
        """
        try:
            session = self.memory_provider.retrieve_by_id(memory_id, MemoryType.SHARED_MEMORY)
            if not session:
                return []
            
            entries = session.get("blackboard", [])
            
            # Apply filters
            if agent_id:
                entries = [e for e in entries if e.get("agent_id") == agent_id]
            if entry_type:
                entries = [e for e in entries if e.get("entry_type") == entry_type]
            
            return entries
            
        except Exception as e:
            logger.error(f"Error retrieving blackboard entries: {e}")
            return []
    
    def update_session_status(self, memory_id: str, status: str) -> bool:
        """Update the status of a shared session."""
        try:
            session = self.memory_provider.retrieve_by_id(memory_id, MemoryType.SHARED_MEMORY)
            if session:
                session["status"] = status
                return self.memory_provider.update_by_id(memory_id, session, MemoryType.SHARED_MEMORY)
            return False
        except Exception as e:
            logger.error(f"Error updating session status: {e}")
            return False
    
    def is_root_agent(self, memory_id: str, agent_id: str) -> bool:
        """Check if an agent is the root agent for a session."""
        try:
            session = self.memory_provider.retrieve_by_id(memory_id, MemoryType.SHARED_MEMORY)
            return session and session.get("root_agent_id") == agent_id
        except Exception as e:
            logger.error(f"Error checking root agent status: {e}")
            return False
    
    def get_session_by_root_agent(self, root_agent_id: str) -> Optional[Dict[str, Any]]:
        """Get active shared session by root agent ID."""
        try:
            # This would need to be implemented in the memory provider
            # For now, we'll need to search through sessions
            return None
        except Exception as e:
            logger.error(f"Error getting session by root agent: {e}")
            return None
    
    def find_active_session_for_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Find an active shared memory session where the agent is already participating.
        
        This enables hierarchical multi-agent coordination by allowing sub-agents
        to join existing sessions rather than creating isolated ones.
        
        Parameters:
            agent_id (str): The ID of the agent to search for
            
        Returns:
            Optional[Dict[str, Any]]: The active session if found, None otherwise
        """
        try:
            # Get all active shared memory sessions
            all_sessions = self.memory_provider.list_all(MemoryType.SHARED_MEMORY)
            
            # Handle case where list_all returns None or empty list
            if not all_sessions:
                return None
            
            for session in all_sessions:
                # Only consider active sessions
                if session.get("status") != "active":
                    continue
                
                # Check if agent is root, delegate, or sub-agent in this session
                if (session.get("root_agent_id") == agent_id or
                    agent_id in session.get("delegate_agent_ids", []) or
                    agent_id in session.get("sub_agent_ids", [])):
                    return session
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding active session for agent {agent_id}: {e}")
            return None
    
    def register_sub_agents(self, 
                           memory_id: str, 
                           parent_agent_id: str, 
                           sub_agent_ids: List[str]) -> bool:
        """
        Register sub-agents in an existing shared memory session.
        
        This method enables hierarchical agent coordination by tracking the complete
        agent hierarchy within a single shared memory session. When a delegate agent
        has its own sub-agents, they are registered here rather than creating a new session.
        
        Parameters:
            memory_id (str): The shared memory session ID
            parent_agent_id (str): The ID of the agent that owns these sub-agents
            sub_agent_ids (List[str]): List of sub-agent IDs to register
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Registering sub-agents {sub_agent_ids} under parent {parent_agent_id} in session {memory_id}")
            
            # Get the shared session
            session = self.memory_provider.retrieve_by_id(memory_id, MemoryType.SHARED_MEMORY)
            if not session:
                logger.error(f"Session not found: {memory_id}")
                return False
            
            # Ensure sub_agent_ids field exists and is a list
            if "sub_agent_ids" not in session:
                session["sub_agent_ids"] = []
            
            # Add new sub-agents (avoid duplicates)
            existing_sub_agents = set(session["sub_agent_ids"])
            new_sub_agents = [agent_id for agent_id in sub_agent_ids if agent_id not in existing_sub_agents]
            
            if new_sub_agents:
                session["sub_agent_ids"].extend(new_sub_agents)
                
                # Log the hierarchy registration for debugging
                self.add_blackboard_entry(
                    memory_id=memory_id,
                    agent_id=parent_agent_id,
                    content={
                        "action": "sub_agent_registration",
                        "parent_agent": parent_agent_id,
                        "registered_sub_agents": new_sub_agents,
                        "total_sub_agents": len(session["sub_agent_ids"])
                    },
                    entry_type="hierarchy_update"
                )
                
                # Update in storage
                update_result = self.memory_provider.update_by_id(memory_id, session, MemoryType.SHARED_MEMORY)
                logger.info(f"Successfully registered {len(new_sub_agents)} new sub-agents")
                return update_result
            else:
                logger.info("All sub-agents already registered")
                return True
                
        except Exception as e:
            logger.error(f"Error registering sub-agents: {e}", exc_info=True)
            return False
    
    def get_agent_hierarchy(self, memory_id: str) -> Dict[str, Any]:
        """
        Get the complete agent hierarchy for a shared memory session.
        
        This provides visibility into the full multi-level agent structure,
        useful for debugging, monitoring, and coordination decisions.
        
        Parameters:
            memory_id (str): The shared memory session ID
            
        Returns:
            Dict[str, Any]: Hierarchy information including all agent levels
        """
        try:
            session = self.memory_provider.retrieve_by_id(memory_id, MemoryType.SHARED_MEMORY)
            if not session:
                return {}
            
            hierarchy = {
                "root_agent": session.get("root_agent_id"),
                "delegate_agents": session.get("delegate_agent_ids", []),
                "sub_agents": session.get("sub_agent_ids", []),
                "total_agents": 1 + len(session.get("delegate_agent_ids", [])) + len(session.get("sub_agent_ids", [])),
                "session_status": session.get("status"),
                "created_at": session.get("created_at")
            }
            
            return hierarchy
            
        except Exception as e:
            logger.error(f"Error getting agent hierarchy: {e}")
            return {} 