# Episodic Memory Module

The Episodic Memory module manages personal experiences and time-stamped event memories within the MemoRizz library. This memory type stores autobiographical information about specific events, conversations, and experiences from the agent's perspective.

## Overview

Episodic memory represents one of the three core long-term memory systems in cognitive science. It stores:
- **Personal experiences**: Specific events and interactions from the agent's life
- **Temporal context**: When events occurred and their sequence
- **Experiential details**: Rich contextual information about experiences
- **Compressed memories**: Summarized versions of extended experiences

## Components

The episodic memory system consists of two integrated components:

### 1. **Conversational Memory Unit** - Individual Experience Records
Stores individual conversation turns and interactions as discrete episodic memories with full contextual detail.

### 2. **Summary Component** - Compressed Experience Records
Stores compressed summaries of multiple experiences over time periods, preserving important patterns while reducing memory load.

## When to Use Episodic Memory

Use episodic memory when you need to:
- Remember specific conversations and interactions
- Maintain context across extended dialogues
- Track the history of relationships and interactions
- Preserve personal experiences and their emotional context
- Enable agents to reference past events and experiences
- Manage memory efficiently through summarization

**Examples of episodic memory content:**
- "Yesterday at 3:PM, the user asked about their order status and seemed frustrated"
- "In our conversation last week, the user mentioned they prefer email over phone"
- "During the technical support session on Monday, we resolved a database issue"

## Core Features

- **Experience Recording**: Store detailed records of individual conversations and interactions
- **Temporal Organization**: Organize memories by time and conversation context
- **Memory Compression**: Automatically summarize extended interactions to manage memory load
- **Contextual Retrieval**: Find relevant past experiences based on current context
- **Emotional Tagging**: Track emotional themes and situational contexts in memories
- **Relationship Tracking**: Maintain understanding of ongoing relationships and interactions

## Usage

### Basic Usage (Without MemAgent)

```python
from memorizz.long_term_memory.episodic import ConversationMemoryUnit, SummaryComponent
from memorizz.memory_provider.mongodb import MongoDBProvider
from memorizz.memory_unit import MemoryUnit
from memorizz.enums.memory_type import MemoryType
import time

# Initialize a memory provider
memory_provider = MongoDBProvider({
    "connection_string": "mongodb://localhost:27017",
    "database_name": "memorizz_db"
})

# Initialize memory unit for episodic memory management
memory_unit = MemoryUnit(
    application_mode="assistant",
    memory_provider=memory_provider
)

# === CONVERSATIONAL MEMORY USAGE ===
# Create individual conversation memories
conversation_memory = ConversationMemoryUnit(
    role="user",
    content="I need help with setting up my new email account",
    timestamp=str(time.time()),
    memory_id="session_123",
    conversation_id="conv_456",
    embedding=[0.1, 0.2, 0.3, 0.4],  # Would be generated automatically
    recall_recency=time.time(),
    associated_conversation_ids=[]
)

# Store the conversational memory
memory_unit._save_memory_unit(conversation_memory, MemoryType.CONVERSATION_MEMORY)

# Create agent response memory
agent_response_memory = ConversationMemoryUnit(
    role="assistant",
    content="I'd be happy to help you set up your email account. Let's start by...",
    timestamp=str(time.time()),
    memory_id="session_123",
    conversation_id="conv_456",
    embedding=[0.2, 0.3, 0.4, 0.5],
    recall_recency=time.time(),
    associated_conversation_ids=[]
)

memory_unit._save_memory_unit(agent_response_memory, MemoryType.CONVERSATION_MEMORY)

# === SUMMARY COMPONENT USAGE ===
# Create a summary of multiple conversation memories
summary = SummaryComponent(
    memory_id="session_123",
    agent_id="customer_service_bot",
    summary_content="User requested help with email setup. Provided step-by-step guidance for Gmail configuration. User was satisfied with the resolution and successfully configured their account.",
    period_start=time.time() - 3600,  # 1 hour ago
    period_end=time.time(),
    memory_units_count=8,  # Summarized 8 conversation turns
    emotional_tags=["helpful", "resolved", "satisfied"],
    situational_tags=["technical_support", "email_setup", "first_time_user"],
    importance_score=0.8
)

# Store the summary
memory_unit._save_memory_unit(summary, MemoryType.SUMMARIES)

# === MEMORY RETRIEVAL ===
# Retrieve conversation history for a specific memory ID
conversation_history = memory_unit.retrieve_memory_units_by_memory_id(
    memory_id="session_123",
    memory_type=MemoryType.CONVERSATION_MEMORY
)

# Query for similar past experiences
similar_memories = memory_unit.retrieve_memory_units_by_query(
    query="email setup help",
    memory_id="session_123",
    memory_type=MemoryType.CONVERSATION_MEMORY,
    limit=5
)

print(f"Found {len(similar_memories)} similar experiences")
for memory in similar_memories:
    print(f"Content: {memory['content']}")
    print(f"Relevance: {memory['memory_signal']}")
```

### Advanced Episodic Memory Management

```python
# Create a comprehensive episodic memory system
def manage_customer_relationship(customer_id: str, memory_provider):
    memory_unit = MemoryUnit("assistant", memory_provider)
    
    # Record multiple interaction episodes
    interactions = [
        {
            "role": "user",
            "content": "I'm having trouble with my order delivery",
            "context": "frustrated_customer"
        },
        {
            "role": "assistant", 
            "content": "I understand your frustration. Let me check your order status immediately.",
            "context": "empathetic_response"
        },
        {
            "role": "user",
            "content": "Thank you, that would be great",
            "context": "appreciative"
        },
        {
            "role": "assistant",
            "content": "I've located your order and expedited shipping. You'll receive it tomorrow.",
            "context": "problem_solved"
        }
    ]
    
    # Store each interaction as episodic memory
    conversation_memories = []
    for i, interaction in enumerate(interactions):
        memory = ConversationMemoryUnit(
            role=interaction["role"],
            content=interaction["content"],
            timestamp=str(time.time() + i),
            memory_id=customer_id,
            conversation_id=f"support_{customer_id}_{int(time.time())}",
            embedding=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i],
            recall_recency=time.time()
        )
        memory_unit._save_memory_unit(memory, MemoryType.CONVERSATION_MEMORY)
        conversation_memories.append(memory)
    
    # Create summary of the customer relationship
    relationship_summary = SummaryComponent(
        memory_id=customer_id,
        agent_id="customer_service",
        summary_content="Customer experienced delivery delay, initially frustrated but became appreciative after quick resolution. Values prompt service and clear communication.",
        period_start=time.time() - 1800,  # 30 minutes ago
        period_end=time.time(),
        memory_units_count=len(interactions),
        emotional_tags=["frustrated", "appreciative", "satisfied"],
        situational_tags=["delivery_issue", "quick_resolution", "customer_service"],
        importance_score=0.9
    )
    
    memory_unit._save_memory_unit(relationship_summary, MemoryType.SUMMARIES)
    
    return conversation_memories, relationship_summary

# Use the system
customer_memories, summary = manage_customer_relationship("customer_789", memory_provider)
```

### Using with MemAgent

```python
from memorizz.memagent import MemAgent
from memorizz.enums.application_mode import ApplicationMode

# Create an agent with episodic memory access
agent = MemAgent(
    application_mode=ApplicationMode.ASSISTANT,  # Includes episodic memory
    memory_provider=memory_provider,
    instruction="You are a customer service agent who remembers past interactions."
)

# The agent automatically manages episodic memory during conversations
# Each interaction is stored as a ConversationMemoryUnit

# Start a conversation (episodic memories are created automatically)
response1 = agent.run("Hi, I ordered something last week but haven't received it yet")
# This creates episodic memory for both user input and agent response

# Continue conversation (agent can reference past context)
response2 = agent.run("Can you check on that order I mentioned?")
# Agent can retrieve episodic memory from earlier in conversation

# Later conversation (agent remembers past customer)
# If customer returns with same memory_id, agent recalls past episodes
response3 = agent.run("Hi again, I wanted to follow up on my delivery")
# Agent retrieves relevant episodic memories from past interactions

print("Agent response:", response3)
# Agent response includes context from remembered episodes

# === WORKING WITH AGENT EPISODIC MEMORIES ===
# Access agent's episodic memories
conversation_history = agent.memory_unit.retrieve_memory_units_by_memory_id(
    memory_id=agent.memory_id,
    memory_type=MemoryType.CONVERSATION_MEMORY
)

print(f"Agent has {len(conversation_history)} episodic memories")

# Query agent's past experiences
relevant_experiences = agent.memory_unit.retrieve_memory_units_by_query(
    query="delivery problems",
    memory_id=agent.memory_id,
    memory_type=MemoryType.CONVERSATION_MEMORY
)

print(f"Found {len(relevant_experiences)} relevant past experiences")
```

### Memory Summarization and Compression

```python
# Advanced memory management with automatic summarization
class EpisodicMemoryManager:
    def __init__(self, memory_provider):
        self.memory_provider = memory_provider
        self.memory_unit = MemoryUnit("assistant", memory_provider)
    
    def create_periodic_summaries(self, memory_id: str, period_hours: int = 24):
        """Create summaries for extended conversation periods."""
        # Retrieve memories from the specified period
        recent_memories = self.get_memories_in_period(memory_id, period_hours)
        
        if len(recent_memories) < 3:  # Not enough memories to summarize
            return None
        
        # Generate summary content (in practice, would use LLM)
        summary_text = self.generate_summary(recent_memories)
        
        # Extract emotional and situational themes
        emotional_tags = self.extract_emotional_themes(recent_memories)
        situational_tags = self.extract_situational_themes(recent_memories)
        
        # Create summary component
        summary = SummaryComponent(
            memory_id=memory_id,
            agent_id="memory_manager",
            summary_content=summary_text,
            period_start=time.time() - (period_hours * 3600),
            period_end=time.time(),
            memory_units_count=len(recent_memories),
            emotional_tags=emotional_tags,
            situational_tags=situational_tags,
            importance_score=self.calculate_importance(recent_memories)
        )
        
        # Store summary and optionally remove original memories
        self.memory_unit._save_memory_unit(summary, MemoryType.SUMMARIES)
        
        return summary
    
    def get_memories_in_period(self, memory_id: str, hours: int):
        """Helper to retrieve memories from specific time period."""
        # Implementation would query memory provider with time filters
        return []
    
    def generate_summary(self, memories):
        """Generate natural language summary of memory sequence."""
        return "Summary of conversation period..."
    
    def extract_emotional_themes(self, memories):
        """Extract emotional themes from memories."""
        return ["collaborative", "problem-solving"]
    
    def extract_situational_themes(self, memories):
        """Extract situational themes from memories."""
        return ["technical_discussion", "planning_session"]
    
    def calculate_importance(self, memories):
        """Calculate importance score for memory sequence."""
        return 0.7

# Use the memory manager
memory_manager = EpisodicMemoryManager(memory_provider)
summary = memory_manager.create_periodic_summaries("project_team_123", 8)
```

## Memory Integration

### Application Mode Configuration

Episodic memory is included in these application modes:
- **ASSISTANT**: Stores conversation history and summaries for personalization
- **WORKFLOW**: Tracks task execution episodes and process memories
- **DEEP_RESEARCH**: Maintains research session histories and findings

```python
# Different modes use episodic memory differently
assistant_agent = MemAgent(
    application_mode=ApplicationMode.ASSISTANT,
    memory_provider=memory_provider
)

# Check active episodic memory types
print(assistant_agent.memory_unit.active_memory_types)
# Includes: CONVERSATION_MEMORY, SUMMARIES
```

### Cross-Memory System Integration

Episodic memory integrates with semantic and procedural memory:

```python
# Complete memory integration example
integrated_agent = MemAgent(
    application_mode=ApplicationMode.ASSISTANT,
    memory_provider=memory_provider
)

# Example conversation demonstrating all memory types
user_input = "Can you help me with the Python issue we discussed yesterday?"

# Agent response uses:
# - Episodic: Recalls yesterday's conversation about Python
# - Semantic: Accesses factual knowledge about Python
# - Procedural: Uses debugging tools and follow systematic approach

response = integrated_agent.run(user_input)

# The response demonstrates integration:
# "I remember yesterday we were working on the database connection issue (episodic). 
#  The error was related to SQLAlchemy configuration (semantic).
#  Let me run the diagnostic tool to check the current status (procedural)."
```

## Implementation Notes

### Conversational Memory Units
- Store individual conversation turns with full context
- Include embeddings for semantic similarity search
- Track recall patterns and associated conversations
- Support rich metadata for filtering and organization

### Summary Components
- Compress multiple memories into coherent narratives
- Preserve emotional and situational context
- Include importance scoring for memory prioritization
- Support automatic and manual summary generation

### Memory Retrieval Patterns
- **Temporal**: Retrieve memories from specific time periods
- **Similarity**: Find memories similar to current context
- **Associative**: Retrieve related memories through conversation IDs
- **Importance**: Prioritize high-importance episodic memories

### Best Practices
- Regular summarization to manage memory load
- Rich metadata tagging for better organization
- Importance scoring to prioritize critical episodes
- Cross-conversation linking for relationship tracking
- Emotional context preservation for empathetic responses

### Performance Considerations
- Automatic memory compression for long conversations
- Efficient indexing by time, memory_id, and conversation_id
- Embedding-based retrieval for context relevance
- Background summarization to maintain responsiveness

This episodic memory system enables agents to maintain rich personal histories, remember specific interactions, and build meaningful relationships through accumulated experiences over time.