# Working Memory Module

The Working Memory module manages the active, conscious workspace for AI agents within the MemoRizz library. This memory type handles context window management, temporary information processing, and the integration of information from different memory systems for immediate use.

## Overview

Working memory represents the conscious, active workspace of cognition. In AI agents, it manages:
- **Context window management**: Optimizing limited attention spans and token limits
- **Active information**: Currently relevant information from all memory systems
- **Processing workspace**: Temporary storage for ongoing cognitive tasks
- **Memory integration**: Combining semantic, episodic, and procedural memories for current use

## When to Use Working Memory

Use working memory when you need to:
- Manage limited context windows in language models
- Integrate information from multiple memory systems
- Maintain focus on current task-relevant information
- Optimize token usage and attention allocation
- Handle temporary information that doesn't need long-term storage
- Coordinate between different memory types for immediate processing

**Examples of working memory content:**
- "Current conversation context from the last 10 exchanges"
- "Relevant facts retrieved from semantic memory for this query"
- "Active tools and procedures for the current task"
- "Temporary variables and intermediate results"

## Core Features

- **Context Window Management**: Optimize limited token space in language models
- **Memory System Integration**: Combine information from semantic, episodic, and procedural memory
- **Attention Management**: Focus on task-relevant information
- **Temporary Storage**: Handle intermediate processing results
- **Dynamic Prompting**: Generate context-aware prompts based on active memory

## Usage

### Basic Usage (Without MemAgent)

```python
from memorizz.short_term_memory.working_memory import CWM
from memorizz.enums.memory_type import MemoryType

# === CONTEXT WINDOW MANAGEMENT ===
# Generate prompts based on active memory types
active_memory_types = [
    MemoryType.CONVERSATION_MEMORY,
    MemoryType.LONG_TERM_MEMORY,
    MemoryType.PERSONAS
]

# Generate context-aware system prompt
context_prompt = CWM.get_prompt_from_memory_types(active_memory_types)
print("Generated context prompt:")
print(context_prompt)

# The prompt will include guidance on how to use each memory type:
# - How to leverage conversation history
# - How to access semantic knowledge  
# - How to maintain persona consistency

# === WORKING MEMORY WORKSPACE ===
class WorkingMemoryWorkspace:
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.active_information = {}
        self.current_context = ""
        self.processing_variables = {}
    
    def add_context(self, context_type: str, content: str, priority: int = 1):
        """Add information to working memory with priority."""
        self.active_information[context_type] = {
            "content": content,
            "priority": priority,
            "timestamp": time.time()
        }
    
    def optimize_context(self):
        """Optimize context to fit within token limits."""
        # Sort by priority and recency
        sorted_info = sorted(
            self.active_information.items(),
            key=lambda x: (x[1]["priority"], x[1]["timestamp"]),
            reverse=True
        )
        
        # Build context within token limits
        total_tokens = 0
        optimized_context = ""
        
        for context_type, info in sorted_info:
            content_tokens = len(info["content"].split())  # Simple approximation
            if total_tokens + content_tokens <= self.max_tokens:
                optimized_context += f"\\n{context_type}: {info['content']}"
                total_tokens += content_tokens
            else:
                break
        
        self.current_context = optimized_context
        return optimized_context

# Use working memory workspace
workspace = WorkingMemoryWorkspace(max_tokens=2000)

# Add various types of active information
workspace.add_context("conversation_history", "User asked about Python debugging", priority=3)
workspace.add_context("semantic_knowledge", "Python debugging involves print statements, debuggers, and logging", priority=2)
workspace.add_context("current_task", "Help user debug their Python script", priority=3)
workspace.add_context("procedural_knowledge", "Follow systematic debugging approach", priority=2)

# Optimize for token limits
optimized_context = workspace.optimize_context()
print("Optimized working memory context:")
print(optimized_context)
```

### Advanced Working Memory Management

```python
import time
from typing import Dict, List, Any

class AdvancedWorkingMemory:
    def __init__(self, max_context_tokens: int = 3000):
        self.max_context_tokens = max_context_tokens
        self.active_memories = {}
        self.processing_cache = {}
        self.attention_weights = {}
        
    def integrate_memory_systems(self, 
                                semantic_memories: List[Dict],
                                episodic_memories: List[Dict], 
                                procedural_memories: List[Dict],
                                current_query: str):
        """Integrate information from all memory systems for current processing."""
        
        # Calculate relevance scores for each memory type
        semantic_relevance = self._calculate_relevance(semantic_memories, current_query)
        episodic_relevance = self._calculate_relevance(episodic_memories, current_query)
        procedural_relevance = self._calculate_relevance(procedural_memories, current_query)
        
        # Create integrated workspace
        integrated_context = {
            "query": current_query,
            "semantic": {
                "content": semantic_memories[:3],  # Top 3 most relevant
                "relevance": semantic_relevance,
                "weight": 0.4
            },
            "episodic": {
                "content": episodic_memories[:2],  # Top 2 most relevant
                "relevance": episodic_relevance,
                "weight": 0.3
            },
            "procedural": {
                "content": procedural_memories[:2],  # Top 2 most relevant
                "relevance": procedural_relevance,
                "weight": 0.3
            }
        }
        
        return self._optimize_integrated_context(integrated_context)
    
    def _calculate_relevance(self, memories: List[Dict], query: str) -> float:
        """Calculate relevance score for memory set."""
        if not memories:
            return 0.0
        
        # Simple relevance calculation (in practice, would use embeddings)
        total_relevance = sum(memory.get('score', 0.5) for memory in memories)
        return total_relevance / len(memories)
    
    def _optimize_integrated_context(self, context: Dict) -> str:
        """Create optimized prompt from integrated context."""
        prompt_parts = []
        
        # Add query context
        prompt_parts.append(f"Current Query: {context['query']}")
        
        # Add semantic context
        if context['semantic']['content']:
            semantic_text = "\\nRelevant Knowledge:"
            for memory in context['semantic']['content']:
                semantic_text += f"\\n- {memory.get('content', '')}"
            prompt_parts.append(semantic_text)
        
        # Add episodic context  
        if context['episodic']['content']:
            episodic_text = "\\nPast Experiences:"
            for memory in context['episodic']['content']:
                episodic_text += f"\\n- {memory.get('content', '')}"
            prompt_parts.append(episodic_text)
        
        # Add procedural context
        if context['procedural']['content']:
            procedural_text = "\\nAvailable Procedures:"
            for memory in context['procedural']['content']:
                procedural_text += f"\\n- {memory.get('name', '')}: {memory.get('description', '')}"
            prompt_parts.append(procedural_text)
        
        return "\\n".join(prompt_parts)
    
    def manage_attention(self, current_focus: str, distractors: List[str]):
        """Manage attention and filter out irrelevant information."""
        attention_scores = {}
        
        # Calculate attention weights
        for item in [current_focus] + distractors:
            # Simple attention scoring (could be more sophisticated)
            attention_scores[item] = self._calculate_attention_score(item, current_focus)
        
        # Focus on high-attention items
        focused_items = {k: v for k, v in attention_scores.items() if v > 0.5}
        
        self.attention_weights = focused_items
        return focused_items
    
    def _calculate_attention_score(self, item: str, focus: str) -> float:
        """Calculate how much attention an item should receive."""
        # Simple similarity-based attention (in practice, would use embeddings)
        common_words = set(item.lower().split()) & set(focus.lower().split())
        return len(common_words) / max(len(focus.split()), 1)

# Use advanced working memory
working_memory = AdvancedWorkingMemory(max_context_tokens=2500)

# Example memory retrieval from different systems
semantic_memories = [
    {"content": "Python is a programming language", "score": 0.8},
    {"content": "Debugging helps find code errors", "score": 0.9}
]

episodic_memories = [
    {"content": "User struggled with debugging yesterday", "score": 0.7},
    {"content": "Previous session: fixed syntax error", "score": 0.6}
]

procedural_memories = [
    {"name": "debug_tool", "description": "Run Python debugger", "score": 0.8},
    {"name": "error_analyzer", "description": "Analyze error messages", "score": 0.7}
]

# Integrate memories for current processing
integrated_context = working_memory.integrate_memory_systems(
    semantic_memories=semantic_memories,
    episodic_memories=episodic_memories,
    procedural_memories=procedural_memories,
    current_query="Help me debug this Python error"
)

print("Integrated working memory context:")
print(integrated_context)
```

### Using with MemAgent

```python
from memorizz.memagent import MemAgent
from memorizz.enums.application_mode import ApplicationMode

# Create agent with working memory management
agent = MemAgent(
    application_mode=ApplicationMode.ASSISTANT,
    memory_provider=memory_provider,
    instruction="You are a helpful assistant with access to multiple memory systems."
)

# Working memory is automatically managed by the agent
# The agent's context window management optimizes information flow

# Example conversation showing working memory in action
response1 = agent.run("I need help with Python programming")
# Working memory:
# - Stores current query in active context
# - Retrieves relevant semantic knowledge about Python
# - Generates optimized prompt within token limits

response2 = agent.run("Specifically, I'm having trouble with debugging")
# Working memory:
# - Maintains previous conversation context
# - Adds current query to active workspace
# - Retrieves debugging-specific knowledge
# - Optimizes context to include relevant history

response3 = agent.run("Can you show me how to use the debugger?")
# Working memory:
# - Integrates conversation history about Python and debugging
# - Retrieves procedural knowledge about debugging tools
# - Manages context window to maintain relevant information
# - Provides focused, contextually relevant response

print("Agent response integrating multiple memory systems:")
print(response3)

# === MONITORING WORKING MEMORY ===
# Access agent's current working memory state (conceptual)
def monitor_working_memory(agent):
    """Monitor the agent's working memory utilization."""
    
    # Get current active memory types
    active_types = agent.memory_unit.active_memory_types
    
    # Generate current context prompt
    context_prompt = CWM.get_prompt_from_memory_types(active_types)
    
    # Estimate context usage
    estimated_tokens = len(context_prompt.split())
    
    return {
        "active_memory_types": [mt.value for mt in active_types],
        "estimated_context_tokens": estimated_tokens,
        "context_prompt": context_prompt[:200] + "..." if len(context_prompt) > 200 else context_prompt
    }

# Monitor agent's working memory
memory_state = monitor_working_memory(agent)
print("\\nWorking memory state:")
print(f"Active memory types: {memory_state['active_memory_types']}")
print(f"Estimated tokens: {memory_state['estimated_context_tokens']}")
print(f"Context preview: {memory_state['context_prompt']}")
```

### Dynamic Context Management

```python
class DynamicContextManager:
    """Advanced context management for working memory."""
    
    def __init__(self, agent: MemAgent):
        self.agent = agent
        self.context_history = []
        self.token_budget = 3000
        
    def adaptive_context_selection(self, current_query: str):
        """Adaptively select context based on current needs."""
        
        # Analyze query to determine memory priorities
        query_analysis = self._analyze_query(current_query)
        
        # Adjust memory system weights based on query type
        memory_weights = self._calculate_memory_weights(query_analysis)
        
        # Retrieve weighted memories
        selected_context = {}
        
        if memory_weights.get('semantic', 0) > 0:
            semantic_memories = self.agent.memory_unit.retrieve_memory_units_by_query(
                query=current_query,
                memory_id=self.agent.memory_id,
                memory_type=MemoryType.LONG_TERM_MEMORY,
                limit=int(5 * memory_weights['semantic'])
            )
            selected_context['semantic'] = semantic_memories
        
        if memory_weights.get('episodic', 0) > 0:
            episodic_memories = self.agent.memory_unit.retrieve_memory_units_by_query(
                query=current_query,
                memory_id=self.agent.memory_id,
                memory_type=MemoryType.CONVERSATION_MEMORY,
                limit=int(5 * memory_weights['episodic'])
            )
            selected_context['episodic'] = episodic_memories
        
        return self._optimize_context_budget(selected_context)
    
    def _analyze_query(self, query: str) -> Dict[str, float]:
        """Analyze query to determine information needs."""
        analysis = {
            "factual_content": 0.0,
            "personal_reference": 0.0, 
            "task_oriented": 0.0,
            "conversational": 0.0
        }
        
        # Simple keyword-based analysis (could be more sophisticated)
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'how', 'explain', 'definition']):
            analysis['factual_content'] = 0.8
            
        if any(word in query_lower for word in ['we', 'yesterday', 'last time', 'remember']):
            analysis['personal_reference'] = 0.8
            
        if any(word in query_lower for word in ['help', 'do', 'create', 'make']):
            analysis['task_oriented'] = 0.7
            
        if any(word in query_lower for word in ['thanks', 'please', 'can you']):
            analysis['conversational'] = 0.6
        
        return analysis
    
    def _calculate_memory_weights(self, analysis: Dict[str, float]) -> Dict[str, float]:
        """Calculate memory system weights based on query analysis."""
        return {
            'semantic': analysis['factual_content'] + analysis['task_oriented'] * 0.5,
            'episodic': analysis['personal_reference'] + analysis['conversational'] * 0.5,
            'procedural': analysis['task_oriented'] + analysis['factual_content'] * 0.3
        }
    
    def _optimize_context_budget(self, context: Dict) -> str:
        """Optimize context to fit within token budget."""
        # Implementation would manage token allocation across context types
        optimized = ""
        remaining_budget = self.token_budget
        
        for context_type, memories in context.items():
            if remaining_budget <= 0:
                break
                
            type_budget = remaining_budget // len(context)
            type_content = self._format_memories(memories, type_budget)
            optimized += f"\\n\\n{context_type.upper()} CONTEXT:\\n{type_content}"
            remaining_budget -= len(type_content.split())
        
        return optimized
    
    def _format_memories(self, memories: List[Dict], token_budget: int) -> str:
        """Format memories within token budget."""
        formatted = ""
        used_tokens = 0
        
        for memory in memories:
            content = memory.get('content', '')
            content_tokens = len(content.split())
            
            if used_tokens + content_tokens <= token_budget:
                formatted += f"- {content}\\n"
                used_tokens += content_tokens
            else:
                break
        
        return formatted

# Use dynamic context manager
context_manager = DynamicContextManager(agent)
adaptive_context = context_manager.adaptive_context_selection(
    "Can you help me debug the Python script we worked on yesterday?"
)
```

## Memory Integration

### Application Mode Configuration

Working memory is active in all application modes as it coordinates other memory systems:
- **ASSISTANT**: Manages conversation context and knowledge integration
- **WORKFLOW**: Coordinates task execution and process memory
- **DEEP_RESEARCH**: Integrates research findings and maintains focus

### Cross-Memory System Coordination

Working memory serves as the integration hub for all memory systems:

```python
# Working memory coordinates between:
# 1. Semantic memory (facts and knowledge)
# 2. Episodic memory (experiences and history)  
# 3. Procedural memory (skills and behaviors)

# Example integration flow:
def demonstrate_memory_integration():
    query = "Help me implement the authentication feature"
    
    # Working memory retrieves and integrates:
    semantic_info = "Authentication involves user verification, tokens, security best practices"
    episodic_info = "Previously discussed OAuth implementation, user had concerns about security"
    procedural_info = "Available tools: auth_generator, security_scanner, implementation_workflow"
    
    # Working memory creates optimized context:
    integrated_context = f"""
    Current Task: {query}
    
    Knowledge Base: {semantic_info}
    Past Discussion: {episodic_info}  
    Available Tools: {procedural_info}
    """
    
    return integrated_context
```

## Implementation Notes

### Context Window Optimization
- Dynamic prompt generation based on active memory types
- Token budget management and allocation
- Priority-based information selection
- Automatic context compression when needed

### Memory System Integration  
- Retrieval coordination across semantic, episodic, and procedural memory
- Relevance scoring and weighting for different memory types
- Attention management and focus control
- Real-time context adaptation

### Performance Considerations
- Efficient context caching and reuse
- Lazy loading of memory content
- Background context preparation
- Memory usage monitoring and optimization

### Best Practices
- Regular context optimization to prevent bloat
- Priority-based memory selection
- Attention management for focus maintenance
- Integration testing across memory systems
- Performance monitoring and tuning

Working memory serves as the cognitive workspace that brings together information from all memory systems, optimizes it for current processing limits, and maintains the focused attention necessary for effective AI agent operation.