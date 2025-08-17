<div align="center">

# Memorizz 🧠

📊 **[Agent Memory Presentation](https://docs.google.com/presentation/d/1iSu667m5-pOXMrJq_LjkfnfD4V0rW1kbhGaQ2u3TKXQ/edit?usp=sharing)** | 🎥 **[AIEWF Richmond's Talk](https://youtu.be/W2HVdB4Jbjs?si=faaI3cMLc71Efpeu)**

[![PyPI version](https://badge.fury.io/py/memorizz.svg)](https://badge.fury.io/py/memorizz)
[![PyPI downloads](https://img.shields.io/pypi/dm/memorizz.svg)](https://pypistats.org/packages/memorizz)

</div>

> **⚠️ IMPORTANT WARNING ⚠️**
> 
> **MemoRizz is an EXPERIMENTAL library intended for EDUCATIONAL PURPOSES ONLY.**
> 
> **Do NOT use in production environments or with sensitive data.**
> 
> This library is under active development, has not undergone security audits, and may contain bugs or breaking changes in future releases.

## Overview

**MemoRizz is a memory management framework for AI agents designed to create memory-augmented agents with explicit memory type allocation based on application mode.**

The framework enables developers to build context-aware agents capable of sophisticated information retrieval and storage. 

MemoRizz provides flexible single and multi-agent architectures that allow you to instantiate agents with specifically allocated memory types—whether episodic, semantic, procedural, or working memory—tailored to your application's operational requirements.


**Why MemoRizz?**
- 🧠 **Persistent Memory**: Your AI agents remember conversations across sessions
- 🔍 **Semantic Search**: Find relevant information using natural language
- 🛠️ **Tool Integration**: Automatically discover and execute functions
- 👤 **Persona System**: Create consistent, specialized agent personalities
- 📊 **Vector Search**: MongoDB Atlas Vector Search for efficient retrieval
- ⚡ **Semantic Cache**: Speed up responses and reduce costs with intelligent caching

## Key Features

- **Persistent Memory Management**: Long-term memory storage with semantic retrieval
- **MemAgent System**: Complete AI agents with memory, personas, and tools
- **MongoDB Integration**: Built on MongoDB Atlas with vector search capabilities
- **Tool Registration**: Automatically convert Python functions into LLM-callable tools
- **Persona Framework**: Create specialized agent personalities and behaviors
- **Vector Embeddings**: Semantic similarity search across all stored information
- **Semantic Cache**: Intelligent query-response caching with vector similarity matching

## Installation

```bash
pip install memorizz
```

### Prerequisites
- Python 3.7+
- MongoDB Atlas account (or local MongoDB with vector search)
- OpenAI API key (for embeddings and LLM functionality)

## Quick Start

### 1. Basic MemAgent Setup

```python
import os
from memorizz.memory_provider.mongodb.provider import MongoDBConfig, MongoDBProvider
from memorizz.memagent import MemAgent
from memorizz.llms.openai import OpenAI

# Set up your API keys
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Configure MongoDB memory provider
mongodb_config = MongoDBConfig(uri="your-mongodb-atlas-uri")
memory_provider = MongoDBProvider(mongodb_config)

# Create a MemAgent
agent = MemAgent(
    model=OpenAI(model="gpt-4"),
    instruction="You are a helpful assistant with persistent memory.",
    memory_provider=memory_provider
)

# Start conversing - the agent will remember across sessions
response = agent.run("Hello! My name is John and I'm a software engineer.")
print(response)

# Later in another session...
response = agent.run("What did I tell you about myself?")
print(response)  # Agent remembers John is a software engineer
```

# Table of single agent and multi-agent setups, their descriptions, and links to example notebooks
| Agent Type                | Description                                                                 | Example Notebook                                                                 |
|---------------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Single Agent              | A standalone agent with its own memory and persona, suitable for individual tasks | [Single Agent Example](examples/memagent_single_agent.ipynb)                      |
| Multi-Agent               | A system of multiple agents collaborating, each with specialized roles and shared memory | [Multi-Agent Example](examples/memagents_multi_agents.ipynb)                        |



# Memory System Components and Examples

| Memory Component | Memory Category | Use Case / Description | Example Notebook |
|------------------|-----------------|------------------------|------------------|
| **Persona** | Semantic Memory | Agent identity, personality, and behavioral consistency | [Persona Example](examples/persona.ipynb) |
| **Knowledge Base** | Semantic Memory | Persistent facts, concepts, and domain knowledge | [Knowledge Base Example](examples/knowledge_base.ipynb) |
| **Toolbox** | Procedural Memory | Registered functions with semantic discovery for LLM execution | [Toolbox Example](examples/toolbox.ipynb) |
| **Workflow** | Procedural Memory | Multi-step process orchestration and execution tracking | [Workflow Example](examples/workflow.ipynb) |
| **Conversation Memory** | Episodic Memory | Interaction history and conversational context | [Single Agent Example](examples/memagent_single_agent.ipynb) |
| **Summaries** | Episodic Memory | Compressed episodic experiences and events | [Summarization Example](examples/memagent_summarisation.ipynb) |
| **Working Memory** | Short-term Memory | Active context management and current session state | [Single Agent Example](examples/memagent_single_agent.ipynb) |
| **Semantic Cache** | Short-term Memory | Vector-based query-response caching for performance optimization | [Semantic Cache Demo](examples/semantic_cache.ipynb) |
| **Shared Memory** | Multi-Agent Coordination | Blackboard for inter-agent communication and coordination | [Multi-Agent Example](examples/memagents_multi_agents.ipynb) |


### 2. Creating Specialized Agents with Personas

```python
from memorizz.long_term_memory.semantic.persona import Persona
from memorizz.long_term_memory.semantic.persona.role_type import RoleType

# Create a technical expert persona using predefined role types
tech_expert = Persona(
    name="TechExpert",
    role=RoleType.TECHNICAL_EXPERT,  # Use predefined role enum
    goals="Help developers solve complex technical problems with detailed explanations.",
    background="10+ years experience in Python, AI/ML, and distributed systems."
)

# Apply persona to agent
agent.set_persona(tech_expert)
agent.save()

# Now the agent will respond as a technical expert
response = agent.run("How should I design a scalable microservices architecture?")
```

### 3. Tool Registration and Function Calling

```python
from memorizz.database import MongoDBTools, MongoDBToolsConfig
from memorizz.embeddings.openai import get_embedding

# Configure tools database
tools_config = MongoDBToolsConfig(
    mongo_uri="your-mongodb-atlas-uri",
    db_name="my_tools_db",
    get_embedding=get_embedding  # Required embedding function
)

# Register tools using decorator
with MongoDBTools(tools_config) as tools:
    toolbox = tools.mongodb_toolbox()
    
    @toolbox
    def calculate_compound_interest(principal: float, rate: float, time: int) -> float:
        """Calculate compound interest for financial planning."""
        return principal * (1 + rate) ** time
    
    @toolbox
    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        # Your weather API integration here
        return f"Weather in {city}: 72°F, sunny"
    
    # Add tools to your agent
    agent.add_tool(toolbox=toolbox)
    
    # Agent can now discover and use these tools automatically
    response = agent.run("What's the weather in San Francisco and calculate interest on $1000 at 5% for 3 years?")
```

### 4. Semantic Cache for Performance Optimization

Speed up your agents and reduce LLM costs with intelligent semantic caching:

```python
# Enable semantic cache on any MemAgent
agent = MemAgent(
    model=OpenAI(model="gpt-4"),
    instruction="You are a helpful assistant.",
    memory_provider=memory_provider,
    semantic_cache=True,  # Enable semantic cache
    semantic_cache_config={
        "similarity_threshold": 0.85,  # Adjust sensitivity (0.0-1.0)
        "max_cache_size": 1000,        # Maximum cached responses
        "ttl_hours": 24.0              # Cache expiration time
    }
)

# Similar queries will use cached responses
response1 = agent.run("What is the capital of France?")
response2 = agent.run("Tell me France's capital city")  # Cache hit! ⚡
response3 = agent.run("What's the capital of Japan?")   # New query, cache miss

# For external frameworks - use standalone semantic cache
from memorizz.short_term_memory.semantic_cache import StandaloneSemanticCache

cache = StandaloneSemanticCache(
    similarity_threshold=0.8,
    embedding_provider="openai"
)

# Integrate with any agent framework
def my_agent(query: str) -> str:
    # Check cache first
    cached_response = cache.query(query)
    if cached_response:
        return cached_response
    
    # Generate new response with your LLM
    response = your_llm_call(query)
    
    # Cache for future similar queries
    cache.cache_response(query, response)
    return response

# Memory-scoped caching for multi-session isolation
from memorizz.short_term_memory.semantic_cache import create_semantic_cache

user_cache = create_semantic_cache(
    agent_id="assistant",
    memory_id="user_session_123",  # Isolate cache per user session
    similarity_threshold=0.9
)
```

**How Semantic Cache Works:**
1. **Store queries + responses** with vector embeddings
2. **New query arrives** → generate embedding  
3. **Similarity search** in cache using cosine similarity
4. **Cache hit** (similarity ≥ threshold) → return cached response ⚡
5. **Cache miss** → fallback to LLM + cache new response

**Benefits:**
- 🚀 **Faster responses** for similar queries
- 💰 **Reduced LLM costs** by avoiding duplicate API calls
- 🎯 **Configurable precision** with similarity thresholds
- 🔒 **Scoped isolation** by agent, memory, or session ID
- 🔌 **Framework agnostic** - works with any agent system

## Core Concepts

### Memory Types

MemoRizz supports different memory categories for organizing information:

- **CONVERSATION_MEMORY**: Chat history and dialogue context
- **WORKFLOW_MEMORY**: Multi-step process information
- **LONG_TERM_MEMORY**: Persistent knowledge storage with semantic search
- **SHORT_TERM_MEMORY**: Temporary processing information including semantic cache for query-response optimization
- **PERSONAS**: Agent personality and behavior definitions
- **TOOLBOX**: Function definitions and metadata
- **SHARED_MEMORY**: Multi-agent coordination and communication
- **MEMAGENT**: Agent configurations and states
- **SUMMARIES**: Compressed summaries of past interactions for efficient memory management

### Long-Term Knowledge Management

Store and retrieve persistent knowledge with semantic search:

```python
# Add knowledge to long-term memory
knowledge_id = agent.add_long_term_memory(
    "I prefer Python for backend development due to its simplicity and extensive libraries.", 
    namespace="preferences"
)

# Retrieve related knowledge
knowledge_entries = agent.retrieve_long_term_memory(knowledge_id)

# Update existing knowledge
agent.update_long_term_memory(
    knowledge_id, 
    "I prefer Python for backend development and FastAPI for building APIs."
)

# Delete knowledge when no longer needed
agent.delete_long_term_memory(knowledge_id)
```

### Tool Discovery

Tools are semantically indexed, allowing natural language discovery:

```python
# Tools are automatically found based on intent
agent.run("I need to check the weather")  # Finds and uses get_weather tool
agent.run("Help me calculate some financial returns")  # Finds compound_interest tool
```

## Advanced Usage

### Custom Memory Providers

Extend the memory provider interface for custom storage backends:

```python
from memorizz.memory_provider.base import MemoryProvider

class CustomMemoryProvider(MemoryProvider):
    def store(self, data, memory_store_type):
        # Your custom storage logic
        pass
    
    def retrieve_by_query(self, query, memory_store_type, limit=10):
        # Your custom retrieval logic
        pass
```

### Multi-Agent Workflows

Create collaborative agent systems:

```python
# Create specialized delegate agents
data_analyst = MemAgent(
    model=OpenAI(model="gpt-4"),
    instruction="You are a data analysis expert.",
    memory_provider=memory_provider
)

report_writer = MemAgent(
    model=OpenAI(model="gpt-4"), 
    instruction="You are a report writing specialist.",
    memory_provider=memory_provider
)

# Create orchestrator agent with delegates
orchestrator = MemAgent(
    model=OpenAI(model="gpt-4"),
    instruction="You coordinate between specialists to complete complex tasks.",
    memory_provider=memory_provider,
    delegates=[data_analyst, report_writer]
)

# Execute multi-agent workflow
response = orchestrator.run("Analyze our sales data and create a quarterly report.")
```

### Memory Management Operations

Control agent memory persistence:

```python
# Save agent state to memory provider
agent.save()

# Load existing agent by ID
existing_agent = MemAgent.load(
    agent_id="your-agent-id",
    memory_provider=memory_provider
)

# Update agent configuration
agent.update(
    instruction="Updated instruction for the agent",
    max_steps=30
)

# Delete agent and optionally cascade delete memories
MemAgent.delete_by_id(
    agent_id="agent-id-to-delete",
    cascade=True,  # Deletes associated memories
    memory_provider=memory_provider
)
```

## Architecture

```
┌─────────────────┐
│   MemAgent      │  ← High-level agent interface
├─────────────────┤
│   Persona       │  ← Agent personality & behavior
├─────────────────┤
│   Toolbox       │  ← Function registration & discovery
├─────────────────┤
│ Memory Provider │  ← Storage abstraction layer
├─────────────────┤
│ Vector Search   │  ← Semantic similarity & retrieval
├─────────────────┤
│   MongoDB       │  ← Persistent storage backend
└─────────────────┘
```

## Examples

Check out the `examples/` directory for complete working examples:

- **memagent_single_agent.ipynb**: Basic conversational agent with memory
- **memagents_multi_agents.ipynb**: Multi-agent collaboration workflows
- **persona.ipynb**: Creating and using agent personas
- **toolbox.ipynb**: Tool registration and function calling
- **workflow.ipynb**: Workflow memory and process tracking
- **knowledge_base.ipynb**: Long-term knowledge management
- **semantic_cache_demo.py**: Semantic cache for performance optimization and external framework integration

## Configuration

### MongoDB Atlas Setup

1. Create a MongoDB Atlas cluster
2. Enable Vector Search on your cluster
3. Create a database and collection for your agent
4. Get your connection string

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="your-openai-api-key"
export MONGODB_URI="your-mongodb-atlas-uri"

# Optional
export MONGODB_DB_NAME="memorizz"  # Default database name
```

## Troubleshooting

**Common Issues:**

1. **MongoDB Connection**: Ensure your IP is whitelisted in Atlas
2. **Vector Search**: Verify vector search is enabled on your cluster
3. **API Keys**: Check OpenAI API key is valid and has credits
4. **Import Errors**: Ensure you're using the correct import paths shown in examples

## Contributing

This is an educational project. Contributions for learning purposes are welcome:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Educational Resources

This library demonstrates key concepts in:
- **AI Agent Architecture**: Memory, reasoning, and tool use
- **Vector Databases**: Semantic search and retrieval
- **LLM Integration**: Function calling and context management
- **Software Design**: Clean abstractions and extensible architecture
