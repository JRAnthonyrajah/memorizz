# Semantic Memory Module

The Semantic Memory module provides factual, conceptual knowledge storage and retrieval capabilities within the MemoRizz library. This memory type stores structured information about facts, concepts, rules, and general knowledge that is not tied to specific personal experiences or temporal contexts.

## Overview

Semantic memory represents one of the three core long-term memory systems in cognitive science. It stores:
- **Facts and concepts**: Objective information about the world
- **Rules and principles**: Logical relationships and procedures
- **Definitions and meanings**: Understanding of language and concepts
- **General knowledge**: Information that doesn't require episodic context

## When to Use Semantic Memory

Use semantic memory when you need to:
- Store and retrieve factual information (encyclopedic knowledge)
- Build knowledge bases for domain-specific expertise
- Create searchable repositories of structured information
- Enable agents to access general world knowledge
- Implement question-answering capabilities based on facts

**Examples of semantic memory content:**
- "The capital of France is Paris"
- "Photosynthesis converts sunlight into energy in plants"
- "Python is a programming language"
- "The boiling point of water is 100°C at sea level"

## Core Features

- **Knowledge ingestion**: Store text corpus as embedded knowledge chunks
- **Semantic retrieval**: Find relevant information using natural language queries
- **Vector search**: Leverage embeddings for similarity-based knowledge discovery
- **Namespace organization**: Categorize knowledge by domains or topics
- **Memory scoping**: Attach knowledge bases to specific agents or sessions

## Usage

### Basic Usage (Without MemAgent)

```python
from memorizz.long_term_memory.semantic import KnowledgeBase
from memorizz.memory_provider.mongodb import MongoDBProvider

# Initialize a memory provider
memory_provider = MongoDBProvider({
    "connection_string": "mongodb://localhost:27017",
    "database_name": "memorizz_db"
})

# Create a knowledge base instance
kb = KnowledgeBase(memory_provider)

# Ingest knowledge into the semantic memory
knowledge_text = """
Artificial Intelligence (AI) is a branch of computer science that aims to create 
intelligent machines capable of performing tasks that typically require human 
intelligence. AI systems can learn, reason, perceive, and make decisions.

Machine Learning is a subset of AI that enables computers to learn and improve 
from experience without being explicitly programmed for every task.
"""

# Store the knowledge with a namespace
ltm_id = kb.ingest_knowledge(
    corpus=knowledge_text,
    namespace="ai_basics"
)
print(f"Stored knowledge with LTM ID: {ltm_id}")

# Retrieve relevant knowledge
query = "What is machine learning?"
relevant_knowledge = kb.query_knowledge(
    query=query,
    namespace="ai_basics",
    limit=3
)

for knowledge in relevant_knowledge:
    print(f"Relevance Score: {knowledge['score']}")
    print(f"Content: {knowledge['content']}")
    print("---")
```

### Domain-Specific Knowledge Management

```python
# Create specialized knowledge bases for different domains
medical_kb = KnowledgeBase(memory_provider)
legal_kb = KnowledgeBase(memory_provider)

# Ingest medical knowledge
medical_text = """
Hypertension, also known as high blood pressure, is a condition where blood 
pressure in the arteries is persistently elevated. Normal blood pressure is 
typically below 120/80 mmHg. Hypertension is diagnosed when readings consistently 
exceed 140/90 mmHg.
"""

medical_ltm_id = medical_kb.ingest_knowledge(
    corpus=medical_text,
    namespace="medical_conditions"
)

# Ingest legal knowledge
legal_text = """
Contract law governs the formation and enforcement of agreements between parties. 
A valid contract requires offer, acceptance, consideration, and mutual intention 
to create legal relations. Breach of contract occurs when one party fails to 
perform their contractual obligations.
"""

legal_ltm_id = legal_kb.ingest_knowledge(
    corpus=legal_text,
    namespace="contract_law"
)

# Query domain-specific knowledge
medical_query = "What is normal blood pressure?"
medical_results = medical_kb.query_knowledge(
    query=medical_query,
    namespace="medical_conditions"
)

legal_query = "What makes a contract valid?"
legal_results = legal_kb.query_knowledge(
    query=legal_query,
    namespace="contract_law"
)
```

### Using with MemAgent

```python
from memorizz.memagent import MemAgent
from memorizz.enums.memory_type import MemoryType

# Create a MemAgent with access to semantic memory
agent = MemAgent(
    memory_provider=memory_provider,
    application_mode="assistant",  # Includes semantic memory access
    instruction="You are a knowledgeable assistant with access to factual information."
)

# Option 1: Attach specific knowledge base to agent
agent.long_term_memory_id = ltm_id  # Links agent to specific knowledge corpus

# Option 2: Create agent-specific knowledge base
agent_kb = KnowledgeBase(memory_provider)
domain_knowledge = """
Customer service best practices include active listening, empathy, 
clear communication, and problem-solving focus. Always acknowledge 
the customer's concern and work towards a satisfactory resolution.
"""

agent_ltm_id = agent_kb.ingest_knowledge(
    corpus=domain_knowledge,
    namespace="customer_service"
)
agent.long_term_memory_id = agent_ltm_id

# The agent can now access this knowledge during conversations
response = agent.run("What are some customer service best practices?")
print(response)

# Agent automatically retrieves relevant semantic knowledge
response = agent.run("How should I handle an upset customer?")
print(response)
```

### Advanced Knowledge Management

```python
# Batch knowledge ingestion
knowledge_docs = [
    {"text": "Solar energy is renewable energy from the sun...", "namespace": "renewable_energy"},
    {"text": "Wind power harnesses wind to generate electricity...", "namespace": "renewable_energy"},
    {"text": "Hydroelectric power uses water flow...", "namespace": "renewable_energy"}
]

ltm_ids = []
for doc in knowledge_docs:
    ltm_id = kb.ingest_knowledge(
        corpus=doc["text"],
        namespace=doc["namespace"]
    )
    ltm_ids.append(ltm_id)

# Cross-namespace knowledge search
query = "What are renewable energy sources?"
results = kb.query_knowledge(
    query=query,
    namespace="renewable_energy",
    limit=5
)

# Knowledge base management
available_namespaces = kb.list_namespaces()
print(f"Available knowledge domains: {available_namespaces}")

# Remove specific knowledge
kb.delete_knowledge(ltm_id=ltm_ids[0])

# Clear entire namespace
kb.clear_namespace("renewable_energy")
```

## Memory Integration

### Application Mode Configuration

Semantic memory is automatically included in these application modes:
- **ASSISTANT**: General-purpose agents with factual knowledge access
- **WORKFLOW**: Task-oriented agents needing procedural and factual knowledge
- **DEEP_RESEARCH**: Research-focused agents with extensive knowledge requirements

```python
from memorizz.enums.application_mode import ApplicationMode

# Agent with semantic memory enabled
research_agent = MemAgent(
    application_mode=ApplicationMode.DEEP_RESEARCH,
    memory_provider=memory_provider
)

# Check active memory types
print(research_agent.memory_unit.active_memory_types)
# Output includes: MemoryType.LONG_TERM_MEMORY (semantic knowledge)
```

### Memory Retrieval Flow

1. **Query Processing**: Natural language queries are converted to embeddings
2. **Similarity Search**: Vector search finds relevant knowledge chunks
3. **Context Integration**: Retrieved knowledge is integrated into agent responses
4. **Memory Updates**: Access patterns can influence future retrieval relevance

## Implementation Notes

### Storage Architecture
- Knowledge is stored as embedded text chunks in the memory provider
- Each knowledge piece includes metadata: namespace, timestamp, relevance scores
- Vector embeddings enable semantic similarity matching

### Performance Considerations
- Chunking large text improves retrieval granularity
- Namespace organization enables efficient domain-specific searches
- Embedding quality directly impacts retrieval accuracy

### Best Practices
- Use descriptive namespaces for knowledge organization
- Include context and examples in knowledge ingestion
- Regular knowledge base maintenance and updates
- Monitor retrieval quality and adjust chunking strategies

### Integration Points
- Seamlessly integrates with MemAgent conversation flow
- Compatible with all memory providers (MongoDB, etc.)
- Works alongside episodic and procedural memory systems

This semantic memory system provides the foundational knowledge layer that enables AI agents to access and reason with factual information, supporting informed decision-making and accurate responses across diverse domains.