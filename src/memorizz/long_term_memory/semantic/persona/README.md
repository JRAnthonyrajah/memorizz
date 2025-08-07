# Persona Module

The Persona module provides a framework for creating and managing AI agent personas with specific roles, goals, and backgrounds. This module is part of the Memorizz library, which handles memory management for AI agents.

## Features

- Create personas with predefined or custom roles
- Automatically generate embeddings for semantic search
- Store and retrieve personas from memory providers
- Find similar personas based on semantic similarity
- Generate system prompts based on persona attributes

## Usage

### Creating a Persona

```python
from src.memorizz.long_term_memory.procedural.persona import Persona
from src.memorizz.memory_provider import MemoryProvider

# Initialize a memory provider
memory_provider = MemoryProvider()

# Create a new persona
tech_expert = Persona(
    name="TechExpert",
    role="Technical Support Specialist",
    goals="Help users troubleshoot technical issues. Provide clear explanations for complex problems.",
    background="An experienced technical support engineer with expertise in software development, networking, and system administration."
)

# Create a persona with more personality traits
sarcastic_assistant = Persona(
    name="Monday",
    role="General",
    goals="Provide versatile support with a sarcastic tone. Add humor to interactions.",
    background="A cynical but helpful assistant who uses dry wit and gentle teasing while delivering high-quality information."
)
```

### Storing Personas

Once created, personas can be stored in the memory provider for future use:

```python
# Store the persona in the memory provider
persona_id = tech_expert.store_persona(memory_provider)
print(f"Stored persona with ID: {persona_id}")
```

### Generating Persona Prompts

Personas can generate system prompts for language models:

```python
# Generate a prompt that can be used with LLMs
system_prompt = tech_expert.generate_system_prompt_input()
print(system_prompt)
```

### Retrieving Personas

Personas can be retrieved by ID:

```python
# Retrieve a persona using its ID
retrieved_persona = Persona.retrieve_persona(persona_id, memory_provider)
print(retrieved_persona)
```

Or by semantic similarity to a query:

```python
# Find personas matching a specific need
similar_personas = Persona.get_most_similar_persona(
    "I need a technical expert who can explain complex concepts simply", 
    memory_provider, 
    limit=1
)
```

### Using Personas with MemAgents

Personas can be assigned to MemAgents to control their behavior:

```python
from src.memorizz.memagent import MemAgent

# Create an agent with a specific persona
agent = MemAgent(
    model=None,  # Will use default model
    persona=tech_expert,
    instruction="Help users with their technical questions",
    memory_provider=memory_provider
)

# Or set/change a persona later
agent.set_persona(sarcastic_assistant)

# Run the agent with its persona influencing responses
response = agent.run("Can you help me fix my computer?")
```

### Persona Persistence

Personas are stored with vector embeddings for efficient retrieval:

```python
# List all available personas
all_personas = memory_provider.list_all(memory_type=MemoryType.PERSONA)

# Delete a persona
memory_provider.delete_by_id(persona_id, memory_type=MemoryType.PERSONA)
```

## Implementation Notes

- Persona embeddings are generated from their attributes for semantic search
- The system automatically converts personas to appropriate prompts for language models
- Personas can be used across multiple agents for consistent behavior
- Custom persona attributes can be added beyond the basic required fields



