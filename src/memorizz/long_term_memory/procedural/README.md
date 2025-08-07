# Procedural Memory Module

The Procedural Memory module manages "how-to" knowledge and behavioral patterns within the MemoRizz library. This memory type stores learned skills, behavioral procedures, and action patterns that agents can execute automatically without conscious deliberation.

## Overview

Procedural memory represents one of the three core long-term memory systems in cognitive science. It stores:
- **Behavioral patterns**: How to act and respond in character (Personas)
- **Skill execution**: How to perform specific tasks and use tools (Toolbox)
- **Process workflows**: How to execute complex multi-step procedures (Workflows)
- **Automatic responses**: Learned behaviors that don't require conscious thought

## Components

The procedural memory system consists of three integrated components:

### 1. **Persona** - Behavioral Procedures
Stores personality traits, response patterns, and behavioral scripts that define how an agent should act.

### 2. **Toolbox** - Skill Procedures  
Stores executable functions and tools that agents can use to perform specific tasks and interact with external systems.

### 3. **Workflow** - Process Procedures
Stores complex multi-step procedures and orchestration patterns for executing sophisticated tasks.

## When to Use Procedural Memory

Use procedural memory when you need to:
- Define consistent agent personalities and behavioral patterns
- Enable agents to use external tools and APIs
- Implement complex multi-step processes and workflows
- Store learned skills that should execute automatically
- Create reusable behavioral and functional capabilities

**Examples of procedural memory content:**
- "How to respond as a helpful customer service agent" (Persona)
- "How to fetch weather data from an API" (Toolbox)
- "How to conduct a research workflow with multiple steps" (Workflow)

## Core Features

- **Persona Management**: Create, store, and manage agent personalities and behavioral patterns
- **Tool Registration**: Register Python functions as AI-callable tools with semantic discovery
- **Workflow Orchestration**: Define and execute complex multi-step processes
- **Automatic Execution**: Execute learned procedures without explicit prompting
- **Memory Integration**: Seamlessly integrate with other memory systems

## Usage

### Basic Usage (Without MemAgent)

```python
from memorizz.long_term_memory.procedural import Persona, Toolbox, Workflow
from memorizz.memory_provider.mongodb import MongoDBProvider

# Initialize a memory provider
memory_provider = MongoDBProvider({
    "connection_string": "mongodb://localhost:27017",
    "database_name": "memorizz_db"
})

# === PERSONA USAGE ===
# Create a behavioral persona
customer_service_persona = Persona(
    name="CustomerServiceAgent",
    role="Customer Support Specialist",
    goals="Provide helpful, empathetic customer support with quick resolution focus",
    background="Experienced support agent with expertise in problem-solving and customer satisfaction"
)

# Store the persona
persona_id = customer_service_persona.store_persona(memory_provider)

# Retrieve persona for use
retrieved_persona = Persona.retrieve_persona(persona_id, memory_provider)
system_prompt = retrieved_persona.generate_system_prompt_input()

# === TOOLBOX USAGE ===
# Create a toolbox for skill procedures
toolbox = Toolbox(memory_provider)

# Define and register a tool function
def get_order_status(order_id: str) -> str:
    """
    Retrieve the current status of a customer order.
    
    Parameters:
    -----------
    order_id : str
        The unique identifier for the customer order
        
    Returns:
    --------
    str
        Current status of the order (e.g., 'Processing', 'Shipped', 'Delivered')
    """
    # Implementation would connect to order system
    return f"Order {order_id} is currently being processed"

# Register the tool
tool_id = toolbox.register_tool(get_order_status)

# === WORKFLOW USAGE ===
# Create a workflow for complex procedures
workflow = Workflow(
    name="CustomerIssueResolution",
    description="Standard procedure for resolving customer issues",
    memory_provider=memory_provider
)

# Define workflow steps
issue_resolution_steps = [
    {"step": "acknowledge", "action": "Acknowledge customer concern with empathy"},
    {"step": "gather_info", "action": "Collect all relevant information about the issue"},
    {"step": "analyze", "action": "Analyze the issue and identify root cause"},
    {"step": "resolve", "action": "Apply appropriate solution or escalate if needed"},
    {"step": "follow_up", "action": "Confirm resolution and ensure customer satisfaction"}
]

workflow_id = workflow.create_workflow(issue_resolution_steps)
```

### Integrated Procedural Memory System

```python
# Create a comprehensive procedural memory setup
def setup_customer_service_procedures(memory_provider):
    # 1. Create service persona
    persona = Persona(
        name="ServiceBot",
        role="Customer Service Representative",
        goals="Resolve customer issues efficiently while maintaining satisfaction",
        background="AI assistant trained in customer service best practices"
    )
    persona_id = persona.store_persona(memory_provider)
    
    # 2. Register service tools
    toolbox = Toolbox(memory_provider)
    
    @toolbox.register_tool
    def check_account_balance(account_id: str) -> float:
        """Check customer account balance."""
        # Implementation here
        return 150.75
    
    @toolbox.register_tool
    def process_refund(order_id: str, amount: float) -> str:
        """Process a refund for a customer order."""
        # Implementation here
        return f"Refund of ${amount} processed for order {order_id}"
    
    # 3. Create service workflow
    workflow = Workflow("ServiceWorkflow", "Customer service procedures", memory_provider)
    service_steps = [
        {"step": "greet", "action": "Greet customer warmly"},
        {"step": "identify_issue", "action": "Understand the customer's concern"},
        {"step": "check_tools", "action": "Use available tools to gather information"},
        {"step": "resolve", "action": "Provide solution or next steps"},
        {"step": "close", "action": "Ensure satisfaction and close interaction"}
    ]
    workflow_id = workflow.create_workflow(service_steps)
    
    return persona_id, toolbox, workflow_id

# Setup the complete procedural system
persona_id, toolbox, workflow_id = setup_customer_service_procedures(memory_provider)
```

### Using with MemAgent

```python
from memorizz.memagent import MemAgent
from memorizz.enums.application_mode import ApplicationMode

# Create an agent with procedural memory access
agent = MemAgent(
    application_mode=ApplicationMode.ASSISTANT,  # Includes procedural memory
    memory_provider=memory_provider,
    instruction="You are a customer service agent with access to support tools."
)

# Attach procedural components to agent
# 1. Set agent persona (behavioral procedure)
agent.set_persona(retrieved_persona)

# 2. Add tools to agent (skill procedures)
agent.add_tool(toolbox=toolbox)  # Add all tools from toolbox
# OR add specific tools
agent.add_tool(tool_id=tool_id)

# 3. Set agent workflow (process procedure)
agent.set_workflow(workflow_id)

# The agent now has complete procedural memory
# - Knows HOW to behave (persona)
# - Knows HOW to use tools (toolbox)  
# - Knows HOW to follow processes (workflow)

response = agent.run("A customer is complaining about a delayed order")
print(response)
# Agent will automatically:
# 1. Use persona behavioral patterns for empathetic response
# 2. Access toolbox functions to check order status
# 3. Follow workflow steps for issue resolution
```

### Advanced Procedural Memory Management

```python
# Create specialized procedural memories for different domains
def create_technical_support_procedures():
    # Technical support persona
    tech_persona = Persona(
        name="TechSupport",
        role="Technical Support Engineer", 
        goals="Diagnose and resolve technical issues with systematic approach",
        background="Expert in troubleshooting software and hardware problems"
    )
    
    # Technical tools
    tech_toolbox = Toolbox(memory_provider)
    
    @tech_toolbox.register_tool
    def run_diagnostics(system_id: str) -> dict:
        """Run system diagnostics and return results."""
        return {"status": "healthy", "issues": [], "recommendations": []}
    
    @tech_toolbox.register_tool
    def restart_service(service_name: str) -> str:
        """Restart a specified system service."""
        return f"Service {service_name} restarted successfully"
    
    # Technical workflow
    tech_workflow = Workflow("TechSupportFlow", "Technical issue resolution", memory_provider)
    tech_steps = [
        {"step": "assess", "action": "Assess technical issue and gather details"},
        {"step": "diagnose", "action": "Run appropriate diagnostic tools"},
        {"step": "isolate", "action": "Isolate the root cause of the problem"},
        {"step": "fix", "action": "Apply technical solution"},
        {"step": "verify", "action": "Verify fix and test functionality"}
    ]
    
    return tech_persona, tech_toolbox, tech_workflow

# Create multiple procedural memory sets
tech_persona, tech_toolbox, tech_workflow = create_technical_support_procedures()

# Agents can switch between different procedural memory sets
tech_agent = MemAgent(memory_provider=memory_provider)
tech_agent.set_persona(tech_persona)
tech_agent.add_tool(toolbox=tech_toolbox)
tech_agent.set_workflow(tech_workflow)

# Now this agent has technical support procedural memory
response = tech_agent.run("The system is running slowly")
```

## Memory Integration

### Application Mode Configuration

Procedural memory components are included in these application modes:
- **ASSISTANT**: Includes personas for behavioral patterns
- **WORKFLOW**: Includes all procedural components for task execution
- **DEEP_RESEARCH**: Includes toolbox for research capabilities

```python
# Different modes activate different procedural components
workflow_agent = MemAgent(
    application_mode=ApplicationMode.WORKFLOW,
    memory_provider=memory_provider
)

# Check which procedural memory types are active
print(workflow_agent.memory_unit.active_memory_types)
# Includes: PERSONAS, TOOLBOX, WORKFLOW_MEMORY
```

### Cross-Memory System Integration

Procedural memory works seamlessly with other memory systems:

```python
# Agent with full memory integration
full_agent = MemAgent(
    application_mode=ApplicationMode.ASSISTANT,
    memory_provider=memory_provider
)

# Procedural memory provides the "HOW"
full_agent.set_persona(customer_service_persona)  # HOW to behave
full_agent.add_tool(toolbox=toolbox)              # HOW to use tools

# Semantic memory provides the "WHAT" (facts and knowledge)
# Episodic memory provides the "WHEN" (conversation history)

# Together they create a complete memory system
response = full_agent.run("Tell me about our return policy")
# Uses: procedural (how to respond), semantic (policy facts), episodic (conversation context)
```

## Implementation Notes

### Behavioral Procedures (Personas)
- Stored as personality attributes with embeddings for semantic matching
- Generate system prompts that guide agent behavior
- Can be shared across multiple agents for consistent behavior

### Skill Procedures (Toolbox)
- Functions are registered with metadata and semantic embeddings
- Automatic discovery based on query relevance
- Runtime execution with error handling and type conversion

### Process Procedures (Workflows)
- Step-by-step procedures stored with execution metadata
- Support for conditional logic and branching
- Integration with tool execution and decision-making

### Best Practices
- Create specific personas for different agent roles and contexts
- Register tools with comprehensive docstrings for better discovery
- Design workflows with clear steps and decision points
- Test procedural combinations to ensure smooth integration
- Use namespacing to organize different procedural memory sets

This procedural memory system enables agents to not just know facts, but to embody behaviors, execute skills, and follow complex processes - providing the "muscle memory" that makes AI agents truly capable and autonomous.