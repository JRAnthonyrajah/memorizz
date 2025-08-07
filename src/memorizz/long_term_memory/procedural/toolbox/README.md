# Toolbox Module

The Toolbox module provides a powerful framework for registering, managing, and utilizing external functions as AI-callable tools within the MemoRizz library. This system enables agents to seamlessly interact with external systems, APIs, and data sources through a standardized interface.

## Core Features

- Register Python functions as semantically discoverable tools
- Generate embeddings for advanced similarity-based tool retrieval
- Store and retrieve tools from memory providers
- Find the most relevant tools based on natural language queries
- Integrate tools with MemAgents for intelligent function execution

## Usage

### Creating a Toolbox

```python
from src.memorizz.long_term_memory.procedural.toolbox import Toolbox
from src.memorizz.memory_provider import MemoryProvider

# Initialize a memory provider
memory_provider = MemoryProvider()

# Create a toolbox instance
toolbox = Toolbox(memory_provider)
```

### Registering Tools

```python
# Define a function to be used as a tool
def get_weather(latitude: float, longitude: float) -> float:
    """
    Get the current temperature at the specified coordinates.
    
    Parameters:
    -----------
    latitude : float
        The latitude coordinate (between -90 and 90)
    longitude : float
        The longitude coordinate (between -180 and 180)
        
    Returns:
    --------
    float
        The current temperature in Celsius
    """
    # Implementation here...
    return temperature

# Register the function as a tool
weather_tool_id = toolbox.register_tool(get_weather)
print(f"Registered tool with ID: {weather_tool_id}")

# You can also use the decorator pattern
@toolbox.register_tool
def get_stock_price(symbol: str, currency: str = "USD") -> str:
    """
    Get the current stock price for a given stock symbol.
    
    Parameters:
    -----------
    symbol : str
        The stock symbol to look up (e.g., 'AAPL' for Apple Inc.)
    currency : str, optional
        The currency code to convert the price into (defaults to 'USD')
        
    Returns:
    --------
    str
        A string with the current stock price
    """
    # Implementation here...
    return f"The current price of {symbol} is {price} {currency}."
```

### Retrieving Tools

```python
# Get a tool by its name
weather_tool = toolbox.get_tool_by_name("get_weather")

# Get a tool by its ID
stock_tool = toolbox.get_tool_by_id(weather_tool_id)

# Find tools based on a natural language query
finance_tools = toolbox.get_most_similar_tools(
    "I need something that can tell me about stocks", 
    limit=3
)

# List all available tools
all_tools = toolbox.list_tools()
```

### Tool Management

```python
# Delete a tool by name
toolbox.delete_tool_by_name("get_weather")

# Delete a tool by ID
toolbox.delete_tool_by_id(stock_tool_id)

# Update tool metadata
toolbox.update_tool_by_id(
    tool_id, 
    {"description": "An improved description of this tool"}
)

# Clear all tools
toolbox.delete_all()
```

### Using with MemAgents

Tools can be added to MemAgents in two ways:

```python
from src.memorizz.memagent import MemAgent

# Create an agent
agent = MemAgent(
    memory_provider=memory_provider,
    instruction="Help users with financial and weather data"
)

# Method 1: Add a specific tool by ID
agent.add_tool(tool_id=weather_tool_id)

# Method 2: Add all tools from a toolbox
agent.add_tool(toolbox=toolbox)

# Control how tools are accessed
agent = MemAgent(
    tool_access="private",  # Only use explicitly added tools (default)
    # OR
    tool_access="global",   # Dynamically discover relevant tools from the toolbox
)

# Run the agent with tool access
response = agent.run("What's the weather like in New York?")
```

## Design Considerations

### Tool Persistence
- Tools are stored in the memory provider with both metadata and embeddings
- Function implementations are kept in memory during runtime
- Tool metadata persists across application restarts

### MemAgent Integration
- A copy of tool metadata is stored in the agent's `tools` attribute
- Function references are maintained in the agent's `_tool_functions` property
- Tools can be refreshed with `refresh_tools(tool_id)` when implementations change
- Missing implementations are handled gracefully during execution

### Access Control
- The `tool_access` attribute controls tool discovery during agent execution:
  - `"private"`: Only access tools explicitly added to the agent
  - `"global"`: Dynamically discover tools from the toolbox based on query relevance

### Performance Optimization
- Embeddings enable efficient semantic search for relevant tools
- Batch tool registration is supported for populating toolboxes efficiently
- Tool augmentation with `augment=True` enhances discoverability with LLM-generated metadata

## Implementation Notes

- Tool docstrings are used to generate rich metadata and improve discoverability
- Function signatures are analyzed to create parameter schemas for LLMs
- The system automatically handles type conversions between Python and JSON
- Error handling ensures failures in one tool don't crash the entire agent

This architecture provides a clean separation between tool definition (in the toolbox) and tool availability (in the agent), enabling fine-grained control over which capabilities are exposed to different agents in your system.
