from memorizz.memory_provider.mongodb.provider import MongoDBConfig, MongoDBProvider

# VOYAGE_API_KEY required in .env
import os 

# set environment variable VOYAGE_API_KEY
os.environ["VOYAGE_API_KEY"] = ""
os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["OPENAI_API_VERSION"] = "2025-04-01-preview" # Use your desired API version

# Create a mongodb config with voyageai embeddings
mongodb_config = MongoDBConfig(
    uri="mongodb://localhost:27017/?retryWrites=true&w=majority&directConnection=true",
    db_name="testing_memorizz",
    embedding_provider="voyageai",
    embedding_config={
        "embedding_type": "contextualized",
        "model": "voyage-context-3",
        "output_dimension": 512,
    }
)

# Create a memory provider
memory_provider = MongoDBProvider(mongodb_config)

from functools import lru_cache
from yahooquery import Ticker
import time

@lru_cache(maxsize=128)
def _fetch_price(symbol: str) -> float:
    """
    Internal helper to fetch the latest market price via yahooquery.
    Caching helps avoid repeated hits for the same symbol.
    """
    ticker = Ticker(symbol)
    # This returns a dict keyed by symbol:
    info = ticker.price or {}
    # regularMarketPrice holds the current trading price
    price = info.get(symbol.upper(), {}).get("regularMarketPrice")
    if price is None:
        raise ValueError(f"No price data for '{symbol}'")
    return price

def get_stock_price(
    symbol: str,
    currency: str = "USD",
    retry: int = 3,
    backoff: float = 0.5
) -> str:
    """
    Get the current stock price for a given symbol using yahooquery,
    with simple retry/backoff to handle occasional rate-limits.

    Parameters
    ----------
    symbol : str
        Stock ticker, e.g. "AAPL"
    currency : str, optional
        Currency code (Currently informational only; yahooquery returns native)
    retry : int, optional
        Number of retries on failure (default: 3)
    backoff : float, optional
        Backoff factor in seconds between retries (default: 0.5s)

    Returns
    -------
    str
        e.g. "The current price of AAPL is 172.34 USD."
    """
    symbol = symbol.upper()
    last_err = None
    for attempt in range(1, retry + 1):
        try:
            price = _fetch_price(symbol)
            return f"The current price of {symbol} is {price:.2f} {currency.upper()}."
        except Exception as e:
            last_err = e
            # simple backoff
            time.sleep(backoff * attempt)
    # if we get here, all retries failed
    raise RuntimeError(f"Failed to fetch price for '{symbol}' after {retry} attempts: {last_err}")

import requests

def get_weather(latitude, longitude):
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()
    return data['current']['temperature_2m']

from memorizz.long_term_memory.procedural.toolbox import Toolbox

from memorizz.embeddings import configure_embeddings, get_embedding
configure_embeddings(
    provider="voyageai",
    config={
        "embedding_type": "text",
        "model": "voyage-3.5",
        "output_dimension": 512,
    }
)

from memorizz.llms.azure import AzureOpenAI
az_openai = AzureOpenAI(
    deployment_name="gpt-4o",
)
toolbox = Toolbox(
    memory_provider=memory_provider,
    llm_provider=az_openai
)

# Now the tools are registered in the memory provider within the toolbox
toolbox.register_tool(get_weather)
toolbox.register_tool(get_stock_price)
print(toolbox.list_tools())

print("PERSONA DEMO TIME")
# PERSONA DEMO

import os
from memorizz.memory_provider.mongodb.provider import MongoDBConfig, MongoDBProvider
from memorizz.memagent import MemAgent
from memorizz.llms.azure import AzureOpenAI

# Create a MemAgent
agent = MemAgent(
    model=az_openai,
    instruction="You are a helpful assistant with persistent memory.",
    memory_provider=memory_provider
)

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
# Start conversing - the agent will remember across sessions
print(agent.run("Hello! My name is John and I'm a software engineer."))

# Later in another session...
response = agent.run("What did I tell you about myself?")
print(response)  # Agent remembers John is a software engineer

print('END OF DEMO')