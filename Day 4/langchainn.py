import os
import sys
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq

# Ensure UTF-8 output on Windows console (for emojis like ☀️)
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# Using qwen/qwen3.8-27b which is available on your Groq account
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=os.environ.get("GROQ_API_KEY"),
)

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)

print(result["messages"][-1].content)
