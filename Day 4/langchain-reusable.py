from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

import sys

# Ensure UTF-8 output on Windows console
sys.stdout.reconfigure(encoding="utf-8")

# Using qwen/qwen3.8-27b with temperature=0 and your Groq API key
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY",),
)

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 5 simple points."
)

chain = prompt | llm | StrOutputParser()

def explain_topic(topic):
    return chain.invoke({"topic": topic})

print(explain_topic("Docker"))
print(explain_topic("Kubernetes"))
print(explain_topic("Terraform"))
