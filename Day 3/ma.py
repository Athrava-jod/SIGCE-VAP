import json
import psutil
import subprocess
from groq import Groq


import os

# ==========================================
# 1. GROQ CONFIGURATION
# ==========================================

# Available models with tool calling on this key: 'qwen/qwen3.8-27b', 'openai/gpt-oss-120b', 'openai/gpt-oss-20b'
MODEL_NAME = "qwen/qwen3.8-27b"

api_key = os.environ.get("GROQ_API_KEY") or "groq api key"

client = Groq(api_key=api_key)


# ==========================================
# 2. SYSTEM TOOLS
# ==========================================

def get_system_metrics() -> str:
    """Returns local host system resource utilization (CPU, RAM, Disk)."""

    metrics = {
        "cpu_usage_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }

    return json.dumps(metrics)


SYSTEM_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_system_metrics",
            "description": "Returns local host system resource utilization including CPU, RAM and Disk.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


SYSTEM_TOOLS_MAP = {
    "get_system_metrics": get_system_metrics
}


# ==========================================
# 3. NETWORK TOOLS
# ==========================================

def ping_host(hostname: str) -> str:
    """
    Pings a network host to evaluate connectivity.
    """

    try:

        result = subprocess.run(
            ["ping", "-n", "4", hostname],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:

            return (
                f"Host {hostname} is reachable.\n"
                f"{result.stdout}"
            )

        else:

            return (
                f"Host {hostname} is not reachable.\n"
                f"{result.stdout}"
            )

    except Exception as exc:

        return f"Error while pinging {hostname}: {exc}"


NETWORK_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "ping_host",
            "description": "Pings a network host to evaluate connectivity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hostname": {
                        "type": "string",
                        "description": "Hostname or IP address to ping."
                    }
                },
                "required": ["hostname"]
            }
        }
    }
]


NETWORK_TOOLS_MAP = {
    "ping_host": ping_host
}


# ==========================================
# 4. AGENT RUNNER
# ==========================================

def run_agent(
    agent_name,
    system_prompt,
    user_query,
    tools_schema,
    tools_map
):

    print(f"\n[{agent_name}]")

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    # ==========================================
    # 1. Ask Groq to decide whether a tool is needed
    # ==========================================

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools_schema,
        tool_choice="auto"
    )

    assistant_message = response.choices[0].message

    # Convert Groq message into dictionary
    assistant_dict = {
        "role": "assistant",
        "content": assistant_message.content or ""
    }

    # Add tool calls if present
    if assistant_message.tool_calls:

        assistant_dict["tool_calls"] = []

        for call in assistant_message.tool_calls:

            assistant_dict["tool_calls"].append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments
                    }
                }
            )

    messages.append(assistant_dict)

    # ==========================================
    # Check tool calls
    # ==========================================

    tool_calls = assistant_message.tool_calls or []

    if not tool_calls:

        print(f"[{agent_name} Response]:")
        print(assistant_message.content)

        return

    # ==========================================
    # 2. Execute tools locally
    # ==========================================

    for call in tool_calls:

        fn_name = call.function.name

        # Groq returns arguments as JSON string
        fn_args = json.loads(call.function.arguments)

        print(
            f"  └─ Executing Tool: `{fn_name}` "
            f"with parameters: {fn_args}"
        )

        if fn_name in tools_map:

            try:

                result = tools_map[fn_name](**fn_args)

            except Exception as exc:

                result = f"Tool execution error: {exc}"

        else:

            result = f"Unknown tool: {fn_name}"

        # ==========================================
        # Feed tool result back to Groq
        # ==========================================

        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result)
            }
        )

    # ==========================================
    # 3. Final response synthesis by Groq
    # ==========================================

    final_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )

    print(f"\n[{agent_name} Final Answer]:")

    print(
        final_response.choices[0].message.content
    )


# ==========================================
# 5. ORCHESTRATOR ROUTER
# ==========================================

def orchestrate_query(user_query: str):

    print("\n==========================================")
    print(f'USER QUERY: "{user_query}"')
    print("==========================================")

    router_prompt = f"""
You are a query router.

Analyze the user prompt and respond with ONLY ONE word:

- 'SYSTEM' if the query asks about local CPU, disk, memory,
  or system hardware status.

- 'NETWORK' if the query asks about pinging, network latency,
  or internet connectivity.

- 'UNKNOWN' if it fits neither.

Query: {user_query}
"""

    # ==========================================
    # Router call to Groq
    # ==========================================

    route_res = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": router_prompt
            }
        ]
    )

    decision = (
        route_res
        .choices[0]
        .message
        .content
        .strip()
        .upper()
    )

    print(f"\n[Router Decision]: {decision}")

    # ==========================================
    # SYSTEM AGENT
    # ==========================================

    if "SYSTEM" in decision:

        run_agent(
            agent_name="System Health Agent",

            system_prompt=(
                "You are a system administration agent. "
                "Use tools to check CPU, RAM, or Disk metrics."
            ),

            user_query=user_query,

            tools_schema=SYSTEM_TOOLS_SCHEMA,

            tools_map=SYSTEM_TOOLS_MAP
        )

    # ==========================================
    # NETWORK AGENT
    # ==========================================

    elif "NETWORK" in decision:

        run_agent(
            agent_name="Network Agent",

            system_prompt=(
                "You are a network diagnostic agent. "
                "Use tools to check network status "
                "and ping target hosts."
            ),

            user_query=user_query,

            tools_schema=NETWORK_TOOLS_SCHEMA,

            tools_map=NETWORK_TOOLS_MAP
        )

    # ==========================================
    # UNKNOWN
    # ==========================================

    else:

        print(
            "\n[Router]: Request does not match "
            "active agent domains."
        )


# ==========================================
# 6. MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    print("==========================================")
    print("       GROQ MULTI-AGENT SYSTEM")
    print("==========================================")

    print("Type 'exit' to quit.")

    while True:

        user_query = input(
            "\nEnter your query: "
        ).strip()

        if user_query.lower() == "exit":
            print("\nExiting...")
            break

        if not user_query:
            continue

        orchestrate_query(user_query)
