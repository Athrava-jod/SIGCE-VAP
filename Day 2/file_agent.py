from pathlib import Path

import ollama

MODEL = "qwen3:8b"

ROOT = Path(r"C:\Users\Atharva\cvadapav\I")

ROOT.mkdir(parents=True, exist_ok=True)


def safe_path(path: str) -> Path:
    """Resolve a path and prevent access outside the workspace."""
    import os
    norm = os.path.abspath(ROOT / path)
    if not (norm == str(ROOT) or norm.startswith(str(ROOT) + os.sep)):
        raise ValueError("Access outside the workspace is not allowed.")

    return Path(norm)


def create_file(path: str, content: str) -> str:
    """Create a file with the given content."""
    target = safe_path(path)

    if target.exists():
        return "Error: File already exists."

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    return f"Created: {target}"


def rename_file(old_path: str, new_path: str) -> str:
    """Rename a file."""
    source = safe_path(old_path)
    destination = safe_path(new_path)

    if not source.is_file():
        return "Error: Source file does not exist."

    if destination.exists():
        return "Error: Destination file already exists."

    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)

    return f"Renamed to: {destination}"


def delete_file(path: str) -> str:
    """Delete a file."""
    target = safe_path(path)

    if not target.is_file():
        return "Error: File does not exist."

    answer = input(f"Confirm deletion of {target}? (yes/no): ")

    if answer.strip().lower() != "yes":
        return "Deletion cancelled."

    target.unlink()

    return f"Deleted: {target}"


def list_files(path: str) -> str:
    """List files in a directory."""
    target = safe_path(path)

    if not target.is_dir():
        return "Error: Directory does not exist."

    return "\n".join(
        ("[DIR] " if p.is_dir() else "[FILE] ") + p.name
        for p in target.iterdir()
    ) or "Directory is empty."


TOOLS = [create_file, rename_file, delete_file, list_files]

SYSTEM_PROMPT="""

You are a local file-management assistant.

Use the available tools to create, rename, delete and list files.

All paths are relative to the provided workspace.

Never claim that an operation succeeded unless its tool confirms it.

Ask the user for missing file names or content.

Do not attempt to bypass tool restrictions"""

def run_agent():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Local Qwen3 File Agent")
    print("Workspace:", ROOT)
    print("Type 'exit' to quit.")

    while True:
        prompt = input("\nEnter file details: ").strip()

        if prompt.lower() == "exit":
            break

        if not prompt:
            continue

        messages.append({"role": "user", "content": prompt})

        while True:
            print("\n[Thinking... please wait]")
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
            )

            messages.append(response.message)
            if not response.message.tool_calls:
                print("\nAgent:", response.message.content)
                break

            for call in response.message.tool_calls:
                name = call.function.name
                args = call.function.arguments

                tool = next((t for t in TOOLS if t.__name__ == name), None)
                if tool is None:
                    result = "Error: Unknown tool."
                else:
                    try:
                        result = tool(**args)
                    except Exception as exc:
                        result = f"Error: {exc}"

                print(f"\n[Tool: {name}]")
                print(result)

                messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": str(result),
                })


if __name__ == "__main__":
    run_agent()