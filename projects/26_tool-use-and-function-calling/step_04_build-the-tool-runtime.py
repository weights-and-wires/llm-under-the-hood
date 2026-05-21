"""
Project 26: Step 4 — Build the tool runtime

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

import json
import re

def get_weather(city: str) -> str:
    # In real code, hit an API. In the lab, mock it.
    table = {"Toronto": "12C, light rain", "Tokyo": "22C, clear"}
    return table.get(city, f"Weather data unavailable for {city}.")

def calculator(expression: str) -> str:
    try:
        # eval() is only safe here because the lab is mocked.
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"ERROR: {e}"

REGISTRY = {"get_weather": get_weather, "calculator": calculator}

ACTION_PATTERN = re.compile(r"Action:\s*(\{.*\})", re.DOTALL)

def parse_action(text: str):
    match = ACTION_PATTERN.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None

def dispatch(action: dict) -> str:
    name = action.get("name")
    args = action.get("arguments", {})
    fn = REGISTRY.get(name)
    if fn is None:
        return f"ERROR: unknown tool '{name}'"
    try:
        return fn(**args)
    except TypeError as e:
        return f"ERROR: bad arguments: {e}"
    except Exception as e:
        return f"ERROR: tool raised: {e}"
