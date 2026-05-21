"""
Project 26: Step 1 — Define a tool schema

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

TOOLS = [
    {
        "name": "get_weather",
        "description": "Look up the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Toronto'."
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression and return the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A Python-style math expression, e.g. '8347 * 91256'."
                }
            },
            "required": ["expression"]
        }
    }
]
