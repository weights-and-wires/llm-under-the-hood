"""
Project 26: Step 8 — Error handling, when the tool itself fails

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def make_error_recovery_example():
    user = "What is the weather in Torronto?"  # typo
    turns = [
        ("assistant",
         "Thought: The user wants the weather. I'll try get_weather.\n"
         "Action: {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Torronto\"}}"),
        ("tool",
         "Observation: ERROR: Weather data unavailable for Torronto."),
        ("assistant",
         "Thought: 'Torronto' looks like a typo for Toronto. I'll retry.\n"
         "Action: {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Toronto\"}}"),
    ]
    return {"user": user, "turns": turns}
