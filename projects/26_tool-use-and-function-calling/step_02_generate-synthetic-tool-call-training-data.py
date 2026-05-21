"""
Project 26: Step 2 — Generate synthetic tool-call training data

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def make_weather_example(city):
    user = f"What is the weather in {city}?"
    assistant = (
        f"Thought: The user is asking about current weather. I should use get_weather.\n"
        f"Action: {{\"name\": \"get_weather\", \"arguments\": {{\"city\": \"{city}\"}}}}"
    )
    return {"user": user, "assistant": assistant}

def make_calc_example(a, b):
    user = f"What is {a} times {b}?"
    assistant = (
        f"Thought: The user is asking for a multiplication. I should use the calculator.\n"
        f"Action: {{\"name\": \"calculator\", \"arguments\": {{\"expression\": \"{a} * {b}\"}}}}"
    )
    return {"user": user, "assistant": assistant}

def make_no_tool_example(question, answer):
    user = question
    assistant = f"Thought: I can answer this from my own knowledge.\n{answer}"
    return {"user": user, "assistant": assistant}
