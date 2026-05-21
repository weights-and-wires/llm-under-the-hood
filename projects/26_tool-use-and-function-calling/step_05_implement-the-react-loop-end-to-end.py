"""
Project 26: Step 5 — Implement the ReAct loop end-to-end

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def run_react(model, tokenizer, user_question, max_steps=6):
    messages = [
        {"role": "system", "content": build_system_prompt(TOOLS)},
        {"role": "user", "content": user_question},
    ]
    for step in range(max_steps):
        response = model.generate_chat(messages)
        messages.append({"role": "assistant", "content": response})
        action = parse_action(response)
        if action is None:
            return response, messages
        observation = dispatch(action)
        messages.append({
            "role": "tool",
            "content": f"Observation: {observation}",
        })
    return response, messages
