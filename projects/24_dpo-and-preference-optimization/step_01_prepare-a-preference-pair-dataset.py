"""
Project 24: Step 1 — Prepare a preference-pair dataset

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

from datasets import load_dataset

dataset = load_dataset("openbmb/UltraFeedback", split="train")
print(dataset[0])

def to_preference_pair(example):
    completions = example["completions"]
    scored = sorted(completions, key=lambda c: c["overall_score"], reverse=True)
    return {
        "prompt": example["instruction"],
        "chosen": scored[0]["response"],
        "rejected": scored[-1]["response"],
    }

paired = dataset.map(to_preference_pair, remove_columns=dataset.column_names)

def format_pair(example, tokenizer):
    chat = [{"role": "user", "content": example["prompt"]}]
    formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    return {
        "prompt": formatted_prompt,
        "chosen": formatted_prompt + example["chosen"],
        "rejected": formatted_prompt + example["rejected"],
    }
