"""
Project 25: Step 1 — Direct prompting and CoT prompting on GSM8K

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def run_one(prompt, max_new_tokens=256, temperature=0.0):
    out = model.generate(prompt, max_new_tokens=max_new_tokens,
                         temperature=temperature, do_sample=temperature > 0)
    return out

def extract_answer(text):
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return float(nums[-1]) if nums else None
