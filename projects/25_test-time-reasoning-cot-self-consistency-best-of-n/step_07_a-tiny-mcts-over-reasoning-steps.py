"""
Project 25: Step 7 — A tiny MCTS over reasoning steps

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def mcts(question, n_iter=32, k_expand=4):
    root = Node(prompt=cot_prompt(question))
    for _ in range(n_iter):
        node = select(root)
        children = expand(node, k=k_expand)
        for child in children:
            rollout_value = rollout_and_grade(child)
            backup(child, rollout_value)
    return best_leaf_answer(root)
