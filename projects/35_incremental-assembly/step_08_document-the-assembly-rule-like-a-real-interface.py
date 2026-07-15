"""
Project 35: Step 8 — Document the assembly rule like a real interface

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

# specialists are fixed modules with the same boundary contract
specialists = [spec_A, spec_B, spec_C]
router = load_router()

baseline = evaluate_system(router, specialists, eval_sets)

spec_D = train_independent_specialist(domain="D")

# attach without retraining the router weights
specialists_plus = specialists + [spec_D]

after_add = evaluate_system(router, specialists_plus, eval_sets_plus_D)

order_results = {}
for order in [
    ["A", "B", "C"],
    ["C", "A", "B"],
    ["B", "C", "A"],
]:
    system = assemble_in_order(order)
    order_results[tuple(order)] = evaluate_system(router, system, eval_sets)

best_router_cfg = autoresearch_search(
    fixed_specialists=specialists_plus,
    search_space={
        "temperature": [0.5, 1.0, 1.5],
        "top_k": [1, 2],
        "entropy_penalty": [0.0, 0.01, 0.05],
    },
    objective=assembly_objective,
)
