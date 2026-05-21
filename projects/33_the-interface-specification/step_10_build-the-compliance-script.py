"""
Project 33: Step 10 — Build the compliance script

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def run_compliance(spec: dict, ckpt_path: str) -> tuple[list, list]:
    model, metadata = load_specialist(ckpt_path)
    failures = []
    warnings = []

    # Declaration checks
    if metadata["interface_name"] != spec["interface_name"]:
        failures.append((
            "interface_name",
            spec["interface_name"],
            metadata["interface_name"],
        ))

    # Structural checks
    actual_norm = detect_boundary_norm(model)
    expected_norm = spec["normalization"]["type"]
    if actual_norm != expected_norm:
        failures.append(("normalization.type", expected_norm, actual_norm))

    actual_d_model = detect_d_model(model)
    if actual_d_model != spec["boundary"]["d_model"]:
        failures.append((
            "boundary.d_model",
            spec["boundary"]["d_model"],
            actual_d_model,
        ))

    # Dynamic checks
    dyn_msg = run_probe(model, spec)
    if dyn_msg is not None:
        failures.append(("dynamic_probe", "pass", dyn_msg))

    return failures, warnings
