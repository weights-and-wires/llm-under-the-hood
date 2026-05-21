"""
Project 17: Step 7 — Simulate 100 concurrent users

Pedagogical reference: this file shows the code for this step in isolation.
For the full assembled, runnable build, use build.py in this same folder.
"""

def simulate(num_users, scheduler, duration_seconds):
    users = [User(scheduler) for _ in range(num_users)]
    start = time.time()
    total_tokens = 0
    latencies = []

    while time.time() - start < duration_seconds:
        # Each user submits a new request if it has none in flight
        for user in users:
            if user.in_flight is None:
                user.submit()

        # Run one scheduler step
        scheduler.step()

        # Collect finished requests
        for user in users:
            if user.in_flight and user.in_flight.finished:
                latencies.append(user.in_flight.latency())
                total_tokens += len(user.in_flight.output_tokens)
                user.in_flight = None

    return {
        "throughput_tokens_per_sec": total_tokens / duration_seconds,
        "p50_latency": np.percentile(latencies, 50),
        "p99_latency": np.percentile(latencies, 99),
    }
