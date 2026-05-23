from __future__ import annotations

from time import perf_counter

import numpy as np

from python_template_cpp import median


def run_benchmark() -> None:
    rng = np.random.default_rng(12345)
    array = rng.integers(1, 65536, size=(32, 32), dtype=np.uint16)

    median(array)

    timings = []
    result = None
    for _ in range(10):
        array = rng.integers(1, 65536, size=(32, 32), dtype=np.uint16)
        start = perf_counter()
        result = median(array)
        timings.append(perf_counter() - start)

    average = sum(timings) / len(timings)
    best = min(timings)
    print(f"median={result}")
    print(f"runs={len(timings)} average_ms={average * 1000:.6f} best_ms={best * 1000:.6f} worst_ms={max(timings) * 1000:.6f}")


if __name__ == "__main__":
    run_benchmark()