import time
import numpy as np
import torch


def time_fn(fn, warmup=50, iters=200):
    """Time a zero-argument callable. Returns per-iteration latency in ms."""
    for _ in range(warmup):          # discarded: context setup, clock ramp-up
        fn()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        torch.cuda.synchronize()     # nothing left over from the last iteration
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()     # wait until the GPU is actually done
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return times_ms


def summarize(times_ms, batch_size):
    a = np.asarray(times_ms)
    p50 = float(np.median(a))
    return {
        "p50_ms": p50,
        "p95_ms": float(np.percentile(a, 95)),
        "mean_ms": float(a.mean()),
        "std_ms": float(a.std()),
        "throughput_img_s": batch_size / (p50 / 1000.0),  # images per second
    }