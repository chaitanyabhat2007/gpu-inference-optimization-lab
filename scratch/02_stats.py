import time
import numpy as np
import torch

x = torch.randn(2048, 2048, device="cuda")

# warmup, discarded
for _ in range(50):
    x @ x
torch.cuda.synchronize()

times = []
for _ in range(200):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    x @ x
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000)

a = np.array(times)
print(f"min:    {a.min():.3f} ms")
print(f"median: {np.median(a):.3f} ms")
print(f"mean:   {a.mean():.3f} ms")
print(f"p95:    {np.percentile(a, 95):.3f} ms")
print(f"max:    {a.max():.3f} ms")