import torch
import time

x = torch.randn(8192, 8192, device="cuda")

for _ in range(5):
    x @ x
torch.cuda.synchronize()

t0 = time.perf_counter()
x @ x
naive = (time.perf_counter() - t0)*1000
torch.cuda.synchronize()

t0 = time.perf_counter()
x @ x
torch.cuda.synchronize()
correct = (time.perf_counter() - t0)*1000

print(f"Naive: {naive:.2f} ms")
print(f"Correct: {correct:.2f} ms")