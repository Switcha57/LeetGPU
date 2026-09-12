import importlib.util

import torch


spec = importlib.util.spec_from_file_location(
    "matrix_mult", "Matrix Multiplication/matrix_mult.py"
)
if spec is None or spec.loader is None:
    raise ImportError("Could not load matrix_mult.py")

matrix_mult = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix_mult)


def run_example(a_values, b_values, M, N, K):
    a = torch.tensor(a_values, device="cuda", dtype=torch.float32).reshape(M, N)
    b = torch.tensor(b_values, device="cuda", dtype=torch.float32).reshape(N, K)
    c = torch.empty((M, K), device="cuda", dtype=torch.float32)

    matrix_mult.solve(a, b, c, M, N, K)
    torch.cuda.synchronize()
    print(c.cpu())


run_example([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], 2, 2, 2)
run_example([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 1, 3, 1)

M = N = K = 512
a = torch.arange(1, M * N + 1, device="cuda", dtype=torch.float32).reshape(M, N) % 100 / 100
b = torch.arange(1, N * K + 1, device="cuda", dtype=torch.float32).reshape(N, K) % 100 / 100
c = torch.empty((M, K), device="cuda", dtype=torch.float32)
matrix_mult.solve(a, b, c, M, N, K)
torch.cuda.synchronize()
print(f"{M}x{N} * {N}x{K} checksum: {c.sum().item():.6e}")
