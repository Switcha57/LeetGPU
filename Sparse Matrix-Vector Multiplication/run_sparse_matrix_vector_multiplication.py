from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


module_path = Path(__file__).with_name("sparse_matrix_vector_multiplication.py")
spec = spec_from_file_location("sparse_matrix_vector_multiplication", module_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {module_path}")

module = module_from_spec(spec)
spec.loader.exec_module(module)

example_M = 3
example_N = 4
example_A = torch.tensor(
    [
        5.0, 0.0, 0.0, 1.0,
        0.0, 2.0, 3.0, 0.0,
        0.0, 0.0, 0.0, 4.0,
    ],
    device="cuda",
)
example_x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
example_y = torch.empty(example_M, device="cuda")
module.solve(example_A, example_x, example_y, example_M, example_N, 5)
torch.cuda.synchronize()
print(f"Example Triton solve output: {example_y.cpu().tolist()}")

performance_M = 1000
performance_N = 10000
performance_rows = torch.arange(performance_M, device="cuda").view(-1, 1)
performance_columns = torch.arange(performance_N, device="cuda").view(1, -1)
performance_pattern = (performance_rows * 131 + performance_columns * 17) % 10
performance_values = (performance_rows * 7 + performance_columns * 3) % 5 + 1
performance_A = torch.where(
    performance_pattern < 6,
    torch.zeros_like(performance_values),
    performance_values,
).to(torch.float32)
performance_x = (torch.arange(performance_N, device="cuda", dtype=torch.float32) % 11) - 5
performance_y = torch.empty(performance_M, device="cuda")
module.solve(performance_A, performance_x, performance_y, performance_M, performance_N, 4000000)
torch.cuda.synchronize()
print(
    "Performance Triton solve: "
    f"M={performance_M}, N={performance_N}, sum(output)={performance_y.sum().item():.1f}"
)
