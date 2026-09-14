from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


module_path = Path(__file__).with_name("swiglu.py")
spec = spec_from_file_location("swiglu", module_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {module_path}")

module = module_from_spec(spec)
spec.loader.exec_module(module)


def run_example(values):
    input = torch.tensor(values, device="cuda", dtype=torch.float32)
    output = torch.empty(input.numel() // 2, device="cuda", dtype=torch.float32)
    module.solve(input, output, input.numel())
    torch.cuda.synchronize()
    print(output.cpu().tolist())


print("Example 1:")
run_example([1.0, 2.0, 3.0, 4.0])

print("Example 2:")
run_example([0.5, 1.0])

N = 100_000
performance_input = (torch.arange(N, device="cuda", dtype=torch.float32) % 2001 - 1000) / 100
performance_output = torch.empty(N // 2, device="cuda", dtype=torch.float32)
module.solve(performance_input, performance_output, N)
torch.cuda.synchronize()
print(f"Performance: N = {N}, output_size = {N // 2}, checksum = {performance_output.sum().item()}")
