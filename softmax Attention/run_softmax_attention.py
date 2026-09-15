from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


module_path = Path(__file__).with_name("softmax_attention.py")
spec = spec_from_file_location("softmax_attention", module_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {module_path}")

module = module_from_spec(spec)
spec.loader.exec_module(module)

def run_example(values):
    input_tensor = torch.tensor(values, device="cuda", dtype=torch.float32)
    output = torch.empty_like(input_tensor)
    module.solve(input_tensor, output, input_tensor.numel())
    torch.cuda.synchronize()
    print(output.cpu().tolist())


print("Example 1 (Triton solve(input, output, N), flattened Q):")
run_example([1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 1.0, -1.0])

print("Example 2 (Triton solve(input, output, N), flattened Q):")
run_example([1.0, 0.0])

M = 512
N = 256
d = 128
performance_input = (torch.arange(M * d, device="cuda", dtype=torch.float32) * 17 % 101 - 50) / 50
performance_output = torch.empty_like(performance_input)
module.solve(performance_input, performance_output, performance_input.numel())
torch.cuda.synchronize()
print(f"Performance: M = {M}, N = {N}, d = {d}, output sum = {performance_output.sum().item()}")
