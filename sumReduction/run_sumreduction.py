from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


module_path = Path(__file__).with_name("sumreduction.py")
spec = spec_from_file_location("sumreduction", module_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {module_path}")

module = module_from_spec(spec)
spec.loader.exec_module(module)

def run_example(values):
    input = torch.tensor(values, device="cuda", dtype=torch.float32)
    output = torch.empty(1, device="cuda", dtype=torch.float32)
    module.solve(input, output, input.numel())
    torch.cuda.synchronize()
    print(output.item())


print("Example 1:")
run_example([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

print("Example 2:")
run_example([-2.5, 1.5, -1.0, 2.0])

N = 4_194_304
input = torch.ones(N, device="cuda", dtype=torch.float32)
output = torch.empty(1, device="cuda", dtype=torch.float32)
module.solve(input, output, N)
torch.cuda.synchronize()
print(f"Performance: N = {N}, sum = {output.item()}")
