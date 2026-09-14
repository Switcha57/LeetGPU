from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


module_path = Path(__file__).with_name("silu.py")
spec = spec_from_file_location("silu", module_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {module_path}")

module = module_from_spec(spec)
spec.loader.exec_module(module)


def run_example(values):
    input = torch.tensor(values, device="cuda", dtype=torch.float32)
    output = torch.empty_like(input)
    module.solve(input, output, input.numel())
    torch.cuda.synchronize()
    print(output.cpu().tolist())


print("Example 1:")
run_example([0.5, 1.0, -0.5])

print("Example 2:")
run_example([-1.0, -2.0, -3.0, -4.0, -5.0])
