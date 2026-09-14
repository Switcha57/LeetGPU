from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


module_path = Path(__file__).with_name("1d_convolution.py")
spec = spec_from_file_location("1d_convolution", module_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {module_path}")

module = module_from_spec(spec)
spec.loader.exec_module(module)

input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
kernel = torch.tensor([1.0, 0.0, -1.0], device="cuda")
output = torch.empty(input.numel() - kernel.numel() + 1, device="cuda")
module.solve(input, kernel, output, input.numel(), kernel.numel())
torch.cuda.synchronize()
print(output.cpu().tolist())
