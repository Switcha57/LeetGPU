from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


module_path = Path(__file__).with_name("matrix_addition.py")
spec = spec_from_file_location("matrix_addition", module_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {module_path}")

module = module_from_spec(spec)
spec.loader.exec_module(module)

input_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
input_b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda")
output = torch.empty_like(input_a)
module.solve(input_a, input_b, output, input_a.shape[0])
torch.cuda.synchronize()
print(output.cpu().tolist())
