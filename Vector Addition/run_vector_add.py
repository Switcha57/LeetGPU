import importlib.util

import torch


spec = importlib.util.spec_from_file_location(
    "vector_add", "Vector Addition/vector_add.py"
)
if spec is None or spec.loader is None:
    raise ImportError("Could not load vector_add.py")

vector_add = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vector_add)

a = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
b = torch.tensor([5.0, 6.0, 7.0, 8.0], device="cuda")
c = torch.empty_like(a)

vector_add.solve(a, b, c, c.numel())
torch.cuda.synchronize()

print(c.cpu().tolist())
