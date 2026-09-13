from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


module_path = Path(__file__).with_name("matrix_transpose.py")
spec = spec_from_file_location("matrix_transpose", module_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {module_path}")

module = module_from_spec(spec)
spec.loader.exec_module(module)


def run_example(values, rows, cols):
    input = torch.tensor(values, device="cuda", dtype=torch.float32)
    output = torch.empty_like(input)

    module.solve(input, output, rows, cols)
    torch.cuda.synchronize()

    print(output.reshape(cols, rows).cpu())


run_example([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
run_example([1.0, 2.0, 3.0], 3, 1)
