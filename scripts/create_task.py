#!/usr/bin/env python3
"""Create a CUDA/Triton task scaffold for this repository."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import stat
from pathlib import Path


CUDA_SOLUTION_TEMPLATE = """#include <cuda_runtime.h>

__global__ void {stem}_kernel(const float* input, float* output, int N) {{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {{
        output[index] = input[index];
    }}
}}

// input and output are device pointers.
extern \"C\" void solve(const float* input, float* output, int N) {{
    constexpr int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    {stem}_kernel<<<blocks_per_grid, threads_per_block>>>(input, output, N);
    cudaDeviceSynchronize();
}}
"""


CUDA_RUNNER_TEMPLATE = """#include <cuda_runtime.h>

#include <iostream>

extern \"C\" void solve(const float* input, float* output, int N);

int main() {{
    constexpr float host_input[] = {{1.0f, 2.0f, 3.0f, 4.0f}};
    constexpr int N = sizeof(host_input) / sizeof(host_input[0]);
    float host_output[N]{{}};

    float* device_input = nullptr;
    float* device_output = nullptr;
    cudaMalloc(&device_input, N * sizeof(float));
    cudaMalloc(&device_output, N * sizeof(float));

    cudaMemcpy(device_input, host_input, N * sizeof(float), cudaMemcpyHostToDevice);
    solve(device_input, device_output, N);
    cudaMemcpy(host_output, device_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (float value : host_output) {{
        std::cout << value << ' ';
    }}
    std::cout << '\\n';

    cudaFree(device_input);
    cudaFree(device_output);
}}
"""


TRITON_SOLUTION_TEMPLATE = """import torch
import triton
import triton.language as tl


@triton.jit
def {stem}_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, values, mask=mask)


def solve(input: torch.Tensor, output: torch.Tensor, N: int) -> None:
    block_size = 256
    grid = (triton.cdiv(N, block_size),)
    {stem}_kernel[grid](input, output, N, BLOCK_SIZE=block_size)
"""


TRITON_RUNNER_TEMPLATE = """from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


module_path = Path(__file__).with_name(\"{stem}.py\")
spec = spec_from_file_location(\"{stem}\", module_path)
if spec is None or spec.loader is None:
    raise ImportError(f\"Could not load {{module_path}}\")

module = module_from_spec(spec)
spec.loader.exec_module(module)

input = torch.tensor([1.0, 2.0, 3.0, 4.0], device=\"cuda\")
output = torch.empty_like(input)
module.solve(input, output, input.numel())
torch.cuda.synchronize()
print(output.cpu().tolist())
"""


RUN_SCRIPT_TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)
CUDA_HOME="${{CUDA_HOME:-{cuda_home}}}"
CUDA_BINARY="$SCRIPT_DIR/{stem}_cuda"

"$CUDA_HOME/bin/nvcc" -std=c++17 \\
    "$SCRIPT_DIR/{stem}.cu" "$SCRIPT_DIR/run_{stem}.cu" \\
    -o "$CUDA_BINARY"

echo "CUDA:"
"$CUDA_BINARY"
echo "Triton:"
python "$SCRIPT_DIR/run_{stem}.py"
"""


PROFILE_SCRIPT_TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)
CUDA_HOME="${{CUDA_HOME:-{cuda_home}}}"
CUDA_BINARY="$SCRIPT_DIR/{stem}_cuda"
PROFILE_DIR="$SCRIPT_DIR/profiles"
mkdir -p "$PROFILE_DIR"

"$CUDA_HOME/bin/nvcc" -std=c++17 \\
    "$SCRIPT_DIR/{stem}.cu" "$SCRIPT_DIR/run_{stem}.cu" \\
    -o "$CUDA_BINARY"

if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys is required for CUDA profiling" >&2
    exit 1
fi

nsys profile --force-overwrite=true \\
    -o "$PROFILE_DIR/{stem}_cuda" "$CUDA_BINARY"
nsys profile --force-overwrite=true \\
    -o "$PROFILE_DIR/{stem}_triton" \\
    python "$SCRIPT_DIR/profile_{stem}.py"
"""


TRITON_PROFILE_TEMPLATE = """from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


module_path = Path(__file__).with_name("{stem}.py")
spec = spec_from_file_location("{stem}", module_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {{module_path}}")

module = module_from_spec(spec)
spec.loader.exec_module(module)

input = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
output = torch.empty_like(input)

with torch.profiler.profile() as profiler:
    for _ in range(10):
        module.solve(input, output, input.numel())
    torch.cuda.synchronize()

print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
"""


def to_stem(task_name: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", task_name).strip("_").lower()
    if not stem:
        raise ValueError("Task name must contain at least one letter or number")
    return stem


def compile_entry(root: Path, source: Path, object_name: str, cuda_home: Path) -> dict[str, str]:
    absolute_source = source.resolve()
    command = [
        str(cuda_home / "bin" / "nvcc"),
        "-std=c++17",
        "-x",
        "cu",
        "-ccbin",
        "/usr/bin/g++",
        "-c",
        str(absolute_source),
        "-o",
        f"/tmp/{object_name}.o",
    ]
    return {
        "directory": str(root.resolve()),
        "file": str(absolute_source),
        "command": shlex.join(command),
    }


def update_compile_commands(root: Path, sources: list[tuple[Path, str]], cuda_home: Path) -> None:
    compile_commands_path = root / "compile_commands.json"
    if compile_commands_path.exists():
        with compile_commands_path.open(encoding="utf-8") as file:
            entries = json.load(file)
    else:
        entries = []

    existing_files = {entry["file"] for entry in entries}
    for source, object_name in sources:
        entry = compile_entry(root, source, object_name, cuda_home)
        if entry["file"] not in existing_files:
            entries.append(entry)

    with compile_commands_path.open("w", encoding="utf-8") as file:
        json.dump(entries, file, indent=2)
        file.write("\n")


def create_task(task_name: str, root: Path, cuda_home: Path, with_profiling: bool) -> Path:
    stem = to_stem(task_name)
    task_directory = root / task_name
    if task_directory.exists():
        raise FileExistsError(f"Task directory already exists: {task_directory}")

    task_directory.mkdir(parents=True)
    files = {
        f"{stem}.cu": CUDA_SOLUTION_TEMPLATE.format(stem=stem),
        f"run_{stem}.cu": CUDA_RUNNER_TEMPLATE.format(stem=stem),
        f"{stem}.py": TRITON_SOLUTION_TEMPLATE.format(stem=stem),
        f"run_{stem}.py": TRITON_RUNNER_TEMPLATE.format(stem=stem),
        f"run_{stem}.sh": RUN_SCRIPT_TEMPLATE.format(
            stem=stem,
            cuda_home=str(cuda_home),
        ),
    }
    if with_profiling:
        files[f"profile_{stem}.sh"] = PROFILE_SCRIPT_TEMPLATE.format(
            stem=stem,
            cuda_home=str(cuda_home),
        )
        files[f"profile_{stem}.py"] = TRITON_PROFILE_TEMPLATE.format(stem=stem)

    for filename, contents in files.items():
        path = task_directory / filename
        path.write_text(contents, encoding="utf-8")
        if path.suffix == ".sh":
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    update_compile_commands(
        root,
        [
            (task_directory / f"{stem}.cu", stem),
            (task_directory / f"run_{stem}.cu", f"run_{stem}"),
        ],
        cuda_home,
    )
    return task_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a CUDA/Triton LeetGPU task scaffold")
    parser.add_argument("task_name", help='Task directory name, for example "Matrix Addition"')
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--cuda-home", type=Path, default=Path("/opt/cuda"))
    parser.add_argument(
        "--with-profiling",
        action="store_true",
        help="also create Nsight Systems and Triton profiler runners",
    )
    args = parser.parse_args()

    task_directory = create_task(
        args.task_name,
        args.root.resolve(),
        args.cuda_home.resolve(),
        args.with_profiling,
    )
    print(f"Created {task_directory}")


if __name__ == "__main__":
    main()