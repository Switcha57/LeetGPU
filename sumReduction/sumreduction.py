import torch
import triton
import triton.language as tl


@triton.jit
def sumreduction_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    partial_sum = tl.sum(values, axis=0)
    tl.atomic_add(output_ptr, partial_sum)


def solve(input: torch.Tensor, output: torch.Tensor, N: int) -> None:
    output.zero_()
    block_size = 1024
    grid = (triton.cdiv(N, block_size),)
    sumreduction_kernel[grid](input, output, N, BLOCK_SIZE=block_size)
