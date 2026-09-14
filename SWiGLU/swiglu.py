import torch
import triton
import triton.language as tl


@triton.jit
def swiglu_kernel(input_ptr, output_ptr, half_n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < half_n
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(input_ptr + half_n + offsets, mask=mask, other=0.0)
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    tl.store(output_ptr + offsets, y * sigmoid, mask=mask)


def solve(input: torch.Tensor, output: torch.Tensor, N: int) -> None:
    block_size = 256
    half_n = N // 2
    grid = (triton.cdiv(half_n, block_size),)
    swiglu_kernel[grid](input, output, half_n, BLOCK_SIZE=block_size)
