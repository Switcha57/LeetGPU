import torch
import triton
import triton.language as tl


@triton.jit
def geglu_kernel(input_ptr, output_ptr, half_n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < half_n
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(input_ptr + half_n + offsets, mask=mask, other=0.0)
    gelu = 0.5 * y * (1.0 + tl.math.erf(y * 0.70710678118))
    tl.store(output_ptr + offsets, x * gelu, mask=mask)


def solve(input: torch.Tensor, output: torch.Tensor, N: int) -> None:
    block_size = 256
    half_n = N // 2
    grid = (triton.cdiv(half_n, block_size),)
    geglu_kernel[grid](input, output, half_n, BLOCK_SIZE=block_size)
