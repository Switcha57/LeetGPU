import torch
import triton
import triton.language as tl


@triton.jit
def matrix_addition_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a_values = tl.load(a_ptr + offsets, mask=mask)
    b_values = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a_values + b_values, mask=mask)


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int) -> None:
    block_size = 256
    grid = (triton.cdiv(N * N, block_size),)
    matrix_addition_kernel[grid](a, b, c, N * N, BLOCK_SIZE=block_size)
