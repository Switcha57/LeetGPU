import torch
import triton
import triton.language as tl


@triton.jit
def 1d_convolution_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, values, mask=mask)


def solve(input: torch.Tensor, output: torch.Tensor, N: int) -> None:
    block_size = 256
    grid = (triton.cdiv(N, block_size),)
    1d_convolution_kernel[grid](input, output, N, BLOCK_SIZE=block_size)
