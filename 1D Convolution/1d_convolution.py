import torch
import triton
import triton.language as tl


@triton.jit
def convolution_1d_kernel(input_ptr, kernel_ptr, output_ptr, output_size, kernel_size,
                          BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    output_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    output_mask = output_offsets < output_size
    kernel_offsets = tl.arange(0, BLOCK_SIZE)
    kernel_mask = kernel_offsets < kernel_size
    values = tl.load(input_ptr + output_offsets[:, None] + kernel_offsets[None, :],
                     mask=output_mask[:, None] & kernel_mask[None, :], other=0.0)
    kernel_values = tl.load(kernel_ptr + kernel_offsets, mask=kernel_mask, other=0.0)
    result = tl.sum(values * kernel_values[None, :], axis=1)
    tl.store(output_ptr + output_offsets, result, mask=output_mask)


def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int,
          kernel_size: int) -> None:
    block_size = 256
    output_size = input_size - kernel_size + 1
    grid = (triton.cdiv(output_size, block_size),)
    convolution_1d_kernel[grid](input, kernel, output, output_size, kernel_size,
                                 BLOCK_SIZE=block_size)
