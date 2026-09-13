import torch
import triton
import triton.language as tl


@triton.jit
def matrix_transpose_kernel(input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    col = tl.program_id(1) * 16 + tl.arange(0, 16)

    mask = (row[:, None] < rows) & (col[None, :] < cols)

    tile = tl.load(
        input + row[:, None] * stride_ir + col[None, :] * stride_ic,
        mask=mask,
        other=0.0,
    )
    transposed = tl.trans(tile)

    tl.store(
        output + col[:, None] * stride_or + row[None, :] * stride_oc,
        transposed,
        mask=mask.trans(),
    )


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1
    stride_or, stride_oc = rows, 1

    grid = (triton.cdiv(rows, 16), triton.cdiv(cols, 16))
    matrix_transpose_kernel[grid](
        input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc
    )
