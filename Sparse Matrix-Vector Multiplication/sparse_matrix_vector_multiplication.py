import triton
import triton.language as tl
import torch


@triton.jit
def sparse_matrix_vector_multiplication_kernel(
    matrix_ptr, vector_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < N
    matrix_values = tl.load(matrix_ptr + row * N + columns, mask=mask, other=0.0)
    vector_values = tl.load(vector_ptr + columns, mask=mask, other=0.0)
    row_sum = tl.sum(matrix_values * vector_values, axis=0)
    tl.store(output_ptr + row, row_sum)


def solve(
    A: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    M: int,
    N: int,
    nnz: int,
) -> None:
    del nnz
    block_size = triton.next_power_of_2(N)
    grid = (M,)
    sparse_matrix_vector_multiplication_kernel[grid](
        A, x, y, N, BLOCK_SIZE=block_size
    )
