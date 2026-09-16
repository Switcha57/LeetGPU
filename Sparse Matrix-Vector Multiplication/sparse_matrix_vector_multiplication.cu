#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void gemv_cub_block(const float *A, const float *x, float *y, int M, int N)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int row = blockIdx.x;
    if (row >= M)
        return;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < N; col += blockDim.x)
    {
        sum += A[row * N + col] * x[col];
    }
    sum = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0)
    {
        y[row] = sum;
    }
}

// A, x, y are device pointers
extern "C" void solve(const float *A, const float *x, float *y, int M, int N, int nnz)
{
    const int threads_per_block = 256;
    gemv_cub_block<<<M, threads_per_block>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}
