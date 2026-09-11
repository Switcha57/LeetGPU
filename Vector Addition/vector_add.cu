#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        C[index] = A[index] + B[index];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    constexpr int threads_per_block = 1024;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    vector_add<<<blocks_per_grid, threads_per_block>>>(
        A, B, C, N);
    cudaDeviceSynchronize();
}