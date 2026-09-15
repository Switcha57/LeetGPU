#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void softmax_kernel(const float* input, float* output, float* sum_value, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index>= N) return;
    output[index] = output[index] / *sum_value;



}

__global__ void exp_normalize_input(const float* input, float* output, int N,float* max_value) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index>= N) return;
    output[index] = expf(input[index] - *max_value);
}
// input and output are device pointers.
extern "C" void solve(const float* input, float* output, int N) {
    constexpr int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    void* temporary_storage = nullptr;
    size_t temporary_storage_bytes = 0;
    float* max_value = nullptr;
    cudaMalloc(&max_value, sizeof(float));
    cub::DeviceReduce::Max(temporary_storage, temporary_storage_bytes, input, max_value, N);
    cudaMalloc(&temporary_storage, temporary_storage_bytes);
    cub::DeviceReduce::Max(temporary_storage, temporary_storage_bytes, input, max_value, N);
    
    exp_normalize_input<<<blocks_per_grid, threads_per_block>>>(input, output, N, max_value);
    cudaFree(temporary_storage);
    cudaFree(max_value);
    cub::DeviceReduce::Sum(temporary_storage, temporary_storage_bytes, output, max_value, N);
    cudaMalloc(&temporary_storage, temporary_storage_bytes);
    cub::DeviceReduce::Sum(temporary_storage, temporary_storage_bytes, output, max_value, N);
    

    softmax_kernel<<<blocks_per_grid, threads_per_block>>>(input, output, max_value, N);
    cudaDeviceSynchronize();
}
