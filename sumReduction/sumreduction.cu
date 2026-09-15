#include <cuda_runtime.h>
#include <cub/cub.cuh>

// come indicato su Cuda programming Guide 13.4(5.4.6) è suggerito usare direttamente cub
// input and output are device pointers.
extern "C" void solve(const float* input, float* output, int N) {
    void* temporary_storage = nullptr;
    size_t temporary_storage_bytes = 0;
    cub::DeviceReduce::Sum(temporary_storage, temporary_storage_bytes, input, output, N);
    cudaMalloc(&temporary_storage, temporary_storage_bytes);
    cub::DeviceReduce::Sum(temporary_storage, temporary_storage_bytes, input, output, N);
    cudaFree(temporary_storage);
    cudaDeviceSynchronize();
}
