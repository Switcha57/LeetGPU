#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int worker_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_id >= halfN) return;

    float xi = input[worker_id];
    float yi = input[worker_id + halfN];
    auto gelu = [](float x) {
            return 0.5f * x * (1.0f + erff(x * 0.70710678118f)); // 1/sqrt(2)
        };
    output[worker_id] = xi * gelu(yi);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
