#include <cuda_runtime.h>

__global__ void silu_kernel(const float* input, float* output, int N) {
    int worker_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_id >= N) return;
     auto sigmoid = [](float x) {
            return 1.0f / (1.0f + expf(-x));
        };
    output[worker_id] = input[worker_id] * sigmoid(input[worker_id]);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
