#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int worker_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_id >= halfN) return;

    float xi = input[worker_id];
    float yi = input[worker_id + halfN];
    auto sigmoid = [](float x) {
            return 1.0f / (1.0f + expf(-x));
        };
    output[worker_id] = yi *( xi*sigmoid(xi));
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
