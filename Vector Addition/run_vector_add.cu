#include <cuda_runtime.h>

#include <iostream>

extern "C" void solve(const float* A, const float* B, float* C, int N);

int main() {
    constexpr int N = 4;
    float host_a[N] = {1.0f, 2.0f, 3.0f, 4.0f};
    float host_b[N] = {5.0f, 6.0f, 7.0f, 8.0f};
    float host_c[N];

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;

    cudaMalloc(&device_a, N * sizeof(float));
    cudaMalloc(&device_b, N * sizeof(float));
    cudaMalloc(&device_c, N * sizeof(float));

    cudaMemcpy(device_a, host_a, sizeof(host_a), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, sizeof(host_b), cudaMemcpyHostToDevice);

    solve(device_a, device_b, device_c, N);

    cudaMemcpy(host_c, device_c, sizeof(host_c), cudaMemcpyDeviceToHost);

    for (float value : host_c) {
        std::cout << value << ' ';
    }
    std::cout << '\n';

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
}
