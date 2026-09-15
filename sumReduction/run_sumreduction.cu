#include <cuda_runtime.h>

#include <iostream>
#include <vector>

extern "C" void solve(const float* input, float* output, int N);

void run_example(const std::vector<float>& host_input) {
    const int N = static_cast<int>(host_input.size());
    float host_output = 0.0f;

    float* device_input = nullptr;
    float* device_output = nullptr;
    cudaMalloc(&device_input, N * sizeof(float));
    cudaMalloc(&device_output, sizeof(float));

    cudaMemcpy(device_input, host_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    solve(device_input, device_output, N);
    cudaMemcpy(&host_output, device_output, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << host_output << '\n';

    cudaFree(device_input);
    cudaFree(device_output);
}

void run_performance_case() {
    constexpr int N = 4'194'304;
    std::vector<float> host_input(N, 1.0f);
    float host_output = 0.0f;

    float* device_input = nullptr;
    float* device_output = nullptr;
    cudaMalloc(&device_input, N * sizeof(float));
    cudaMalloc(&device_output, sizeof(float));
    cudaMemcpy(device_input, host_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    solve(device_input, device_output, N);
    cudaMemcpy(&host_output, device_output, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Performance: N = " << N << ", sum = " << host_output << '\n';

    cudaFree(device_input);
    cudaFree(device_output);
}

int main() {
    std::cout << "Example 1:\n";
    run_example({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});

    std::cout << "Example 2:\n";
    run_example({-2.5f, 1.5f, -1.0f, 2.0f});

    run_performance_case();
}
