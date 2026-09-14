#include <cuda_runtime.h>

#include <iostream>
#include <vector>

extern "C" void solve(const float* input, float* output, int N);

void run_example(const std::vector<float>& host_input) {
    const int N = static_cast<int>(host_input.size());
    std::vector<float> host_output(N);

    float* device_input = nullptr;
    float* device_output = nullptr;
    cudaMalloc(&device_input, N * sizeof(float));
    cudaMalloc(&device_output, N * sizeof(float));

    cudaMemcpy(device_input, host_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    solve(device_input, device_output, N);
    cudaMemcpy(host_output.data(), device_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << '[';
    for (int index = 0; index < N; ++index) {
        std::cout << host_output[index];
        if (index + 1 < N) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";

    cudaFree(device_input);
    cudaFree(device_output);
}

int main() {
    std::cout << "Example 1:\n";
    run_example({0.5f, 1.0f, -0.5f});

    std::cout << "Example 2:\n";
    run_example({-1.0f, -2.0f, -3.0f, -4.0f, -5.0f});
}
