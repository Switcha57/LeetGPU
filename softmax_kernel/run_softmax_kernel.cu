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

void run_performance_case() {
    constexpr int N = 500'000;
    std::vector<float> host_input(N);
    std::vector<float> host_output(N);
    for (int index = 0; index < N; ++index) {
        host_input[index] = static_cast<float>((index % 2001) - 1000) / 100.0f;
    }

    float* device_input = nullptr;
    float* device_output = nullptr;
    cudaMalloc(&device_input, N * sizeof(float));
    cudaMalloc(&device_output, N * sizeof(float));
    cudaMemcpy(device_input, host_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    solve(device_input, device_output, N);
    cudaMemcpy(host_output.data(), device_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    double output_sum = 0.0;
    for (float value : host_output) {
        output_sum += value;
    }
    std::cout << "Performance: N = " << N << ", output sum = " << output_sum << '\n';

    cudaFree(device_input);
    cudaFree(device_output);
}

int main() {
    std::cout << "Example 1:\n";
    run_example({1.0f, 2.0f, 3.0f});

    std::cout << "Example 2:\n";
    run_example({-10.0f, -5.0f, 0.0f, 5.0f, 10.0f});

    run_performance_case();
}
