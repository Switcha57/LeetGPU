#include <cuda_runtime.h>

#include <iostream>
#include <vector>

extern "C" void solve(const float* input, float* output, int N);

void run_example(const std::vector<float>& host_input) {
    const int N = static_cast<int>(host_input.size());
    const int output_size = N / 2;
    std::vector<float> host_output(output_size);

    float* device_input = nullptr;
    float* device_output = nullptr;
    cudaMalloc(&device_input, N * sizeof(float));
    cudaMalloc(&device_output, output_size * sizeof(float));

    cudaMemcpy(device_input, host_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    solve(device_input, device_output, N);
    cudaMemcpy(host_output.data(), device_output, output_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::cout << '[';
    for (int index = 0; index < output_size; ++index) {
        std::cout << host_output[index];
        if (index + 1 < output_size) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";

    cudaFree(device_input);
    cudaFree(device_output);
}

void run_performance_case() {
    constexpr int N = 100'000;
    constexpr int output_size = N / 2;
    std::vector<float> host_input(N);
    std::vector<float> host_output(output_size);
    for (int index = 0; index < N; ++index) {
        host_input[index] = static_cast<float>((index % 2001) - 1000) / 100.0f;
    }

    float* device_input = nullptr;
    float* device_output = nullptr;
    cudaMalloc(&device_input, N * sizeof(float));
    cudaMalloc(&device_output, output_size * sizeof(float));
    cudaMemcpy(device_input, host_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    solve(device_input, device_output, N);
    cudaMemcpy(host_output.data(), device_output, output_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    double checksum = 0.0;
    for (float value : host_output) {
        checksum += value;
    }
    std::cout << "Performance: N = " << N << ", output_size = " << output_size
              << ", checksum = " << checksum << '\n';

    cudaFree(device_input);
    cudaFree(device_output);
}

int main() {
    std::cout << "Example 1:\n";
    run_example({1.0f, 2.0f, 3.0f, 4.0f});

    std::cout << "Example 2:\n";
    run_example({0.5f, 1.0f});

    run_performance_case();
}
