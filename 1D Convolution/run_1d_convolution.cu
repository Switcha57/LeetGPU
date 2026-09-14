#include <cuda_runtime.h>

#include <iostream>
#include <vector>

extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size);

void run_example(const std::vector<float>& host_input, const std::vector<float>& host_kernel) {
    const int input_size = static_cast<int>(host_input.size());
    const int kernel_size = static_cast<int>(host_kernel.size());
    const int output_size = input_size - kernel_size + 1;
    std::vector<float> host_output(output_size);

    float* device_input = nullptr;
    float* device_kernel = nullptr;
    float* device_output = nullptr;
    cudaMalloc(&device_input, input_size * sizeof(float));
    cudaMalloc(&device_kernel, kernel_size * sizeof(float));
    cudaMalloc(&device_output, output_size * sizeof(float));

    cudaMemcpy(device_input, host_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_kernel, host_kernel.data(), kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    solve(device_input, device_kernel, device_output, input_size, kernel_size);
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
    cudaFree(device_kernel);
    cudaFree(device_output);
}

void run_performance_case() {
    constexpr int input_size = 1'500'000;
    constexpr int kernel_size = 2'047;
    std::vector<float> input(input_size);
    std::vector<float> kernel(kernel_size);
    for (int index = 0; index < input_size; ++index) {
        input[index] = static_cast<float>(index % 11);
    }
    for (int index = 0; index < kernel_size; ++index) {
        kernel[index] = (index % 2 == 0) ? 0.5f : -0.5f;
    }

    const int output_size = input_size - kernel_size + 1;
    std::vector<float> output(output_size);
    float* device_input = nullptr;
    float* device_kernel = nullptr;
    float* device_output = nullptr;
    cudaMalloc(&device_input, input.size() * sizeof(float));
    cudaMalloc(&device_kernel, kernel.size() * sizeof(float));
    cudaMalloc(&device_output, output.size() * sizeof(float));
    cudaMemcpy(device_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_kernel, kernel.data(), kernel.size() * sizeof(float), cudaMemcpyHostToDevice);
    solve(device_input, device_kernel, device_output, input_size, kernel_size);
    cudaMemcpy(output.data(), device_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Performance: input_size = " << input_size
              << ", kernel_size = " << kernel_size << ", output_size = " << output_size << '\n';

    cudaFree(device_input);
    cudaFree(device_kernel);
    cudaFree(device_output);
}

int main() {
    std::cout << "Example 1:\n";
    run_example({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {1.0f, 0.0f, -1.0f});

    std::cout << "Example 2:\n";
    run_example({2.0f, 4.0f, 6.0f, 8.0f}, {0.5f, 0.2f});

    run_performance_case();
}
