#include <cuda_runtime.h>

#include <iostream>

extern "C" void solve(const float* input, float* output, int rows, int cols);

void run_example(const float* host_input, int rows, int cols) {
    const int element_count = rows * cols;
    float* host_output = new float[element_count];

    float* device_input = nullptr;
    float* device_output = nullptr;

    cudaMalloc(&device_input, element_count * sizeof(float));
    cudaMalloc(&device_output, element_count * sizeof(float));

    cudaMemcpy(device_input, host_input, element_count * sizeof(float), cudaMemcpyHostToDevice);

    solve(device_input, device_output, rows, cols);

    cudaMemcpy(host_output, device_output, element_count * sizeof(float), cudaMemcpyDeviceToHost);

    for (int row = 0; row < cols; ++row) {
        for (int col = 0; col < rows; ++col) {
            std::cout << host_output[row * rows + col] << ' ';
        }
        std::cout << '\n';
    }
    std::cout << '\n';

    cudaFree(device_input);
    cudaFree(device_output);
    delete[] host_output;
}

int main() {
    constexpr float example_input[] = {1.0f, 2.0f, 3.0f,
                                       4.0f, 5.0f, 6.0f};
    run_example(example_input, 2, 3);

    constexpr float example_input_2[] = {1.0f, 2.0f, 3.0f};
    run_example(example_input_2, 3, 1);
}
