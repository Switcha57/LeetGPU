#include <cuda_runtime.h>

#include <iostream>

extern "C" void solve(const float* A, const float* B, float* C, int N);

void run_example(const float* host_a, const float* host_b, int N) {
    float host_output[N * N]{};

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    cudaMalloc(&device_a, N * N * sizeof(float));
    cudaMalloc(&device_b, N * N * sizeof(float));
    cudaMalloc(&device_c, N * N * sizeof(float));

    cudaMemcpy(device_a, host_a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, N * N * sizeof(float), cudaMemcpyHostToDevice);
    solve(device_a, device_b, device_c, N);
    cudaMemcpy(host_output, device_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int row = 0; row < N; ++row) {
        std::cout << '[';
        for (int col = 0; col < N; ++col) {
            std::cout << host_output[row * N + col];
            if (col + 1 < N) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
}

int main() {
    constexpr float example_a[] = {1.0f, 2.0f,
                                   3.0f, 4.0f};
    constexpr float example_b[] = {5.0f, 6.0f,
                                   7.0f, 8.0f};
    std::cout << "Example 1:\n";
    run_example(example_a, example_b, 2);

    constexpr float example_a_2[] = {1.5f, 2.5f, 3.5f,
                                     4.5f, 5.5f, 6.5f,
                                     7.5f, 8.5f, 9.5f};
    constexpr float example_b_2[] = {0.5f, 0.5f, 0.5f,
                                     0.5f, 0.5f, 0.5f,
                                     0.5f, 0.5f, 0.5f};
    std::cout << "Example 2:\n";
    run_example(example_a_2, example_b_2, 3);
}
