#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int M, int N, int d);

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

std::vector<float> run_case(const std::vector<float>& host_q,
                            const std::vector<float>& host_k,
                            const std::vector<float>& host_v,
                            int M, int N, int d) {
    const size_t output_elements = static_cast<size_t>(M) * std::max(N, d);
    std::vector<float> host_output(static_cast<size_t>(M) * d);

    float* device_input = nullptr;
    float* device_key = nullptr;
    float* device_value = nullptr;
    float* device_output = nullptr;
    check_cuda(cudaMalloc(&device_input, host_q.size() * sizeof(float)), "cudaMalloc(Q)");
    check_cuda(cudaMalloc(&device_key, host_k.size() * sizeof(float)), "cudaMalloc(K)");
    check_cuda(cudaMalloc(&device_value, host_v.size() * sizeof(float)), "cudaMalloc(V)");
    check_cuda(cudaMalloc(&device_output, output_elements * sizeof(float)),
               "cudaMalloc(output)");

    check_cuda(cudaMemcpy(device_input, host_q.data(), host_q.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(Q)");
    check_cuda(cudaMemcpy(device_key, host_k.data(), host_k.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(K)");
    check_cuda(cudaMemcpy(device_value, host_v.data(), host_v.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(V)");
    solve(device_input, device_key, device_value, device_output, M, N, d);
    check_cuda(cudaMemcpy(host_output.data(), device_output,
                          host_output.size() * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy(output)");

    check_cuda(cudaFree(device_input), "cudaFree(Q)");
    check_cuda(cudaFree(device_key), "cudaFree(K)");
    check_cuda(cudaFree(device_value), "cudaFree(V)");
    check_cuda(cudaFree(device_output), "cudaFree(output)");
    return host_output;
}

void print_output(const std::vector<float>& output, int M, int d) {
    std::cout << '[';
    for (int index = 0; index < M * d; ++index) {
        if (index != 0) {
            std::cout << ", ";
        }
        std::cout << output[index];
    }
    std::cout << "]\n";
}

void run_example(const std::vector<float>& q, const std::vector<float>& k,
                 const std::vector<float>& v, int M, int N, int d) {
    std::vector<float> output = run_case(q, k, v, M, N, d);
    print_output(output, M, d);
}

void run_performance_case() {
    constexpr int M = 512;
    constexpr int N = 256;
    constexpr int d = 128;
    std::vector<float> q(M * d);
    std::vector<float> k(N * d);
    std::vector<float> v(N * d);
    for (size_t index = 0; index < q.size(); ++index) {
        q[index] = static_cast<float>(static_cast<int>((index * 17) % 101) - 50) / 50.0f;
    }
    for (size_t index = 0; index < k.size(); ++index) {
        k[index] = static_cast<float>(static_cast<int>((index * 13) % 97) - 48) / 48.0f;
        v[index] = static_cast<float>(static_cast<int>((index * 19) % 89) - 44) / 44.0f;
    }

    const std::vector<float> output = run_case(q, k, v, M, N, d);
    double checksum = 0.0;
    for (float value : output) {
        checksum += value;
    }
    std::cout << "Performance: M = " << M << ", N = " << N << ", d = " << d
              << ", output sum = " << checksum << '\n';
}

int main() {
    std::cout << "Example 1 (Q 2x4, K 3x4, V 3x4):\n";
    run_example({1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f},
                {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f},
                {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                2, 3, 4);
    std::cout << "Example 2 (Q 1x2, K 2x2, V 2x2):\n";
    run_example({1.0f, 2.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {3.0f, 4.0f, 5.0f, 6.0f},
                1, 2, 2);
    run_performance_case();
}
