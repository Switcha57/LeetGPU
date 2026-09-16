#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz);

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void run_case(const char* name, const std::vector<float>& host_A,
              const std::vector<float>& host_x, int M, int N) {
    const int nnz = static_cast<int>(std::count_if(
        host_A.begin(), host_A.end(), [](float value) { return value != 0.0f; }));
    std::vector<float> host_y(static_cast<std::size_t>(M), 0.0f);

    float* device_A = nullptr;
    float* device_x = nullptr;
    float* device_y = nullptr;
    check_cuda(cudaMalloc(&device_A, host_A.size() * sizeof(float)), "cudaMalloc(A)");
    check_cuda(cudaMalloc(&device_x, host_x.size() * sizeof(float)), "cudaMalloc(x)");
    check_cuda(cudaMalloc(&device_y, host_y.size() * sizeof(float)), "cudaMalloc(y)");

    check_cuda(cudaMemcpy(device_A, host_A.data(), host_A.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "cudaMemcpy(A)");
    check_cuda(cudaMemcpy(device_x, host_x.data(), host_x.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "cudaMemcpy(x)");
    check_cuda(cudaMemset(device_y, 0, host_y.size() * sizeof(float)), "cudaMemset(y)");
    solve(device_A, device_x, device_y, M, N, nnz);
    check_cuda(cudaGetLastError(), "solve");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    check_cuda(cudaMemcpy(host_y.data(), device_y, host_y.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "cudaMemcpy(y)");

    std::cout << name << " (M=" << M << ", N=" << N << ", nnz=" << nnz << "):\n";
    if (M <= 10) {
        std::cout << "  y = [";
        for (int row = 0; row < M; ++row) {
            if (row != 0) {
                std::cout << ", ";
            }
            std::cout << host_y[row];
        }
        std::cout << "]\n";
    } else {
        const float output_sum = std::accumulate(host_y.begin(), host_y.end(), 0.0f);
        std::cout << "  sum(y) = " << output_sum << '\n';
    }

    check_cuda(cudaFree(device_A), "cudaFree(A)");
    check_cuda(cudaFree(device_x), "cudaFree(x)");
    check_cuda(cudaFree(device_y), "cudaFree(y)");
}

}  // namespace

int main() {
    constexpr int example_M = 3;
    constexpr int example_N = 4;
    const std::vector<float> example_A = {
        5.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 2.0f, 3.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 4.0f,
    };
    const std::vector<float> example_x = {1.0f, 2.0f, 3.0f, 4.0f};
    run_case("Example", example_A, example_x, example_M, example_N);

    constexpr int performance_M = 1000;
    constexpr int performance_N = 10000;
    std::vector<float> performance_A(static_cast<std::size_t>(performance_M) * performance_N);
    std::vector<float> performance_x(performance_N);
    for (int row = 0; row < performance_M; ++row) {
        for (int column = 0; column < performance_N; ++column) {
            const int pattern = (row * 131 + column * 17) % 10;
            performance_A[static_cast<std::size_t>(row) * performance_N + column] =
                pattern < 6 ? 0.0f : static_cast<float>((row * 7 + column * 3) % 5 + 1);
        }
    }
    for (int column = 0; column < performance_N; ++column) {
        performance_x[column] = static_cast<float>((column % 11) - 5);
    }
    run_case("Performance", performance_A, performance_x, performance_M, performance_N);
}
