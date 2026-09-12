#include <cuda_runtime.h>

#include <iostream>

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K);

void run_example(const float* host_a, const float* host_b, int M, int N, int K) {
	float* host_c = new float[M * K];

	float* device_a = nullptr;
	float* device_b = nullptr;
	float* device_c = nullptr;

	cudaMalloc(&device_a, M * N * sizeof(float));
	cudaMalloc(&device_b, N * K * sizeof(float));
	cudaMalloc(&device_c, M * K * sizeof(float));

	cudaMemcpy(device_a, host_a, M * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b, N * K * sizeof(float), cudaMemcpyHostToDevice);

	solve(device_a, device_b, device_c, M, N, K);

	cudaMemcpy(host_c, device_c, M * K * sizeof(float), cudaMemcpyDeviceToHost);

	for (int row = 0; row < M; ++row) {
		for (int col = 0; col < K; ++col) {
			std::cout << host_c[row * K + col] << ' ';
		}
		std::cout << '\n';
	}
	std::cout << '\n';

	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);
	delete[] host_c;
}

int main() {
	constexpr float example_a[] = {1.0f, 2.0f, 3.0f, 4.0f};
	constexpr float example_b[] = {5.0f, 6.0f, 7.0f, 8.0f};
	run_example(example_a, example_b, 2, 2, 2);

	constexpr float example_a_2[] = {1.0f, 2.0f, 3.0f};
	constexpr float example_b_2[] = {4.0f, 5.0f, 6.0f};
	run_example(example_a_2, example_b_2, 1, 3, 1);
}
