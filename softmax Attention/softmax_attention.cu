#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>

__global__ void softmax_rows(float *data, int M, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M)
        return;

    float max_value = data[row * N];
    for (int column = 1; column < N; ++column) {
        max_value = fmaxf(max_value, data[row * N + column]);
    }

    float sum = 0.0f;
    for (int column = 0; column < N; ++column) {
        float value = expf(data[row * N + column] - max_value);
        data[row * N + column] = value;
        sum += value;
    }

    for (int column = 0; column < N; ++column) {
        data[row * N + column] /= sum;
    }
}

__global__ void softmax_kernel(const float *input, float *output, float *sum_value, int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N)
        return;
    output[index] = output[index] / *sum_value;
}

__global__ void exp_normalize_input(const float *input, float *output, int N, float *max_value)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N)
        return;
    output[index] = expf(input[index] - *max_value);
}

extern "C" void softy(const float *input, float *output, int N)
{
    constexpr int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    void *temporary_storage = nullptr;
    size_t temporary_storage_bytes = 0;
    float *max_value = nullptr;
    cudaMalloc(&max_value, sizeof(float));
    cub::DeviceReduce::Max(temporary_storage, temporary_storage_bytes, input, max_value, N);
    cudaMalloc(&temporary_storage, temporary_storage_bytes);
    cub::DeviceReduce::Max(temporary_storage, temporary_storage_bytes, input, max_value, N);

    exp_normalize_input<<<blocks_per_grid, threads_per_block>>>(input, output, N, max_value);
    cudaFree(temporary_storage);
    cudaFree(max_value);
    cub::DeviceReduce::Sum(temporary_storage, temporary_storage_bytes, output, max_value, N);
    cudaMalloc(&temporary_storage, temporary_storage_bytes);
    cub::DeviceReduce::Sum(temporary_storage, temporary_storage_bytes, output, max_value, N);
    softmax_kernel<<<blocks_per_grid, threads_per_block>>>(input, output, max_value, N);
    cudaDeviceSynchronize();
}

#define TILE 32

__global__ void transpose(const float *in, float *out, int W, int H)
{                                  // better transpose kernel
    __shared__ float tile[TILE][TILE + 1]; // +1 avoids bank conflicts

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < W && y < H)
        tile[threadIdx.y][threadIdx.x] = in[y * W + x];

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < H && y < W)
        out[y * H + x] = tile[threadIdx.x][threadIdx.y];
}
__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) { //better matmul kernel
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    // Loop over tiles of dimension K
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // Load into shared memory
        sA[threadIdx.y][threadIdx.x] = (row < M && (t * TILE + threadIdx.x) < K) ? A[row * K + t * TILE + threadIdx.x] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = ((t * TILE + threadIdx.y) < K && col < N) ? B[(t * TILE + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();
        // Accumulate partial dot product from shared memory
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void divide_by_sqrt_d(float *data, int count, float inv_sqrt_d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
        data[idx] *= inv_sqrt_d;
}

// Q, K, V, output are device pointers
extern "C" void solve(const float *Q, const float *K, const float *V, float *output, int M, int N,
                      int d)
{
    dim3 block(TILE, TILE);
    dim3 scores_grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    dim3 output_grid((d + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    float *K_transposed = nullptr;
    float *scores = nullptr;
    cudaMalloc(&K_transposed, N * d * sizeof(float));
    cudaMalloc(&scores, M * N * sizeof(float));
    transpose<<<dim3((d + TILE - 1) / TILE, (N + TILE - 1) / TILE), block>>>(K,
                                                                                K_transposed,
                                                                                d, N);

    matmul<<<scores_grid, block>>>(Q, K_transposed, scores, M, N, d);

    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    int elements = M * N;
    divide_by_sqrt_d<<<(elements + 255) / 256, 256>>>(scores, elements, inv_sqrt_d);

    softmax_rows<<<(M + 127) / 128, 128>>>(scores, M, N);
    matmul<<<output_grid, block>>>(scores, V, output, M, d, N);
    cudaFree(scores);
    cudaFree(K_transposed);
    cudaDeviceSynchronize();
}
