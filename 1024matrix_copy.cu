#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int MATRIX_SIZE = 1024;
constexpr int TILE_DIM = 32;
constexpr int WARMUP_RUNS = 10;
constexpr int TIMED_RUNS = 200;

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
    }
}

template <int BLOCK_X, int BLOCK_Y>
__global__ void matrixAdd(const float* a,
                          const float* b,
                          float* out,
                          int width) {
    const int tileRow = blockIdx.y * TILE_DIM;
    const int tileCol = blockIdx.x * TILE_DIM;

    for (int localRow = threadIdx.y; localRow < TILE_DIM; localRow += BLOCK_Y) {
        for (int localCol = threadIdx.x; localCol < TILE_DIM;
             localCol += BLOCK_X) {
            const int row = tileRow + localRow;
            const int col = tileCol + localCol;
            const int index = row * width + col;
            out[index] = a[index] + b[index];
        }
    }
}

template <int BLOCK_X, int BLOCK_Y>
float benchmark(const float* a, const float* b, float* out, dim3 grid) {
    const dim3 block(BLOCK_X, BLOCK_Y);
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        matrixAdd<BLOCK_X, BLOCK_Y><<<grid, block>>>(a, b, out, MATRIX_SIZE);
    }
    checkCuda(cudaGetLastError(), "warmup kernel launch");
    checkCuda(cudaDeviceSynchronize(), "warmup synchronization");

    cudaEvent_t start;
    cudaEvent_t stop;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");

    checkCuda(cudaEventRecord(start), "record start event");
    for (int i = 0; i < TIMED_RUNS; ++i) {
        matrixAdd<BLOCK_X, BLOCK_Y><<<grid, block>>>(a, b, out, MATRIX_SIZE);
    }
    checkCuda(cudaGetLastError(), "timed kernel launch");
    checkCuda(cudaEventRecord(stop), "record stop event");
    checkCuda(cudaEventSynchronize(stop), "wait for timed kernels");

    float elapsedMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsedMs, start, stop),
              "measure elapsed time");
    checkCuda(cudaEventDestroy(start), "destroy start event");
    checkCuda(cudaEventDestroy(stop), "destroy stop event");
    return elapsedMs / TIMED_RUNS;
}

bool matches(const std::vector<float>& actual,
             const std::vector<float>& expected) {
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > 1e-6f) {
            std::cerr << "Mismatch at index " << i << ": expected "
                      << expected[i] << ", got " << actual[i] << '\n';
            return false;
        }
    }
    return true;
}
}  // namespace

int main() {
    const size_t elementCount = static_cast<size_t>(MATRIX_SIZE) * MATRIX_SIZE;
    const size_t bytes = elementCount * sizeof(float);
    std::vector<float> hostA(elementCount);
    std::vector<float> hostB(elementCount);
    std::vector<float> expected(elementCount);
    std::vector<float> result32x8(elementCount);
    std::vector<float> result8x32(elementCount);

    for (size_t i = 0; i < elementCount; ++i) {
        hostA[i] = static_cast<float>(i % 97) * 0.25f;
        hostB[i] = static_cast<float>(i % 53) * 0.5f;
        expected[i] = hostA[i] + hostB[i];
    }

    checkCuda(cudaSetDevice(0), "select CUDA device");
    cudaDeviceProp deviceProp{};
    checkCuda(cudaGetDeviceProperties(&deviceProp, 0),
              "get CUDA device properties");

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceOut = nullptr;
    checkCuda(cudaMalloc(&deviceA, bytes), "allocate matrix A");
    checkCuda(cudaMalloc(&deviceB, bytes), "allocate matrix B");
    checkCuda(cudaMalloc(&deviceOut, bytes), "allocate output matrix");
    checkCuda(cudaMemcpy(deviceA, hostA.data(), bytes, cudaMemcpyHostToDevice),
              "copy matrix A");
    checkCuda(cudaMemcpy(deviceB, hostB.data(), bytes, cudaMemcpyHostToDevice),
              "copy matrix B");

    const dim3 grid(MATRIX_SIZE / TILE_DIM, MATRIX_SIZE / TILE_DIM);
    const float time32x8 = benchmark<32, 8>(deviceA, deviceB, deviceOut, grid);
    checkCuda(
        cudaMemcpy(result32x8.data(), deviceOut, bytes, cudaMemcpyDeviceToHost),
        "copy 32x8 result");

    const float time8x32 = benchmark<8, 32>(deviceA, deviceB, deviceOut, grid);
    checkCuda(
        cudaMemcpy(result8x32.data(), deviceOut, bytes, cudaMemcpyDeviceToHost),
        "copy 8x32 result");

    std::cout << "GPU: " << deviceProp.name << '\n'
              << "Matrix: " << MATRIX_SIZE << " x " << MATRIX_SIZE << '\n'
              << "Tile: " << TILE_DIM << " x " << TILE_DIM << ", "
              << (TILE_DIM * TILE_DIM / (32 * 8)) << " elements per thread\n"
              << "32x8 block average kernel time: " << time32x8 << " ms\n"
              << "8x32 block average kernel time: " << time8x32 << " ms\n";

    if (time8x32 > 0.0f) {
        std::cout << "32x8 / 8x32 time ratio: " << time32x8 / time8x32 << "x\n";
    }

    std::cout << "32x8 result: "
              << (matches(result32x8, expected) ? "OK" : "FAILED") << '\n'
              << "8x32 result: "
              << (matches(result8x32, expected) ? "OK" : "FAILED") << '\n';

    checkCuda(cudaFree(deviceA), "free matrix A");
    checkCuda(cudaFree(deviceB), "free matrix B");
    checkCuda(cudaFree(deviceOut), "free output matrix");
    return 0;
}