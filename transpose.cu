#include <cuda_runtime.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int MATRIX_ROWS = 1024;
constexpr int MATRIX_COLS = 1024;
constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;
constexpr int WARMUP_RUNS = 10;
constexpr int TIMED_RUNS = 200;

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
    }
}

__global__ void transpose_naive(const float* input, float* output,
                                int rows, int cols) {
    const int col = blockIdx.x * TILE_DIM + threadIdx.x;
    for (int offset = 0; offset < TILE_DIM; offset += BLOCK_ROWS) {
        const int row = blockIdx.y * TILE_DIM + threadIdx.y + offset;
        if (row < rows && col < cols) {
            output[col * rows + row] = input[row * cols + col];
        }
    }
}

__global__ void transpose_sharedMem(const float* input, float* output,
                                   int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    const int col = blockIdx.x * TILE_DIM + threadIdx.x;
    for (int offset = 0; offset < TILE_DIM; offset += BLOCK_ROWS) {
        const int row = blockIdx.y * TILE_DIM + threadIdx.y + offset;
        tile[threadIdx.y + offset][threadIdx.x] =
            (row < rows && col < cols) ? input[row * cols + col] : 0.0f;
    }
    __syncthreads();

    const int outputCol = blockIdx.y * TILE_DIM + threadIdx.x;
    for (int offset = 0; offset < TILE_DIM; offset += BLOCK_ROWS) {
        const int outputRow =
            blockIdx.x * TILE_DIM + threadIdx.y + offset;
        if (outputRow < cols && outputCol < rows) {
            output[outputRow * rows + outputCol] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

__global__ void Transpose_Padding3(const float* input, float* output,
                                  int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    const int col = blockIdx.x * TILE_DIM + threadIdx.x;
    for (int offset = 0; offset < TILE_DIM; offset += BLOCK_ROWS) {
        const int row = blockIdx.y * TILE_DIM + threadIdx.y + offset;
        tile[threadIdx.y + offset][threadIdx.x] =
            (row < rows && col < cols) ? input[row * cols + col] : 0.0f;
    }
    __syncthreads();

    const int outputCol = blockIdx.y * TILE_DIM + threadIdx.x;
    for (int offset = 0; offset < TILE_DIM; offset += BLOCK_ROWS) {
        const int outputRow =
            blockIdx.x * TILE_DIM + threadIdx.y + offset;
        if (outputRow < cols && outputCol < rows) {
            output[outputRow * rows + outputCol] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

template <typename Launch>
float benchmark(Launch launch) {
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        launch();
    }
    checkCuda(cudaGetLastError(), "warmup kernel launch");
    checkCuda(cudaDeviceSynchronize(), "warmup synchronization");

    cudaEvent_t start;
    cudaEvent_t stop;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");
    checkCuda(cudaEventRecord(start), "record start event");
    for (int i = 0; i < TIMED_RUNS; ++i) {
        launch();
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

void printMetrics(const char* name, float timeMs, size_t bytesMoved) {
    const double bandwidthGBs =
        static_cast<double>(bytesMoved) / (timeMs * 1.0e6);
    std::cout << std::left << std::setw(22) << name << std::right
              << " time: " << std::setw(10) << timeMs << " ms, bandwidth: "
              << std::setw(10) << bandwidthGBs << " GB/s\n";
}
}  // namespace

int main() {
    try {
        const size_t elementCount =
            static_cast<size_t>(MATRIX_ROWS) * MATRIX_COLS;
        const size_t bytes = elementCount * sizeof(float);
        const size_t bytesMoved = 2 * bytes;
        std::vector<float> hostInput(elementCount);
        std::vector<float> expected(elementCount);
        std::vector<float> result(elementCount);

        for (int row = 0; row < MATRIX_ROWS; ++row) {
            for (int col = 0; col < MATRIX_COLS; ++col) {
                const size_t inputIndex =
                    static_cast<size_t>(row) * MATRIX_COLS + col;
                hostInput[inputIndex] =
                    static_cast<float>((row * MATRIX_COLS + col) % 997);
                expected[static_cast<size_t>(col) * MATRIX_ROWS + row] =
                    hostInput[inputIndex];
            }
        }

        checkCuda(cudaSetDevice(0), "select CUDA device");
        cudaDeviceProp deviceProp{};
        checkCuda(cudaGetDeviceProperties(&deviceProp, 0),
                  "get CUDA device properties");

        float* deviceInput = nullptr;
        float* deviceOutput = nullptr;
        checkCuda(cudaMalloc(&deviceInput, bytes), "allocate input matrix");
        checkCuda(cudaMalloc(&deviceOutput, bytes), "allocate output matrix");
        checkCuda(cudaMemcpy(deviceInput, hostInput.data(), bytes,
                             cudaMemcpyHostToDevice),
                  "copy input matrix");

        const dim3 grid((MATRIX_COLS + TILE_DIM - 1) / TILE_DIM,
                        (MATRIX_ROWS + TILE_DIM - 1) / TILE_DIM);
        const dim3 block(TILE_DIM, BLOCK_ROWS);
        float times[3];
        bool valid[3];

        std::cout << "GPU: " << deviceProp.name << '\n'
                  << "Matrix: " << MATRIX_ROWS << " x " << MATRIX_COLS
                  << " -> " << MATRIX_COLS << " x " << MATRIX_ROWS << '\n'
                  << "Threads per block: " << TILE_DIM * BLOCK_ROWS << " ("
                  << TILE_DIM << " x " << BLOCK_ROWS << "), tile: "
                  << TILE_DIM << " x " << TILE_DIM << '\n'
                  << "Warmup runs: " << WARMUP_RUNS
                  << ", timed runs: " << TIMED_RUNS << '\n'
                  << std::fixed << std::setprecision(3);

        times[0] = benchmark([&] {
            transpose_naive<<<grid, block>>>(deviceInput, deviceOutput,
                                              MATRIX_ROWS, MATRIX_COLS);
        });
        checkCuda(cudaMemcpy(result.data(), deviceOutput, bytes,
                             cudaMemcpyDeviceToHost),
                  "copy naive result");
        valid[0] = matches(result, expected);
        printMetrics("transpose_naive", times[0], bytesMoved);
        std::cout << "  Correctness: " << (valid[0] ? "OK" : "FAILED") << '\n';

        times[1] = benchmark([&] {
            transpose_sharedMem<<<grid, block>>>(deviceInput, deviceOutput,
                                                  MATRIX_ROWS, MATRIX_COLS);
        });
        checkCuda(cudaMemcpy(result.data(), deviceOutput, bytes,
                             cudaMemcpyDeviceToHost),
                  "copy shared-memory result");
        valid[1] = matches(result, expected);
        printMetrics("transpose_sharedMem", times[1], bytesMoved);
        std::cout << "  Correctness: " << (valid[1] ? "OK" : "FAILED") << '\n';

        times[2] = benchmark([&] {
            Transpose_Padding3<<<grid, block>>>(deviceInput, deviceOutput,
                                                 MATRIX_ROWS, MATRIX_COLS);
        });
        checkCuda(cudaMemcpy(result.data(), deviceOutput, bytes,
                             cudaMemcpyDeviceToHost),
                  "copy padded shared-memory result");
        valid[2] = matches(result, expected);
        printMetrics("Transpose_Padding3", times[2], bytesMoved);
        std::cout << "  Correctness: " << (valid[2] ? "OK" : "FAILED") << '\n';

        for (int i = 1; i < 3; ++i) {
            if (times[i] > 0.0f) {
                const char* name =
                    i == 1 ? "transpose_sharedMem" : "Transpose_Padding3";
                std::cout << name << " speedup vs naive: "
                          << times[0] / times[i] << "x\n";
            }
        }

        checkCuda(cudaFree(deviceInput), "free input matrix");
        checkCuda(cudaFree(deviceOutput), "free output matrix");
        return valid[0] && valid[1] && valid[2] ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
