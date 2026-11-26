#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
// 所有方案都用一个线程块处理 1024 个数，最后把总和写入 output[0]。
constexpr int ELEMENT_COUNT = 1024;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int ELEMENTS_PER_THREAD = ELEMENT_COUNT / THREADS_PER_BLOCK;
constexpr int WARMUP_RUNS = 10;
constexpr int TIMED_RUNS = 100;

// 将 CUDA 运行时错误转换成可读的异常信息，避免错误被忽略。
void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
    }
}

__global__ void reduceAtomicGlobal(const int* input, int* output) {
    const int tid = threadIdx.x;
    // 每个线程负责若干个元素，并直接对显存中的总和做原子加法。
    // 原子操作保证多个线程同时更新 output 时不会互相覆盖。
    for (int i = tid; i < ELEMENT_COUNT; i += blockDim.x) {
        atomicAdd(output, input[i]);
    }
}

__global__ void reduceAtomicShare(const int* input, int* output) {
    // 先在较快的共享内存中累加，最后只由一个线程更新显存结果。
    __shared__ int sharedSum;
    if (threadIdx.x == 0) {
        sharedSum = 0;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < ELEMENT_COUNT; i += blockDim.x) {
        atomicAdd(&sharedSum, input[i]);
    }
    // 等所有线程完成共享内存累加后，再读取最终值。
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(output, sharedSum);
    }
}

__global__ void reduceInterleaved(const int* input, int* output) {
    // 先由每个线程计算一部分，再把线程间的部分和放到共享内存归并。
    __shared__ int values[THREADS_PER_BLOCK];
    const int tid = threadIdx.x;
    int sum = 0;
    for (int i = tid; i < ELEMENT_COUNT; i += blockDim.x) {
        sum += input[i];
    }
    values[tid] = sum;
    // 确保所有线程都写完 values 后才开始归并。
    __syncthreads();

    // 交错归并：每轮 stride 翻倍，参与相加的位置也随之交错。
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        const int index = 2 * stride * tid;
        if (index < blockDim.x) {
            values[index] += values[index + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[0] = values[0];
    }
}

__global__ void reduceSequential(const int* input, int* output) {
    __shared__ int values[THREADS_PER_BLOCK];
    const int tid = threadIdx.x;
    int sum = 0;
    for (int i = tid; i < ELEMENT_COUNT; i += blockDim.x) {
        sum += input[i];
    }
    values[tid] = sum;
    __syncthreads();

    // 连续归并：每轮将参与计算的线程数减半。
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            values[tid] += values[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[0] = values[0];
    }
}

__global__ void reduceUnroll(const int* input, int* output) {
    __shared__ int values[THREADS_PER_BLOCK];
    const int tid = threadIdx.x;
    const int base = tid * ELEMENTS_PER_THREAD;
    int sum = 0;
    // 每个线程先把连续的 4 个数加起来，减少后续归并的数据量。
    // #pragma unroll 提示编译器展开这个固定次数的小循环。
#pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        sum += input[base + i];
    }
    values[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            values[tid] += values[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[0] = values[0];
    }
}

__device__ int warpReduceSum(int value) {
    // warp 内的线程通过 shuffle 互相传递数值，不需要共享内存。
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void reduceWarpReduceFunc(const int* input, int* output) {
    // 每个 warp 先独立求和；之后第一个 warp 再归并各 warp 的结果。
    __shared__ int values[THREADS_PER_BLOCK / 32];
    const int tid = threadIdx.x;
    int sum = 0;
    for (int i = tid; i < ELEMENT_COUNT; i += blockDim.x) {
        sum += input[i];
    }

    sum = warpReduceSum(sum);
    // 每个 warp 的 lane 0 保存该 warp 的部分和。
    if ((tid & 31) == 0) {
        values[tid >> 5] = sum;
    }
    __syncthreads();

    if (tid < 32) {
        const int warpCount = blockDim.x / 32;
        // 第一个 warp 的前 warpCount 个线程读取各 warp 的部分和。
        sum = tid < warpCount ? values[tid] : 0;
        sum = warpReduceSum(sum);
        if (tid == 0) {
            output[0] = sum;
        }
    }
}

__global__ void reduceMultipleload(const int* input, int* output) {
    // 每个线程一次读入 4 个相隔 THREADS_PER_BLOCK 的元素并先局部求和。
    __shared__ int values[THREADS_PER_BLOCK];
    const int tid = threadIdx.x;
    int sum = input[tid] + input[tid + THREADS_PER_BLOCK] +
              input[tid + 2 * THREADS_PER_BLOCK] +
              input[tid + 3 * THREADS_PER_BLOCK];
    values[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            values[tid] += values[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[0] = values[0];
    }
}

__global__ void reduceCompleteUnroll(const int* input, int* output) {
    __shared__ int values[THREADS_PER_BLOCK];
    const int tid = threadIdx.x;
    int sum = 0;
#pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        sum += input[tid + i * THREADS_PER_BLOCK];
    }
    values[tid] = sum;
    __syncthreads();

    // 固定大小的归并循环提示编译器完全展开，减少循环控制开销。
#pragma unroll
    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            values[tid] += values[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[0] = values[0];
    }
}

using ReductionKernel = void (*)(const int*, int*);

// 保存方案名称及对应的 CUDA kernel，方便用同一套流程逐个测试。
struct Reduction {
    const char* name;
    ReductionKernel kernel;
};

float benchmark(ReductionKernel kernel, const int* input, int* output) {
    // 预热 GPU，尽量避免首次启动等额外开销影响计时。
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        kernel<<<1, THREADS_PER_BLOCK>>>(input, output);
    }
    checkCuda(cudaGetLastError(), "warmup kernel launch");
    checkCuda(cudaDeviceSynchronize(), "warmup synchronization");

    cudaEvent_t start;
    cudaEvent_t stop;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");
    checkCuda(cudaEventRecord(start), "record start event");
    // CUDA kernel 是异步启动的；事件可以测量 GPU 上实际执行的时间。
    for (int i = 0; i < TIMED_RUNS; ++i) {
        kernel<<<1, THREADS_PER_BLOCK>>>(input, output);
    }
    checkCuda(cudaGetLastError(), "timed kernel launch");
    checkCuda(cudaEventRecord(stop), "record stop event");
    checkCuda(cudaEventSynchronize(stop), "wait for timed kernels");

    float elapsedMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsedMs, start, stop),
              "measure elapsed time");
    checkCuda(cudaEventDestroy(start), "destroy start event");
    checkCuda(cudaEventDestroy(stop), "destroy stop event");
    // 返回每次 kernel 的平均耗时，单位为毫秒。
    return elapsedMs / TIMED_RUNS;
}
}  // namespace

int main() {
    try {
        // 使用固定、可重复的数据，并在 CPU 上算出正确答案供 GPU 对照。
        std::vector<int> hostInput(ELEMENT_COUNT);
        int expected = 0;
        for (int i = 0; i < ELEMENT_COUNT; ++i) {
            hostInput[i] = i % 17 + 1;
            expected += hostInput[i];
        }

        checkCuda(cudaSetDevice(0), "select CUDA device");
        cudaDeviceProp deviceProp{};
        checkCuda(cudaGetDeviceProperties(&deviceProp, 0),
                  "get CUDA device properties");

        int* deviceInput = nullptr;
        int* deviceOutput = nullptr;
        // 分配显存，并把输入数组从 CPU 内存复制到 GPU。
        checkCuda(cudaMalloc(&deviceInput, hostInput.size() * sizeof(int)),
                  "allocate input");
        checkCuda(cudaMalloc(&deviceOutput, sizeof(int)), "allocate output");
        checkCuda(
            cudaMemcpy(deviceInput, hostInput.data(),
                       hostInput.size() * sizeof(int), cudaMemcpyHostToDevice),
            "copy input");

        const Reduction reductions[] = {
            {"AtomicGlobal", reduceAtomicGlobal},
            {"AtomicShare", reduceAtomicShare},
            {"Interleaved", reduceInterleaved},
            {"Sequential", reduceSequential},
            {"Unroll", reduceUnroll},
            {"WarpReduceFunc", reduceWarpReduceFunc},
            {"Multipleload", reduceMultipleload},
            {"CompleteUnroll", reduceCompleteUnroll},
        };

        std::cout << "GPU: " << deviceProp.name << '\n'
                  << "Array size: " << ELEMENT_COUNT
                  << ", expected sum: " << expected << '\n'
                  << std::left << std::setw(20) << "Method" << std::right
                  << std::setw(14) << "Time (us)" << std::setw(20)
                  << "Effective GB/s" << "  Result\n"
                  << std::fixed << std::setprecision(3);

        bool allCorrect = true;
        int actual = 0;
        for (const Reduction& reduction : reductions) {
            const float timeMs =
                benchmark(reduction.kernel, deviceInput, deviceOutput);

            // 原子累加方案会保留前一次调用的总和，所以先清零再单独运行一次
            // 来检查结果；计时循环中的累加结果不用于正确性校验。
            checkCuda(cudaMemset(deviceOutput, 0, sizeof(int)), "clear output");
            reduction.kernel<<<1, THREADS_PER_BLOCK>>>(deviceInput,
                                                       deviceOutput);
            checkCuda(cudaGetLastError(), "reduction kernel launch");
            checkCuda(cudaMemcpy(&actual, deviceOutput, sizeof(int),
                                 cudaMemcpyDeviceToHost),
                      "copy reduction result");

            const bool correct = actual == expected;
            allCorrect = allCorrect && correct;
            const double bandwidthGBs =
                // 这里按读取 1024 个 int 估算有效带宽，单位为十进制 GB/s。
                static_cast<double>(ELEMENT_COUNT * sizeof(int)) /
                (timeMs * 1.0e6);
            std::cout << std::left << std::setw(20) << reduction.name
                      << std::right << std::setw(14) << timeMs * 1000.0
                      << std::setw(20) << bandwidthGBs << "  "
                      << (correct ? "OK"
                                  : "FAILED (" + std::to_string(actual) + ")")
                      << '\n';
        }

        checkCuda(cudaFree(deviceInput), "free input");
        checkCuda(cudaFree(deviceOutput), "free output");
        return allCorrect ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
