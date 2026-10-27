#include <iostream>
#include <random>
#include <cuda_runtime.h>
#define NUM_DATA 1000
#define MIN_DATA 1
#define MAX_DATA 100000

namespace ParallelComputingCommon
{

    class CDataManager
    {
    public:
        int *generateRandomArray(int num_data, int mn, int mx)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> distrib(mn, mx);
            int *res = new int[num_data];
            for (int i = 0; i < num_data; ++i)
            {
                res[i] = distrib(gen);
            }
            return res;
        }

        void compareTwoArrays(int *o1, std::string a1, int *o2, std::string a2, int num)
        {
            std::cout << a1 << " and " << a2 << " is ";
            for (int i = 0; i < num; ++i)
            {
                if (o1[i] != o2[i])
                {
                    std::cout << "different!" << std::endl;
                    return;
                }
            }
            std::cout << "same!" << std::endl;
            return;
        }
    };

}

void _asserte() // 因_ASSERTE是 Microsoft Visual C++ 的调试宏，不是标准 C++ 或 CUDA 宏，故手动实现
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << '\n';
        return;
    }
}

__global__ void kernelAdd(const int *vSrc1, const int *vSrc2, int *vOutput)
{
    int i = threadIdx.x;
    vOutput[i] = vSrc1[i] + vSrc2[i];
}

void add_cuda(const int *vInput1, const int *vInput2, int *vOutput, unsigned int vNumData)
{
    int *pInput1GPU = nullptr;
    int *pInput2GPU = nullptr;
    int *pOutputGPU = nullptr;

    cudaMalloc((void **)&pInput1GPU, vNumData * sizeof(int));
    cudaMalloc((void **)&pInput2GPU, vNumData * sizeof(int));
    cudaMalloc((void **)&pOutputGPU, vNumData * sizeof(int));

    cudaMemcpy(pInput1GPU, vInput1, vNumData * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pInput2GPU, vInput2, vNumData * sizeof(int), cudaMemcpyHostToDevice);

    _asserte();

    kernelAdd<<<1, vNumData>>>(pInput1GPU, pInput2GPU, pOutputGPU);

    _asserte();

    cudaDeviceSynchronize();

    cudaMemcpy(vOutput, pOutputGPU, vNumData * sizeof(int), cudaMemcpyDeviceToHost);
}

void add_cpu(const int *vInput1, const int *vInput2, int *vOutput, unsigned int vNumData)
{
    for (int i = 0; i < vNumData; ++i)
    {
        vOutput[i] = vInput1[i] + vInput2[i];
    }
}

int main()
{
    ParallelComputingCommon::CDataManager DataManager;

    int *pInput1 = DataManager.generateRandomArray(NUM_DATA, MIN_DATA, MAX_DATA);
    int *pInput2 = DataManager.generateRandomArray(NUM_DATA, MIN_DATA, MAX_DATA);
    int *pOutputByCPU = new int[NUM_DATA];
    int *pOutputByGPU = new int[NUM_DATA];

    cudaSetDevice(0);
    add_cuda(pInput1, pInput2, pOutputByGPU, NUM_DATA);
    cudaDeviceReset();

    add_cpu(pInput1, pInput2, pOutputByCPU, NUM_DATA);

    DataManager.compareTwoArrays(pOutputByCPU, "CPU results", pOutputByGPU,
                                 "GPU results", NUM_DATA);

    delete[] pInput1;
    delete[] pInput2;
    delete[] pOutputByCPU;
    delete[] pOutputByGPU;

    return 0;
}