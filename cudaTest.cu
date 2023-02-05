#include <cuda_runtime.h>
#include <iostream>
#include "cudalibrary.h"

__device__ void innerCall(double *ptr, int th_i)
{
    for (int i = 0; i < 32; i++)
        printf("[%d]%lf, ", i, ptr[i]);
    __syncthreads();
} 

__global__ void extercall(double *a, double *b)
{
    __shared__ double vals[32][33];

    vals[threadIdx.x][threadIdx.y] = a[threadIdx.x + threadIdx.y * 32];

    // printf("%f\n", a[i + threadIdx.y * 16]);
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        innerCall(vals[threadIdx.x], threadIdx.x);
        printf("\n");
    }

    __syncthreads();
    b[threadIdx.x + threadIdx.y * 32] = vals[threadIdx.x][threadIdx.y];
}

int main()
{
    freopen("io/cudaTest.csv", "w", stdout);

    double *a, *b;

    int n = 32 * 32;

    cudaMallocManaged((void **)&a, n * sizeof(double));
    cudaMallocManaged((void **)&b, n * sizeof(double));

    initilizeData(a, n);

    for (int j = 0; j < 32; j++)
    {
        for (int i = 0; i < 32; i++)
        {
            std::cout << a[i + j * 32] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    cudaSetDevice(0);
    dim3 grid(1);
    dim3 block(32, 32);
    extercall<<<grid, block>>>(a, b);
    cudaDeviceSynchronize();

    for (int j = 0; j < 32; j++)
    {
        for (int i = 0; i < 32; i++)
        {
            std::cout << b[i + j * 32] << ", ";
        }
        std::cout << std::endl;
    }
}