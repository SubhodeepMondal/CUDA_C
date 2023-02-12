#include <cuda_runtime.h>
#include <stdarg.h>

void initilizeData(double *data, int nElem)
{
    time_t t;
    srand((unsigned)time(&t));

    for (int i = 0; i < nElem; i++)
        data[i] = (float)(rand() & 0xFF) / 117.00000;
}


namespace cuda
{
__global__ void cudaTranspose(unsigned int *a, unsigned int *b, int xsize, int ysize)
{
    int ix, iy, mat_in, mat_tra;

    __shared__ unsigned int smallblock[32][32];

    ix = blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    mat_in = ix * xsize + iy;

    int bidx, icol, irow;
    bidx = threadIdx.y * blockDim.x + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    mat_tra = iy * ysize + ix;

    smallblock[threadIdx.x][threadIdx.y] = mat_in;
    __syncthreads();
    b[mat_tra] = smallblock[icol][irow];
}

__global__ void cudaDotMul(double *a, double *b, double *c, int x, int y, int a_m, int a_n, int b_m, int b_n)
{
    int n = b_m;
    int ind, i;
    double val;
    unsigned mask = 0xffffffff;

    ind = (threadIdx.x + x * b_m) + (y * a_m * b_m);
    if (threadIdx.x < a_n)
    {
        c[ind] = a[threadIdx.x + x * a_n] * b[threadIdx.x * b_n + y];
    }
    else
        c[ind] = 0.0f;

    __syncthreads();

    if (threadIdx.x < a_n)
    {
        for (i = n / 2; i > 0; i /= 2)
        {
            val = c[ind];
            val = __shfl_down_sync(mask, val, i);
            c[ind] += val;
        }
    }
}

__global__ void cudaMatrixMulMultiParallesied(double *a, double *b, double *c, double *d, int a_m, int a_n, int b_m, int b_n)
{
    int ix, iy, rowDim;
    ix = threadIdx.x + blockIdx.x * blockDim.x;
    iy = threadIdx.y + blockIdx.y * blockDim.y;
    rowDim = b_m;
    d[ix + iy * a_m] = 0;
    if (ix < a_m && iy < b_n)
    {
        cudaDotMul<<<1, rowDim>>>(a, b, c, ix, iy, a_m, a_n, b_m, b_n);
        d[ix + iy * a_m] += c[ix * b_m * a_m + iy * b_m];
    }
}

__global__ void cudaRollingSum(double *a)
{
    int n = blockDim.x;
    int ind, i, k;
    double val;
    ind = threadIdx.x;
    unsigned x = 0xffffffff;

    k = n;
    if (threadIdx.x < n)
    {
        for (i = n / 2; i > 0; i /= 2)
        {
            val = a[ind];
            val = __shfl_down_sync(x, val, i);
            a[ind] += val;
            k /= 2;
        }
    }
}

__device__ double cudaSubDotMul(double *a, double *b, int a_m, int a_n, int b_m, int b_n, int n)
{
    double sum = 0;

    for (int i = 0; i < n; i++)
    {
        sum += a[i] * b[i];
    }

    return sum;
}

__global__ void cudaSubMul(double *a, double *b, double *d, int a_m, int a_n, int b_m, int b_n, int i, int j)
{
    __shared__ double Y_shared[32][33];

    double val;

    int Ai, Bi, Bj, n;

    Bi = j + threadIdx.x;
    Bj = i + threadIdx.y;
    Ai = i + (threadIdx.y + blockDim.y * blockIdx.y) * a_n;
    (a_n - i) >= 32 ? n = 32 : n = (a_n - i);

    if (Bi < b_n && Bj < b_m)
    {
        Y_shared[threadIdx.x][threadIdx.y] = b[Bi + Bj * b_n];
    }
    __syncthreads();

    if (Bi < b_n && (threadIdx.y + blockDim.y * blockIdx.y) < a_m)
    {

        val = cudaSubDotMul((a + Ai), Y_shared[threadIdx.x], a_m, a_n, b_m, b_n, n);
        __syncthreads();

        d[Bi + (threadIdx.y + blockDim.y * blockIdx.y) * b_n] += val;
    }
}

__global__ void print(double *a)
{
    int idx = threadIdx.x;
    printf("%lf, ", a[idx]);
}
}


template <class T>
class NDArray
{
public:
    int nDim;
    int *dimention;
    int nElem;
    T *data;

public:
    NDArray(int n, ...)
    {
        va_list valist;

        nDim = n;
        nElem = 1;
        dimention = new int[n];
        va_start(valist, n);

        for (int i = 0; i < n; i++)
        {
            dimention[i] = va_arg(valist, int);
        }

        for (int i = 0; i < nDim; i++)
        {
            nElem *= dimention[i];
        }

        this->data = new T[nElem];
    }

    NDArray()
    {
    }

    void printDimentions()
    {
        for (int i = 0; i < nDim; i++)
            std::cout << dimention[i] << ", ";
        std::cout << std::endl;
    }

    void printData()
    {
        for (int i = 0; i < nElem; i++)
        {
            std::cout << data[i] << ", ";
        }
    }

    void initData(T *data)
    {
        for (int i = 0; i < nElem; i++)
            this->data[i] = data[i];
    }

    void initRandData()
    {
        time_t t;
        srand((unsigned)time(&t));

        for (int i = 0; i < nElem; i++)
            data[i] = (float)(rand() & 0xFF) / 117.00000;
    }
    void copyData(T *data)
    {
        for (int i = 0; i < nElem; i++)
            data[i] = this->data[i];
        std::cout << std::endl;
    }
};


class NDMath
{
public:
    NDArray<double> multiplication(NDArray<double> a, NDArray<double> b)
    {
        int i, j, a_m, a_n, b_m, b_n;
        dim3 block, grid;
        double *ptrA, *ptrB, *ptrC;
        a_m = a.dimention[0];
        a_n = a.dimention[1];
        b_m = b.dimention[0];
        b_n = b.dimention[1];

        NDArray<double> c(2, a_m, b_n);

        if (a_n == b_m)
        {
            cudaMallocManaged((double **)&ptrA, sizeof(double) * a.nElem);
            cudaMallocManaged((double **)&ptrB, sizeof(double) * b.nElem);
            cudaMallocManaged((double **)&ptrC, sizeof(double) * c.nElem);

            a.copyData(ptrA);
            b.copyData(ptrB);

            block.x = 32;
            block.y = 32;
            grid.x = 1;
            grid.y = ceil(a_m / 32.0f);

            std::cout << block.x << " " << block.y << " " << grid.x << " " << grid.y << std::endl;

            cudaSetDevice(0);

            for (i = 0; i < b_m; i += 32)
            {
                for (j = 0; j < b_n; j += 32)
                {
                    cuda::cudaSubMul<<<grid, block>>>(ptrA, ptrB, ptrC, a_m, a_n, b_m, b_n, i, j);
                }
            }
            cudaDeviceSynchronize();

            c.initData(ptrC);

            for (int i = 0; i < 10; i++)
                std::cout << ptrC[i] << ", ";
            std::cout << std::endl;
        }

        return c;
    }
};
