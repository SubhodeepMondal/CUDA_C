#pragma ONCE

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdarg.h>
#include <random>
#include <math.h>

namespace gpu
{

    __global__ void printData(double *a, unsigned x, unsigned y, unsigned z)
    {
        int i, j, k;
        for (i = 0; i < z; i++)
            for (j = 0; j < y; j++)
                for (k = 0; k < x; k++)
                {
                    if (k == x - 1)
                        if (j == y - 1)
                            printf(" %lf\n\n", a[k + j * x + i * x * y]);
                        else
                            printf(" %lf\n", a[k + j * x + i * x * y]);
                    else
                        printf(" %lf", a[k + j * x + i * x * y]);
                }
    }

    __global__ void print(double *a)
    {
        printf("%lf", a[0]);
    }

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

    __global__ void cudaMatrixMul(double *a, double *b, double *d, int a_m, int a_n, int b_m, int b_n, int i, int j)
    {
        __shared__ double Y_shared[32][33];
        __shared__ double X_shared[32][32];

        double val;

        int Ai, Bi, Bj, n, idx_Ai;

        Bi = j + threadIdx.x;
        Bj = i + threadIdx.y;
        Ai = i + (threadIdx.y + blockDim.y * blockIdx.y) * a_n;
        idx_Ai = (threadIdx.y + blockDim.y * blockIdx.y);

        (a_n - i) >= 32 ? n = 32 : n = (a_n - i);

        if (Bi < b_n && Bj < b_m)
        {
            Y_shared[threadIdx.x][threadIdx.y] = b[Bi + Bj * b_n];
        }
        if (idx_Ai < a_m && threadIdx.x < a_n)
        {
            X_shared[threadIdx.y][threadIdx.x] = a[Ai + threadIdx.x];
        }
        __syncthreads();

        if (Bi < b_n && (threadIdx.y + blockDim.y * blockIdx.y) < a_m)
        {

            val = cudaSubDotMul(X_shared[threadIdx.y], Y_shared[threadIdx.x], a_m, a_n, b_m, b_n, n);
            __syncthreads();

            d[Bi + (threadIdx.y + blockDim.y * blockIdx.y) * b_n] += val;
        }
    }

    __global__ void matrixDotMul(double *a, double *b, unsigned x, unsigned y, unsigned m, unsigned n, unsigned index)
    {
        // m : features, n : neurons;
        // x : max feature, y : max neuron;
        unsigned intr_x, intr_y, inp_lin, w_lin, res_lin;

        intr_x = threadIdx.x + (blockIdx.x * blockDim.x); // neuron axis
        intr_y = threadIdx.y + (blockIdx.y * blockDim.y); // feature axis

        if (intr_x < n && intr_y < m)
        {
            w_lin = intr_x + (intr_y * x) + (index * x * y);         // z = features + 1 for bias.
            inp_lin = intr_y + (index * x * y);                      // input linear index.
            res_lin = intr_x + (intr_y * x) + ((index + 1) * x * y); // resluting array input index.

            a[res_lin] = (double)a[inp_lin] * b[w_lin];
            // a[res_lin] = w_lin;
        }
    }

    __global__ void matrixSubSum(double *a, int x, int y, int m, unsigned n, unsigned index)
    {
        // x: neurons, y: features, m: max_feature, a[0][i] should contain the rolling sum
        unsigned intr_x, inp_lin, out_lin, i;
        double val;

        intr_x = threadIdx.x + (blockIdx.x * blockDim.x);
        out_lin = intr_x + (index * m * n);
        // printf("%ld ", (double*) a);

        if (intr_x < x)
        {
            val = 0.0f;

            for (i = 0; i < y; i++)
            {
                inp_lin = out_lin + (i * m);
                val += a[inp_lin];
                // if(i!=0)
                //     a[inp_lin] = 0;
            }

            a[out_lin] = val;
        }
    }

    __global__ void matrixRelu(double *a, int x, int y, int m, int flag)
    {
        // x: neuron, y: feature. m: max_neuron.
        unsigned id_x;
        id_x = threadIdx.x + (blockDim.x * blockIdx.x);

        if (id_x < x && a[id_x] < 0)
            a[id_x] = 0;

        if (id_x == x && flag == 1)
            a[id_x] = 1;
    }

    __global__ void matrixSigmoid(double *a, int x, int y, int m, int flag)
    {
        // x: neuron, y: feature. m: max_neuron.
        unsigned id_x;
        id_x = threadIdx.x + (blockDim.x * blockIdx.x);

        if (id_x < x)
            a[id_x] = 1.0f / (1 + exp(-1 * a[id_x]));

        if (id_x == x && flag == 1)
            a[id_x] = 1;
        
    }
}

namespace cpu
{
    void initilizeData(double *data, int nElem)
    {
        time_t t;
        srand((unsigned)time(&t));

        for (int i = 0; i < nElem; i++)
            data[i] = (float)(rand() & 0xFF) / 117.00000;
    }

    void matrixMul(double *A, double *B, double *C, int a_m, int a_n, int b_m, int b_n)
    {
        double sum = 0;
        for (int i = 0; i < a_m; i++)
            for (int j = 0; j < b_n; j++)
            {
                sum = 0;
                for (int k = 0; k < a_n; k++)
                    sum += A[i * a_n + k] * B[k * b_n + j];
                C[i * b_n + j] = sum;
            }
    }

    void compareArray(double *A, double *B, int n)
    {
        int flag = 0;
        for (int i = 0; i < n; i++)
            if (abs(A[i] - B[i]) > 0.01f)
            {
                flag = 1;
                break;
            }
        if (flag == 0)
            std::cout << "The arrays are exact match." << std::endl;
        else
            std::cout << "The arrays are not a match." << std::endl;
    }
}

template <class T, int typeFlag>
class NDArray
{
    int type = typeFlag;
    int no_of_gpu;
    unsigned nDim;
    unsigned *dimension;
    unsigned nElem;
    T *data;

public:
    NDArray(unsigned n, ...)
    {
        va_list valist;

        nDim = n;
        nElem = 1;
        dimension = new unsigned[n];
        va_start(valist, n);

        for (int i = 0; i < n; i++)
            dimension[i] = va_arg(valist, unsigned);

        va_end(valist);

        for (int i = 0; i < nDim; i++)
            nElem *= dimension[i];

        cudaGetDeviceCount(&no_of_gpu);
        if (no_of_gpu && type)
            cudaMalloc((T **)&data, nElem * sizeof(T));
        else
            this->data = new T[nElem];
    }

    NDArray(unsigned n, unsigned *arr)
    {
        nDim = n;
        nElem = 1;
        dimension = new unsigned[n];

        for (int i = 0; i < n; i++)
            dimension[i] = arr[i];

        for (int i = 0; i < nDim; i++)
            nElem *= dimension[i];

        cudaGetDeviceCount(&no_of_gpu);
        if (no_of_gpu && type)
            cudaMalloc((T **)&data, nElem * sizeof(T));
        else
            this->data = new T[nElem];
    }

    NDArray(NDArray &ndarray)
    {
        this->nDim = ndarray.nDim;
        this->dimension = ndarray.dimension;
        this->nElem = ndarray.nElem;
        this->data = ndarray.data;
        this->type = type;
    }

    NDArray() {}

    ~NDArray()
    {
        // if(typeFlag && no_of_gpu)
        //     cudaFree(data);
        // else
        //     delete [] data;
    }

    unsigned *getDimensions()
    {
        unsigned *ptr;
        ptr = dimension;
        return ptr;
    }

    unsigned getNoOfDimensions()
    {
        return nDim;
    }

    unsigned getNoOfElem()
    {
        return nElem;
    }

    T *getData()
    {
        return data;
    }

    void printDimensions()
    {
        for (int i = 0; i < nDim; i++)
            std::cout << dimension[i] << ", ";
        std::cout << std::endl;
    }

    void printData()
    {
        int Elem;

        int *dim;
        dim = new int[nDim];
        for (int i = 0; i < nDim; i++)
            dim[i] = dimension[i];

        // dim[0] += dim[1];
        // dim[1] = dim[0] - dim[1];
        // dim[0] -= dim[1];
        if (type && no_of_gpu)
            cudaSetDevice(0);

        for (int i = 0; i < nElem; i++)
        {
            if (dim[0] == 1)
                std::cout << "[";

            Elem = 1;
            for (int j = 0; j < nDim; j++)
            {
                Elem *= dim[j];
                if ((i + 1) % Elem == 1)
                    std::cout << "[";
            }

            std::cout << "\t";

            if (type && no_of_gpu)
            {
                gpu::print<<<1, 1>>>(data + i);
                cudaDeviceSynchronize();
            }
            else
            {
                std::cout.precision(6);
                std::cout.setf(std::ios::showpoint);
                std::cout << data[i];
            }

            if ((i + 1) % dim[0] != 0)
                std::cout << ",";

            Elem = 1;
            for (int j = 0; j < nDim; j++)
            {
                Elem *= dim[j];
                if ((i + 1) % Elem == 0)
                {
                    if (j == 0)
                        std::cout << "\t";
                    std::cout << "]";
                }
            }

            if ((i + 1) % dim[0] == 0)
                std::cout << std::endl;
        }
        std::cout << std::endl;
        free(dim);
    }

    void printLinearData()
    {
        for (int i = 0; i < nElem; i++)
        {
            std::cout << data[i] << ", ";
        }
        std::cout << std::endl;
    }

    void initData(T *data)
    {
        if (type && no_of_gpu)
            cudaMemcpy(this->data, data, sizeof(T) * nElem, cudaMemcpyHostToDevice);
        else
            for (int i = 0; i < nElem; i++)
                this->data[i] = data[i];
    }

    void initPartialData(unsigned index, unsigned n, T *data_source)
    {
        if (type && no_of_gpu)
            cudaMemcpy((data + index), data_source, sizeof(T) * n, cudaMemcpyHostToDevice);
        else
            for (int i = index; i < n; i++)
                data[i] = data_source[i];
    }

    void initRandData(int lower_limit, int upper_limit)
    {
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution((0 + lower_limit), (1 * upper_limit));

        for (int i = 0; i < nElem; i++)
            data[i] = distribution(generator);
    }

    void copyData(T *data)
    {
        for (int i = 0; i < nElem; i++)
            data[i] = this->data[i];
        std::cout << std::endl;
    }
};

template <class T, int typeFlag>
class NDMath:public NDArray<T,typeFlag>
{
    NDArray<double, 1> *nd_ptr;
    double **ptr;
    unsigned ptrDim, max_feature, max_neuron;
    dim3 grid, block;

public:
    NDMath(NDArray<unsigned, 0> input, unsigned max_feature, unsigned max_neuron)
    {
        ptrDim = input.getDimensions()[0];
        ptr = new double *[ptrDim];
        this->max_feature = max_feature;
        this->max_neuron = max_neuron;
        std::cout << this->max_feature << " " << this->max_neuron << "\n";

        for (int i = 0; i < ptrDim; i++)
        {
            cudaMalloc((double **)&ptr[i], input.getData()[i] * sizeof(double));
        }
    }

    NDMath(unsigned max_feature, unsigned max_neuron, unsigned layer_count)
    {
        nd_ptr = new NDArray<double, 1>[2];
        nd_ptr[0] = NDArray<double, 1>(3, max_feature, max_neuron, layer_count + 1);
        nd_ptr[1] = NDArray<double, 1>(3, max_neuron, max_feature, layer_count);
    }

    NDMath() {}

    void printData(unsigned idx, unsigned x, unsigned y, unsigned z)
    {
        gpu::printData<<<1, 1>>>(ptr[idx], x, y, z);
    }

    void printData(unsigned idx)
    {
        nd_ptr[idx].printData();
    }

    void transferData(unsigned idx, unsigned offset, unsigned no_of_data, double *from)
    {
        nd_ptr[idx].initPartialData(offset, no_of_data, from);
        // cudaMemcpy((ptr[idx] + offset), from, no_of_data * sizeof(double), cudaMemcpyHostToDevice);
    }

    NDArray<double, 0> multiplication(NDArray<double, 0> a, NDArray<double, 0> b, int gpu = 1)
    {
        int noDevice;
        int i, j, a_m, a_n, b_m, b_n;
        dim3 block, grid;
        double *ptrA, *ptrB, *ptrC;

        a_m = a.getDimensions()[0];
        a_n = a.getDimensions()[1];
        b_m = b.getDimensions()[0];
        b_n = b.getDimensions()[1];

        NDArray<double, 0> c(2, a_m, b_n);
        NDArray<double, 0> e(2, a_m, b_n);

        cudaGetDeviceCount(&noDevice);

        if (a_n == b_m)
        {
            if (noDevice > 0)
            {

                cudaMalloc((double **)&ptrA, sizeof(double) * a.getNoOfElem());
                cudaMalloc((double **)&ptrB, sizeof(double) * b.getNoOfElem());
                cudaMalloc((double **)&ptrC, sizeof(double) * c.getNoOfElem());

                cudaMemcpy(ptrA, a.getData(), sizeof(double) * a.getNoOfElem(), cudaMemcpyHostToDevice);
                cudaMemcpy(ptrB, b.getData(), sizeof(double) * b.getNoOfElem(), cudaMemcpyHostToDevice);

                block.x = 32;
                block.y = 32;
                grid.x = 1;
                grid.y = ceil(a_m / 32.0f);

                cudaSetDevice(0);

                for (i = 0; i < b_m; i += 32)
                {
                    for (j = 0; j < b_n; j += 32)
                    {
                        gpu::cudaMatrixMul<<<grid, block>>>(ptrA, ptrB, ptrC, a_m, a_n, b_m, b_n, i, j);
                    }
                }
                cudaDeviceSynchronize();

                cudaMemcpy(c.getData(), ptrC, sizeof(double) * c.getNoOfElem(), cudaMemcpyDeviceToHost);

                cudaDeviceSynchronize();

                cudaFree(ptrA);
                cudaFree(ptrB);
                cudaFree(ptrC);
            }
            else
            {
                double *ptrD;
                ptrD = new double[a_m * b_n];
                cpu::matrixMul(ptrA, ptrB, ptrD, a_m, a_n, b_m, b_n);
                c.initData(ptrD);
            }

            // ptrE = new double[a_m * b_n];
            // cpu::matrixMul(ptrA, ptrB, ptrE, a_m, a_n, b_m, b_n);
            // cpu::compareArray(c.data,ptrE,c.getNoOfElem());
        }
        return c;
    }

    void matrixDotMultiplication(unsigned neurons, unsigned no_of_features, unsigned index)
    {
        // unsigned intr_x, intr_y;
        int no_of_gpu; // getting no of cuda capable gpus

        cudaGetDeviceCount(&no_of_gpu);

        // printData(1);

        // std::cout << no_of_features << " " << neurons << "\n";


        if (no_of_gpu)
        {

            dim3 block, grid;

            cudaSetDevice(0);

            block.x = (neurons > 32) ? 32 : neurons;
            block.y = (no_of_features > 32) ? 32 : no_of_features;
            grid.x = ceil((float)neurons / block.x);
            grid.y = ceil((float)no_of_features / block.y);
            gpu::matrixDotMul<<<grid, block>>>(nd_ptr[0].getData(), nd_ptr[1].getData(), nd_ptr[0].getDimensions()[0], nd_ptr[1].getDimensions()[0], no_of_features, neurons, index);
            cudaDeviceSynchronize();

            block.y = 1;
            grid.y = 1;
            // nd_ptr[0].printData();

            gpu::matrixSubSum<<<grid, block>>>(nd_ptr[0].getData(), neurons, no_of_features, nd_ptr[0].getDimensions()[0], nd_ptr[1].getDimensions()[0], index + 1);
            cudaDeviceSynchronize();

            // nd_ptr[0].printData();
        }
        else
        {
        }
    }

    void reluActivation(unsigned neurons, unsigned no_of_features, unsigned index, int flag)
    {
        int x, y;

        dim3 block, grid;
        y = no_of_features;
        x = neurons;

        block.x = ((neurons+1) > 32) ? 32 : (neurons+1);
        grid.x = ceil((float)(neurons+1 )/ block.x);

        gpu::matrixRelu<<<grid, block>>>(nd_ptr[0].getData() + (index + 1) * nd_ptr[0].getDimensions()[0] * nd_ptr[1].getDimensions()[0], x, y, nd_ptr[1].getDimensions()[0], flag);
        cudaDeviceSynchronize();
    }

    void sigmoidActivation(unsigned neurons, unsigned no_of_features, unsigned index, int flag)
    {
        int x, y;
        dim3 block, grid;

        x = neurons;
        y = no_of_features;

        block.x = ((neurons+1) > 32) ? 32 : (neurons+1);
        grid.x = ceil((float)(neurons+1 )/ block.x);



        gpu::matrixSigmoid<<<grid, block>>>(nd_ptr[0].getData() + (index + 1) * nd_ptr[0].getDimensions()[0] * nd_ptr[1].getDimensions()[0], x, y, nd_ptr[1].getDimensions()[0], flag);
        cudaDeviceSynchronize();


    }

    // ~NDMath()
    // {
    // }
};
