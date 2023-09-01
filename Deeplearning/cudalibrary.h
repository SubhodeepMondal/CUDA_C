#pragma ONCE

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdarg.h>
#include <random>
#include <math.h>

namespace gpu
{

    // __device__ double learning_rate;

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
        printf("%.6lf", *(a));
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

        // std::cout << "delta_output:\n";
        // delta_output.printData();
        // std::cout << "delta_output:\n";
        // delta_output.printData();

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

    __global__ void matrixDotMul(double *input_A, double *input_B, double *input_C, double *output, unsigned x, unsigned y, unsigned z)
    {
        // m : features, n : neurons;
        // x : max feature, y : max neuron;
        unsigned intr_x, intr_y, intr_z, inp_lin, w_lin, res_lin;

        intr_x = threadIdx.x + (blockIdx.x * blockDim.x); // neuron axis
        intr_y = threadIdx.y + (blockIdx.y * blockDim.y); // batch axis
        intr_z = threadIdx.z + (blockIdx.z * blockDim.z); // batch axis

        inp_lin = intr_z + (intr_y * z);                    // input linear index.
        w_lin = intr_x + (intr_z * x);                      // z = features + 1 for bias.
        res_lin = intr_x + (intr_y * x) + (intr_z * x * y); // resluting array input index.

        if (intr_x < x && intr_y < y && intr_z < z)
        {
            output[res_lin] = input_A[inp_lin] * input_B[w_lin]; // (double)
            // c[res_lin] = a[inp_lin];
        }
        else if (intr_x < x && intr_y < y && intr_z == z)
        {
            output[res_lin] = input_C[intr_x];
        }
    }

    __global__ void matrixDifferentialWeights(double *input, double *delta_output, double *difference, double *d_weights_biases, unsigned x, unsigned y, unsigned z)
    {
        unsigned indx_x, indx_y, indx_z, out_lin, inp_lin, diff_lin;

        indx_x = threadIdx.x + blockIdx.x * blockDim.x;
        indx_y = threadIdx.y + blockIdx.y * blockDim.y;
        indx_z = threadIdx.z + blockIdx.z * blockDim.z;

        if (indx_x < x && indx_y < y && indx_z < z)
        {
            out_lin = indx_x + indx_y * x + indx_z * x * y;
            inp_lin = indx_y + indx_z * y;
            diff_lin = indx_x + indx_z * x;

            d_weights_biases[out_lin] = 2 * delta_output[diff_lin] * difference[diff_lin] * input[inp_lin]; //
        }
    }

    __global__ void matrixDifferentialBiases(double *delta_output, double *difference, double *delta_biases, unsigned x, unsigned y)
    {
        unsigned indx_x, indx_y, out_lin, diff_lin;

        indx_x = threadIdx.x + blockIdx.x * blockDim.x;
        indx_y = threadIdx.y + blockIdx.y * blockDim.y;

        if (indx_x < x && indx_y < y)
        {
            out_lin = indx_x + indx_y * x;
            diff_lin = indx_x + indx_y * x;

            delta_biases[out_lin] = 2 * delta_output[diff_lin] * difference[diff_lin]; //
        }
    }

    __global__ void matrixDifferentialInput(double *weights, double *delta_output, double *difference, double *delta_input, unsigned x, unsigned y, unsigned z)
    {
        unsigned indx_x, indx_y, indx_z, out_lin, inp_lin, diff_lin;

        indx_x = threadIdx.x + blockIdx.x * blockDim.x;
        indx_y = threadIdx.y + blockIdx.y * blockDim.y;
        indx_z = threadIdx.z + blockIdx.z * blockDim.z;
        out_lin = indx_x + indx_y * x + indx_z * x * y;

        if (indx_x < x && indx_y < y & indx_z < z)
        {
            out_lin = indx_x + indx_y * x + indx_z * x * y;
            diff_lin = indx_y + indx_z * y;
            inp_lin = indx_x + indx_z * x;

            delta_input[out_lin] = 2 * weights[inp_lin] * difference[diff_lin] * delta_output[diff_lin]; //
        }
    }

    __global__ void matrixRollingSum(double *input, double *output, unsigned x, unsigned y, unsigned z)
    {
        // x: neurons, y: features
        unsigned intr_x, intr_y, inp_lin, out_lin, i;
        double val;

        intr_x = threadIdx.x + (blockIdx.x * blockDim.x);
        intr_y = threadIdx.y + (blockIdx.y * blockDim.y);
        out_lin = intr_x + intr_y * x;

        if (intr_x < x && intr_y < y)
        {
            val = 0.0;
            for (i = 0; i < z; i++)
            {
                inp_lin = out_lin + i * x * y;
                val += input[inp_lin];
            }
            output[out_lin] = val;
        }
    }

    __global__ void matrixRelu(double *a, double *d_a, int x, int y)
    {
        // x: neuron, y: feature. m: max_neuron.
        unsigned id_x, id_y, lin_idx;
        id_x = threadIdx.x + (blockDim.x * blockIdx.x);
        id_y = threadIdx.y + (blockDim.y * blockIdx.y);

        if (id_x < x && id_y < y)
        {
            lin_idx = id_x + id_y * x;
            if (a[lin_idx] > 0)
            {
                d_a[lin_idx] = 1;
            }
            else
            {

                a[lin_idx] = 0;
                d_a[lin_idx] = 0;
            }
        }
    }

    __global__ void matrixSigmoid(double *a, double *d_a, int x, int y)
    {
        // x: neuron, y: feature. m: max_neuron.
        unsigned id_x, id_y, lin_idx;
        id_x = threadIdx.x + (blockDim.x * blockIdx.x);
        id_y = threadIdx.y + (blockDim.y * blockIdx.y);

        if (id_x < x && id_y < y)
        {

            lin_idx = id_x + id_y * x;
            a[lin_idx] = 1.0f / (1 + exp(-1 * a[lin_idx]));
            d_a[lin_idx] = a[lin_idx] * (1 - a[lin_idx]);
        }
    }

    __global__ void matrixLinear(double *a, double *d_a, int x, int y)
    {
        // x: neuron, y: feature. m: max_neuron.
        unsigned id_x, id_y, lin_idx;
        id_x = threadIdx.x + (blockDim.x * blockIdx.x);
        id_y = threadIdx.y + (blockDim.y * blockIdx.y);
        lin_idx = id_x + id_y * x;

        if (id_x < x && id_y < y)
        {
            d_a[lin_idx] = 1;
        }
    }

    __global__ void matrixSquaredError(double *a, double *b, unsigned x, unsigned y)
    {
        unsigned id_x, id_y, lin_idx;
        id_x = threadIdx.x + (blockDim.x * blockIdx.x);
        id_y = threadIdx.y + (blockDim.y * blockIdx.y);
        lin_idx = id_x + id_y * x;

        if (id_x < x && id_y < y)
            b[lin_idx] = pow((a[lin_idx]), 2);
    }

    __global__ void matrixFindMean(double *a, unsigned x, unsigned y, unsigned mean)
    {
        unsigned indx_x, indx_y, inp_lin;

        indx_x = threadIdx.x + blockIdx.x * blockDim.x;
        indx_y = threadIdx.y + blockIdx.y * blockDim.y;
        inp_lin = indx_x + indx_y * x;

        if (indx_x < x & indx_y < y)
            a[inp_lin] /= mean;
    }

    __global__ void matrixDifference(double *input_A, double *input_B, double *output_C, unsigned x, unsigned y)
    {
        unsigned id_x, id_y, lin_idx;
        id_x = threadIdx.x + (blockDim.x * blockIdx.x);
        id_y = threadIdx.y + (blockDim.y * blockIdx.y);

        lin_idx = id_x + id_y * x;

        if (id_x < x && id_y < y)
        {
            output_C[lin_idx] = input_A[lin_idx] - input_B[lin_idx]; // a[lin_idx] - b[id_y];
        }
    }

    __global__ void matrixUpdateWeightsBiases(double *weights_biases, double *learning_rate, double *d_weights_biases, unsigned a_m, unsigned a_n)
    {
        unsigned indx_x, indx_y, index;

        indx_x = threadIdx.x + blockIdx.x * blockDim.x;
        indx_y = threadIdx.y + blockIdx.y * blockDim.y;

        if (indx_x < a_m && indx_y < a_n)
        {
            index = indx_x + indx_y * a_m;
            weights_biases[index] -= learning_rate[index] * d_weights_biases[index]; //  ;
        }
    }

    __global__ void matrixUpdateWeightsBiasesSGDmomentum(double* sigma, double *learning_rate, double* sum_d_weights_biases, double *d_weights_biases, double *weights_biases, unsigned a_m, unsigned a_n)
    {
        unsigned indx_x, indx_y, index;
        // double weighted_delta_weights_biases;

        indx_x = threadIdx.x + blockIdx.x * blockDim.x;
        indx_y = threadIdx.y + blockIdx.y * blockDim.y;

        if (indx_x < a_m && indx_y < a_n)
        {
            index = indx_x + indx_y * a_m;

            sum_d_weights_biases[index] = (*sigma) * sum_d_weights_biases[index] + (1 - (*sigma)) * d_weights_biases[index];
            weights_biases[index] -= learning_rate[index] * sum_d_weights_biases[index]; //  ;
        }
    }

    __global__ void matrixUpdateWeightsBiasesRMSprop(double* sigma, double* epsalon, double *learning_rate, double* sum_d_weights_biases, double *d_weights_biases, double *weights_biases, unsigned a_m, unsigned a_n)
    {
        unsigned indx_x, indx_y, index;
        double squared_delta_weights_biases;

        indx_x = threadIdx.x + blockIdx.x * blockDim.x;
        indx_y = threadIdx.y + blockIdx.y * blockDim.y;

        if (indx_x < a_m && indx_y < a_n)
        {
            index = indx_x + indx_y * a_m;

            squared_delta_weights_biases = pow(d_weights_biases[index], 2);

            sum_d_weights_biases[index] = (*sigma) * sum_d_weights_biases[index] + (1 - (*sigma)) * squared_delta_weights_biases;
            weights_biases[index] -= learning_rate[index] * d_weights_biases[index] / (sqrt( sum_d_weights_biases[index]) + (*epsalon)); //  ;
        }
    }

    __global__ void matrixUpdateWeightsBiasesADAM(double* sigma, double* epsalon, double *learning_rate, double* sum_d_weights_biases, double* sum_d_weights_biases_squared, double *d_weights_biases, double *weights_biases, unsigned a_m, unsigned a_n)
    {
        unsigned indx_x, indx_y, index;
        double squared_delta_weights_biases;

        indx_x = threadIdx.x + blockIdx.x * blockDim.x;
        indx_y = threadIdx.y + blockIdx.y * blockDim.y;

        if (indx_x < a_m && indx_y < a_n)
        {
            index = indx_x + indx_y * a_m;

            squared_delta_weights_biases = pow(d_weights_biases[index], 2);

            sum_d_weights_biases[index] = (*sigma) * sum_d_weights_biases[index] + (1 - (*sigma)) * d_weights_biases[index];
            sum_d_weights_biases_squared[index] = (*sigma) * sum_d_weights_biases_squared[index] + (1 - (*sigma)) * squared_delta_weights_biases;

            weights_biases[index] -= learning_rate[index] * sum_d_weights_biases[index] / (sqrt( sum_d_weights_biases_squared[index]) + (*epsalon)); //  ;
        }
    }

    __global__ void matrixUpdateLearningRateAdagrad(double *epsalon, double *delta_weights_biases, double *sum_delta_weights, double *learning_rate, unsigned a_m, unsigned a_n)
    {
        unsigned indx_x, indx_y, index;
        double squared_delta_weights_biases;

        indx_x = threadIdx.x + blockIdx.x * blockDim.x;
        indx_y = threadIdx.y + blockIdx.y * blockDim.y;

        if (indx_x < a_m && indx_y < a_n)
        {
            index = indx_x + indx_y * a_m;

            squared_delta_weights_biases = pow(delta_weights_biases[index], 2);
            sum_delta_weights[index] += squared_delta_weights_biases;

            learning_rate[index] /= sqrt(sum_delta_weights[index] + epsalon[0]);
            learning_rate[index] = (learning_rate[index] < 0.001) ? 0.001 : learning_rate[index];
        }
    }

    __global__ void matrixUpdateLearningRateAdadelta(double *epsalon, double *sigma, double *delta_weights_biases, double *sum_delta_weights, double *learning_rate, unsigned a_m, unsigned a_n)
    {
        unsigned indx_x, indx_y, index;
        double squared_delta_weights_biases;

        indx_x = threadIdx.x + blockIdx.x * blockDim.x;
        indx_y = threadIdx.y + blockIdx.y * blockDim.y;

        if (indx_x < a_m && indx_y < a_n)
        {
            index = indx_x + indx_y * a_m;

            squared_delta_weights_biases = pow(delta_weights_biases[index], 2);
            sum_delta_weights[index] = (*sigma) * sum_delta_weights[index] + (1 - (*sigma)) * squared_delta_weights_biases;

            learning_rate[index] /= sqrt(sum_delta_weights[index] + (*epsalon));
            learning_rate[index] = (learning_rate[index] < 0.001) ? 0.001 : learning_rate[index];
        }
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
    unsigned nDim, isInitilized;
    unsigned *dimension;
    unsigned nElem;
    T *data;

public:
    NDArray(unsigned n, ...)
    {
        va_list valist;
        int no_of_gpu;

        nDim = n;
        nElem = 1;
        dimension = new unsigned[n];
        isInitilized = 1;
        va_start(valist, n);

        for (int i = 0; i < n; i++)
            dimension[i] = va_arg(valist, unsigned);

        va_end(valist);

        for (int i = 0; i < nDim; i++)
            nElem *= dimension[i];

        cudaGetDeviceCount(&no_of_gpu);
        if (no_of_gpu && type)
        {
            type = 1;
            cudaMalloc((T **)&data, nElem * sizeof(T));
        }
        else
            this->data = new T[nElem];
    }

    NDArray(unsigned n, unsigned *arr, unsigned isInitilized = 1)
    {

        int no_of_gpu;
        nDim = n;
        nElem = 1;
        dimension = new unsigned[n];

        for (int i = 0; i < n; i++)
            dimension[i] = arr[i];

        for (int i = 0; i < nDim; i++)
            nElem *= dimension[i];

        this->isInitilized = isInitilized;
        cudaGetDeviceCount(&no_of_gpu);

        if (isInitilized)
        {
            if (no_of_gpu && type)
            {
                type = 1;
                cudaMalloc((T **)&data, nElem * sizeof(T));
            }
            else
                this->data = new T[nElem];
        }
    }

    NDArray(NDArray &ndarray)
    {
        this->nDim = ndarray.nDim;
        this->dimension = ndarray.dimension;
        this->nElem = ndarray.nElem;
        this->data = ndarray.data;
    }

    NDArray() {}

    ~NDArray()
    {
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
        std::cout << "[ ";
        for (int i = 0; i < nDim; i++)
            std::cout << dimension[i] << ", ";
        std::cout << "]";
    }

    void printData()
    {
        int Elem;

        int *dim;
        dim = new int[nDim];
        for (int i = 0; i < nDim; i++)
            dim[i] = dimension[i];

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

            if (type)
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

    void initData(T data)
    {
        if (type)
        {
            T *tempPtr = new T[nElem];
            for (int i = 0; i < nElem; i++)
                tempPtr[i] = data;

            cudaMemcpy(this->data, tempPtr, sizeof(T) * nElem, cudaMemcpyHostToDevice);

            delete[] tempPtr;
        }
        else
            for (int i = 0; i < nElem; i++)
                this->data[i] = data;
    }

    void initData(T *data)
    {
        if (type)
            cudaMemcpy(this->data, data, sizeof(T) * nElem, cudaMemcpyHostToDevice);
        else
            for (int i = 0; i < nElem; i++)
                this->data[i] = data[i];
    }

    void initData(NDArray<double, 1> data)
    {
        cudaMemcpy(this->data, data.getData(), sizeof(T) * nElem, cudaMemcpyDeviceToDevice);
    }

    void initPartialData(unsigned index, unsigned n, T *data_source)
    {
        int j = 0;
        if (type)
            cudaMemcpy((data + index), data_source, sizeof(T) * n, cudaMemcpyHostToDevice);
        else
            for (int i = index; i < (index + n); i++)
                data[i] = data_source[j++];
    }

    void initRandData(int lower_limit, int upper_limit)
    {
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution((0 + lower_limit), (1 * upper_limit));

        for (int i = 0; i < nElem; i++)
            data[i] = distribution(generator);
    }

    void initPreinitilizedData(double *Data)
    {
        if (!isInitilized)
            this->data = Data;
        else
            std::cout << "This NDArray is Initilized, can't initiate with different data!\n";
    }

    void copyData(T *data)
    {
        for (int i = 0; i < nElem; i++)
            data[i] = this->data[i];
        std::cout << std::endl;
    }

    void destroy()
    {
        if (typeFlag)
            cudaFree(data);
        else
            delete[] data;

        delete[] dimension;
    }
};

class NDMath
{
    NDArray<double, 1> *nd_ptr;

    NDArray<double, 1> intermediate_output;
    unsigned isIntermediateOutputUpdated = 0, isSymbolCopied = 0;
    // cudaStream_t stream;

public:
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

    void matrixDotMultiplication(NDArray<double, 1> input, NDArray<double, 1> weights, NDArray<double, 1> biases, NDArray<double, 1> output, cudaStream_t stream)
    {
        unsigned intr_x, intr_y, intr_z;
        dim3 block, grid;

        intr_x = weights.getDimensions()[0];     // no of neurons
        intr_y = input.getDimensions()[1];       // no of batches
        intr_z = weights.getDimensions()[1] + 1; // no of features

        if (!isIntermediateOutputUpdated)
        {
            intermediate_output = NDArray<double, 1>(3, intr_x, intr_y, intr_z);
            isIntermediateOutputUpdated = 1;
        }

        block.x = (intr_x > 8) ? 8 : intr_x;
        block.y = (intr_y > 8) ? 8 : intr_y;
        block.z = (intr_z > 16) ? 16 : intr_z;

        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);
        grid.z = ceil((float)intr_z / block.z);

        // std::cout << "\nInput:\n";
        // input.printData();
        // std::cout << "Weights:" << weights.getData() << "\n";
        // weights.printData();
        // std::cout << "Biases:\n";
        // biases.printData();

        gpu::matrixDotMul<<<grid, block, 0, stream>>>(input.getData(), weights.getData(), biases.getData(), intermediate_output.getData(), intr_x, intr_y, intr_z - 1);

        block.z = 1;
        grid.z = 1;

        gpu::matrixRollingSum<<<grid, block, 0, stream>>>(intermediate_output.getData(), output.getData(), intr_x, intr_y, intr_z);

        // std::cout << "Intermediate_output:\n";
        // intermediate_output.printData();

        // std::cout << "Output:\n";
        // output.printData();
    }

    void updateLearningRateWeightsAdagrad(NDArray<double, 1> epsalon, NDArray<double, 1> sum_delta_weights, NDArray<double, 1> delta_weights, NDArray<double, 1> learning_rate_weights, cudaStream_t stream)
    {
        unsigned intr_x, intr_y;
        dim3 grid, block;

        intr_x = delta_weights.getDimensions()[0];
        intr_y = delta_weights.getDimensions()[1];

        block.x = (intr_x > 32) ? 32 : intr_x;
        block.y = (intr_y > 32) ? 32 : intr_y;

        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);

        gpu::matrixUpdateLearningRateAdagrad<<<grid, block, 0, stream>>>(epsalon.getData(), delta_weights.getData(), sum_delta_weights.getData(), learning_rate_weights.getData(), intr_x, intr_y);
    }

    void updateLearningRateBiasesAdagrad(NDArray<double, 1> epsalon, NDArray<double, 1> sum_delta_biases, NDArray<double, 1> delta_biases, NDArray<double, 1> learning_rate_biases, cudaStream_t stream)
    {
        unsigned intr_x;
        dim3 grid, block;

        intr_x = delta_biases.getDimensions()[0];

        block.x = (intr_x > 32) ? 32 : intr_x;

        grid.x = ceil((float)intr_x / block.x);

        gpu::matrixUpdateLearningRateAdagrad<<<grid, block, 0, stream>>>(epsalon.getData(), delta_biases.getData(), sum_delta_biases.getData(), learning_rate_biases.getData(), intr_x, 1);
    }

    void updateLearningRateWeightsAdadelta(NDArray<double, 1> epsalon, NDArray<double, 1> sigma, NDArray<double, 1> sum_delta_weights, NDArray<double, 1> delta_weights, NDArray<double, 1> learning_rate_weights, cudaStream_t stream)
    {
        unsigned intr_x, intr_y;
        dim3 grid, block;

        intr_x = delta_weights.getDimensions()[0];
        intr_y = delta_weights.getDimensions()[1];

        block.x = (intr_x > 32) ? 32 : intr_x;
        block.y = (intr_y > 32) ? 32 : intr_y;

        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);

        gpu::matrixUpdateLearningRateAdadelta<<<grid, block, 0, stream>>>(epsalon.getData(), sigma.getData(), delta_weights.getData(), sum_delta_weights.getData(), learning_rate_weights.getData(), intr_x, intr_y);
    }

    void updateLearningRateBiasesAdadelta(NDArray<double, 1> epsalon, NDArray<double, 1> sigma, NDArray<double, 1> sum_delta_biases, NDArray<double, 1> delta_biases, NDArray<double, 1> learning_rate_biases, cudaStream_t stream)
    {
        unsigned intr_x;
        dim3 grid, block;

        intr_x = delta_biases.getDimensions()[0];

        block.x = (intr_x > 32) ? 32 : intr_x;

        grid.x = ceil((float)intr_x / block.x);

        gpu::matrixUpdateLearningRateAdadelta<<<grid, block, 0, stream>>>(epsalon.getData(), sigma.getData(), delta_biases.getData(), sum_delta_biases.getData(), learning_rate_biases.getData(), intr_x, 1);
    }

    void updateWeights(NDArray<double, 1> weights, NDArray<double, 1> learning_rate, NDArray<double, 1> delta_weights, cudaStream_t stream)
    {
        unsigned intr_x, intr_y;
        dim3 grid, block;
        intr_x = weights.getDimensions()[0];
        intr_y = weights.getDimensions()[1];

        // std::cout << "Delta weights:\n";
        // weights.printData();

        // std::cout << "weights:\n";
        // weights.printData();

        block.x = (intr_x > 32) ? 32 : intr_x;
        block.y = (intr_y > 32) ? 32 : intr_y;

        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);

        gpu::matrixUpdateWeightsBiases<<<grid, block, 0, stream>>>(weights.getData(), learning_rate.getData(), delta_weights.getData(), intr_x, intr_y);

        // std::cout << "Updated weights:\n";
        // weights.printData();
        // std::cout << "\n\n\n";
    }

    void updateBiases(NDArray<double, 1> biases, NDArray<double, 1> learning_rate, NDArray<double, 1> delta_biases, cudaStream_t stream)
    {
        unsigned intr_x;
        dim3 grid, block;
        intr_x = biases.getDimensions()[0];

        // std::cout << "Biases:\n";
        // biases.printData();
        // std::cout << "Delta weights:\n";
        // delta_biases.printData();

        block.x = (intr_x > 32) ? 32 : intr_x;

        grid.x = ceil((float)intr_x / block.x);

        gpu::matrixUpdateWeightsBiases<<<grid, block, 0, stream>>>(biases.getData(), learning_rate.getData(), delta_biases.getData(), intr_x, 1);

        // std::cout << "Biases:\n";
        // biases.printData();
        // std::cout << "\n\n\n";
    }

    void updateWeightsSGDmomentum(NDArray<double, 1> sigma, NDArray<double, 1> weights, NDArray<double, 1> learning_rate, NDArray<double, 1> sum_delta_weights, NDArray<double, 1> delta_weights, cudaStream_t stream)
    {
        unsigned intr_x, intr_y;
        dim3 grid, block;
        intr_x = weights.getDimensions()[0];
        intr_y = weights.getDimensions()[1];

        block.x = (intr_x > 32) ? 32 : intr_x;
        block.y = (intr_y > 32) ? 32 : intr_y;

        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);

        gpu::matrixUpdateWeightsBiasesSGDmomentum<<<grid, block, 0, stream>>>(sigma.getData(), learning_rate.getData(), sum_delta_weights.getData(), delta_weights.getData(), weights.getData(), intr_x, intr_y);
    }

    void updateBiasesSGDmomentum(NDArray<double, 1> sigma, NDArray<double, 1> biases, NDArray<double, 1> learning_rate, NDArray<double, 1> sum_delta_biases, NDArray<double, 1> delta_biases, cudaStream_t stream)
    {
        unsigned intr_x;
        dim3 grid, block;
        intr_x = biases.getDimensions()[0];

        block.x = (intr_x > 32) ? 32 : intr_x;

        grid.x = ceil((float)intr_x / block.x);

        gpu::matrixUpdateWeightsBiasesSGDmomentum<<<grid, block, 0, stream>>>(sigma.getData(), learning_rate.getData(), sum_delta_biases.getData(), delta_biases.getData(), biases.getData(), intr_x, 1);
    }

    void updateWeightsRMSpropDense(NDArray<double, 1> sigma, NDArray<double, 1> epsalon, NDArray<double, 1> weights, NDArray<double, 1> learning_rate, NDArray<double, 1> sum_delta_weights, NDArray<double, 1> delta_weights, cudaStream_t stream)
    {
        unsigned intr_x, intr_y;
        dim3 grid, block;
        intr_x = weights.getDimensions()[0];
        intr_y = weights.getDimensions()[1];

        block.x = (intr_x > 32) ? 32 : intr_x;
        block.y = (intr_y > 32) ? 32 : intr_y;

        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);

        gpu::matrixUpdateWeightsBiasesRMSprop<<<grid, block, 0, stream>>>(sigma.getData(), epsalon.getData(), learning_rate.getData(), sum_delta_weights.getData(), delta_weights.getData(), weights.getData(), intr_x, intr_y);
    }

    void updateBiasesRMSpropDense(NDArray<double, 1> sigma, NDArray<double, 1> epsalon, NDArray<double, 1> biases, NDArray<double, 1> learning_rate, NDArray<double, 1> sum_delta_biases, NDArray<double, 1> delta_biases, cudaStream_t stream)
    {
        unsigned intr_x;
        dim3 grid, block;
        intr_x = biases.getDimensions()[0];

        block.x = (intr_x > 32) ? 32 : intr_x;

        grid.x = ceil((float)intr_x / block.x);

        gpu::matrixUpdateWeightsBiasesRMSprop<<<grid, block, 0, stream>>>(sigma.getData(), epsalon.getData(), learning_rate.getData(), sum_delta_biases.getData(), delta_biases.getData(), biases.getData(), intr_x, 1);
    }

    void updateWeightsADAMDense(NDArray<double, 1> sigma, NDArray<double, 1> epsalon, NDArray<double, 1> weights, NDArray<double, 1> learning_rate, NDArray<double, 1> sum_delta_weights, NDArray<double, 1> sum_delta_weights_square, NDArray<double, 1> delta_weights, cudaStream_t stream)
    {
        unsigned intr_x, intr_y;
        dim3 grid, block;
        intr_x = weights.getDimensions()[0];
        intr_y = weights.getDimensions()[1];

        block.x = (intr_x > 32) ? 32 : intr_x;
        block.y = (intr_y > 32) ? 32 : intr_y;

        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);

        gpu::matrixUpdateWeightsBiasesADAM<<<grid, block, 0, stream>>>(sigma.getData(), epsalon.getData(), learning_rate.getData(), sum_delta_weights.getData(), sum_delta_weights_square.getData(), delta_weights.getData(), weights.getData(), intr_x, intr_y);
    }

    void updateBiasesADAMDense(NDArray<double, 1> sigma, NDArray<double, 1> epsalon, NDArray<double, 1> biases, NDArray<double, 1> learning_rate, NDArray<double, 1> sum_delta_biases, NDArray<double, 1> sum_delta_biases_squared, NDArray<double, 1> delta_biases, cudaStream_t stream)
    {
        unsigned intr_x;
        dim3 grid, block;
        intr_x = biases.getDimensions()[0];

        block.x = (intr_x > 32) ? 32 : intr_x;

        grid.x = ceil((float)intr_x / block.x);

        gpu::matrixUpdateWeightsBiasesADAM<<<grid, block, 0, stream>>>(sigma.getData(), epsalon.getData(), learning_rate.getData(), sum_delta_biases.getData(), sum_delta_biases_squared.getData(), delta_biases.getData(), biases.getData(), intr_x, 1);
    }

    void getDifferentialWeights(NDArray<double, 1> input, NDArray<double, 1> delta_output, NDArray<double, 1> difference, NDArray<double, 1> delta_weights, NDArray<double, 1> delta_weights_intermediate, cudaStream_t stream)
    {
        int intr_x, intr_y, intr_z;
        dim3 grid, block;

        // NDArray<double, 1> delta_weights_intermediate;

        intr_x = difference.getDimensions()[0]; // no of output neuron
        intr_y = input.getDimensions()[0];      // no of input feature
        intr_z = input.getDimensions()[1];      // no of batches

        // delta_weights_intermediate = NDArray<double, 1>(3, intr_x, intr_y, intr_z);

        // std::cout << "input:\n";
        // input.printData();
        // std::cout << "delta_output:\n";
        // delta_output.printData();
        // std::cout << "difference:\n";
        // difference.printData();

        block.x = (intr_x > 8) ? 8 : intr_x;
        block.y = (intr_y > 8) ? 8 : intr_y;
        block.z = (intr_z > 16) ? 16 : intr_z;

        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);
        grid.z = ceil((float)intr_z / block.z);

        gpu::matrixDifferentialWeights<<<grid, block, 0, stream>>>(input.getData(), delta_output.getData(), difference.getData(), delta_weights_intermediate.getData(), intr_x, intr_y, intr_z);

        block.z = 1;
        grid.z = 1;

        gpu::matrixRollingSum<<<grid, block, 0, stream>>>(delta_weights_intermediate.getData(), delta_weights.getData(), intr_x, intr_y, intr_z);

        gpu::matrixFindMean<<<grid, block, 0, stream>>>(delta_weights.getData(), intr_x, intr_y, intr_z);
        // delta_weights_intermediate.destroy();

        // std::cout << "Delta weights:\n";
        // delta_weights.printData();
    }

    void getDifferentialBiases(NDArray<double, 1> delta_output, NDArray<double, 1> difference, NDArray<double, 1> delta_biases, NDArray<double, 1> delta_biases_intermediate, cudaStream_t stream)
    {
        int intr_x, intr_y;
        dim3 grid, block;
        // NDArray<double, 1> delta_biases_intermediate;

        intr_x = delta_output.getDimensions()[0]; // no of output neuron
        intr_y = delta_output.getDimensions()[1]; // no of batches

        // delta_biases_intermediate = NDArray<double, 1>(2, intr_x, intr_y);

        block.x = (intr_x > 8) ? 8 : intr_x;
        block.y = (intr_y > 8) ? 8 : intr_y;

        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);

        // std::cout << "Biasest:\n";
        // delta_output.printData();
        // std::cout << "delta_output:\n";
        // delta_output.printData();
        // std::cout << "difference:\n";
        // difference.printData();

        gpu::matrixDifferentialBiases<<<grid, block, 0, stream>>>(delta_output.getData(), difference.getData(), delta_biases_intermediate.getData(), intr_x, intr_y);

        block.y = 1;
        grid.y = 1;

        gpu::matrixRollingSum<<<grid, block, 0, stream>>>(delta_biases_intermediate.getData(), delta_biases.getData(), intr_x, 1, intr_y);

        gpu::matrixFindMean<<<grid, block, 0, stream>>>(delta_biases.getData(), intr_x, 1, intr_y);
        // delta_biases_intermediate.destroy();
        // d_weights_biases.printData();
        // std::cout << "Delta Biases Intr:\n";
        // delta_biases_intermediate.printData();
        // std::cout << "Delta Biases:\n";
        // delta_biases.printData();
    }

    void getDifferentialInput(NDArray<double, 1> weights, NDArray<double, 1> delta_output, NDArray<double, 1> difference, NDArray<double, 1> difference_input, NDArray<double, 1> delta_input_intermediate, NDArray<double, 1> delta_input, cudaStream_t stream)
    {
        // NDArray<double, 1> delta_input_intermediate;
        unsigned intr_x, intr_y, intr_z; // intr_x: no of input + bias, intr_y: batch size, intr_z = no of neuron
        dim3 grid, block;
        intr_x = delta_input.getDimensions()[0]; // no of input feature
        intr_y = delta_input.getDimensions()[1]; // no of batchs
        intr_z = weights.getDimensions()[0];     // no of neuron
        // delta_input_intermediate = NDArray<double, 1>(3, intr_x, intr_y, intr_z);

        block.x = (intr_x > 8) ? 8 : intr_x;
        block.y = (intr_y > 8) ? 8 : intr_y;
        block.z = (intr_y > 16) ? 16 : intr_z;

        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);
        grid.z = ceil((float)intr_z / block.z);

        // std::cout << "\nweights_biases:\n";
        // weights.printData();
        // std::cout << "d_activation:\n";
        // delta_output.printData();
        // std::cout << "difference:\n";
        // difference.printData();

        gpu::matrixDifferentialInput<<<grid, block, 0, stream>>>(weights.getData(), delta_output.getData(), difference.getData(), delta_input_intermediate.getData(), intr_x, intr_y, intr_z);

        block.z = 1;
        grid.z = 1;

        gpu::matrixRollingSum<<<grid, block, 0, stream>>>(delta_input_intermediate.getData(), difference_input.getData(), intr_x, intr_y, intr_z);

        // delta_input_intermediate.destroy();

        // difference_input.initData(delta_input_intermediate.getData());

        // delta_input.initData(delta_input.getData());
        // gpu::matrixFindMean<<<grid, block, 0, stream>>>(delta_input.getData(), intr_x, intr_y, intr_z);

        // std::cout << "delta_input intermediate:\n";
        // delta_input_intermediate.printData();
        // std::cout << "delta_input:\n";
        // difference_input.printData();
    }

    void reluActivation(NDArray<double, 1> input, NDArray<double, 1> d_activation, cudaStream_t stream)
    {
        int intr_x, intr_y;
        dim3 block, grid;

        intr_x = input.getDimensions()[0];
        intr_y = input.getDimensions()[1];

        block.x = (intr_x > 32) ? 32 : intr_x;
        block.y = (intr_y > 32) ? 32 : intr_y;
        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);

        gpu::matrixRelu<<<grid, block, 0, stream>>>(input.getData(), d_activation.getData(), intr_x, intr_y);
    }

    void sigmoidActivation(NDArray<double, 1> input, NDArray<double, 1> d_activation, cudaStream_t stream)
    {
        int intr_x, intr_y;
        dim3 block, grid;
        intr_x = input.getDimensions()[0];
        intr_y = input.getDimensions()[1];

        block.x = ((intr_x + 1) > 32) ? 32 : (intr_x + 1);
        block.y = ((intr_y + 1) > 32) ? 32 : (intr_y + 1);
        grid.x = ceil((float)(intr_x + 1) / block.x);
        grid.y = ceil((float)(intr_y + 1) / block.y);

        gpu::matrixSigmoid<<<grid, block, 0, stream>>>(input.getData(), d_activation.getData(), intr_x, intr_y);
    }

    void linearActivation(NDArray<double, 1> input, NDArray<double, 1> d_activation, cudaStream_t stream)
    {
        int intr_x, intr_y;
        dim3 grid, block;

        intr_x = input.getDimensions()[0];
        intr_y = input.getDimensions()[1];

        block.x = ((intr_x + 1) > 32) ? 32 : (intr_x + 1);
        block.y = ((intr_y + 1) > 32) ? 32 : (intr_y + 1);
        grid.x = ceil((float)(intr_x + 1) / block.x);
        grid.y = ceil((float)(intr_y + 1) / block.y);

        gpu::matrixLinear<<<grid, block, 0, stream>>>(input.getData(), d_activation.getData(), intr_x, intr_y);
    }

    void squaredError(NDArray<double, 1> Difference, NDArray<double, 1> Squared_Error, cudaStream_t stream)
    {
        unsigned intr_x, intr_y;
        dim3 grid, block;

        intr_x = Difference.getDimensions()[0];
        intr_y = Difference.getDimensions()[1];

        block.x = intr_x > 32 ? 32 : intr_x;
        block.y = intr_y > 32 ? 32 : intr_y;
        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);

        gpu::matrixSquaredError<<<grid, block, 0, stream>>>(Difference.getData(), Squared_Error.getData(), intr_x, intr_y);
    }

    void findMean(NDArray<double, 1> X, NDArray<double, 1> y, cudaStream_t stream)
    {
        unsigned intr_x, intr_y;
        dim3 grid, block;
        intr_x = X.getDimensions()[0];
        intr_y = X.getDimensions()[1];

        gpu::matrixRollingSum<<<grid, block, 0, stream>>>(X.getData(), X.getData(), intr_x, 1, intr_y);
        y.initData(X);

        gpu::matrixFindMean<<<grid, block, 0, stream>>>(y.getData(), intr_x, intr_y, intr_y);
    }

    void findDifference(NDArray<double, 1> Y_predict, NDArray<double, 1> Y_target, NDArray<double, 1> Difference, cudaStream_t stream)
    {
        unsigned intr_x, intr_y;
        dim3 grid, block;

        intr_x = Y_target.getDimensions()[0];
        intr_y = Y_target.getDimensions()[1];

        block.x = intr_x > 32 ? 32 : intr_x;
        block.y = intr_y > 32 ? 32 : intr_y;

        grid.x = ceil((float)intr_x / block.x);
        grid.y = ceil((float)intr_y / block.y);

        gpu::matrixDifference<<<grid, block, 0, stream>>>(Y_predict.getData(), Y_target.getData(), Difference.getData(), intr_x, intr_y);

        // std::cout << "Y_target:\n";
        // Y_target.printData();
        // std::cout << "Y_predict:\n";
        // Y_predict.printData();
        // std::cout << "Difference\n";
        // Difference.printData();
    }
};
