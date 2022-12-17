#include<cuda_runtime.h>
#include<stdio.h>

void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned) time(&t));

    for(int i=0; i<size; i++)
    {
        ip[i] = (float)(rand() & 0xFF )/10.0f;
    }
}

__global__ void sumArrayOnGPU(float *A, float *B, float *C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main(int argc, char **argv)
{
    int dev = 0;
    cudaSetDevice(dev);

    int nElem = 32;

    int byteSize = nElem * sizeof(float);

    printf("Vector size : %d\n",nElem);

    float *h_A, *h_B, *h_C;

    h_A = (float *) malloc(nElem*sizeof(float));
    h_B = (float *) malloc(nElem*sizeof(float));
    h_C = (float *) malloc(nElem*sizeof(float));

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(h_C,0,byteSize);

    float *d_A, *d_B, *d_C;

    cudaMalloc((float **)&d_A, byteSize);
    cudaMalloc((float **)&d_B, byteSize);
    cudaMalloc((float **)&d_C, byteSize);

    cudaMemcpy(d_A, h_A, byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_A, byteSize, cudaMemcpyHostToDevice);

    dim3 block (nElem);
    dim3 grid (nElem/block.x);

    sumArrayOnGPU<<<grid,block >>> (d_A,d_B,d_C);

    cudaMemcpy(h_C,d_C, byteSize, cudaMemcpyDeviceToHost);

    printf("Vector A:\n");
    for(int i=0; i<nElem ; i++)
        printf("%2.2f\n",h_A[i]);
    
    printf("Vector B:\n");
    for(int i=0; i<nElem ; i++)
        printf("%2.2f\n",h_B[i]);

    printf("Vector C:\n");
    for(int i=0; i<nElem ; i++)
        printf("%2.2f\n",h_C[i]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
}