#include<cuda_runtime.h>
#include<iostream>

void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned) time(&t));

    for(int i=0; i<size; i++)
    {
        ip[i] = (float)(rand() & 0xFF )/10.0f;
    }
}
__global__ void matrixSumOnGPUUnifiedMemory(float *a,float *b, float *c)
{
    int i;
    i = blockDim.x * threadIdx.x + threadIdx.y;
    //printf("Index: %d  threadIdx.x : %d threadIdx.y : %d blockdim.x : %d \n",i,threadIdx.x , threadIdx.y,blockDim.x );
    c[i] = a[i] + b[i];
}

int main()
{
    freopen("outputSumUnified.csv","w",stdout);

    int nElement, xdim,ydim;
    float *a, *b, *c;



    nElement = 1<<12;
    xdim = ydim = 1<<6;

    cudaMallocManaged(&a, nElement * sizeof(float));
    cudaMallocManaged(&b, nElement * sizeof(float));
    cudaMallocManaged(&c, nElement * sizeof(float));

    initialData(a,nElement);
    initialData(b,nElement);


    std::cout << nElement << "," << xdim << "," << ydim << std::endl;
    std::cout << "A:" << std::endl;
    for(int i=0; i<xdim;i++){
        for(int j=0;j<ydim;j++){
            std::cout <<  a[i*xdim+j] << ",";
        }
        std::cout << std::endl;
    }
    std::cout << "B:" << std::endl;
    for(int i=0; i<xdim;i++){
        for(int j=0;j<ydim;j++){
            std::cout <<  b[i*xdim+j] << ",";
        }
        std::cout << std::endl;
    }
    dim3 block(xdim,ydim);
    dim3 grid(1);

    matrixSumOnGPUUnifiedMemory<<<grid,block>>>(a,b,c);
    cudaDeviceSynchronize();

    std::cout << "C:" << std::endl;
    for(int i=0; i<xdim;i++){
        for(int j=0;j<ydim;j++){
            std::cout <<  c[i*xdim+j] << ",";
        }
        std::cout << std::endl;
    }
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

}