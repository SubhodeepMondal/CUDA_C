#include <cuda_runtime.h>
#include <iostream>

void initializeData(double *a,int size){
    time_t t;
    srand((unsigned) time(&t));
    for(int i =0;i<size;i++){
        a[i] = (double) (rand() & 0xFF)/10.0;
    }
}

void initializeData(float *ip, int size)
{
    time_t t;
    srand((unsigned) time(&t));

    for(int i=0; i<size; i++)
    {
        ip[i] = (float)(rand() & 0xFF )/10.0f;
    }
}

__global__ void cudaMatrixSum(double * a, double * b, double * c){
    int i;
    i = threadIdx.x * blockDim.x + threadIdx.y;
    //printf("Index: %d  threadIdx.x : %d threadIdx.y : %d blockdim.x : %d \n",i,threadIdx.x , threadIdx.y,blockDim.x );
    c[i] = a[i] + b[i];
}

int main(){

    freopen("outputSum.csv","w",stdout);
    int nElem, xdim,ydim;
    double *a, *b, *c,*d_a,*d_b,*d_c;

    nElem  = 1<<14;
    xdim = ydim = 1<<7;

    cudaSetDevice(0);
    std::cout << xdim << "," <<ydim << std::endl;

    cudaMalloc(&d_a, nElem * sizeof(double));
    cudaMalloc(&d_b, nElem * sizeof(double));
    cudaMalloc(&d_c, nElem * sizeof(double));

    a = (double *)malloc(nElem * sizeof(double));
    b = (double *)malloc(nElem * sizeof(double));
    c = (double *)malloc(nElem * sizeof(double));

    initializeData(a,nElem);
    initializeData(b,nElem);

    for(int i = 0; i<xdim;i++){
        for(int j=0;j<ydim;j++){
            std::cout << a[i*xdim + j] << "," ;
        }
        std::cout << std::endl;
    }

    for(int i = 0; i<xdim;i++){
        for(int j=0;j<ydim;j++){
            std::cout << b[i*xdim + j] << "," ;
        }

        std::cout << std::endl;
    }

    cudaMemcpy(d_a,a,nElem * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,nElem * sizeof(double),cudaMemcpyHostToDevice);

    dim3 block(xdim,ydim);
    dim3 grid(1);
    
    cudaMatrixSum<<<grid,block>>>(d_a,d_b,d_c);
    cudaDeviceSynchronize();

    cudaMemcpy(c,d_c,nElem * sizeof(double),cudaMemcpyDeviceToHost);
    
    for(int i = 0; i<xdim;i++){
        for(int j=0;j<ydim;j++){
            std::cout << c[i*xdim + j] << "," ;
        }
        std::cout << std::endl;
    }

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}