#include<cuda_runtime.h>


void initilizeData(double *ptr, int size)
{
    time_t t;
    srand((unsigned) time(&t));

    for( int i=0; i<size; i++)
        ptr[i] =(float)(rand() & 0xFF)/100.0f;
}

__global__ void cudaTranspose(unsigned int *a, unsigned int *b, int xsize,int ysize){
    int ix,iy, mat_in,mat_tra;


    __shared__ unsigned int smallblock[32][32];

    ix = blockDim.x * blockIdx.x + threadIdx.x ;
    iy = blockDim.y * blockIdx.y + threadIdx.y ;

    mat_in = ix * xsize + iy;

    int bidx, icol, irow;
    bidx = threadIdx.y * blockDim.x + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    mat_tra = iy * ysize + ix;

        smallblock[threadIdx.x][threadIdx.y] =mat_in;
        __syncthreads();
        b[mat_tra] = smallblock[icol][irow];
}

 __global__ void cudaDotMul(double *a,double *b, double *c,int x,int y){
    int n = blockDim.x;
    int ind,i,k ;
    double val;

    ind = (x * n + threadIdx.x) + (y * n * n);
    
    c[ind] = a[threadIdx.x+x*n] * b[y+threadIdx.x*n];
    __syncthreads();

    k=n;
    for(i =n/2;i>0; i/=2){
        val = c[ind];
        val = __shfl_down_sync(k,val,i);
        c[ind] += val;
        k /=2;
    }

}

__global__ void cudaMatrixMulMultiParallesied(double *a,double *b,double *c, double *d){
    int ix,iy,rowDim;
    ix = threadIdx.x + blockIdx.x * blockDim.x;
    iy = threadIdx.y + blockIdx.y * blockDim.y;
    rowDim = blockDim.x * gridDim.x;

    cudaDotMul<<<1,rowDim>>>(a,b,d,ix,iy);

}

__global__ void cudaRollingSum(double *a){
    int n = blockDim.x;
    int ind,i,k ;
    double val;
    ind = threadIdx.x;

    printf("%d, %lf\n",threadIdx.x, a[ind]);
    k=n;
    for(i =n/2;i>0; i/=2){
        val =  a[ind];
        val =  __shfl_down_sync(k,val,i);
        a[ind] += val;
        if(ind<i)
            printf("%d, %lf\n",threadIdx.x, a[ind]);
        k /=2;

    }

}