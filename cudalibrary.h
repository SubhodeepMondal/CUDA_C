#include<cuda_runtime.h>


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