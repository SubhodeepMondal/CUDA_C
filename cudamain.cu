#include<iostream>
#include "cudalibrary.h"
#define xsize  64
#define ysize  64

int main(){
    freopen("outputmain.csv","w",stdout);
    unsigned int *a,*b, i,j,val,index;

    val = 1;

    cudaMallocManaged((void **)&a,sizeof(unsigned int)* xsize * ysize);
    cudaMallocManaged((void **)&b,sizeof(unsigned int)* xsize * ysize);

    for(i=0;i<  xsize ;i++){
        for(j=0;j<ysize; j++){
            index = i * xsize + j ;
            a[index] = val ++ ;
        }
    }

    for(i=0;i<xsize;i++){
        for(j=0;j<ysize;j++){
            index = i * xsize + j;
            std::cout << a[index] << " ";
        }
        std::cout << std::endl;
    }

    dim3 block(32,32);
    dim3 grid(2,2);

    cudaTranspose<<<grid,block>>>(a,b,xsize,ysize);
    cudaDeviceSynchronize();


    std::cout << std::endl;
    std::cout << std::endl;

    for(i=0;i<xsize;i++){
        for(j=0;j<ysize;j++){
            index = i * xsize + j;
            std::cout << b[index] << " ";
        }
        std::cout << std::endl;
    }
}