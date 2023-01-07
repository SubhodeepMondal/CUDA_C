#include<iostream>
#include"cudalibrary.h"

int main(){

    freopen("cudarollingsum.csv","w",stdout);
    double *a,b;
    int i,n=16;

    cudaSetDevice(0);

    cudaMallocManaged((double**)&a, n * sizeof(double));
    cudaMallocManaged((double**)&b, n * sizeof(double));

    for (i=0;i<n;i++)
        a[i] =(double) 13*i/32;

    b = 0;
    for (i=0;i<n;i++)
        b += a[i];
    

    //dim3 grid,block;
    dim3 block(n);
    dim3 grid(1,1);
    
    cudaRollingSum<<<grid,block>>>(a);
    cudaDeviceSynchronize();

    std::cout << b << ", ";
    
    std::cout << a[0] << ",";
}