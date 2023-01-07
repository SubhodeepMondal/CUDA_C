#include<iostream>
#include"cudalibrary.h"

int main(){
    freopen("io/cudamulMultithreaded.csv","w",stdout);
    freopen("io/input.txt","r",stdin);

    double *a, *b, *c, *d;
    int n;

    std::cin>>n;
    int nElem = 1 << (n*2);
    n=(int)sqrt(nElem);


    cudaSetDevice(0);

    cudaMallocManaged((double**)&a,nElem*sizeof(double));
    cudaMallocManaged((double**)&b,nElem*sizeof(double));
    cudaMallocManaged((double**)&c,nElem*sizeof(double));
    cudaMallocManaged((double**)&d,nElem*sqrt(nElem)*sizeof(double));

    initilizeData(a,nElem);
    initilizeData(b,nElem);

    dim3 block(16,16);
    dim3 grid((int)sqrt(nElem/(block.x * block.y)),(int)sqrt(nElem/(block.x * block.y)));

    std::cout<<(int)pow(n,2) << " " << std::endl;
    std::cout <<block.x <<" " <<block.y <<" " <<grid.x <<" " <<grid.y<< std::endl;

    cudaMatrixMulMultiParallesied<<<grid,block>>>(a,b,c,d);
    cudaDeviceSynchronize();

    

    for(int i =0; i<n; i++){
        for(int j=0; j<n; j++)
            std::cout << a[(i * n + j)] << ",";
        std::cout << std::endl;
    }
    std::cout << std::endl<<std::endl;


    for(int i =0; i<n; i++){
        for(int j=0; j<n; j++)
            std::cout << b[(i * n + j)] << ",";
        std::cout << std::endl;
    }
    std::cout << std::endl<<std::endl;


    double sum =0;
    for (int i = 0; i<n; i++){
        for(int j =0; j<n;j++){
            sum =0;
            for(int k=0; k<n; k++){
                sum += a[i*n+k] * b[k*n+j];
            }
            c[i*n+j] =sum;
            std::cout << c[i*n+j] << ",";
        }
        std::cout << std::endl;
    }


    std:: cout << std::endl;
    for(int i =0; i<n; i++){
        for(int j=0; j<n; j++)
            std::cout << d[(i * n + j*n*n)] << ",";
        std::cout << std::endl;
    }
    std::cout << std::endl<<std::endl;
}