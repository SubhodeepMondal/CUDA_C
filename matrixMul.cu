#include<cuda_runtime.h>
#include<stdio.h>

void initilizeData(double *ptr, int size)
{
    time_t t;
    srand((unsigned) time(&t));

    for( int i=0; i<size; i++)
        ptr[i] =(float)(rand() & 0xFF)/100.0f;
}

void matrixMultiplicationOnHost(double *a, double *b, double *c, int n)
{
    int i, j, k;
    double sum ;

    for( i=0; i<n; i++ )
        for( j=0; j<n; j++ )
        {
            sum = 0;
            for( k=0; k<n; k++ )
                sum += a[i*n+k] * b[k*n+j];
            c[i*n+j] = sum;
        }
            
}

__global__ void matrixMultiplication(double *a, double *b, double *c)
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    int n = blockDim.x;
    double sum = 0;

    for(int k = 0; k < n; k++ )
        sum += a[i*n+k] * b[k*n+j];

    c[i*n+j] = sum;
}

int main()
{
    freopen("outputMul.csv","w",stdout);

    cudaSetDevice(0);
    
    int prec,pred;
    scanf("%d%d",&prec,&pred);
    int nElem = 1<<prec;
    int n = 1<<pred;

    double *hA, *hB, *hC, *dRes;
    double *dA, *dB, *dC;
    int byteSize = nElem * sizeof(double);

    hA = (double *) malloc(byteSize);
    hB = (double *) malloc(byteSize);
    hC = (double *) malloc(byteSize);
    dRes = (double *) malloc(byteSize);

    initilizeData(hA,nElem);
    initilizeData(hB,nElem);

    cudaMalloc((double **)&dA,byteSize);
    cudaMalloc((double **)&dB,byteSize);
    cudaMalloc((double **)&dC,byteSize);

    cudaMemcpy(dA,hA,byteSize,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,byteSize,cudaMemcpyHostToDevice);

    dim3 block(n,n);
    dim3 grid(1);
    
    matrixMultiplication<<<grid,block>>>(dA,dB,dC);
    matrixMultiplicationOnHost(hA, hB, hC, n);

    cudaMemcpy(dRes,dC,byteSize,cudaMemcpyDeviceToHost);

/*    for( int i=0; i<n; i++ )
    {
        for( int j=0; j<n; j++)
        {
            printf("%2.2lf,",hA[i*n+j]);
        }
        printf("\n");
    }
    printf("\n");

    for( int i=0; i<n; i++ )
    {
        for( int j=0; j<n; j++)
        {
            printf("%2.2lf,",hB[i*n+j]);
        }
        printf("\n");
    }
    printf("\n");*/
/*
    for( int i=0; i<n; i++ )
    {
        for( int j=0; j<n; j++)
        {
            printf("%lf,",hC[i*n+j]);
        }
        printf("\n");
    }*/
    printf("\n");

    for( int i=0; i<n; i++ )
    {
        for( int j=0; j<n; j++)
        {
            printf("%lf,",dRes[i*n+j]);
        }
        printf("\n");
    }
        
}