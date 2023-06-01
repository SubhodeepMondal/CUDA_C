#include <iostream>
#include "cudalibrary.h"
#include "cuda.h"

int main()
{
    int noOfDevice, deviceId;
    int a_m, a_n, b_m, b_n;

    freopen("io/input.txt", "r", stdin);
    freopen("io/cudaTest.csv", "w", stdout);

    std::cin >> a_m >> a_n >> b_m >> b_n;

    NDArray<double> a(2, a_m, a_n);
    NDArray<double> b(2, b_m, b_n);
    NDArray<double> c(2, a_m, b_n);
    NDMath math;

    cudaDeviceProp devProp;
    cudaGetDeviceCount(&noOfDevice);
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&devProp, deviceId);

    std::cout << noOfDevice << std::endl;
    std::cout << deviceId << std::endl;
    std::cout << "Device name: " << devProp.name << std::endl;
    std::cout << "Device major: " << devProp.major << std::endl;
    std::cout << "Device minor: " << devProp.minor << std::endl;
    std::cout << "Memory size: " << devProp.totalGlobalMem / (1024 * 1024) << std::endl;
    std::cout << "No of Streaming multiprocessor: " << devProp.multiProcessorCount << std::endl;
    std::cout << "No of cuda cores: " << 128 * devProp.multiProcessorCount << std::endl;
    std::cout << "Amount of Shared memory per block: " << devProp.sharedMemPerBlock << std::endl;
    std::cout << "No of registers per block: " << devProp.regsPerBlock << std::endl;
    std::cout << "Wrap size: " << devProp.warpSize << std::endl;

    std::cout << a.nElem << ", ";
    a.printDimentions();

    std::cout << b.nElem << ", ";
    b.printDimentions();

    a.initRandData();
    b.initRandData();

    // a.printData();
    std::cout << std::endl;
    // b.printData();

    c = math.multiplication(a, b, 0);
    // c.printData();

    
}