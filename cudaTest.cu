#include <iostream>
#include "cudalibrary.h"

int main()
{

    cudaSetDevice(0);
    int a_m, a_n, b_m, b_n;
    freopen("io/input.txt", "r", stdin);
    freopen("io/cudaTest.csv", "w", stdout);
    std::cin >> a_m >> a_n >> b_m >> b_n;

    NDArray<double> a(2, a_m, a_n);
    NDArray<double> b(2, b_m, b_n);
    NDArray<double> c(2, a_m, b_n);
    NDMath math;

    std::cout << a.nElem << ", ";
    a.printDimentions();

    std::cout << a.nElem << ", ";
    b.printDimentions();

    a.initRandData();
    b.initRandData();

    // a.printData();
    // b.printData();

    c = math.multiplication(a, b);

    // c.printData();
}