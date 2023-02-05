#include <iostream>
#include "cudalibrary.h"

int main()
{
    freopen("io/cudamulMultithreaded.csv", "w", stdout);
    freopen("io/input.txt", "r", stdin);

    double *a, *b, *d, *e;
    int a_m, a_n, b_m, b_n;
    int aElem, bElem, nElem;

    std::cin >> a_m >> a_n >> b_m >> b_n;

    if (a_n == b_m)
    {
        aElem = a_m * a_n;
        bElem = a_n * b_n;
        nElem = a_m * b_n;

        cudaMallocManaged((double **)&a, aElem * sizeof(double));
        cudaMallocManaged((double **)&b, bElem * sizeof(double));
        cudaMallocManaged((double **)&d, nElem * sizeof(double));
        cudaMallocManaged((double **)&e, nElem * sizeof(double));

        initilizeData(a, aElem);
        initilizeData(b, bElem);

        // for (int i = 0; i < a_m; i++)
        // {
        //     for (int j = 0; j < a_n; j++)
        //     {
        //         std::cout << a[j + i * a_n] << ", ";
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << std::endl;

        // for (int i = 0; i < a_n; i++)
        // {
        //     for (int j = 0; j < b_n; j++)
        //     {
        //         std::cout << b[j + i * b_n] << ", ";
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << std::endl;

        matrixMultiplication(a, b, e, a_m, a_n, b_m, b_n);
        std::cout << std::endl
                  << std::endl;

        double sum = 0;
        // for (int i = 0; i < a_m; i++)
        // {
        //     for (int j = 0; j < b_n; j++)
        //     {
        //         sum = 0;
        //         for (int k = 0; k < a_n; k++)
        //         {
        //             sum += a[i * a_n + k] * b[k * b_n + j];
        //         }
        //         d[i * b_n + j] = sum;
        //         // std::cout << d[i * b_n + j] << ",";
        //     }
        //     // std::cout << std::endl;
        // }

        // std::cout << std::endl;
        // for (int i = 0; i < a_m; i++)
        // {
        //     for (int j = 0; j < b_n; j++)
        //         std::cout << e[i * b_n + j] << ",";
        //     std::cout << std::endl;
        // }
    }
    else
    {
        std::cout << "Two metrices are not in proper order";
    }

    // checkMultiplication(d, e, a_m, b_n);
}
