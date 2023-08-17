#include<iostream>
#include<random>

int main()
{
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, 100);
    for(int i = 0; i< 50; i++)
        std::cout << distribution(generator) << "\n";
}