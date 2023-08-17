#include <iostream>
#include "deeplearning.h"

void generatorFunction(NDArray<double, 0> x, NDArray<double, 0> y, unsigned no_of_feature, unsigned no_of_sample)
{
    unsigned i;
    float angle;
    double *ptrA = x.getData();
    double *ptrB = y.getData();

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0f, 1.0f);
    for (i = 0; i < no_of_sample; i++)
    {
        angle = distribution(generator);
        // ptrA[i] = angle * (3.14159 / 180.0f);
        // ptrB[i] = sin(ptrA[i]);
        ptrA[i] = angle;
        ptrB[i] = 3.5 * angle + 1.25;
    }
}

int main()
{

    freopen("io/DLearning.csv", "w", stdout);
    unsigned no_of_feature, no_of_sample;
    unsigned a[1], b[1];
    no_of_feature = 1;
    no_of_sample = 100;
    a[0] = no_of_feature;
    b[0] = 16;

    NDArray<double, 1> input_shape(1, a);
    NDArray<unsigned, 0> intermediate_shape(1, b);
    NDArray<double, 0> X_train(2, 1, 100);
    NDArray<double, 0> y_train(2, 1, 100);
    NDArray<double, 0> X_test(2, 1, 250);
    NDArray<double, 0> y_test(2, 1, 250);

    generatorFunction(X_train, y_train, no_of_feature, no_of_sample);
    generatorFunction(X_test, y_test, 1, 250); 

    Model *model = new Sequential();
    // model->add(new Dense{16, input_shape, "relu", "Dense_1"});
    // model->add(new Dense{8, "relu", "Dense_2"});
    model->add(new Dense{1, input_shape, "linear", "Dense_1"});
    model->compile("mean_squared_error", "sgd", "accuracy");
    model->summary();
    model->fit(X_train, y_train, 10000, 32);
}