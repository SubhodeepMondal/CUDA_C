#include <iostream>
#include "deeplearning.h"

void generatorFunction1(NDArray<double, 0> x, NDArray<double, 0> y, unsigned no_of_feature, unsigned no_of_sample)
{
    unsigned i, j, lin_index;
    float angle;
    double *ptrA = x.getData();
    double *ptrB = y.getData();

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0f, 1.0f);
    for (i = 0; i < no_of_sample; i++)
    {
        for (j = 0; j < no_of_feature; j++)
        {
            lin_index = j + i * no_of_feature;
            angle = distribution(generator);
            // ptrA[i] = angle * (3.14159 / 180.0f);
            // ptrB[i] = sin(ptrA[i]);
            ptrA[lin_index] = angle;
            switch (j)
            {
            case 0:
                ptrB[i] = 1.5 * angle + 0.25;
                break;
            case 1:
                ptrB[i] += 0.75 * angle;
                break;
            case 2:
                ptrB[i] += 0.25 * angle;
                break;
            case 3:
                ptrB[i] += 1.75 * angle;
                break;
            case 4:
                ptrB[i] += 2.35 * angle;
                break;
            case 5:
                ptrB[i] += 3.05 * angle;
                break;
            case 6:
                ptrB[i] += 1.12 * angle;
                break;
            case 7:
                ptrB[i] += 4.012 * angle;
                break;

            default:
                break;
            }
        }
    }
}

void generatorFunctionSinWave(NDArray<double, 0> x, NDArray<double, 0> y)
{
    unsigned i;
    double angle;
    unsigned no_of_samples;
    double *ptrA, *ptrB;

    no_of_samples = x.getDimensions()[1];
    ptrA = x.getData();
    ptrB = y.getData();

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0f, 180.0f);

    for (i = 0; i < no_of_samples; i++)
    {
        angle = distribution(generator);
        ptrA[i] = angle * (3.14159 / 180.0f);
        ptrB[i] =  sin(ptrA[i]) * sin(ptrA[i] - 0.45) * sin(ptrA[i] + 1.25 );
    }
}

int main()
{
    NDMath math;
    freopen("io/input.txt", "r", stdin);
    freopen("io/DLearning.csv", "w", stdout);
    unsigned no_of_feature, no_of_sample, epochs, batch_size;
    unsigned a[1];
    no_of_feature = 1;
    no_of_sample = 10000;
    a[0] = no_of_feature;

    std::cin >> epochs;
    std::cin >> batch_size;

    NDArray<double, 0> input_shape(1, a);
    NDArray<double, 0> X_train(2, no_of_feature, no_of_sample);
    NDArray<double, 0> y_train(2, 1, no_of_sample);
    NDArray<double, 0> X_test(2, no_of_feature, 100);
    NDArray<double, 0> y_test(2, 1, 100);
    NDArray<double, 0> y_predict;


    generatorFunctionSinWave(X_train, y_train);
    generatorFunctionSinWave(X_test, y_test);


    Model *model = new Sequential();
    model->add(new Dense{32, input_shape, "relu", "Dense_1"});
    model->add(new Dense{32, "relu", "Dense_2"});
    model->add(new Dense{1, "softmax", "Dense_3"});
    model->compile("mean_squared_error", "ADAM", "accuracy");
    model->summary();

    model->fit(X_train, y_train, epochs, batch_size);

    y_predict = model->predict(X_test);
    NDArray<double, 0> RMS = math.findSquareRoot(math.findSquare(math.findDifference(y_predict,y_test)));
    std::cout << " " << math.findMean(RMS);
}