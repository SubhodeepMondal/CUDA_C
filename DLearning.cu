#include <iostream>
#include "deeplearning.h"

void generatorFunction(NDArray<double,0> x,NDArray<double,0> y, unsigned no_of_feature, unsigned no_of_sample)
{
    unsigned i;
    float angle;
    double * ptrA = x.getData();
    double *ptrB = y.getData();

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(1.0f, 360.0f);
    for(i =0; i< no_of_sample; i++)
    {
        angle = distribution(generator);
        ptrA[i] = angle * (3.14159/180.0f);
        ptrB[i] = sin(ptrA[i]);
    }
}

int main()
{

    freopen("io/DLearning.csv", "w", stdout);
    unsigned no_of_feature, no_of_sample;
    unsigned a[1],b[1];
    no_of_feature = 1;
    no_of_sample = 1000;
    a[0] = no_of_feature;
    b[0] = 16;


    NDArray<unsigned,0> input_shape(1, a);
    NDArray<unsigned,0> intermediate_shape(1, b);
    NDArray<double,0> X_train(2,1,1000);
    NDArray<double,0> y_train(2,1,1000);

    generatorFunction(X_train,y_train,no_of_feature,no_of_sample);
    // input_shape.initData(a);

    // Dense dense_layer(16, input_shape, "relu");
    // Dense dense_layer2(16, "relu");
    // Dense dense_output(1, "sigmoid");


    Model<NDArray<double,0>> model("Sequential");
    model.add(Dense{16,input_shape,"sigmoid"});
    model.add(Dense{16,intermediate_shape,"sigmoid"});
    model.add(Dense{1,intermediate_shape,"sigmoid"});
    model.compile("mean_squared_error", "adam", "accuracy");
    model.summary();
    model.fit(X_train, y_train, 10, 1);
}
