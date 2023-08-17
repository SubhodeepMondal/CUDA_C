#pragma ONCE

typedef struct struct_Optimizer
{
    enum optimizers
    {
        sgd,
        adadelta,
        adafactor,
        adagrad,
        adam,
        adamw,
        ftrl,
        nadam,
        rmsprop
    };
    std::map<std::string, optimizers> optimizer;

    struct_Optimizer()
    {
        optimizer["sgd"] = sgd;
        optimizer["adadelta"] = adadelta;
        optimizer["adam"] = adam;
        optimizer["rmsprop"] = rmsprop;
        optimizer["nadam"] = nadam;
    }

} struct_Optimizer;

class Optimizer
{
protected:
    /* data */
    NDArray<double, 1> delta_weights;
    NDArray<double, 1> delta_biases;
    NDArray<double, 1> delta_input;
    NDMath math;

public:
    virtual void optimize(NDArray<double, 1> input, NDArray<double, 1> weights, NDArray<double, 1> biases, NDArray<double, 1> delta_activation, NDArray<double, 1> difference, NDArray<double, 1> difference_input, cudaStream_t stream) {}
};

class SGD : public Optimizer
{
    unsigned *arr;
    double learning_rate = 0.001;

public:
    void optimize(NDArray<double, 1> input, NDArray<double, 1> weights, NDArray<double, 1> biases, NDArray<double, 1> delta_activation,  NDArray<double, 1> difference, NDArray<double, 1> difference_input, cudaStream_t stream)
    {
        delta_weights = NDArray<double, 1>(weights.getNoOfDimensions(), weights.getDimensions());
        delta_biases  = NDArray<double, 1>(biases.getNoOfDimensions(), biases.getDimensions());
        delta_input  = NDArray<double, 1>(input.getNoOfDimensions(), input.getDimensions());

        // std::cout << "Input:\n";
        // input.printData();
        // std::cout << "Weights:\n";
        // weights.printData();
        // std::cout << "Biases:\n";
        // biases.printData();
        // std::cout << "Difference:\n";
        // difference.printData();

        math.getDifferentialBiases(delta_activation, difference, delta_biases, stream);
        math.getDifferentialWeights(input, delta_activation, difference, delta_weights, stream);
        math.getDifferentialInput(weights, delta_activation, difference, delta_input, stream);

        // std::cout << "Delta Input:\n";
        // delta_input.printData();
        // std::cout << "Delta Weights:\n";
        // delta_weights.printData();
        // std::cout << "Delta Biases:\n";
        // delta_biases.printData();

        math.updateBiases(biases, delta_biases, learning_rate, stream);
        math.updateWeights(weights, delta_weights, learning_rate, stream);
        math.updateDifferentialInput(input,delta_input,difference_input,learning_rate,stream);

        // std::cout << "updated Weights:\n";
        // weights.printData();
        // std::cout << "updated biases:\n";
        // biases.printData();
        // std::cout << "Difference propagation:\n";
        // difference_input.printData();
    }
};