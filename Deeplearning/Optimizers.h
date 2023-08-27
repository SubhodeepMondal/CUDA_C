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
    NDArray<double, 1> delta_weights_intermediate;
    NDArray<double, 1> delta_biases_intermediate;
    NDArray<double, 1> delta_input_intermediate;

public:
    SGD() {}
    SGD(NDArray<double, 1> weights, NDArray<double, 1> biases, NDArray<double, 1> input)
    {
        unsigned neurons, features, batches;

        neurons = weights.getDimensions()[0];
        features = input.getDimensions()[0];
        batches = input.getDimensions()[1];

        delta_weights = NDArray<double, 1>(weights.getNoOfDimensions(), weights.getDimensions());
        delta_biases = NDArray<double, 1>(biases.getNoOfDimensions(), biases.getDimensions());
        delta_input = NDArray<double, 1>(input.getNoOfDimensions(), input.getDimensions());

        delta_biases_intermediate = NDArray<double, 1>(2, neurons, batches);
        delta_weights_intermediate = NDArray<double, 1>(3, neurons, features, batches);
        delta_biases_intermediate = NDArray<double, 1>(3, features, batches,neurons);
    }
    void optimize(NDArray<double, 1> input, NDArray<double, 1> weights, NDArray<double, 1> biases, NDArray<double, 1> delta_activation, NDArray<double, 1> difference, NDArray<double, 1> difference_input, cudaStream_t stream)
    {

        // std::cout << "Input:\n";
        // input.printData();
        // std::cout << "Weights:" << weights.getData() << "\n";
        // weights.printData();
        // std::cout << "Biases:\n";
        // biases.printData();
        // std::cout << "Difference:\n";
        // difference.printData();

        math.getDifferentialBiases(delta_activation, difference, delta_biases, delta_biases_intermediate, stream);
        math.getDifferentialWeights(input, delta_activation, difference, delta_weights, delta_weights_intermediate, stream);
        math.getDifferentialInput(weights, delta_activation, difference, difference_input, delta_biases_intermediate, delta_input, stream);

        // std::cout << "Delta Input:\n";
        // delta_input.printData();
        // std::cout << "Delta Weights:\n";
        // delta_weights.printData();
        // std::cout << "Delta Biases:\n";
        // delta_biases.printData();

        math.updateBiases(biases, delta_biases, learning_rate, stream);
        math.updateWeights(weights, delta_weights, learning_rate, stream);
        // math.updateDifferentialInput(input, delta_input, difference_input, learning_rate, stream);

        // std::cout << "updated Weights:\n";
        // weights.printData();
        // std::cout << "updated biases:\n";
        // biases.printData();
        // std::cout << "Difference propagation:\n";
        // difference_input.printData();
    }
};