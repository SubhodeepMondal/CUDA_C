#pragma ONCE

typedef struct struct_Optimizer
{
    enum optimizers
    {
        sgd,
        sgd_momentum,
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
        optimizer["sgd_momentum"] = sgd_momentum;
        optimizer["adagrad"] = adagrad;
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
    NDArray<double, 1> learning_rate_weights;
    NDArray<double, 1> learning_rate_biases;
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

        learning_rate_weights = NDArray<double, 1>(weights.getNoOfDimensions(), weights.getDimensions());
        learning_rate_biases = NDArray<double, 1>(biases.getNoOfDimensions(), biases.getDimensions());

        learning_rate_biases.initData(learning_rate);
        learning_rate_weights.initData(learning_rate);

        // learning_rate_weights.printData();
        // learning_rate_biases.printData();

        delta_biases_intermediate = NDArray<double, 1>(2, neurons, batches);
        delta_weights_intermediate = NDArray<double, 1>(3, neurons, features, batches);
        delta_biases_intermediate = NDArray<double, 1>(3, features, batches, neurons);
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

        math.updateBiases(biases, learning_rate_biases, delta_biases, stream);
        math.updateWeights(weights, learning_rate_weights, delta_weights, stream);
        // math.updateDifferentialInput(input, delta_input, difference_input, learning_rate, stream);

        // std::cout << "updated Weights:\n";
        // weights.printData();
        // std::cout << "updated biases:\n";
        // biases.printData();
        // std::cout << "Difference propagation:\n";
        // difference_input.printData();
    }
};

class Adagrad : public Optimizer
{
    unsigned *arr;
    double learning_rate = 0.01;
    NDArray<double, 1> learning_rate_weights;
    NDArray<double, 1> learning_rate_biases;
    NDArray<double, 1> sum_delta_weights;
    NDArray<double, 1> sum_delta_biases;
    NDArray<double, 1> delta_weights_intermediate;
    NDArray<double, 1> delta_biases_intermediate;
    NDArray<double, 1> delta_input_intermediate;
    NDArray<double, 1> epsalon = NDArray<double, 1>(1, 1);

public:
    Adagrad() {}
    Adagrad(NDArray<double, 1> weights, NDArray<double, 1> biases, NDArray<double, 1> input)
    {
        unsigned neurons, features, batches;

        neurons = weights.getDimensions()[0];
        features = input.getDimensions()[0];
        batches = input.getDimensions()[1];
        double e = 1.1025;

        epsalon.initData(&e);

        delta_weights = NDArray<double, 1>(weights.getNoOfDimensions(), weights.getDimensions());
        delta_biases = NDArray<double, 1>(biases.getNoOfDimensions(), biases.getDimensions());
        delta_input = NDArray<double, 1>(input.getNoOfDimensions(), input.getDimensions());

        sum_delta_biases = NDArray<double, 1>(biases.getNoOfDimensions(), biases.getDimensions());
        sum_delta_weights = NDArray<double, 1>(weights.getNoOfDimensions(), weights.getDimensions());

        learning_rate_biases = NDArray<double, 1>(biases.getNoOfDimensions(), biases.getDimensions());
        learning_rate_weights = NDArray<double, 1>(weights.getNoOfDimensions(), weights.getDimensions());

        learning_rate_biases.initData(learning_rate);
        learning_rate_weights.initData(learning_rate);

        delta_biases_intermediate = NDArray<double, 1>(2, neurons, batches);
        delta_weights_intermediate = NDArray<double, 1>(3, neurons, features, batches);
        delta_biases_intermediate = NDArray<double, 1>(3, features, batches, neurons);
    }
    void optimize(NDArray<double, 1> input, NDArray<double, 1> weights, NDArray<double, 1> biases, NDArray<double, 1> delta_activation, NDArray<double, 1> difference, NDArray<double, 1> difference_input, cudaStream_t stream)
    {
        math.getDifferentialBiases(delta_activation, difference, delta_biases, delta_biases_intermediate, stream);
        math.getDifferentialWeights(input, delta_activation, difference, delta_weights, delta_weights_intermediate, stream);
        math.getDifferentialInput(weights, delta_activation, difference, difference_input, delta_biases_intermediate, delta_input, stream);

        math.updateLearningRateBiasesAdagrad(epsalon, sum_delta_biases, delta_biases, learning_rate_biases, stream);
        math.updateLearningRateWeightsAdagrad(epsalon, sum_delta_weights, delta_weights, learning_rate_weights, stream);

        math.updateBiases(biases, learning_rate_biases, delta_biases, stream);
        math.updateWeights(weights, learning_rate_weights, delta_weights, stream);
    }
};

class Adadelta : public Optimizer
{
    unsigned *arr;
    double learning_rate = 0.01;
    NDArray<double, 1> learning_rate_weights;
    NDArray<double, 1> learning_rate_biases;
    NDArray<double, 1> sum_delta_weights;
    NDArray<double, 1> sum_delta_biases;
    NDArray<double, 1> delta_weights_intermediate;
    NDArray<double, 1> delta_biases_intermediate;
    NDArray<double, 1> delta_input_intermediate;
    NDArray<double, 1> epsalon = NDArray<double, 1>(1, 1);
    NDArray<double, 1> sigma = NDArray<double, 1>(1, 1);

public:
    Adadelta() {}
    Adadelta(NDArray<double, 1> weights, NDArray<double, 1> biases, NDArray<double, 1> input)
    {
        unsigned neurons, features, batches;

        neurons = weights.getDimensions()[0];
        features = input.getDimensions()[0];
        batches = input.getDimensions()[1];
        double e = 1.1025;
        double s = 0.9;

        epsalon.initData(&e);
        sigma.initData(&s);

        delta_weights = NDArray<double, 1>(weights.getNoOfDimensions(), weights.getDimensions());
        delta_biases = NDArray<double, 1>(biases.getNoOfDimensions(), biases.getDimensions());
        delta_input = NDArray<double, 1>(input.getNoOfDimensions(), input.getDimensions());

        sum_delta_biases = NDArray<double, 1>(biases.getNoOfDimensions(), biases.getDimensions());
        sum_delta_weights = NDArray<double, 1>(weights.getNoOfDimensions(), weights.getDimensions());

        learning_rate_biases = NDArray<double, 1>(biases.getNoOfDimensions(), biases.getDimensions());
        learning_rate_weights = NDArray<double, 1>(weights.getNoOfDimensions(), weights.getDimensions());

        learning_rate_biases.initData(learning_rate);
        learning_rate_weights.initData(learning_rate);

        delta_biases_intermediate = NDArray<double, 1>(2, neurons, batches);
        delta_weights_intermediate = NDArray<double, 1>(3, neurons, features, batches);
        delta_biases_intermediate = NDArray<double, 1>(3, features, batches, neurons);
    }
    void optimize(NDArray<double, 1> input, NDArray<double, 1> weights, NDArray<double, 1> biases, NDArray<double, 1> delta_activation, NDArray<double, 1> difference, NDArray<double, 1> difference_input, cudaStream_t stream)
    {
        math.getDifferentialBiases(delta_activation, difference, delta_biases, delta_biases_intermediate, stream);
        math.getDifferentialWeights(input, delta_activation, difference, delta_weights, delta_weights_intermediate, stream);
        math.getDifferentialInput(weights, delta_activation, difference, difference_input, delta_biases_intermediate, delta_input, stream);

        math.updateLearningRateBiasesAdadelta(epsalon, sigma, sum_delta_biases, delta_biases, learning_rate_biases, stream);
        math.updateLearningRateWeightsAdadelta(epsalon, sigma, sum_delta_weights, delta_weights, learning_rate_weights, stream);

        math.updateBiases(biases, learning_rate_biases, delta_biases, stream);
        math.updateWeights(weights, learning_rate_weights, delta_weights, stream);
    }
};

class SGD_momentum : public Optimizer
{
    unsigned *arr;
    double learning_rate = 0.01;
    NDArray<double, 1> learning_rate_weights;
    NDArray<double, 1> learning_rate_biases;
    NDArray<double, 1> sum_delta_weights;
    NDArray<double, 1> sum_delta_biases;
    NDArray<double, 1> delta_weights_intermediate;
    NDArray<double, 1> delta_biases_intermediate;
    NDArray<double, 1> delta_input_intermediate;
    NDArray<double, 1> epsalon = NDArray<double, 1>(1, 1);
    NDArray<double, 1> sigma = NDArray<double, 1>(1, 1);

public:
    SGD_momentum() {}
    SGD_momentum(NDArray<double, 1> weights, NDArray<double, 1> biases, NDArray<double, 1> input)
    {
        unsigned neurons, features, batches;

        neurons = weights.getDimensions()[0];
        features = input.getDimensions()[0];
        batches = input.getDimensions()[1];
        double e = 1.1025;
        double s = 0.99;

        epsalon.initData(&e);
        sigma.initData(&s);

        delta_weights = NDArray<double, 1>(weights.getNoOfDimensions(), weights.getDimensions());
        delta_biases = NDArray<double, 1>(biases.getNoOfDimensions(), biases.getDimensions());
        delta_input = NDArray<double, 1>(input.getNoOfDimensions(), input.getDimensions());

        sum_delta_biases = NDArray<double, 1>(biases.getNoOfDimensions(), biases.getDimensions());
        sum_delta_weights = NDArray<double, 1>(weights.getNoOfDimensions(), weights.getDimensions());

        learning_rate_biases = NDArray<double, 1>(biases.getNoOfDimensions(), biases.getDimensions());
        learning_rate_weights = NDArray<double, 1>(weights.getNoOfDimensions(), weights.getDimensions());

        learning_rate_biases.initData(learning_rate);
        learning_rate_weights.initData(learning_rate);

        delta_biases_intermediate = NDArray<double, 1>(2, neurons, batches);
        delta_weights_intermediate = NDArray<double, 1>(3, neurons, features, batches);
        delta_biases_intermediate = NDArray<double, 1>(3, features, batches, neurons);
    }
    void optimize(NDArray<double, 1> input, NDArray<double, 1> weights, NDArray<double, 1> biases, NDArray<double, 1> delta_activation, NDArray<double, 1> difference, NDArray<double, 1> difference_input, cudaStream_t stream)
    {
        math.getDifferentialBiases(delta_activation, difference, delta_biases, delta_biases_intermediate, stream);
        math.getDifferentialWeights(input, delta_activation, difference, delta_weights, delta_weights_intermediate, stream);
        math.getDifferentialInput(weights, delta_activation, difference, difference_input, delta_biases_intermediate, delta_input, stream);

        math.updateBiasesSGDmomentum(sigma, biases, learning_rate_biases, sum_delta_biases, delta_biases, stream);
        math.updateWeightsSGDmomentum(sigma, weights, learning_rate_weights, sum_delta_weights, delta_weights, stream);
    }
};

class RMSprop : public Optimizer
{
    unsigned *arr;
    double learning_rate = 0.01;
    NDArray<double, 1> learning_rate_weights;
    NDArray<double, 1> learning_rate_biases;
    NDArray<double, 1> sum_delta_weights;
    NDArray<double, 1> sum_delta_biases;
    NDArray<double, 1> delta_weights_intermediate;
    NDArray<double, 1> delta_biases_intermediate;
    NDArray<double, 1> delta_input_intermediate;
    NDArray<double, 1> epsalon = NDArray<double, 1>(1, 1);
    NDArray<double, 1> sigma = NDArray<double, 1>(1, 1);

public:
    RMSprop() {}
    RMSprop(NDArray<double, 1> weights, NDArray<double, 1> biases, NDArray<double, 1> input)
    {
        unsigned neurons, features, batches;

        neurons = weights.getDimensions()[0];
        features = input.getDimensions()[0];
        batches = input.getDimensions()[1];
        double e = 1.001;
        double s = 0.99;

        epsalon.initData(&e);
        sigma.initData(&s);

        delta_weights = NDArray<double, 1>(weights.getNoOfDimensions(), weights.getDimensions());
        delta_biases = NDArray<double, 1>(biases.getNoOfDimensions(), biases.getDimensions());
        delta_input = NDArray<double, 1>(input.getNoOfDimensions(), input.getDimensions());

        sum_delta_biases = NDArray<double, 1>(biases.getNoOfDimensions(), biases.getDimensions());
        sum_delta_weights = NDArray<double, 1>(weights.getNoOfDimensions(), weights.getDimensions());

        learning_rate_biases = NDArray<double, 1>(biases.getNoOfDimensions(), biases.getDimensions());
        learning_rate_weights = NDArray<double, 1>(weights.getNoOfDimensions(), weights.getDimensions());

        learning_rate_biases.initData(learning_rate);
        learning_rate_weights.initData(learning_rate);

        delta_biases_intermediate = NDArray<double, 1>(2, neurons, batches);
        delta_weights_intermediate = NDArray<double, 1>(3, neurons, features, batches);
        delta_biases_intermediate = NDArray<double, 1>(3, features, batches, neurons);
    }
    void optimize(NDArray<double, 1> input, NDArray<double, 1> weights, NDArray<double, 1> biases, NDArray<double, 1> delta_activation, NDArray<double, 1> difference, NDArray<double, 1> difference_input, cudaStream_t stream)
    {
        math.getDifferentialBiases(delta_activation, difference, delta_biases, delta_biases_intermediate, stream);
        math.getDifferentialWeights(input, delta_activation, difference, delta_weights, delta_weights_intermediate, stream);
        math.getDifferentialInput(weights, delta_activation, difference, difference_input, delta_biases_intermediate, delta_input, stream);

        math.updateBiasesRMSpropDense(sigma, epsalon, biases, learning_rate_biases, sum_delta_biases, delta_biases, stream);
        math.updateWeightsRMSpropDense(sigma, epsalon, weights, learning_rate_weights, sum_delta_weights, delta_weights, stream);
    }
};

class ADAM : public Optimizer
{
    unsigned *arr;
    double learning_rate = 0.01;
    NDArray<double, 1> learning_rate_weights;
    NDArray<double, 1> learning_rate_biases;
    NDArray<double, 1> sum_delta_weights;
    NDArray<double, 1> sum_delta_biases;
    NDArray<double, 1> sum_delta_weights_squared;
    NDArray<double, 1> sum_delta_biases_squared;
    NDArray<double, 1> delta_weights_intermediate;
    NDArray<double, 1> delta_biases_intermediate;
    NDArray<double, 1> delta_input_intermediate;
    NDArray<double, 1> epsalon = NDArray<double, 1>(1, 1);
    NDArray<double, 1> sigma = NDArray<double, 1>(1, 1);

public:
    ADAM() {}
    ADAM(NDArray<double, 1> weights, NDArray<double, 1> biases, NDArray<double, 1> input)
    {
        unsigned neurons, features, batches;

        neurons = weights.getDimensions()[0];
        features = input.getDimensions()[0];
        batches = input.getDimensions()[1];
        double e = 1.001;
        double s = 0.99;

        epsalon.initData(&e);
        sigma.initData(&s);

        delta_weights = NDArray<double, 1>(weights.getNoOfDimensions(), weights.getDimensions());
        delta_biases = NDArray<double, 1>(biases.getNoOfDimensions(), biases.getDimensions());
        delta_input = NDArray<double, 1>(input.getNoOfDimensions(), input.getDimensions());

        sum_delta_biases = NDArray<double, 1>(biases.getNoOfDimensions(), biases.getDimensions());
        sum_delta_weights = NDArray<double, 1>(weights.getNoOfDimensions(), weights.getDimensions());

        sum_delta_biases_squared = NDArray<double, 1>(biases.getNoOfDimensions(), biases.getDimensions());
        sum_delta_weights_squared = NDArray<double, 1>(weights.getNoOfDimensions(), weights.getDimensions());

        learning_rate_biases = NDArray<double, 1>(biases.getNoOfDimensions(), biases.getDimensions());
        learning_rate_weights = NDArray<double, 1>(weights.getNoOfDimensions(), weights.getDimensions());

        learning_rate_biases.initData(learning_rate);
        learning_rate_weights.initData(learning_rate);

        delta_biases_intermediate = NDArray<double, 1>(2, neurons, batches);
        delta_weights_intermediate = NDArray<double, 1>(3, neurons, features, batches);
        delta_biases_intermediate = NDArray<double, 1>(3, features, batches, neurons);
    }
    void optimize(NDArray<double, 1> input, NDArray<double, 1> weights, NDArray<double, 1> biases, NDArray<double, 1> delta_activation, NDArray<double, 1> difference, NDArray<double, 1> difference_input, cudaStream_t stream)
    {
        math.getDifferentialBiases(delta_activation, difference, delta_biases, delta_biases_intermediate, stream);
        math.getDifferentialWeights(input, delta_activation, difference, delta_weights, delta_weights_intermediate, stream);
        math.getDifferentialInput(weights, delta_activation, difference, difference_input, delta_biases_intermediate, delta_input, stream);

        // math.updateLearningRateBiasesAdadelta(epsalon, sigma, sum_delta_biases, delta_biases, learning_rate_biases, stream);
        // math.updateLearningRateWeightsAdadelta(epsalon, sigma, sum_delta_weights, delta_weights, learning_rate_weights, stream);

        
        math.updateBiasesADAMDense(sigma, epsalon, biases, learning_rate_biases, sum_delta_biases, sum_delta_biases_squared, delta_biases, stream);
        math.updateWeightsADAMDense(sigma, epsalon, weights, learning_rate_weights, sum_delta_weights, sum_delta_weights_squared,delta_weights, stream);
    }
};