#pragma ONCE

typedef struct struct_Activations
{
    enum activationFunc
    {
        relu,
        sigmoid,
        softmax,
        tanh,
        e_relu,
        leaky_relu,
        linear
    };
    std::map<std::string, activationFunc> activations;

    struct_Activations()
    {
        activations["relu"]         =   relu;
        activations["sigmoid"]      =   sigmoid;
        activations["softmax"]      =   softmax;
        activations["e_relu"]       =   e_relu;
        activations["leaky_relu"]   =   leaky_relu;
        activations["tanh"]         =   tanh;
        activations["linear"]       =   linear;
    }
} struct_Activations;

class Activation
{
protected:
    NDMath math;

public:
    virtual void activate(NDArray<double, 1> output, NDArray<double, 1> delta_activation, cudaStream_t stream) {}
    virtual void activate(NDArray<double, 0> output) {}
};

class Relu_Activation : public Activation
{
public:
    void activate(NDArray<double, 1> output, NDArray<double, 1> delta_activation, cudaStream_t stream) override
    {
        math.reluActivation(output, delta_activation, stream);
    }
    void activate(NDArray<double, 0> output) override
    {
        math.reluActivation(output);
    }
};

class Sigmoid_Activation : public Activation
{
public:
    void activate(NDArray<double, 1> output, NDArray<double, 1> delta_activation, cudaStream_t stream) override
    {
        math.sigmoidActivation(output, delta_activation, stream);
    }
    void activate(NDArray<double, 0> output) override
    {
        math.sigmoidActivation(output);
    }
};

class Linear_Activation : public Activation
{
public:
    void activate(NDArray<double, 1> output, NDArray<double, 1> delta_activation, cudaStream_t stream) override
    {
        math.linearActivation(output, delta_activation, stream);
    }
    void activate(NDArray<double, 0> output) override
    {
        
    }
};

class Softmax_Activation : public Activation
{
    NDArray<double, 1> softmax_sum;

    public:
        void activate(NDArray<double, 1> output, NDArray<double, 1> delta_activation, cudaStream_t stream) override
        {
            math.softmaxActivation(output, softmax_sum, delta_activation,stream);
        }
        
        void activate(NDArray<double, 0> output) override
        {

        }
};
