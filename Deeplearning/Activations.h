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
        activations["relu"] = relu;
        activations["sigmoid"] = sigmoid;
        activations["softmax"] = softmax;
        activations["e_relu"] = e_relu;
        activations["leaky_relu"] = leaky_relu;
        activations["tanh"] = tanh;
        activations["linear"] = linear;
    }
} struct_Activations;

class Activation
{
protected:
    NDMath math;

public:
    virtual void activate(NDArray<double, 1> output, NDArray<double, 1> delta_activation, cudaStream_t stream) {}
};

class relu : public Activation
{
public:
    void activate(NDArray<double, 1> output, NDArray<double, 1> delta_activation, cudaStream_t stream) override
    {
        // std::cout << "In relu activation:\n";
        math.reluActivation(output, delta_activation, stream);
    }
};

class sigmoid : public Activation
{
public:
    void activate(NDArray<double, 1> output, NDArray<double, 1> delta_activation, cudaStream_t stream) override
    {
        // std::cout << "In sigmoid activation:\n";
        math.sigmoidActivation(output, delta_activation, stream);
    }
};

class linear : public Activation
{
public:
    void activate(NDArray<double, 1> output, NDArray<double, 1> delta_activation, cudaStream_t stream) override
    {
        // std::cout << "In linear activation:\n";
        math.linearActivation(output, delta_activation, stream);
    }
};

// class softmax : public Activation
// {
// };

// class e_relu : public Activation
// {
// };

// class leaky_rely : public Activation
// {
// };

// class tanh : public Activation
// {
// };
