#pragma ONCE

#include "cudalibrary.h"
#include <map>

typedef struct Activations
{
    enum activationFunc
    {
        relu,
        sigmoid,
        softmax,
        tanh,
        e_relu,
        leaky_relu
    };
    std::map<std::string, activationFunc> activations;

    Activations()
    {
        activations["relu"] = relu;
        activations["sigmoid"] = sigmoid;
        activations["softmax"] = softmax;
        activations["e_relu"] = e_relu;
        activations["leaky_relu"] = leaky_relu;
    }
} Activation;

class Dense
{
protected:
    Activation activation;
    std::string dense_activation;          // dense activation function
    NDArray<unsigned, 0> dense_dimensions; // input dimension for dense layer
    unsigned dense_unit;                   // no of neuron of dense layer

public:
    Dense(unsigned unit, NDArray<unsigned, 0> input_shape, std::string activation)
    {
        this->dense_unit = unit;
        this->dense_activation = activation;
        this->dense_dimensions = input_shape;
    }

    Dense(unsigned unit, std::string activation)
    {
        this->dense_unit = unit;
        this->dense_activation = activation;
    }

    Dense() {}

    int getNoOfDimensions()
    {
        return dense_dimensions.getNoOfDimensions();
    }

    unsigned *getDimensions()
    {
        return dense_dimensions.getDimensions();
    }

    std::string getActivationFunction()
    {
        return dense_activation;
    }

    int getNoOfNeuron()
    {
        return dense_unit;
    }

    void denseForwardPropagation(unsigned index, NDMath<double,1> math)
    {

        math.matrixDotMultiplication(dense_unit, dense_dimensions.getDimensions()[0] + 1, index);

        switch (activation.activations[dense_activation])
        {
        case Activation::relu:
            math.reluActivation(dense_unit, dense_dimensions.getDimensions()[0] + 1, index, 1); // 1 is for flag it is appending 1 at the last row of input NDarray
            break;

        case Activation::sigmoid:
            math.sigmoidActivation(dense_unit, dense_dimensions.getDimensions()[0] + 1, index, 1); // 1 is for flag it is appending 1 at the last row of input NDarray
            break;
        }
        // math.printData(0);

    }

    void denseBackwardPropagation(unsigned index, NDMath<double,1> math)
    {
    }
};

class Layers : public Dense
{
    enum layers
    {
        dense
    };
    std::map<std::string, layers> layer;
    std::string layer_type;

    void registerTypes()
    {
        layer["Dense"] = dense;
    }

public:
    Layers *next, *previous;
    void initilizeLayer(Dense dense)
    {
        this->layer_type = "dense";
        this->dense_activation = dense.getActivationFunction();
        this->dense_unit = dense.getNoOfNeuron();
        NDArray<unsigned, 0> ndarray(dense.getNoOfDimensions(), dense.getDimensions());

        this->dense_dimensions = ndarray;
        registerTypes();
    }

    void layerInfo()
    {
        switch (layer[layer_type])
        {
        case dense:
            /* code */
            std::cout << "layer : " << layer_type;
            std::cout << "\tunit : " << dense_unit;
            std::cout << "\tactivation :" << dense_activation;
            break;

        default:
            break;
        }
    }

    void forwardPropagation(unsigned index, NDMath<double,1> math)
    {
        switch (layer[layer_type])
        {
        case dense:
            denseForwardPropagation(index, math);
            break;

        default:
            break;
        }
    }
    void backwardPropagation(unsigned index, NDMath<double,1> math)
    {
        switch (layer[layer_type])
        {
        case dense:
            denseBackwardPropagation(index,math);
            break;

        default:
            break;
        }
    }
};

class Sequential
{
protected:
    Layers *head = NULL;
    Layers *tail = NULL;

public:
    void add(Dense denseLayer)
    {
        Layers *ptr;
        ptr = head;

        Layers *layer = new Layers;

        if (ptr != NULL)
        {
            while (ptr->next != NULL)
            {
                ptr = ptr->next;
            }
        }

        if (head == NULL)
        {
            head = layer;
            tail = layer;
            layer->next = NULL;
            layer->previous = NULL;
            layer->initilizeLayer(denseLayer);
        }
        else
        {
            tail = layer;
            ptr->next = layer;
            layer->next = NULL;
            layer->previous = ptr;
            layer->initilizeLayer(denseLayer);
        }
    }
};

template <class T>
class Model : public Sequential
{

    enum losses
    {
        categorical_crossentropy,
        binary_crossentropy,
        mean_squared_error,
        mean_absolute_error
    };
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
    enum metrics
    {
        accuracy,
        binary_accuracy,
        categorical_accuracy,
        precision,
        recall,

    };

    std::map<std::string, losses> Loss;
    std::map<std::string, optimizers> Optimizer;
    std::map<std::string, metrics> Metric;

    class Weights_Biases
    {
    public:
        Weights_Biases *next, *previous;
        NDArray<double, 0> weights_biases;
        Weights_Biases(unsigned n, unsigned *arr)
        {
            weights_biases = NDArray<double, 0>(n, arr);
        }
        Weights_Biases() {}

        void getDimensions()
        {
            weights_biases.printDimensions();
        }
    };

    unsigned layer_count, max_feature, max_neuron;
    std::string model_type, loss, optimizer, metric;
    Weights_Biases *weights_biases = NULL;

    void initilizeLosses()
    {
        Loss["categorical_corssentropy"] = categorical_crossentropy;
        Loss["binary_corssentropy"] = binary_crossentropy;
        Loss["mean_squared_error"] = mean_squared_error;
        Loss["mean_absolute_error"] = mean_absolute_error;
    }
    void initilizeOptimizers()
    {
        Optimizer["sgd"] = sgd;
        Optimizer["adadelta"] = adadelta;
        Optimizer["adafactor"] = adafactor;
        Optimizer["adagrad"] = adagrad;
        Optimizer["adam"] = adam;
        Optimizer["adamw"] = adamw;
        Optimizer["nadam"] = nadam;
        Optimizer["rmsprop"] = rmsprop;
    }
    void initilizeMetrics()
    {
        Metric["accuracy"] = accuracy;
        Metric["binary_accuracy"] = binary_accuracy;
        Metric["categorical_accuracy"] = categorical_accuracy;
        Metric["precision"] = precision;
        Metric["recall"] = recall;
    }

public:
    Model(std::string str)
    {
        model_type = str;
        layer_count = max_feature = max_neuron = 0;
    }

    void compile(std::string loss, std::string optimizer, std::string metrics)
    {

        unsigned weights_biases_dimention[2];
        unsigned no_of_weights, neurons, total_weights;
        unsigned *weights;

        this->loss = loss;
        this->optimizer = optimizer;
        this->metric = metrics;

        initilizeLosses();
        initilizeOptimizers();
        initilizeMetrics();

        Layers *ptr = head;

        if (ptr == NULL)
        {
            std::cout << "model has no layers to compile" << std::endl;
        }
        else
        {
            while (ptr != NULL)
            {
                layer_count++;
                ptr = ptr->next;
            }

            weights_biases = new Weights_Biases[layer_count];
            ptr = head;

            for (int i = 0; i < layer_count; i++)
            {
                if (ptr->previous == NULL)
                {
                    no_of_weights = ptr->getNoOfDimensions();
                    weights = ptr->getDimensions();
                    neurons = ptr->getNoOfNeuron();
                    total_weights = 1;

                    for (int j = 0; j < no_of_weights; j++)
                        total_weights *= weights[j];

                    total_weights++;
                    weights_biases_dimention[0] = neurons;
                    weights_biases_dimention[1] = total_weights;

                    weights_biases[i] = Weights_Biases((unsigned)2, weights_biases_dimention);
                    max_feature = neurons;
                    max_neuron = total_weights;
                }
                else
                {
                    total_weights = ptr->previous->getNoOfNeuron();
                    neurons = ptr->getNoOfNeuron();
                    total_weights++;
                    weights_biases_dimention[0] = neurons;
                    weights_biases_dimention[1] = total_weights;


                    weights_biases[i] = Weights_Biases((unsigned)2, weights_biases_dimention);

                    if (max_feature < total_weights)
                        max_feature = total_weights;
                    if (max_neuron < neurons)
                        max_neuron = neurons;
                }
                ptr = ptr->next;
            }
        }
    }

    void summary()
    {
        Layers *ptr;
        Weights_Biases *weights_biases_ptr;
        ptr = head;
        weights_biases_ptr = weights_biases;
        int i = 0;

        if (ptr == NULL)
        {
            std::cout << "model has no layers." << std::endl;
        }
        else
        {
            while (ptr)
            {
                ptr->layerInfo();
                std::cout << "\tTrainable parameters : ";
                weights_biases_ptr[i++].getDimensions();
                ptr = ptr->next;
            }
        }
    }

    void fit(T X, T Y, int epochs, int batch_size)
    {
        unsigned no_of_sample, batch_index, features, i, j;
        double error;

        Layers *layer_ptr;
        NDMath<double,1> math;
        NDArray<double,0> predicted_values;
        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution;


        layer_ptr = head;
        features = X.getDimensions()[0];
        no_of_sample = X.getDimensions()[1];
        distribution = std::uniform_int_distribution(0, (int)no_of_sample);


        for (i = 0; i < layer_count; i++)
        {
            weights_biases[i].weights_biases.initRandData(0, 1);
            // weights_biases[i].weights_biases.printData();
        }

        math = NDMath<double,1>(max_feature, max_neuron + 1, layer_count);

        // std::cout << max_neuron << " " << max_feature << "\n";
        for (int i = 0; i < layer_count; i++)
        {
            for (int k = 0; k < weights_biases[i].weights_biases.getDimensions()[1]; k++)
            {

                unsigned offset, no_of_data, data_offset;
                offset = k * max_feature + i * (max_neuron + 1) * max_feature;
                no_of_data = weights_biases[i].weights_biases.getDimensions()[0];
                data_offset = k * weights_biases[i].weights_biases.getDimensions()[0];
                math.transferData(1, offset, no_of_data, weights_biases[i].weights_biases.getData() + data_offset);
            }
        }

        // math.printData(1);

        for (i = 0; i < epochs; i++)
        {
            batch_index = distribution(generator);

            // copying data from input to layer.
            unsigned b_idx = ((no_of_sample - batch_index) > batch_size) ? batch_index : (no_of_sample - batch_size);
            unsigned m = 0;

            for (int k = b_idx; k < (b_idx + batch_size); k++)
            {
                double *one = new double;
                *one = 1;
                math.transferData(0, 0, features, (X.getData() + k * X.getDimensions()[0]));
                math.transferData(0, features, 1, one);

                j = 0;

                layer_ptr = head;
                while (layer_ptr)
                {
                    layer_ptr->forwardPropagation(j++, math);
                    layer_ptr = layer_ptr->next;
                }
                m++;

                switch (Loss[loss])
                {
                case mean_squared_error:
                    
                    break;
                
                default:
                    break;
                }

                layer_ptr = tail;
                while(layer_ptr)
                {
                    j--;
                    layer_ptr->backwardPropagation(j, math);
                    layer_ptr = layer_ptr->previous;
                }
            }
        }
    }
};