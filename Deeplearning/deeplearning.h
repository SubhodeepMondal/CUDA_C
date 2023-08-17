#pragma ONCE

#include <map>
#include "cudalibrary.h"
#include "Metrics.h"
#include "Losses.h"
#include "Activations.h"
#include "Optimizers.h"
#include "Layers.h"

typedef struct struct_Models
{
    enum Models
    {
        sequential,
        asequential,
        boltzman_machine
    };
    std::map<std::string, Models> model;
    struct_Models()
    {
        model["sequential"] = sequential;
        model["asequential"] = asequential;
        model["boltzman_machine"] = boltzman_machine;
    }
} Models;

class Model
{

protected:
    NDArray<double, 1> input_data;
    Layer *input, *output;
    std::string model_type;
    unsigned layer_count, max_feature, max_neuron;
    Optimizer *optimizer;
    Loss *loss;
    Metric *metric;
    struct_Loss st_loss;
    struct_Optimizer st_optimizer;
    search_parameter search;

public:
    Model(std::string str) : input(NULL), output(NULL)
    {
        model_type = str;
        layer_count = max_feature = max_neuron = 0;
    }

    virtual void add(Layer *layers)
    {
        std::cout << " in virtual add\n";
    }

    void getModelType()
    {
        std::cout << model_type << "\n";
    }

    virtual void compile(std::string loss, std::string optimizer, std::string metrics)
    {
    }

    void summary()
    {
        Layer *ptr = input;
        getModelType();
        search.search_param = "summary";
        ptr->searchDFS(search);
    }

    void fit(NDArray<double, 0> X, NDArray<double, 0> Y, int epochs, int batch_size)
    {
        unsigned no_of_sample, no_of_input_feature, batch_index;
        Layer *ptr;
        std::random_device generator;
        std::uniform_int_distribution<int> distribution;
        // NDArray<double, 1> input;

        cudaStream_t stream;
        cudaSetDevice(0);

        no_of_input_feature = X.getDimensions()[0];
        no_of_sample = X.getDimensions()[1];

        distribution = std::uniform_int_distribution<int>(0, no_of_sample);

        ptr = input;

        cudaSetDevice(0);
        cudaStreamCreate(&stream);

        search.search_param = "update_stream";
        search.cuda_Stream = stream;
        ptr->searchDFS(search);

        search.search_param = "initilize_output_intermidiate";
        search.Integer = batch_size;
        ptr->searchDFS(search);

        search.search_param = "initilize_input_intermidiate";
        ptr->searchDFS(search);

        search.search_param = "initilize_weights_biases_gpu";
        ptr->searchDFS(search);

        for (int i = 0; i < epochs; i++)
        {
            unsigned no_of_data, index;
            batch_index = distribution(generator);
            batch_index = (no_of_sample - batch_index) > batch_size ? batch_index : (no_of_sample - batch_size);

            ptr = input;
            no_of_data = no_of_input_feature * batch_size;
            index = batch_index * no_of_input_feature;
            ptr->initilizeInput(0, no_of_data, X.getData() + index);

            ptr = output;
            no_of_data = Y.getDimensions()[0] * batch_size;
            index = batch_index * Y.getDimensions()[0];
            ptr->initilizeTarget(0, no_of_data, Y.getData() + index);

            ptr = input;
            ptr->printInputIntermediate();

            ptr = output;
            ptr->printTarget();

            ptr = input;
            search.search_param = "forward_propagation";
            ptr->searchDFS(search);

            ptr = output;
            ptr->findCost(loss);

            search.search_param = "backward_propagation";
            ptr->searchBFS(search);

            // ptr = input;
            // search.search_param = "print_parameters";
            // ptr->searchDFS(search);

            // cudaStreamSynchronize(stream);
        }

        // ptr = input;
        // search.search_param = "commit_weights_biases";
        // ptr->searchDFS(search);
    }
};

class Sequential : public Model
{

public:
    Sequential() : Model("sequential"){};

    void add(Layer *layer)
    {
        if (output)
        {
            *(output) = layer;
            output = layer;
        }
        else
        {
            input = layer;
            output = layer;
        }
    }

    void compile(std::string loss, std::string optimizer, std::string metrics)
    {
        // str_loss = loss;
        // str_optimizer = optimizer;
        // str_metric = metrics;

        Layer *ptr = output;

        search.search_param = "compile";
        ptr->searchBFS(search);

        switch (st_loss.loss[loss])
        {
        case this->st_loss.mean_squared_error:
            this->loss = new Mean_Squared_Error;
            break;

        default:
            break;
        }

        switch (st_optimizer.optimizer[optimizer])
        {
        case this->st_optimizer.sgd:
        {
            this->optimizer = new SGD;
            break;
        }

        default:
            break;
        }

        ptr = input;
        ptr->updateOptimizer(this->optimizer);

        search.search_param = "initilize_weights_biases";
        ptr->searchDFS(search);

        search.search_param = "initilize_output";
        ptr->searchDFS(search);

        search.search_param = "initilize_optimizer";
        ptr->searchDFS(search);

        ptr = output;
        search.search_param = "set_input_pointer";
        ptr->searchBFS(search);

        // ptr = input;
        // search.search_param = "print_parameters";
        // ptr->searchDFS(search);
    }
};