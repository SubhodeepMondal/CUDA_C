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
    unsigned input_layer_count, output_layer_cout;
    NDArray<double, 1> X_input, y_target;
    NDArray<unsigned, 0> input_shape, output_shape;
    Layer *input, *output;
    std::string model_type;
    Optimizer *optimizer;
    Loss *loss;
    Metric *metric;
    struct_Models st_models;
    struct_Loss st_loss;
    struct_Optimizer st_optimizer;
    search_parameter search;

public:
    Model(std::string str) : input(NULL), output(NULL)
    {
        switch (st_models.model[str])
        {
        case this->st_models.sequential:
        {
            model_type = str;
            input_layer_count = output_layer_cout = 1;
            break;
        }

        default:
            break;
        }
    }

    virtual void add(Layer *layers) {}

    void getModelType()
    {
        std::cout << model_type << "\n";
    }

    virtual void compile(std::string loss, std::string optimizer, std::string metrics) {}

    void summary()
    {
        Layer *ptr = input;
        getModelType();
        search.search_param = "summary";
        ptr->searchDFS(search);
    }

    void fit(NDArray<double, 0> X, NDArray<double, 0> Y, int epochs, int batch_size)
    {
        unsigned no_of_sample, no_of_input_feature, batch_index, data_range, epoch_range, mod;
        Layer *ptr;
        std::random_device generator;
        std::uniform_int_distribution<int> distribution;
        NDArray<double, 1> Cost;

        cudaStream_t stream;
        cudaSetDevice(0);

        no_of_input_feature = X.getDimensions()[0];
        no_of_sample = X.getDimensions()[1];
        data_range = (unsigned)no_of_sample / 4;
        epoch_range = epochs / 4;

        X_input = NDArray<double, 1>(2, no_of_input_feature, data_range);
        y_target = NDArray<double, 1>(Y.getNoOfDimensions(), data_range);
        Cost = NDArray<double, 1>(2, 1, epochs);

        distribution = std::uniform_int_distribution<int>(0, data_range);

        cudaStreamCreate(&stream);

        ptr = input;
        search.cuda_Stream = stream;
        search.Integer = batch_size;
        search.search_param = "prepare_training";
        ptr->searchDFS(search);

        // std::cout << "Parameters before training:\n";
        // search.search_param = "print_parameters";
        // ptr->searchDFS(search);

        for (int i = 0; i < epochs; i++)
        {

            unsigned index;
            batch_index = distribution(generator);
            batch_index = (data_range - batch_index) > batch_size ? batch_index : (data_range - batch_size);

            // std::cout << "Epoch: " << i+1 << ", batch index: " << batch_index << " " ;
            mod = i % epoch_range;
            if (!mod)
            {
                X_input.initData(X.getData() + mod * data_range);
                y_target.initData(Y.getData() + mod * data_range);
            }

            ptr = input;
            // no_of_data = no_of_input_feature * batch_size;
            index = batch_index * no_of_input_feature;
            ptr->initilizeInputGPU(X_input.getData() + index);

            ptr = output;
            // no_of_data = Y.getDimensions()[0] * batch_size;
            index = batch_index * Y.getDimensions()[0];
            ptr->initilizeTarget(y_target.getData() + index);

            // ptr = input;
            // ptr->printInputIntermediate();

            // ptr = output;
            // ptr->printTarget();

            ptr = input;
            search.search_param = "forward_propagation";
            ptr->searchDFS(search);

            ptr = output;
            Cost.initPartialData(i, 1, ptr->findCost(loss));

            search.search_param = "backward_propagation";
            ptr->searchBFS(search);


            // std::cout <<"___________________________________________________________________________________________________________________________________\n";
            // std:: cout << "Epoch: " << i << "\n";
            // ptr = output;
            // ptr->printOutputGPU();
            // ptr->printTarget();
            // ptr->printDifference();
        }

        // Cost.printData();
        Cost.destroy();

        ptr = input;
        search.search_param = "commit_weights_biases";
        ptr->searchDFS(search);

        // std::cout << "Parameters after training:\n";
        // search.search_param = "print_parameters";
        // ptr->searchDFS(search);
        
    }

    NDArray<double, 0> predict(NDArray<double, 0> X)
    {
        unsigned i, no_of_input_features, no_of_dimensions, no_of_instances, prediction_dimension;

        NDArray<double, 0> X_input, y_predict;
        Layer *ptr;

        prediction_dimension = output->getNoOfNeuron();
        no_of_dimensions = X.getNoOfDimensions();
        no_of_instances = X.getDimensions()[no_of_dimensions-1];
        no_of_input_features = 0;

        for (i = 0; i < no_of_dimensions - 1; i++)
        {
            no_of_input_features += X.getDimensions()[i];
        }

        X_input = NDArray<double, 0>(no_of_dimensions - 1, X.getDimensions());
        y_predict = NDArray<double, 0>(2, prediction_dimension, X.getDimensions()[no_of_dimensions - 1]);

        for (i = 0; i < no_of_instances; i++)
        {
            X_input.initData(X.getData()[i * no_of_input_features]);
            // X_input.printData();

            ptr = input;
            ptr->initilizeInputData(X_input);
            search.search_param = "predict";
            ptr->searchDFS(search);

            ptr = output;
            y_predict.initPartialData(i, y_predict.getDimensions()[0], ptr->getOutputData());
        }

        return y_predict;
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

        Layer *ptr = input;

        search.search_param = "compile";
        search.String = optimizer;
        ptr->searchDFS(search);

        switch (st_loss.loss[loss])
        {
        case this->st_loss.mean_squared_error:
            this->loss = new Mean_Squared_Error;
            break;

        default:
            break;
        }
    }
};