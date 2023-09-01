#pragma ONCE

typedef struct struct_Layers
{
    enum layers
    {
        dense,
        conv2d,
        batch_normalization,
        dropout,
    };
    std::map<std::string, layers> layer;

    struct_Layers()
    {
        layer["dense"] = dense;
        layer["conv2d"] = conv2d;
        layer["batch_normalization"] = batch_normalization;
        layer["dropout"] = dropout;
    }
} struct_Layers;

typedef struct search_parameter
{
    std::string search_param;
    std::string String;
    unsigned Integer;
    double Double;
    cudaStream_t cuda_Stream;
    int *int_Ptr;
    double *double_Ptr;
} search_parameter;

typedef struct search_Flags
{
    enum flags
    {
        compile,
        train,
        predict,
        summary,
        prepare_training,
        initilize_weights_biases,
        initilize_weights_biases_gpu,
        forward_propagation,
        backward_propagation,
        initilize_optimizer,
        initilize_output,
        initilize_output_gpu,
        print_output_gpu,
        update_stream,
        set_input_pointer,
        print_pointer,
        print_parameters,
        commit_weights_biases,
        initilize_input_intermidiate,
        initilize_output_intermidiate
    };

    std::map<std::string, flags> search_flags;

    search_Flags()
    {
        search_flags["compile"] = compile;
        search_flags["train"] = train;
        search_flags["predict"] = predict;
        search_flags["summary"] = summary;
        search_flags["prepare_training"] = prepare_training;
        search_flags["initilize_weights_biases"] = initilize_weights_biases;
        search_flags["initilize_weights_biases_gpu"] = initilize_weights_biases_gpu;
        search_flags["forward_propagation"] = forward_propagation;
        search_flags["backward_propagation"] = backward_propagation;
        search_flags["initilize_optimizer"] = initilize_optimizer;
        search_flags["initilize_output"] = initilize_output;
        search_flags["initilize_output_gpu"] = initilize_output_gpu;
        search_flags["initilize_input_intermidiate"] = initilize_input_intermidiate;
        search_flags["initilize_output_intermidiate"] = initilize_output_intermidiate;
        search_flags["print_parameters"] = print_parameters;
        search_flags["update_stream"] = update_stream;
        search_flags["set_input_pointer"] = set_input_pointer;
        search_flags["print_pointer"] = print_pointer;
        search_flags["commit_weights_biases"] = commit_weights_biases;
    }
} search_Flags;

typedef struct search_Positions
{
    enum positions
    {
        input,
        output,
    };
    std::map<std::string, positions> search_positions;

    search_Positions()
    {
        search_positions["input"] = input;
        search_positions["output"] = output;
    }
} search_Positions;

class Layer
{
protected:
    typedef struct Layer_ptr
    {
        struct Layer_ptr *next, *previous;
        Layer *layer;

        Layer_ptr()
        {
            next = previous = NULL;
            layer = NULL;
        }
    } Layer_ptr;

public:
    std::string layer_type;
    Layer_ptr *in_vertices, *out_vertices;

    Layer(std::string Layer_Type) : in_vertices(NULL), out_vertices(NULL), layer_type(Layer_Type) {}

    void operator=(Layer *layer)
    {
        Layer_ptr *out_lyr_ptr, *out_prev_lyr_ptr;
        Layer_ptr *in_lyr_ptr, *in_prev_lyr_ptr;
        Layer_ptr *in_ptr, *out_ptr;

        in_ptr = new Layer_ptr;
        out_ptr = new Layer_ptr;

        // For outgoing edge
        if (out_vertices) // if out vertex is not NULL
        {

            out_prev_lyr_ptr = out_lyr_ptr = out_vertices;

            while (out_lyr_ptr)
            {
                if (out_lyr_ptr != out_prev_lyr_ptr)
                    out_prev_lyr_ptr = out_prev_lyr_ptr->next;
                out_lyr_ptr = out_lyr_ptr->next;
            }

            out_ptr->layer = layer;
            out_ptr->previous = out_prev_lyr_ptr;
            out_prev_lyr_ptr->next = out_ptr;

            // std::cout << "out vertex tail: " << out_ptr << "\n";
        }
        else // if out vertex is NULL
        {
            out_ptr->layer = layer;
            out_vertices = out_ptr;
            // std::cout << "out vertex head: " << out_ptr << "\n";
        }

        if (layer->in_vertices)
        {
            in_prev_lyr_ptr = in_lyr_ptr = layer->in_vertices;

            while (in_lyr_ptr)
            {
                if (in_lyr_ptr != in_prev_lyr_ptr)
                    in_prev_lyr_ptr = in_prev_lyr_ptr->next;
                in_lyr_ptr = in_lyr_ptr->next;
            }

            in_ptr->layer = this;
            in_ptr->previous = in_prev_lyr_ptr;
            in_prev_lyr_ptr->next = in_ptr;
            // std::cout << "in vertex tail:" << in_ptr << "\n";
        }
        else
        {
            in_ptr->layer = this;
            layer->in_vertices = in_ptr;
            // std::cout << "in vertex head:" << in_ptr << "\n";
        }
    }

    virtual void forwardPropagation() {}

    virtual void LayerProperties() {}

    virtual void searchDFS(search_parameter search) {}

    virtual void searchBFS(search_parameter search) {}

    virtual unsigned getNoOfNeuron() { return 0; }

    virtual double *getOutputDataGPU() { return NULL; }

    virtual NDArray<double, 1> getDifferencefromPrevious() { return 0; }

    virtual NDArray<double, 1> getDifference() { return 0; }

    virtual void initilizeInput(unsigned, unsigned no_of_data, double *ptr) {}

    virtual void initilizeTarget(unsigned index, unsigned no_of_data, double *ptr) {}

    virtual void printInput() {}

    virtual void updateStream(cudaStream_t stream) {}

    virtual cudaStream_t getCUDAStream() { return NULL; }

    virtual Optimizer *getOptimizer() { return NULL; }

    virtual void updateOptimizer(std::string optimizer) {}

    virtual double *findCost(Loss *loss) { return NULL; }

    virtual void printCost() {}

    virtual void printInputIntermediate() {}

    virtual void printTarget() {}

    virtual void printWeight_Biases() {}

    virtual void printWeightGPU() {}

    //    virtual unsigned getLayerUnit() {}

    //    virtual unsigned getNoOfDimension() {}

    //    virtual unsigned *getDimensions() {}

    //    virtual std::string getActivationFunction() {}

    //    virtual unsigned updateLayerUnit(unsigned unit) {}
};

class Dense : public Layer
{
protected:
    Activation *activation;
    struct_Activations acti_func;
    std::string dense_activation; // dense activation function
    // NDArray<unsigned, 0> input; // input dimension for dense layer
    NDArray<double, 0> input;
    NDArray<double, 0> weights;
    NDArray<double, 0> biases;
    NDArray<double, 1> input_gpu;
    NDArray<double, 1> weights_gpu;
    NDArray<double, 1> biases_gpu;
    NDArray<double, 1> delta_activation;
    NDArray<double, 1> delta_weights;
    NDArray<double, 1> delta_biases;
    NDArray<double, 1> delta_prev_input;
    NDArray<double, 1> delta_input;
    NDArray<double, 0> output;
    NDArray<double, 1> output_gpu;
    NDArray<double, 1> target;
    NDArray<double, 1> difference;
    NDArray<double, 1> cost = NDArray<double, 1>(1, 1);
    NDMath math;
    cudaStream_t stream;
    Optimizer *optimizer;
    struct_Optimizer optimizers;

    unsigned dense_unit, batch_size, isInputInitilized, isOptimizerUpdated = 0, isTrainable, isCUDAStreamUpdated = 0; // no of neuron of dense layer
    std::string layer_name, layer_optimizer;
    search_Flags Flag;

    unsigned getInputDimension()
    {
        Layer_ptr *ptr = in_vertices;
        unsigned count = 0;

        while (ptr)
        {
            count += ptr->layer->getNoOfNeuron();
            ptr = ptr->next;
        }

        return count;
    }

    Optimizer *getOptimizerfromPrevious()
    {

        Layer_ptr *ptr = in_vertices;

        if (ptr)
        {
            return ptr->layer->getOptimizer();
        }
        else
            return NULL;
    }

    NDArray<double, 1> getDifference()
    {
        return difference;
    }

    NDArray<double, 1> getDifferencefromPrevious()
    {

        Layer_ptr *ptr = in_vertices;

        if (ptr)
        {
            return ptr->layer->getDifference();
        }
        else
            return 0;
    }

    void initilizeWeightsBiases()
    {
        weights = NDArray<double, 0>(2, dense_unit, input.getDimensions()[0]);
        biases = NDArray<double, 0>(1, dense_unit);
        weights.initRandData(-1, 1);
        biases.initRandData(-1, 1);
        // printWeight_Biases();
    }

    void initilizeWeightsBiasesGPU(unsigned batch_size)
    {
        unsigned no_of_features;
        no_of_features = input.getDimensions()[0];
        weights_gpu = NDArray<double, 1>(2, dense_unit, no_of_features);
        biases_gpu = NDArray<double, 1>(1, dense_unit);

        weights_gpu.initData(weights.getData());
        biases_gpu.initData(biases.getData());

        delta_input = getDifferencefromPrevious();
    }

    void initilizeOutput()
    {
        output = NDArray<double, 0>(1, dense_unit);
    }

    void initilizeOutputGPU(unsigned batch_size)
    {
        target = NDArray<double, 1>(2, dense_unit, batch_size);
        output_gpu = NDArray<double, 1>(2, dense_unit, batch_size);
        difference = NDArray<double, 1>(2, dense_unit, batch_size);
        delta_activation = NDArray<double, 1>(2, dense_unit, batch_size);
        // output_gpu.printDimensions();
    }

    void initilizeActivation()
    {
        switch (acti_func.activations[dense_activation])
        {
        case this->acti_func.relu:
        {
            activation = new relu;
            break;
        }
        case this->acti_func.sigmoid:
        {
            activation = new sigmoid;
            break;
        }
        case this->acti_func.linear:
        {
            activation = new linear;
            break;
        }
        default:
        {
            activation = NULL;
            break;
        }
        }
    }

    void initilizeOptimizer(Optimizer *optimizer)
    {
        this->optimizer = optimizer;
    }

    void initilizeInputIntermidiate(unsigned batch_size)
    {
        unsigned dims = input.getNoOfDimensions() + 1;
        unsigned i;
        unsigned *a = new unsigned[dims];

        Layer_ptr *ptr = in_vertices;

        for (i = 0; i < dims - 1; i++)
            a[i] = input.getDimensions()[i];
        a[i] = batch_size;

        if (ptr)
        {
            input_gpu = NDArray<double, 1>(dims, a, 0);
            input_gpu.initPreinitilizedData(ptr->layer->getOutputDataGPU());
        }
        else
        {
            input_gpu = NDArray<double, 1>(dims, a);
        }
    }

public:
    Dense(unsigned unit, NDArray<double, 0> input_shape, std::string activation, std::string layer_name = "dense", unsigned isTrainable = 1) : Layer("dense")
    {
        this->dense_unit = unit;
        this->dense_activation = activation;
        this->input = NDArray<double, 0>(input_shape.getNoOfDimensions(), input_shape.getDimensions()[0]);
        this->isInputInitilized = 1;
        this->layer_name = layer_name;
        this->isTrainable = isTrainable;
        initilizeActivation();
    }

    Dense(unsigned unit, std::string activation, std::string layer_name = "dense") : Layer("dense")
    {
        this->dense_unit = unit;
        this->dense_activation = activation;
        this->isInputInitilized = 0;
        this->layer_name = layer_name;
        initilizeActivation();
    }

    void layerProperties()
    {
        std::cout << layer_name << ": " << getNoOfNeuron();
        std::cout << ", [ " << getDimensions()[0] << ", "
                  << "], " << getActivationFunction() << "\n";
    }

    unsigned getNoOfDimensions()
    {
        return input.getNoOfDimensions();
    }

    unsigned *getDimensions()
    {
        return input.getDimensions();
    }

    double *getOutputDataGPU()
    {
        return output_gpu.getData();
    }

    cudaStream_t getCUDAStream()
    {
        return stream;
    }

    std::string getActivationFunction()
    {
        return dense_activation;
    }

    unsigned getNoOfNeuron()
    {
        return dense_unit;
    }

    Optimizer *getOptimizer()
    {
        return optimizer;
    }

    unsigned updateLayerUnit(unsigned unit)
    {
        unsigned *arr = new unsigned;
        arr[0] = unit;
        input = NDArray<double, 0>(1, arr, 0);
        return 0;
    }

    void updateStream(cudaStream_t stream)
    {
        if (!isCUDAStreamUpdated)
        {
            this->stream = stream;
            isCUDAStreamUpdated = 1;
            // std::cout << "updated stream:" << stream << "\n";
        }
    }

    void updateOptimizer(std::string optimizer)
    {
        switch (optimizers.optimizer[optimizer])
        {
        case this->optimizers.sgd:
        {
            if (!isOptimizerUpdated)
            {
                this->optimizer = new SGD(weights_gpu, biases_gpu, input_gpu);
                isOptimizerUpdated = 1;
            }
            break;
        }
        case this->optimizers.adagrad:
        {
            if (!isOptimizerUpdated)
            {
                this->optimizer = new Adagrad(weights_gpu, biases_gpu, input_gpu);
                isOptimizerUpdated = 1;
            }
            break;
        }
        case this->optimizers.adadelta:
        {
            if (!isOptimizerUpdated)
            {
                this->optimizer = new Adadelta(weights_gpu, biases_gpu, input_gpu);
                isOptimizerUpdated = 1;
            }
            break;
        }
        case this->optimizers.sgd_momentum:
        {
            if (!isOptimizerUpdated)
            {
                this->optimizer = new SGD_momentum(weights_gpu, biases_gpu, input_gpu);
                isOptimizerUpdated = 1;
            }
            break;
        }
        case this->optimizers.rmsprop:
        {
            if (!isOptimizerUpdated)
            {
                this->optimizer = new RMSprop(weights_gpu, biases_gpu, input_gpu);
                isOptimizerUpdated = 1;
            }
            break;
        }
        case this->optimizers.adam:
        {
            if (!isOptimizerUpdated)
            {
                this->optimizer = new ADAM(weights_gpu, biases_gpu, input_gpu);
                isOptimizerUpdated = 1;
            }
            break;
        }
        default:
            break;
        }
    }

    void updateInputPointer(Layer_ptr *ptr)
    {
        if (ptr)
            input.initPreinitilizedData((ptr->layer)->getOutputDataGPU());
    }

    void searchDFS(search_parameter search)
    {
        Layer_ptr *ptr = out_vertices;

        switch (Flag.search_flags[search.search_param])
        {
        case this->Flag.summary:
        {
            layerProperties();
            break;
        }
        case this->Flag.compile:
        {
            unsigned layer_unit = getInputDimension();

            if (layer_unit)
                updateLayerUnit(layer_unit);

            initilizeWeightsBiases();
            initilizeOutput();
            layer_optimizer = search.String;
            break;
        }
        case this->Flag.prepare_training:
        {
            updateStream(search.cuda_Stream);
            initilizeOutputGPU(search.Integer);
            initilizeInputIntermidiate(search.Integer);
            initilizeWeightsBiasesGPU(search.Integer);
            updateOptimizer(layer_optimizer);

            break;
        }

        case this->Flag.initilize_weights_biases_gpu:
        {
            initilizeWeightsBiasesGPU(search.Integer);
            break;
        }
        case this->Flag.forward_propagation:
        {
            forwardPropagation();
            break;
        }
        case this->Flag.update_stream:
        {
            updateStream(search.cuda_Stream);
            break;
        }
        case this->Flag.initilize_input_intermidiate:
        {
            initilizeInputIntermidiate(search.Integer);
            break;
        }
        case this->Flag.initilize_output_intermidiate:
        {
            initilizeOutputGPU(search.Integer);
            break;
        }
        case this->Flag.print_pointer:
        {
            printInputPointer();
            printOutputPointer();
            break;
        }
        case this->Flag.print_parameters:
        {
            // printInput();
            // printWeight_Biases();
            // printWeightGPU();
            printOutputGPU();
        }
        case this->Flag.predict:
        {
        }
        case this->Flag.commit_weights_biases:
        {
        }
        }

        while (ptr)
        {
            // std::cout << "In while.\n";

            ptr->layer->searchDFS(search);
            ptr = ptr->next;
        }
    }

    void searchBFS(search_parameter search)
    {
        Layer_ptr *ptr = in_vertices;

        switch (Flag.search_flags[search.search_param])
        {
        case this->Flag.compile:
        {
            unsigned Layer_Unit = getInputDimension();
            if (Layer_Unit)
            {
                updateLayerUnit(Layer_Unit);
            }
            // input.printData();
            break;
        }
        case this->Flag.set_input_pointer:
        {
            updateInputPointer(ptr);
            break;
        }
        case this->Flag.backward_propagation:
        {
            backwardPropagation();
            break;
        }
        }

        while (ptr)
            ptr = ptr->next;

        ptr = in_vertices;

        while (ptr)
        {

            ptr->layer->searchBFS(search);
            ptr = ptr->next;
        }
    }

    void initilizeInput(unsigned index, unsigned no_of_data, double *ptr) override
    {
        input_gpu.initPartialData(index, no_of_data, ptr);
    }

    void initilizeTarget(unsigned index, unsigned no_of_data, double *ptr) override
    {
        target.initPartialData(index, no_of_data, ptr);
    }

    void printInput()
    {
        std::cout << layer_name << ": input data: \n";
        input.printData();
    }

    void printInputIntermediate()
    {
        std::cout << layer_name << ": Input intermediate: \n";
        input_gpu.printData();
    }

    void printWeight_Biases()
    {
        std::cout << layer_name << ": weight data: \n";
        weights.printData();
        std::cout << layer_name << ": biases data: \n";
        biases.printData();
    }

    void printWeightGPU()
    {

        std::cout << layer_name << ": weight gpu data: \n";
        weights_gpu.printData();
    }

    void printBiasesGPU()
    {

        std::cout << layer_name << ": biases gpu data: \n";
        biases_gpu.printData();
    }

    void printOutput()
    {
        output.printData();
    }

    void printOutputGPU()
    {
        std::cout << layer_name << ": output data: \n";
        output_gpu.printData();
    }

    void printDiffActivation()
    {
        std::cout << layer_name << ": differential_activation: \n";
        delta_activation.printData();
    }

    void printDifference()
    {
        std::cout << layer_name << ": Difference: \n";
        difference.printData();
    }

    void printTarget()
    {
        std::cout << layer_name << ": target data: \n";
        target.printData();
    }

    void printDiffWeightsBiases()
    {
        std::cout << layer_name << ": Differential Weights Biases: \n";
        delta_weights.printData();
    }

    void printCost()
    {
        std::cout << layer_name << ": Cost: ";
        cost.printData();
    }

    void printInputPointer()
    {
        std::cout << layer_name << " Input Pointer:\t" << input.getData() << "\n";
    }

    void printOutputPointer()
    {
        std::cout << layer_name << " Output Pointer:\t" << output_gpu.getData() << "\n";
    }

    void forwardPropagation()
    {
        math.matrixDotMultiplication(input_gpu, weights_gpu, biases_gpu, output_gpu, stream);
        activation->activate(output_gpu, delta_activation, stream);

        // std::cout << "Forward propagation:\n";
        // printInputIntermediate();
        // printWeightGPU();
        // printBiasesGPU();
        // printOutputGPU();
    }

    double *findCost(Loss *loss) override
    {
        loss->findLoss(output_gpu, target, difference, cost, stream);

        // printWeightGPU();
        // printBiasesGPU();
        // std::cout << "Find cost:\n";
        // printOutputGPU();
        // printTarget();
        // printDifference();
        // printCost();
        return cost.getData();
    }

    void backwardPropagation()
    {

        optimizer->optimize(input_gpu, weights_gpu, biases_gpu, delta_activation, difference, delta_input, stream);

        // std::cout <<"Backward propagation:\n";
        // printWeightGPU();
        // printBiasesGPU();
        // std::cout << "\n\n";
    }

};