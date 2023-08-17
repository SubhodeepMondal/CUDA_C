#pragma ONCE


typedef struct struct_Metrics
{
    enum metrics
    {
        accuracy,
        binary_accuracy,
        categorical_accuracy,
        precision,
        recall,

    };
    std::map<std::string, metrics> metric;

    struct_Metrics()
    {
        metric["accuracy"] = accuracy;
        metric["binary_accuracy"] = binary_accuracy;
        metric["categorical_accuracy"] = categorical_accuracy;
        metric["precision"] = precision;
        metric["recall"] = recall;
    }
} struct_Metric;

class Metric
{

};

class Accuracy:public Metric
{

};