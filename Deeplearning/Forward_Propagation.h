#pragma ONCE

class DenseForward
{
    NDMath math;

protected:
    void predict(NDArray<double, 0> input, NDArray<double, 0> weights, NDArray<double, 0> biases, NDArray<double, 0> output)
    {
        math.matrixDotMultiplication(input, weights, biases, output);
    }
    void fit(NDArray<double, 1> input_gpu, NDArray<double, 1> weights_gpu, NDArray<double, 1> biases_gpu, NDArray<double, 1> output_gpu, cudaStream_t stream)
    {
        math.matrixDotMultiplication(input_gpu, weights_gpu, biases_gpu, output_gpu, stream);
    }
};