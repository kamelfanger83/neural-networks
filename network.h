#ifndef FORMATTER_PY_NETWORK_H
#define FORMATTER_PY_NETWORK_H

#include <vector>
#include <iostream>
#include <memory>
#include <functional>


struct hyperparameters {
    std::vector<std::vector<int>> layer_output_shape;
    std::vector<int> layer_type;
    std::vector<std::vector<int>> layer_data;
    int samples;
    int test_samples;
    float wlearning_rate, blearning_rate, w_conv_learning_rate, b_conv_learning_rate;
    float L2_frac;
    float keep_adj;
    int batchsize;
    int epochs;

    hyperparameters() = default;
};

struct layers {
    layers() = default;
    virtual void feedforward(const hyperparameters &params, const std::vector<float> &inp, std::vector<float> &derivatives, std::vector<float> &results) = 0;
    virtual void adjust_adjustments(const hyperparameters &params, const std::vector<float> &derivatives_of_activation, const std::vector<float> &from_previous, const std::vector<float> &derivatives_on_cost, std::vector<float> &derivatives_for_previous, bool last_layer = false) = 0;
    virtual void apply_adjustments(const hyperparameters &params) = 0;
    virtual void print(std::ostream &out) = 0;
};

struct fully_connected : public layers {
    int neurons, previous_neurons;
    std::vector<std::vector<float>> weights, weights_gradients;
    std::vector<float> biases, biases_gradient;
    std::function<std::vector<float>(std::vector<float>)> activation, activation_derivative;

    fully_connected(int inputs, int outputs, int activtion_type);
    void feedforward(const hyperparameters &params, const std::vector<float>& inp, std::vector<float>& derivatives, std::vector<float>& results) override;
    void adjust_adjustments(const hyperparameters &params, const std::vector<float>& derivatives_of_activation, const std::vector<float>& from_previous, const std::vector<float>& derivatives_on_cost, std::vector<float>& derivatives_for_previous, bool last_layer) override;
    void apply_adjustments(const hyperparameters &params) override;
    void print(std::ostream& out) override;
};

struct convolutional : public layers {
    std::vector<int> previous_shape, shape; // n x m x c, n = width, m = height, c = channels
    int stride, padding, kernel_size;
    std::vector<std::vector<std::vector<std::vector<float>>>> weights, weights_gradients; // first dimension: channel, second dimension: previous channel, third dimension: column, fourth dimension: row
    std::vector<float> biases, biases_gradient;
    std::function<std::vector<float>(std::vector<float>)> activation, activation_derivative;
    convolutional(std::vector<int> previous_shape, int channels, int stride, int padding, int kernel_size, int activation_type);
    void feedforward(const hyperparameters &params, const std::vector<float>& inp, std::vector<float>& derivatives, std::vector<float>& results) override;
    void adjust_adjustments(const hyperparameters &params, const std::vector<float>& derivatives_of_activation, const std::vector<float>& from_previous, const std::vector<float>& derivatives_on_cost, std::vector<float>& derivatives_for_previous, bool last_layer) override;
    void apply_adjustments(const hyperparameters &params) override;
    void print(std::ostream& out) override;
};

struct maxpool : public layers {
    std::vector<int> previous_shape;
    int width, height, window_side;
    maxpool (std::vector<int> previous_shape, int window_side);
    void feedforward(const hyperparameters &params, const std::vector<float>& inp, std::vector<float>& derivatives, std::vector<float>& results) override;
    void adjust_adjustments(const hyperparameters &params, const std::vector<float>& derivatives_of_activation, const std::vector<float>& from_previous, const std::vector<float>& derivatives_on_cost, std::vector<float>& derivatives_for_previous, bool last_layer) override;
    void apply_adjustments(const hyperparameters &params) override;
    void print(std::ostream& out) override;
};

struct networks {
    std::vector<std::vector<int>> shape;
    std::vector<std::unique_ptr<layers>> all_layers;
    std::vector<std::vector<float>> results, actderivatives, a_gradients, allzeros; //activated values, derivative of activation at the point it got called, a_gradients of neurons on cost
    hyperparameters params;

    explicit networks(hyperparameters& params);
    std::vector<float> feedforward(const std::vector<float>& inp);
    void train(const std::vector<std::vector<float>>& inp, const std::vector<std::vector<float>>& labels, int s, int e);
};

#endif //FORMATTER_PY_NETWORK_H
