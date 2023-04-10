#include "network.h"

#include <algorithm>
#include <cmath>
#include <cassert>
#include <limits>

#define error(condition, message) if(!(condition)){cerr << "Error: " << message << endl; assert(false);}

using namespace std;

const auto relu = [](vector<float> x){
    for(auto& i : x) i = max(0.0f, i);
    return x;
};
const auto relu_derivative = [](vector<float> x){
    for(auto& i : x) i = (i > 0);
    return x;
};


const auto softmax = [](vector<float> x){
    float max = *max_element(x.begin(), x.end());
    float sum = 0;
    for (auto& i : x){
        i = exp(i - max);
        sum += i;
    }
    for (auto& i : x) i /= sum;
    return x;
};
const auto softmax_derivative = [](vector<float> x){
    float sum = 0;
    for (auto& i : x){
        i = exp(i);
        sum += i;
    }
    for (auto& i : x) i = i/sum * (1 - i/sum);
    return x;
};

fully_connected::fully_connected(int inputs, int outputs, int activtion_type = 0){ //activtion_type 0 = relu, 1 = softmax
    neurons = outputs;
    previous_neurons = inputs;

    weights = vector<vector<float>> (neurons, vector<float> (previous_neurons));
    biases = vector<float> (neurons, 0);
    for (int initializer = 0; initializer < neurons; ++initializer) {
        for (int inner = 0; inner < previous_neurons; ++inner) {
            weights[initializer][inner] = ((float)rand()/RAND_MAX*2-1) / neurons; //should be improved
        }
    }

    weights_gradients = vector<vector<float>> (neurons, vector<float> (previous_neurons, 0));
    biases_gradient = vector<float> (neurons, 0);

    if (activtion_type == 1) {
        activation = softmax;
        activation_derivative = softmax_derivative;
    }
    else {
        activation = relu;
        activation_derivative = relu_derivative;
    }
}

void fully_connected::feedforward(const hyperparameters &params, const vector<float>& inp, vector<float>& derivatives, vector<float>& results){ //standard fully connected feedforward, can be edited for cnn or so
    for (int neuron = 0; neuron < neurons; ++neuron) {
        results[neuron] = 0;
        for (int previous = 0; previous < previous_neurons; ++previous) {
            results[neuron] += weights[neuron][previous] * inp[previous];
        }
        results[neuron] += biases[neuron];
    }
    derivatives = activation_derivative(results);
    results = activation(results);
}

void fully_connected::adjust_adjustments(const hyperparameters &params, const vector<float>& derivatives_of_activation, const vector<float>& from_previous, const vector<float>& derivatives_on_cost, vector<float>& derivatives_for_previous, bool last_layer = false){ // if last_layer is true, derivatives_on_cost is the derivative of the z values, not the activation values i know not the best way to do it but it works
    for (int neuron = 0; neuron < neurons; ++neuron) {
        float preactivated_derivative_on_cost;

        if (last_layer) preactivated_derivative_on_cost = derivatives_on_cost[neuron];
        else preactivated_derivative_on_cost = derivatives_on_cost[neuron] * derivatives_of_activation[neuron]; //chain rule

        // check that there are no nans
        if (isnan(preactivated_derivative_on_cost)) {
            cout << "nan in preactivated_derivative_on_cost, at neuron " << neuron << ", value is " << preactivated_derivative_on_cost << endl;
            exit(1);
        }

        biases_gradient[neuron] -= params.blearning_rate / params.batchsize * preactivated_derivative_on_cost;

        for (int previous = 0; previous < previous_neurons; ++previous) {
            weights_gradients[neuron][previous] -= params.wlearning_rate / params.batchsize * preactivated_derivative_on_cost * from_previous[previous]; //chain rule
            derivatives_for_previous[previous] += preactivated_derivative_on_cost * weights[neuron][previous]; //chain rule
        }
    }
}

void fully_connected::apply_adjustments(const hyperparameters &params){
    // check that there are no nans
    for (int neuron = 0; neuron < neurons; ++neuron) {
        if (isnan(biases_gradient[neuron])) {
            cout << "nan in biases_gradient, at neuron " << neuron << ", value is " << biases_gradient[neuron] << endl;
            exit(1);
        }
        for (int previous = 0; previous < previous_neurons; ++previous) {
            if (isnan(weights_gradients[neuron][previous])) {
                cout << "nan in weights_gradients" << endl;
                exit(1);
            }
        }
    }

    for (int neuron = 0; neuron < neurons; ++neuron) {
        biases[neuron] += biases_gradient[neuron];
        biases_gradient[neuron] *= params.keep_adj;
        for (int previous = 0; previous < previous_neurons; ++previous) {
            weights_gradients[neuron][previous] -= params.wlearning_rate * params.L2_frac / params.samples * weights[neuron][previous]; //apply L2-regularization
            weights[neuron][previous] += weights_gradients[neuron][previous];
            weights_gradients[neuron][previous] *= params.keep_adj;
        }
    }
}

void fully_connected::print(ostream& out){
    for (int neuron = 0; neuron < neurons; ++neuron) {
        for (int previous = 0; previous < previous_neurons; ++previous) {
            out << weights[neuron][previous] << " ";
        }
        out << "\n";
    }

    for (int neuron = 0; neuron < neurons; ++neuron) {
        out << biases[neuron] << "\n";
    }
}


convolutional::convolutional(vector<int> previous_shape, int channels, int kernel_size, int padding, int stride, int activation_type = 0) : previous_shape(previous_shape), stride(stride), padding(padding), kernel_size(kernel_size) {
    if ((previous_shape[0] + 2 * padding - kernel_size) % stride != 0){
        cerr << "CAUTION: stride does not divide the sidelength of your convolutional net";
    }

    int width = (previous_shape[0] + 2 * padding - kernel_size) / stride + 1;
    int height = (previous_shape[1] + 2 * padding - kernel_size) / stride + 1;
    shape = {width, height, channels};

    weights = vector<vector<vector<vector<float>>>>(channels, vector<vector<vector<float>>>(previous_shape[2], vector<vector<float>>(kernel_size, vector<float>(kernel_size, 0))));
    weights_gradients = weights;

    for (int channel = 0; channel < channels; ++channel) {
        for (int previous_channel = 0; previous_channel < previous_shape[2]; ++previous_channel) {
            for (int x = 0; x < kernel_size; ++x) {
                for (int y = 0; y < kernel_size; ++y) {
                    weights[channel][previous_channel][x][y] = 0.002f * (rand() % 1000 - 500); // to be improved
                }
            }
        }
    }

    biases = vector<float> (channels, 0);
    biases_gradient = biases;

    error(activation_type == 0, "activation type not yet supported for convolutional layers")
    if (activation_type == 0){
        activation = relu;
        activation_derivative = relu_derivative;
    }
}

void convolutional::feedforward(const hyperparameters &params, const vector<float> &inp, vector<float> &derivatives, vector<float> &results) {
    error(inp.size() == previous_shape[0] * previous_shape[1] * previous_shape[2], "input size given to convolutional layer does not match expectation")
    error(results.size() == shape[0] * shape[1] * shape[2], "output size given to convolutional layer does not match expectation")

    for (int channel = 0; channel < shape[2]; ++channel) {
        for (int x = 0; x < shape[0]; ++x) {
            for (int y = 0; y < shape[1]; ++y) {
                int result_i = channel * shape[0] * shape[0] + x * shape[1] + y;
                results[result_i] = 0;
                for (int previous_channel = 0; previous_channel < previous_shape[2]; ++previous_channel) {
                    for (int kernel_x = max(0, padding - x); kernel_x < kernel_size && x - padding + kernel_x * stride < previous_shape[0]; ++kernel_x) {
                        for (int kernel_y = max(0, padding - y); kernel_y < kernel_size && y - padding + kernel_y * stride < previous_shape[1]; ++kernel_y) {
                            int previous_x = x - padding + kernel_x * stride;
                            int previous_y = y - padding + kernel_y * stride;
                            int previous_i = previous_channel * previous_shape[0] * previous_shape[1] + previous_x * previous_shape[0] + previous_y;
                            results[result_i] += inp[previous_i] * weights[channel][previous_channel][kernel_x][kernel_y];
                        }
                    }
                }
                results[result_i] += biases[channel];
            }
        }
    }
    derivatives = activation_derivative(results);
    results = activation(results);
}

void convolutional::adjust_adjustments(const hyperparameters &params, const std::vector<float> &derivatives_of_activation, const std::vector<float> &from_previous, const std::vector<float> &derivatives_on_cost, std::vector<float> &derivatives_for_previous, bool last_layer) {

    for (int channel = 0; channel < shape[2]; ++channel) {
        for (int x = 0; x < shape[0]; ++x) {
            for (int y = 0; y < shape[1]; ++y) {
                float preactivated_derivative_on_cost;
                int result_i = channel * shape[0] * shape[1] + x * shape[0] + y;
                if (last_layer) preactivated_derivative_on_cost = derivatives_on_cost[result_i];
                else preactivated_derivative_on_cost = derivatives_on_cost[result_i] * derivatives_of_activation[result_i];
                biases_gradient[channel] -= params.b_conv_learning_rate / params.batchsize / (shape[0] * shape[1]) * pow(stride, 2) * preactivated_derivative_on_cost;
                for (int previous_channel = 0; previous_channel < previous_shape[2]; ++previous_channel) {
                    for (int kernel_x = max(0, padding - x); kernel_x < kernel_size && x - padding + kernel_x * stride < previous_shape[0]; ++kernel_x) {
                        for (int kernel_y = max(0, padding - y); kernel_y < kernel_size && y - padding + kernel_y * stride < previous_shape[1]; ++kernel_y) {
                            int previous_x = x - padding + kernel_x * stride;
                            int previous_y = y - padding + kernel_y * stride;
                            int previous_i = previous_channel * previous_shape[0] * previous_shape[1] + previous_x * previous_shape[0] + previous_y;
                            weights_gradients[channel][previous_channel][kernel_x][kernel_y] -= params.w_conv_learning_rate / params.batchsize / (shape[0] * shape[1]) * pow(stride, 2) * preactivated_derivative_on_cost * from_previous[previous_i];
                            derivatives_for_previous[previous_i] += preactivated_derivative_on_cost * weights[channel][previous_channel][kernel_x][kernel_y];
                        }
                    }
                }
            }
        }
    }
}

void convolutional::apply_adjustments(const hyperparameters &params) {
    for (int channel = 0; channel < shape[2]; ++channel) {
        biases[channel] += biases_gradient[channel];
        biases_gradient[channel] *= params.keep_adj;
        for (int previous_channel = 0; previous_channel < previous_shape[2]; ++previous_channel) {
            for (int kernel_x = 0; kernel_x < kernel_size; ++kernel_x) {
                for (int kernel_y = 0; kernel_y < kernel_size; ++kernel_y) {
                    weights[channel][previous_channel][kernel_x][kernel_y] += weights_gradients[channel][previous_channel][kernel_x][kernel_y];
                    weights_gradients[channel][previous_channel][kernel_x][kernel_y] *= params.keep_adj;
                }
            }
        }
    }
}

void convolutional::print(std::ostream &out) {
    out << kernel_size << "\n";
    out << padding << "\n";
    out << stride << "\n";

    for (int channel = 0; channel < shape[2]; ++channel) {
        out << biases[channel] << "\n";
        for (int previous_channel = 0; previous_channel < previous_shape[2]; ++previous_channel) {
            for (int kernel_x = 0; kernel_x < kernel_size; ++kernel_x) {
                for (int kernel_y = 0; kernel_y < kernel_size; ++kernel_y) {
                    out << weights[channel][previous_channel][kernel_x][kernel_y] << " ";
                }
                out << "\n";
            }
            out << "\n";
        }
    }
}

maxpool::maxpool(std::vector<int> previous_shape, int window_side) : previous_shape(previous_shape), window_side(window_side) {
    if (previous_shape[0] % window_side != 0){
        cerr << "CAUTION: sidelength does not divide side of maxpool layer";
    }
    width = previous_shape[0] / window_side;
    height = previous_shape[1] /window_side;
}

void maxpool::feedforward(const hyperparameters &params, const std::vector<float> &inp, std::vector<float> &derivatives, std::vector<float> &results) {

    for (auto& i : results) i = numeric_limits<float>::min();

    for (int channel = 0; channel < previous_shape[2]; ++channel) {
        for (int x = 0; x < previous_shape[0]; ++x) {
            for (int y = 0; y < previous_shape[1]; ++y) {
                results[channel * width * height + (x/window_side) * width + (y/window_side)] = max (results[channel * width * height + (x/window_side) * width + (y/window_side)], inp[channel * previous_shape[1] * previous_shape[0] + x * previous_shape[0] + y]);
            }
        }
    }

    derivatives = results; // it's not actually the derivatives it is just remembering the maximums so that i can determine the derivatives afterwards
}

void maxpool::adjust_adjustments(const hyperparameters &params, const std::vector<float> &derivatives_of_activation, const std::vector<float> &from_previous, const std::vector<float> &derivatives_on_cost, std::vector<float> &derivatives_for_previous, bool last_layer) {

    for (int channel = 0; channel < previous_shape[2]; ++channel) {
        for (int x = 0; x < previous_shape[0]; ++x) {
            for (int y = 0; y < previous_shape[1]; ++y) {
                if (abs(derivatives_of_activation[channel * width * height + (x/window_side) * width + (y/window_side)] - from_previous[channel * previous_shape[1] * previous_shape[0] + x * previous_shape[0] + y]) < pow(10, -8)){
                    derivatives_for_previous[channel * previous_shape[1] * previous_shape[0] + x * previous_shape[0] + y] = derivatives_on_cost[channel * width * height + (x/window_side) * width + (y/window_side)];
                }
            }
        }
    }
}

void maxpool::apply_adjustments(const hyperparameters &params) {}

void maxpool::print(std::ostream &out) {}

int total_size(vector<int> shape) {return shape[0]*shape[1]*shape[2];}

networks::networks(hyperparameters &params) : shape(params.layer_output_shape), params(params) {

        for (int layer_constructor = 1; layer_constructor < shape.size(); ++layer_constructor) {
            if (params.layer_type[layer_constructor] == 1){
                auto layer = make_unique<fully_connected> (total_size(shape[layer_constructor-1]), total_size(shape[layer_constructor]), params.layer_data[layer_constructor][0]);
                all_layers.push_back(move(layer));
            }
            if (params.layer_type[layer_constructor] == 2){
                vector<int> ld = params.layer_data[layer_constructor]; // for shorter lines

                auto layer = make_unique<convolutional> (shape[layer_constructor - 1], ld[0], ld[1], ld[2], ld[3], ld[4]);

                shape[layer_constructor] = layer->shape;

                all_layers.push_back(move(layer));
            }
            if(params.layer_type[layer_constructor] == 3){

                auto layer = make_unique<maxpool>(shape[layer_constructor - 1], params.layer_data[layer_constructor][0]);

                shape[layer_constructor] = vector<int> {layer->width, layer->height, layer->previous_shape[2]};

                all_layers.push_back(move(layer));
            }
        }

        allzeros = vector<vector<float>>(shape.size());
        for (int allocator = 0; allocator < shape.size(); ++allocator) {
            allzeros[allocator] = vector<float> (total_size(shape[allocator]), 0);
        }

        results = allzeros;
        actderivatives = allzeros;
        a_gradients = allzeros;
    }

vector<float> networks::feedforward(const vector<float> &inp){ //results has to contain the input in the first vector
    results[0] = inp;
    for (int layerfeeder = 1; layerfeeder < shape.size(); ++layerfeeder) {
        all_layers[layerfeeder-1]->feedforward(params, results[layerfeeder-1], actderivatives[layerfeeder], results[layerfeeder]); //-1 in the indexing of all_layers because input_layer is not in all_layers
    }
    return results[shape.size()-1];
}

void networks::train(const vector<vector<float>>& inp, const vector<vector<float>>& labels, int s, int e){
    for (int sample = s; sample < e; ++sample) { //go through whole batch and sum adjustments
        a_gradients = allzeros;
        results[0] = inp[sample];

        for (int layerfeeder = 1; layerfeeder < shape.size(); ++layerfeeder) {
            all_layers[layerfeeder-1]->feedforward(params, results[layerfeeder-1], actderivatives[layerfeeder], results[layerfeeder]); //-1 in the indexing of all_layers because input_layer is not in all_layers
        }

        for (int neuron = 0; neuron < total_size(shape.back()); ++neuron) {
            a_gradients.back()[neuron] = results[shape.size() - 1][neuron] - labels[sample][neuron]; //derivative of cost function (cross entropy) on the output layer (softmax)
        }

        for (int backpropagator = (int)shape.size()-1; backpropagator > 0; --backpropagator) {
            all_layers[backpropagator-1]->adjust_adjustments(params, actderivatives[backpropagator], results[backpropagator-1], a_gradients[backpropagator], a_gradients[backpropagator - 1], backpropagator == shape.size() - 1);
        }
    }

    for (auto& layer : all_layers) {
        layer->apply_adjustments(params);
    }
}
