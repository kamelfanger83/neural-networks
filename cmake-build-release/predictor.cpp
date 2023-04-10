#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>

using namespace std;

const vector<int> gshape {784, 30, 16, 10};
const int samples = 60000;
const float wlearning_rate = 0.1;
const float blearning_rate = 0.1;
const int batchsize = 1;
const int epochs = 30;

void printsample(int n, vector<vector<float>>& inps, vector<vector<float>>& outs){
    for (int sampleprintr = 0; sampleprintr < n; ++sampleprintr) {
        for (int outss = 0; outss < 10; ++outss) {
            cout << outs[sampleprintr][outss] << " ";
        }
        for (int inp = 0; inp < 784; ++inp) {
            if (inp % 28 == 0) cout << endl;
            if(inps[sampleprintr][inp] != 0) cout << "@@";
            else cout << "  ";
        }
        cout << endl << endl << endl;
    }
}

void printvec(vector<float> f){
    for(float t : f) cout << t << ", ";
    cout << "\n";
}

class layers{
    public:
    int neurons, previous_neurons;
    vector<vector<float>> weights, weights_adjustment;
    vector<float> biases, biases_adjustment;
    layers(int size, int previous){
        neurons = size;
        previous_neurons = previous;
        weights = vector<vector<float>> (neurons, vector<float> (previous_neurons));
        biases = vector<float> (neurons, 0);
        for (int initializer = 0; initializer < neurons; ++initializer) {
            for (int inner = 0; inner < previous_neurons; ++inner) {
                weights[initializer][inner] = ((float)rand()/RAND_MAX*2-1)/neurons; //should be improved
            }
        }
        weights_adjustment = vector<vector<float>> (neurons, vector<float> (previous_neurons, 0));
        biases_adjustment = vector<float> (neurons, 0);
    }
    float activate(float x){
        return tanh(x);
    }
    float derive_activation(float x){
        return 1-tanh(x)*tanh(x); //from: https://socratic.org/questions/what-is-the-derivative-of-tanh-x
    }

    void feedforward(const vector<float>& inp, vector<float>& derivatives, vector<float>& results){ //standard fully connected feedforward, can be edited for cnn or so
        for (int neuron = 0; neuron < neurons; ++neuron) {
            float val = 0;
            for (int previous = 0; previous < previous_neurons; ++previous) {
                val += weights[neuron][previous] * inp[previous];
            }
            val += biases[neuron];
            derivatives[neuron] = derive_activation(val);
            results[neuron] = activate(val);
        }
    }

    void adjust_adjustments(const vector<float>& derivatives_of_activation, const vector<float>& from_previous, const vector<float>& derivatives_on_cost, vector<float>& derivatives_for_previous){
        for (int neuron = 0; neuron < neurons; ++neuron) {
            float preactivated_derivative_on_cost = derivatives_on_cost[neuron]*derivatives_of_activation[neuron]; //chain rule
            float nbadjustment = blearning_rate  * preactivated_derivative_on_cost;
            biases_adjustment[neuron] -= blearning_rate  * preactivated_derivative_on_cost; //gradient descent
            for (int previous = 0; previous < previous_neurons; ++previous) {
                float nwadjustment = wlearning_rate  *  preactivated_derivative_on_cost * from_previous[previous];
                weights_adjustment[neuron][previous] -= wlearning_rate  *  preactivated_derivative_on_cost * from_previous[previous]; //chain rule
                derivatives_for_previous[previous] += preactivated_derivative_on_cost*weights[neuron][previous]; //chain rule
            }
        }
    }

    void apply_adjustments(){
        float avb = 0, avw = 0;
        for (int neuron = 0; neuron < neurons; ++neuron) {
            avb += abs(biases_adjustment[neuron])/neurons;
            biases[neuron] += biases_adjustment[neuron];
            biases_adjustment[neuron] *= 0;
            for (int previous = 0; previous < previous_neurons; ++previous) {
                avw += abs(weights_adjustment[neuron][previous])/neurons/previous_neurons;
                weights[neuron][previous] += weights_adjustment[neuron][previous];
                weights_adjustment[neuron][previous] *= 0;
            }
        }
        //cout << "average bias adjustment: " << avb << ", average weight adjustment: " << avw << endl;
    }
};

class networks{
    public:
    vector<int> shape;
    vector<layers> all_layers;
    vector<vector<float>> results, actderivatives, derivatives, zero_derivative; //activated values, derivative of activation at the point it got called, derivatives of neurons on cost

    networks(vector<int> ashape){
        shape = ashape;
        for (int layer_constructor = 1; layer_constructor < shape.size(); ++layer_constructor) {
            layers layer(shape[layer_constructor], shape[layer_constructor-1]);
            all_layers.push_back(layer);
        }
        zero_derivative = vector<vector<float>>(shape.size());
        for (int allocator = 0; allocator < shape.size(); ++allocator) {
            zero_derivative[allocator] = vector<float> (shape[allocator]);
        }
        results = zero_derivative;
        actderivatives = zero_derivative;
        derivatives = zero_derivative;
    }

    vector<float> feedforward(vector<float> inp){ //results has to contain the input in the first vector
        results[0] = inp;
        for (int layerfeeder = 1; layerfeeder < shape.size(); ++layerfeeder) {
            all_layers[layerfeeder-1].feedforward(results[layerfeeder-1], actderivatives[layerfeeder], results[layerfeeder]); //-1 in the indexing of all_layers because input_layer is not in all_layers
        }
        return results[shape.size()-1];
    }

    void train(vector<vector<float>>& inp, vector<vector<float>>& labels, int s, int e){
        for (int sample = s; sample < e; ++sample) { //go through whole batch and sum adjustments
            derivatives = zero_derivative;
            results[0] = inp[sample];
            for (int layerfeeder = 1; layerfeeder < shape.size(); ++layerfeeder) {
                all_layers[layerfeeder-1].feedforward(results[layerfeeder-1], actderivatives[layerfeeder], results[layerfeeder]); //-1 in the indexing of all_layers because input_layer is not in all_layers
            }
            for (int lastlayer = 0; lastlayer < shape.back(); ++lastlayer) {
                derivatives[shape.size()-1][lastlayer] = results[shape.size()-1][lastlayer]-labels[sample][lastlayer]; //derive x^2 -> x
            }
            for (int backpropagator = shape.size()-1; backpropagator > 0; --backpropagator) {
                all_layers[backpropagator-1].adjust_adjustments(actderivatives[backpropagator], results[backpropagator-1], derivatives[backpropagator], derivatives[backpropagator-1]);
            }
        }
        for (int applier = 0; applier < all_layers.size(); ++applier) {
            all_layers[applier].apply_adjustments();
        }
    }

    void printbiases(){
        for(layers layer : all_layers){
            for (int neuron = 0; neuron < layer.neurons; ++neuron) {
                cout << layer.biases[neuron] << endl;
            }
            cout << endl;
        }
    }

    void printweights(){
        for(layers layer : all_layers){
            for (int neuron = 0; neuron < layer.neurons; ++neuron) {
                for (int previous = 0; previous < layer.previous_neurons; ++previous) {
                    cout << layer.weights[neuron][previous] << ", ";
                }
                cout << endl;
            }
            cout << endl;
        }
    }
};

int highest(vector<float> l){
    int in = 0;
    float h = -2;
    for (int i = 0; i < l.size(); ++i) {
        if (l[i] > h){
            h = l[i];
            in = i;
        }
    }
    return in;
}

float getaccuracy(vector<vector<float>>& images, vector<int>& labels, networks& network){
    float ac = 0;
    for (int sample = 0; sample < images.size(); ++sample) {
        int h = highest(network.feedforward(images[sample]));
        if(h == labels[sample]) ac += (float)1/images.size();
    }
    return ac;
}

int main(){
    //srand(time(NULL));

    networks network (gshape);

    fstream params;
    params.open("params.txt", ios::in);
    
    float trash;
    params >> trash;
    params >> trash;
    params >> trash;
    params >> trash;
    params >> trash;
    float read;
    for (int wp = 1; wp < network.shape.size(); ++wp) {
        for (int neuron = 0; neuron < network.shape[wp]; ++neuron) {
            for (int previopus = 0; previopus < network.shape[wp - 1]; ++previopus) {
                params >> read;
                network.all_layers[wp-1].weights[neuron][previopus] = read;
            }
        }
    }
    for (int bp = 1; bp < network.shape.size(); ++bp) {
        for (int neuron = 0; neuron < network.shape[bp]; ++neuron) {
            params >> read;
            network.all_layers[bp-1].biases[neuron] = read;
        }
    }

    params.close();

    /*network.printweights();
    cout << endl;
    network.printbiases();*/

    vector<vector<float>> images (samples, vector<float> (784));
    vector<int> labels (samples);

    fstream inp;
    inp.open("inp.txt", ios::in);
    float sample;
    for (int inpreader = 0; inpreader < samples; ++inpreader) {
        for (int inner = 0; inner < 784; ++inner) {
            inp >> sample;
            images[inpreader][inner] = sample;
        }
    }

    fstream labelsf;
    labelsf.open("labels.txt", ios::in);
    int answer;
    for (int labelreader = 0; labelreader < samples; ++labelreader) {
        labelsf >> answer;
        labels[labelreader] = answer;
    }

    cout << getaccuracy(images, labels, network);

    return 0;
}