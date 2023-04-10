#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cassert>
#include <memory>
#include <algorithm>

#include "network.h"


using namespace std;

template <typename T>
ostream& operator<< (ostream& out, const vector<T>& v) {
    if ( !v.empty() ) {
        out << '[';
        for (const auto& value: v)
            out << value << ", ";
        out << "\b\b]"; // use two ANSI backspace characters '\b' to overwrite final ", "
    }
    return out;
} // From https://stackoverflow.com/questions/10750057/how-do-i-print-out-the-contents-of-a-vector


void printsample(int n, vector<vector<float>>& inps, vector<vector<float>>& outs){
    for (int sampleprintr = 0; sampleprintr < n; ++sampleprintr) {
        cout << "Output: " << outs[sampleprintr];

        for (int inp = 0; inp < 784; ++inp) {
            if (inp % 28 == 0) cout << endl;
            if(inps[sampleprintr][inp] != 0) cout << "@@";
            else cout << "  ";
        }

        cout << endl << endl << endl;
    }
}

float cost(vector<float> &out, vector<float> &target){ // cross entropy
    float sum = 0;
    for (int i = 0; i < out.size(); ++i) {
        sum -= target[i]*log(out[i]);
    }
    return sum;
}

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
        if(h == labels[sample]) ac += 1/(float)images.size();
    }
    return ac;
}

int main(){
    srand(time(NULL));

    hyperparameters params;

    params.layer_output_shape = {{28, 28, 1}, {}, {30, 1, 1}, {10, 1, 1}}; //{} means layer can determine
    params.layer_type = {0, 2, 1, 1}; // 0 = input, 1 = fully connected, 2 = convolutional, 3 = maxpool
    params.layer_data = {{}, {3, 5, 2, 1, 0}, {0}, {1}}; // fully connected layer data: {0 = relu/1 = softmax}, convolutional data : {channels, kernel_size, padding, stride, activation_type (0 = relu)}, maxpool layer: {sidelength of pooling window}
    params.samples = 60000;
    params.test_samples = 10000;
    params.wlearning_rate = 0.1;
    params.blearning_rate = 1;
    params.w_conv_learning_rate = 1;
    params.b_conv_learning_rate = 1;
    params.L2_frac = 1; //how much the weights should be shrunk by afer each batch (gets divided by samples)
    params.keep_adj = 0.0; //momentum based learning -> probably only do when decreasing learning_rate
    params.batchsize = 10;
    params.epochs = 50;

    auto start = chrono::high_resolution_clock::now();
    cin.tie(0);
    ios_base::sync_with_stdio(false);
    vector<vector<float>> inps (params.samples, vector<float> (784));
    vector<vector<float>> inpst (params.test_samples, vector<float> (784));
    vector<vector<float>> outs (params.samples, vector<float> (10, 0));
    vector<vector<float>> outst (params.test_samples, vector<float> (10, 0));

    vector<int> ilabels (params.samples);
    vector<int> ilabelst (params.test_samples);

    fstream inp;
    inp.open("../cmake-build-release/inp.txt", ios::in);
    int sample;
    for (int inpreader = 0; inpreader < params.samples; ++inpreader) {
        for (int inner = 0; inner < 784; ++inner) {
            inp >> sample;
            inps[inpreader][inner] = (float)sample; // i use int for reading for faster parsing
        }
    }
    inp.close();
    inp.open("../cmake-build-release/inpt.txt", ios::in); // test set
    for (int inpreader = 0; inpreader < params.test_samples; ++inpreader) {
        for (int inner = 0; inner < 784; ++inner) {
            inp >> sample;
            inpst[inpreader][inner] = (float)sample;
        }
    }

    fstream labels;
    labels.open("../cmake-build-release/labels.txt", ios::in);
    int answer;
    for (int labelreader = 0; labelreader < params.samples; ++labelreader) {
        labels >> answer;
        ilabels[labelreader] = answer;
        outs[labelreader][answer] = 1;
    }
    labels.close();
    labels.open("../cmake-build-release/labelst.txt", ios::in);
    for (int labelreader = 0; labelreader < params.test_samples; ++labelreader) {
        labels >> answer;
        ilabelst[labelreader] = answer;
        outst[labelreader][answer] = 1;
    }

    long long read_in_seconds = chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now()-start).count();

    cout << "finished reading in: " << read_in_seconds << " seconds" << endl;
    start = chrono::high_resolution_clock::now();

    //printsample(min(10, params.samples), inps, outs);

    networks network (params);

    auto print_stats = [&](){
        cout << "after " << params.epochs << " epochs: training accuracy: " << getaccuracy(inps, ilabels, network) << ", test accuracy: " << getaccuracy(inpst, ilabelst, network) << endl;
    };

    for (int epoch = 0; epoch < params.epochs; ++epoch) {
        if (epoch % 5 == 0) cout << "EPOCH: " << epoch << ", training accuracy: " << getaccuracy(inps, ilabels, network) << ", test accuracy: " << getaccuracy(inpst, ilabelst, network) << endl;
        for (int batch = 0; batch < params.samples / params.batchsize; ++batch) {
            network.train(inps, outs, batch*params.batchsize, (batch+1)*params.batchsize);
        }
    }

    cout << "after " << params.epochs << " epochs: training accuracy: " << getaccuracy(inps, ilabels, network) << ", test accuracy: " << getaccuracy(inpst, ilabelst, network) << endl;

    for (int print_sample = 0; print_sample < min(10,params.samples); ++print_sample) {
        cout << network.feedforward(inps[print_sample]) << endl;
    }

    long long trained_in_seconds = chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now()-start).count();
    cout << "finished training in: " << trained_in_seconds << " seconds\n";


    fstream params_file;
    params_file.open("params.txt", ios::out);

    params_file << network.shape.size() << "\n";

    for (auto shapes : network.shape) params_file << shapes << "\n";

    for (auto& layer : network.all_layers){
        layer->print(params_file);
    }

    params_file.close();

    return 0;
}