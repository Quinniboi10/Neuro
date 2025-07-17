#pragma once

#include "dataloader.h"
#include "util.h"

struct Network {
	vector<Layer> layers;

	Network(usize inputSize, usize outputSize, Activation outputActivation) {
		layers.resize(2);
		layers[0] = InputLayer(inputSize);
		layers[1] = Layer(outputSize, outputActivation);
	}

    explicit Network(const vector<Layer>& layers) : layers(layers) {}

    void init(const bool useXavierInit = true) {
        std::random_device rd;
        std::mt19937 gen(rd());

        // Resize all the weight vectors
        for (usize l = 1; l < layers.size(); l++)
            layers[l].init(layers[l - 1]);

        for (usize l = 1; l < layers.size(); ++l) {
            Layer& layer = layers[l];
            usize fanIn = layers[l - 1].size;
            usize fanOut = layer.size;

            if (useXavierInit) {
                // Xavier Uniform
                float limit = std::sqrt(6.0f / (fanIn + fanOut));
                std::uniform_real_distribution<float> dis(-limit, limit);

                for (usize i = 0; i < layer.weights.size(); i++)
                    for (usize j = 0; j < layer.weights[i].size(); j++)
                        layer.weights[i][j] = dis(gen);

                for (usize i = 0; i < layer.biases.size(); i++)
                    layer.biases[i] = 0.0f;
            }
            else {
                // He Normal
                float stddev = std::sqrt(2.0f / fanIn);
                std::normal_distribution<float> dis(0.0f, stddev);

                for (usize i = 0; i < layer.weights.size(); i++)
                    for (usize j = 0; j < layer.weights[i].size(); j++)
                        layer.weights[i][j] = dis(gen);

                for (usize i = 0; i < layer.biases.size(); i++)
                    layer.biases[i] = 0.0f;
            }
        }
    }

	void load(InputLayer input) {
		layers[0] = input;
	}
	void load(DataPoint data) {
		load(data.input);
	}

	Network& addLayer(usize size, Activation activation) {
		layers.resize(layers.size() + 1);
		layers.back() = layers[layers.size() - 2];
		layers[layers.size() - 2] = Layer(size, activation);

		return *this;
	}

	void forwardPass() {
		for (usize i = 1; i < layers.size(); i++)
			layers[i].forward(layers[i - 1]);
	}

	const vector<float>& output() const {
		return layers.back().activated;
	}

};