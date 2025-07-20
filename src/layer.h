#pragma once

#include "types.h"

struct Layer {
	MultiVector<float, 2> weights; // Indexed [currNeuron][prevLayerNeuron]
	vector<float> biases;

	vector<float> preActivation;
	vector<float> activated;

	Activation activation;

	usize size;

	Layer() = default;

	Layer(const InputLayer& from) {
		size = from.size();
		preActivation = from;
		activated = from;
	}

	Layer(const usize size, const Activation activation) {
		this->activation = activation;
		this->size = size;

		biases.resize(size);\

		preActivation.resize(size);
		activated.resize(size);
	}

	void init(const Layer& previous) {
		weights.resize(size);
		for (vector<float>& w : weights)
			w.resize(previous.size);

	}

	void forward(const Layer& previous) {
		preActivation = biases;
		for (usize curr = 0; curr < preActivation.size(); curr++)
			for (usize prev = 0; prev < previous.activated.size(); prev++)
				preActivation[curr] += previous.activated[prev] * weights[curr][prev];

		activated = activations::activate(activation, preActivation);
	}
};