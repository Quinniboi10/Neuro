#pragma once

#include "types.h"

struct Layer {
	MultiVector<float, 2> weights; // Indexed [currNeuron][prevLayerNeuron]
	vector<float> biases;

	MultiVector<float, 2> weightGradients;
	vector<float> biasGradients;

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

		biases.resize(size);
		biasGradients.resize(size);

		preActivation.resize(size);
		activated.resize(size);
	}

	void construct(const Layer& previous) {
		weights.resize(size);
		for (vector<float>& w : weights)
			w.resize(previous.size);
		weightGradients.resize(size);
		for (vector<float>& w : weightGradients)
			w.resize(previous.size);
	}

	void forward(const Layer& previous) {
		preActivation = biases;
		for (usize curr = 0; curr < preActivation.size(); curr++)
			for (usize prev = 0; prev < previous.activated.size(); prev++)
				preActivation[curr] += previous.activated[prev] * weights[curr][prev];

		activated = activations::activate(activation, preActivation);
	}

	void learn(float lr) {
		// Update weights
		for (usize i = 0; i < weights.size(); i++)
			for (usize j = 0; j < weights[i].size(); j++)
				weights[i][j] -= lr * weightGradients[i][j];

		// Update biases
		for (usize i = 0; i < biases.size(); i++)
			biases[i] -= lr * biasGradients[i];
	}
};

struct Grad {
	vector<float> underlying;

	Grad() = default;
	Grad(usize size) { underlying.resize(size); }

	void resize(usize size) {
		underlying.resize(size);
	}

	auto begin() { return underlying.begin(); }
	auto end() { return underlying.end(); }

	usize size() { return underlying.size(); }

	Grad operator+(const Grad& other) const {
		assert(underlying.size() == other.underlying.size());
		Grad vec(underlying.size());

		for (usize i = 0; i < underlying.size(); i++)
			vec[i] = underlying[i] + other[i];

		return vec;
	}

	Grad operator+=(const Grad& other) {
		assert(underlying.size() == other.underlying.size());

		for (usize i = 0; i < underlying.size(); i++)
			underlying[i] += other[i];

		return *this;
	}

	Grad operator/(const float value) const {
		Grad vec(underlying.size());

		for (usize i = 0; i < underlying.size(); i++)
			vec[i] = underlying[i] / value;

		return vec;
	}

	Grad operator/=(const float value) {

		for (usize i = 0; i < underlying.size(); i++)
			underlying[i] /= value;

		return *this;
	}

	float& operator[](usize idx) { return underlying[idx]; }
	const float& operator[](usize idx) const { return underlying[idx]; }
};