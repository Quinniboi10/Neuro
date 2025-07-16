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