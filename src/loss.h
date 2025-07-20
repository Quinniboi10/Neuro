#pragma once

#include "layer.h"

enum Loss {
	MSE,
	CROSS_ENTROPY
};

namespace lossFunctions {
	inline float mse(const Layer& output, const Target& target) {
		assert(output.size == target.size());

		float loss = 0;

		for (usize i = 0; i < output.size; i++) {
			assert(std::isfinite(output.activated[i] - target[i]));
			loss += std::pow<float>(output.activated[i] - target[i], 2);
		}

		return loss / output.size;
	}

	inline Gradient mseDeriv(const Layer& output, const Target& target) {
		assert(output.size == target.size());

		Gradient grad(output.size);

		for (usize i = 0; i < output.size; i++) {
			assert(std::isfinite(output.activated[i] - target[i]));
			grad[i] = 2 * (output.activated[i] - target[i]) / output.size;
		}

		return grad;
	}

	inline float crossEntropy(const Layer& output, const Target& target) {
		assert(output.size == target.size());

		float loss = 0.0;
		for (usize i = 0; i < output.size; ++i) {
			assert(std::isfinite(output.activated[i] - target[i]));
			loss -= target[i] * std::log(output.activated[i] + FLT_EPSILON);
		}

		return loss;
	}

	inline Gradient crossEntropyDeriv(const Layer& output, const Target& target) {
		assert(output.size == target.size());

		Gradient grad(output.size);

		for (usize i = 0; i < output.size; i++)
			grad[i] = -target[i] / output.activated[i];

		return grad;
	}
}

inline float getLoss(const Loss func, const Layer& output, const Target& target) {
	using namespace lossFunctions;

	switch (func) {
	case MSE: return mse(output, target);
	case CROSS_ENTROPY: return crossEntropy(output, target);
	}
}

inline Gradient lossDeriv(const Loss func, const Layer& output, const Target& target) {
	using namespace lossFunctions;

	switch (func) {
	case MSE: return mseDeriv(output, target);
	case CROSS_ENTROPY: return crossEntropyDeriv(output, target);
	}
}