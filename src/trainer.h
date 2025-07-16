#pragma once

#include <fmt/fmt/format.h>

#include "dataloader.h"
#include "optim.h"

#include <string_view>

struct Trainer {
	static float mse(const Layer& output, const Target& target) {
		assert(output.size == target.size());

		float loss = 0;

		for (usize i = 0; i < output.size; i++) {
			assert(std::isfinite(output.activated[i] - target[i]));
			loss += std::pow<float>(output.activated[i] - target[i], 2);
		}

		return loss / output.size;
	}

	static Grad mseDeriv(const Layer& output, const Target& target) {
		assert(output.size == target.size());

		Grad grad;
		grad.resize(output.size);

		for (usize i = 0; i < output.size; i++) {
			assert(std::isfinite(output.activated[i] - target[i]));
			grad[i] = 2 * (output.activated[i] - target[i]) / output.size;
		}

		return grad;
	}

	static vector<Grad> backward(const Network& net, const Target& target) {
		vector<Grad> grads(net.layers.size());
		for (usize l = 0; l < net.layers.size(); ++l)
			grads[l].resize(net.layers[l].size);

		// Output layer gradient: dL/dA
		grads.back() = mseDeriv(net.layers.back(), target);

		// Hidden layers: Backpropagate error
		for (int l = net.layers.size() - 2; l > 0; --l) {
			const Layer& currLayer = net.layers[l];
			const Layer& nextLayer = net.layers[l + 1];
			for (usize i = 0; i < currLayer.size; ++i) {
				float error = 0.0f;
				for (usize j = 0; j < nextLayer.size; ++j) {
					error += grads[l + 1][j] * nextLayer.weights[j][i];
				}
				grads[l][i] = error * activations::derivActivate(currLayer.activation, currLayer.activated[i]);
			}
		}
		return grads;
	}

	static void applyGradients(Network& net, const usize batchSize, const MultiVector<float, 3>& weightGradAccum, const MultiVector<float, 2>& biasGradAccum) {
		// Apply gradients to weights and biases
		for (usize l = 1; l < net.layers.size(); l++) {
			Layer& currLayer = net.layers[l];
			for (usize i = 0; i < currLayer.size; i++) {
				for (usize j = 0; j < currLayer.weights[i].size(); j++)
					currLayer.weightGradients[i][j] += weightGradAccum[l - 1][i][j] / batchSize;
				currLayer.biasGradients[i] += biasGradAccum[l - 1][i] / batchSize;
			}
		}
	}

	static void train(Network& net, DataLoader& dataLoader, auto& optim, usize batchSize, usize epochs) {
		u64 batches = dataLoader.numSamples * epochs / batchSize;

		cout << "Training for " << batches << " batches" << endl;

		cout << "Batch        Loss        Accuracy" << endl;

		const auto getLossAcc = [&]() {
			float loss = 0;
			usize numCorrect = 0;
			usize testSize = dataLoader.batchData.size();
			while (dataLoader.hasNext()) {
				DataPoint data = dataLoader.next();
				net.load(data);
				net.forwardPass();
				loss += Trainer::mse(net.layers.back(), data.target);
				usize guess = 0, goal = 0;
				for (usize i = 0; i < data.target.size(); i++) {
					if (net.layers.back().activated[i] > net.layers.back().activated[guess])
						guess = i;
					if (data.target[i] > data.target[goal])
						goal = i;
				}
				numCorrect += (guess == goal);
			}
			return std::pair<float, float>{ loss / (testSize ? testSize : 1), numCorrect / static_cast<float>(testSize ? testSize : 1) };
			};

		u64 batch = 0;
		while (batch < batches) {
			MultiVector<float, 3> weightGradAccum;
			MultiVector<float, 2> biasGradAccum;
			for (usize l = 1; l < net.layers.size(); l++) {
				weightGradAccum.push_back(
					vector<vector<float>>(net.layers[l].size, vector<float>(net.layers[l - 1].size, 0.0f))
				);
				biasGradAccum.push_back(vector<float>(net.layers[l].size, 0.0f));
			}

			optim.zeroGrad();

			dataLoader.loadTestSet();
			auto la = getLossAcc();
			dataLoader.loadBatch(batchSize);
			cout << fmt::format("{:>5L}{:>12.5f}{:>15.2f}%", batch, la.first, la.second * 100) << endl;

			for (usize b = 0; b < batchSize; b++) {
				DataPoint data = dataLoader.next();
				net.load(data);
				net.forwardPass();
				auto gradients = backward(net, data.target);

				// Accumulate gradients
				for (usize l = 1; l < net.layers.size(); l++) {
					const Layer& prevLayer = net.layers[l - 1];
					for (usize i = 0; i < net.layers[l].size; i++) {
						for (usize j = 0; j < prevLayer.size; j++) {
							weightGradAccum[l - 1][i][j] += gradients[l][i] * prevLayer.activated[j];
						}
						biasGradAccum[l - 1][i] += gradients[l][i];
					}
				}
			}

			applyGradients(net, batchSize, weightGradAccum, biasGradAccum);
			optim.clipGrad(1);
			optim.step();
			batch++;
		}
	}
};