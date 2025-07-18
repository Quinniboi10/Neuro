#pragma once

#include <fmt/fmt/format.h>

#include "dataloader.h"
#include "lrschedule.h"
#include "progbar.h"
#include "optim.h"

#include <string_view>
#include <numeric>
#include <mutex>
#include <omp.h>

struct Learner {
	Network& net;
	DataLoader& dataLoader;
	optimizers::Optimizer& optimizer;

	Learner(Network& net, DataLoader& dataLoader, optimizers::Optimizer& optimizer) : net(net), dataLoader(dataLoader), optimizer(optimizer) {}

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

	void applyGradients(const Network& net, optimizers::Optimizer& optim, const usize batchSize, const MultiVector<float, 3>& weightGradAccum, const MultiVector<float, 2>& biasGradAccum) {
		// Apply gradients to weights and biases
		for (usize l = 1; l < net.layers.size(); l++) {
			const Layer& currLayer = net.layers[l];
			for (usize i = 0; i < currLayer.size; i++) {
				for (usize j = 0; j < currLayer.weights[i].size(); j++) {
					assert(l - 1 < weightGradAccum.size());
					assert(i < weightGradAccum[l - 1].size());
					assert(j < weightGradAccum[l - 1][i].size());
					assert(l < optim.weightGradients.size());
					assert(i < optim.weightGradients[l].size());
					assert(j < optim.weightGradients[l][i].size());

					optim.weightGradients[l][i][j] += weightGradAccum[l - 1][i][j] / batchSize;
				}
				optim.biasGradients[l][i] += biasGradAccum[l - 1][i] / batchSize;
			}
		}
	}

	void learn(LRSchedule& lrSchedule, usize epochs, usize threads = 0) {
		if (threads == 0)
			threads = std::thread::hardware_concurrency();
		if (threads == 0) {
			cerr << "Failed to detect number of threads" << endl;
			threads = 1;
		}

		const u64 batchSize = dataLoader.batchSize;
		u64 batchesPerEpoch = dataLoader.numSamples / batchSize;

		// Hide cursor
		cout << "\033[?25l";

		cout << "Training for " << batchesPerEpoch * epochs << " batches with " << batchesPerEpoch << " batches per epoch" << endl;

		cout << "Epoch    Train loss    Test loss     Train accuracy     Test accuracy" << endl;
		cout << endl;
		cout << endl;

		const auto getTestLossAcc = [&]() {
			float loss = 0;
			usize numCorrect = 0;
			dataLoader.loadTestSet();
			usize testSize = dataLoader.batchData().size();
			while (dataLoader.hasNext()) {
				DataPoint data = dataLoader.next();
				net.load(data);
				net.forwardPass();
				loss += Learner::mse(net.layers.back(), data.target);
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

		MultiVector<float, 3> weightGradAccum;
		MultiVector<float, 2> biasGradAccum;

		MultiVector<float, 4> threadWeightGradAccum;
		MultiVector<float, 3> threadBiasGradAccum;
		vector<Network> networks;

		vector<float> losses;

		weightGradAccum.reserve(net.layers.size());
		biasGradAccum.reserve(net.layers.size());

		threadWeightGradAccum.reserve(threads);
		threadBiasGradAccum.reserve(threads);
		networks.reserve(threads);

		for (usize l = 1; l < net.layers.size(); l++) {
			weightGradAccum.push_back(MultiVector<float, 2>(net.layers[l].weights.size(), vector<float>(net.layers[l].weights[0].size(), 0.0f)));
			biasGradAccum.push_back(vector<float>(net.layers[l].biases.size(), 0.0f));
		}

		for (usize t = 0; t < threads; t++) {
			threadWeightGradAccum.push_back(weightGradAccum);
			threadBiasGradAccum.push_back(biasGradAccum);
			networks.push_back(net);
		}

		for (usize epoch = 0; epoch < epochs; epoch++) {
			dataLoader.asyncPreloadoadBatch(batchSize);

			ProgressBar progressBar{};

			u64 batch = 0;

			float trainLossSum = 0.0f;
			usize trainCorrect = 0;
			usize trainTotal = 0;

			while (batch < batchesPerEpoch) {
				deepFill(weightGradAccum, 0);
				deepFill(biasGradAccum, 0);

				deepFill(threadWeightGradAccum, 0);
				deepFill(threadBiasGradAccum, 0);

				deepFill(networks, net);

				optimizer.zeroGrad();

				// Dataloader mutex
				std::mutex dlMut;

				dataLoader.waitForBatch();
				dataLoader.swapBuffers();

				// Start async loading the next batch's data into the buffer as soon as possible
				dataLoader.asyncPreloadoadBatch(batchSize);

				#pragma omp parallel for num_threads(threads) reduction(+:trainLossSum, trainCorrect, trainTotal)
				for (usize idx = 0; idx < batchSize; idx++) {
					usize tID = omp_get_thread_num();

					Network& net = networks[tID];

					dlMut.lock();
					DataPoint data = dataLoader.batchData()[idx];
					dlMut.unlock();
					net.load(data);
					net.forwardPass();

					// Accumulate training loss
					float loss = Learner::mse(net.layers.back(), data.target);
					trainLossSum += loss;

					// Accumulate training accuracy
					usize guess = 0, goal = 0;
					for (usize i = 0; i < data.target.size(); i++) {
						if (net.layers.back().activated[i] > net.layers.back().activated[guess])
							guess = i;
						if (data.target[i] > data.target[goal])
							goal = i;
					}
					trainCorrect += (guess == goal);
					trainTotal++;

					// Backward + accumulate gradients
					auto gradients = backward(net, data.target);
					for (usize l = 1; l < net.layers.size(); l++) {
						const Layer& prevLayer = net.layers[l - 1];
						for (usize i = 0; i < net.layers[l].size; i++) {
							for (usize j = 0; j < prevLayer.size; j++) {
								threadWeightGradAccum[tID][l - 1][i][j] += gradients[l][i] * prevLayer.activated[j];
							}
							threadBiasGradAccum[tID][l - 1][i] += gradients[l][i];
						}
					}
				}

				// Accumulate
				for (usize t = 0; t < threads; t++) {
					for (usize l = 1; l < net.layers.size(); l++) {
						for (usize i = 0; i < net.layers[l].size; i++) {
							for (usize j = 0; j < net.layers[l - 1].size; j++) {
								weightGradAccum[l - 1][i][j] += threadWeightGradAccum[t][l - 1][i][j];
							}
							biasGradAccum[l - 1][i] += threadBiasGradAccum[t][l - 1][i];
						}
					}
				}

				applyGradients(net, optimizer, batchSize, weightGradAccum, biasGradAccum);
				optimizer.clipGrad(1);
				optimizer.step(lrSchedule.lr(epoch));
				batch++;

				// Update trainLoss/trainAcc after each batch
				float trainLoss = trainLossSum / (trainTotal ? trainTotal : 1);
				float trainAcc = trainCorrect / static_cast<float>(trainTotal ? trainTotal : 1);

				cursor::up();
				cursor::up();
				cursor::begin();
				cout << fmt::format("{:>5L}{:>14.5f}{:>13}{:>18.2f}%{:>18}", epoch, trainLoss, "Pending", trainAcc * 100, "Pending") << endl;
				cout << progressBar.report(batch, batchesPerEpoch, 63) << "      " << endl;
			}

			float trainLoss = trainLossSum / (trainTotal ? trainTotal : 1);
			float trainAcc = trainCorrect / static_cast<float>(trainTotal ? trainTotal : 1);

			auto testLA = getTestLossAcc();

			cursor::up();
			cursor::clear();
			cursor::up();
			cout << fmt::format("{:>5L}{:>14.5f}{:>13.5f}{:>18.2f}%{:>17.2f}%", epoch, trainLoss, testLA.first, trainAcc * 100, testLA.second * 100) << endl;
			cout << endl;
			cout << endl;
		}

		cursor::up();
		cursor::up();

		// Show cursor
		cout << "\033[?25h";
	}
};