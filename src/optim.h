#pragma once

#include "network.h"

namespace optimizers {
    struct Optimizer {
        Network& net;
        float momentum;

        MultiVector<float, 3> weightGradients;
        MultiVector<float, 2> biasGradients;

        Optimizer(Network& net, float momentum = 0.9f) : net(net), momentum(momentum) {
            for (Layer& l : net.layers) {
                if (!l.weights.empty() && !l.weights[0].empty())
                    weightGradients.emplace_back(l.weights.size(), vector<float>(l.weights[0].size()));
                else
                    weightGradients.emplace_back();
                biasGradients.emplace_back(l.biases.size());
            }
        }

        void zeroGrad() {
            deepFill(weightGradients, 0);
            deepFill(biasGradients, 0);
        }

        void clipGrad(float maxNorm) {
            // Compute total norm of all gradients (weights and biases) across all layers
            float totalNormSq = 0.0f;
            for (const auto& layerGradients : weightGradients) {
                // Weights gradients
                for (const auto& row : layerGradients)
                    for (float wg : row)
                        totalNormSq += wg * wg;
            }
            for (const auto& layerGradients : biasGradients) {
                // Bias gradients
                for (float bg : layerGradients)
                    totalNormSq += bg * bg;
            }

            float totalNorm = std::sqrt(totalNormSq);

            // Scale all gradients if needed
            if (totalNorm > maxNorm && totalNorm > 0.0f) {
                float scale = maxNorm / totalNorm;
                for (auto& layerGradients : weightGradients) {
                    // Weights gradients
                    for (auto& row : layerGradients)
                        for (float& wg : row)
                            wg *= scale;
                }
                for (auto& layerGradients : biasGradients) {
                    // Bias gradients
                    for (float& bg : layerGradients)
                        bg *= scale;
                }
            }
        }

        virtual void step(float lr) = 0;
    };

    struct SGD : Optimizer {
        MultiVector<float, 3> weightVelocities;
        MultiVector<float, 2> biasVelocities;

        SGD(Network& net, float momentum = 0.9f) : Optimizer(net, momentum) {
            for (Layer& l : net.layers) {
                if (!l.weights.empty() && !l.weights[0].empty())
                    weightVelocities.emplace_back(l.weights.size(), vector<float>(l.weights[0].size()));
                else
                    weightVelocities.emplace_back();
                biasVelocities.emplace_back(l.biases.size());
            }
        }

        inline void step(float lr) override {
            for (usize lIdx = 1; lIdx < net.layers.size(); lIdx++) {
                Layer& l = net.layers[lIdx];
                // Update weights with momentum
                for (usize i = 0; i < l.weights.size(); i++) {
                    for (usize j = 0; j < l.weights[i].size(); j++) {
                        weightVelocities[lIdx][i][j] = momentum * weightVelocities[lIdx][i][j] - lr * weightGradients[lIdx][i][j];
                        l.weights[i][j] += weightVelocities[lIdx][i][j];
                    }
                }

                // Update biases with momentum
                for (usize i = 0; i < l.biases.size(); i++) {
                    biasVelocities[lIdx][i] = momentum * biasVelocities[lIdx][i] - lr * biasGradients[lIdx][i];
                    l.biases[i] += biasVelocities[lIdx][i];
                }
            }
        }
    };

    struct RMSprop : Optimizer {
        float beta;
        float epsilon;
        MultiVector<float, 3> weightSqGrads;
        MultiVector<float, 2> biasSqGrads;

        RMSprop(Network& net, float momentum = 0.9f, float beta = 0.9f, float epsilon = 1e-8f) : Optimizer(net, momentum), beta(beta), epsilon(epsilon) {
            for (Layer& l : net.layers) {
                if (!l.weights.empty() && !l.weights[0].empty())
                    weightSqGrads.emplace_back(l.weights.size(), vector<float>(l.weights[0].size()));
                else
                    weightSqGrads.emplace_back();
                biasSqGrads.emplace_back(l.biases.size());
            }
        }

        inline void step(float lr) override {
            for (usize lIdx = 0; lIdx < net.layers.size(); ++lIdx) {
                Layer& l = net.layers[lIdx];

                // Update weights
                for (usize i = 0; i < l.weights.size(); ++i) {
                    for (usize j = 0; j < l.weights[i].size(); ++j) {
                        weightSqGrads[lIdx][i][j] = beta * weightSqGrads[lIdx][i][j] + (1.0f - beta) * weightGradients[lIdx][i][j] * weightGradients[lIdx][i][j];

                        l.weights[i][j] -= lr * weightGradients[lIdx][i][j] / (std::sqrt(weightSqGrads[lIdx][i][j]) + epsilon);
                    }
                }

                // Update biases
                for (usize i = 0; i < l.biases.size(); ++i) {
                    biasSqGrads[lIdx][i] = beta * biasSqGrads[lIdx][i] + (1.0f - beta) * biasGradients[lIdx][i] * biasGradients[lIdx][i];

                    l.biases[i] -= lr * biasGradients[lIdx][i] / (std::sqrt(biasSqGrads[lIdx][i]) + epsilon);
                }
            }
        }
    };

    // Heavily based on code from h1me, the developer of the Astra chess engine
    // Thank you for your contribution!
    struct Adam : Optimizer {
        float beta1;
        float beta2;
        float epsilon;
        float decay;
        usize iteration = 0;

        MultiVector<float, 3> weightMomentums;
        MultiVector<float, 3> weightVelocities;
        MultiVector<float, 2> biasMomentums;
        MultiVector<float, 2> biasVelocities;

        Adam(Network& net, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, float decay = 0.01f)
            : Optimizer(net), beta1(beta1), beta2(beta2), epsilon(epsilon), decay(decay) {
            for (Layer& l : net.layers) {
                if (!l.weights.empty() && !l.weights[0].empty()) {
                    weightMomentums.emplace_back(l.weights.size(), vector<float>(l.weights[0].size()));
                    weightVelocities.emplace_back(l.weights.size(), vector<float>(l.weights[0].size()));
                }
                else {
                    weightMomentums.emplace_back();
                    weightVelocities.emplace_back();
                }
                biasMomentums.emplace_back(l.biases.size());
                biasVelocities.emplace_back(l.biases.size());
            }
        }

        inline void step(float lr) override {
            iteration++;
            float biasCorr1 = 1.0f - std::pow(beta1, iteration);
            float biasCorr2 = 1.0f - std::pow(beta2, iteration);

            for (usize lIdx = 1; lIdx < net.layers.size(); ++lIdx) {
                Layer& l = net.layers[lIdx];

                // Update weights
                for (usize i = 0; i < l.weights.size(); ++i) {
                    for (usize j = 0; j < l.weights[i].size(); ++j) {
                        l.weights[i][j] *= (1.0f - lr * decay);

                        weightMomentums[lIdx][i][j] = beta1 * weightMomentums[lIdx][i][j] + (1.0f - beta1) * weightGradients[lIdx][i][j];
                        weightVelocities[lIdx][i][j] = beta2 * weightVelocities[lIdx][i][j] + (1.0f - beta2) * weightGradients[lIdx][i][j] * weightGradients[lIdx][i][j];

                        // Bias correction
                        float mHat = weightMomentums[lIdx][i][j] / biasCorr1;
                        float vHat = weightVelocities[lIdx][i][j] / biasCorr2;

                        l.weights[i][j] -= lr * mHat / (std::sqrt(vHat) + epsilon);
                    }
                }

                // Update biases
                for (usize i = 0; i < l.biases.size(); ++i) {
                    l.biases[i] *= (1.0f - lr * decay);

                    biasMomentums[lIdx][i] = beta1 * biasMomentums[lIdx][i] + (1.0f - beta1) * biasGradients[lIdx][i];
                    biasVelocities[lIdx][i] = beta2 * biasVelocities[lIdx][i] + (1.0f - beta2) * biasGradients[lIdx][i] * biasGradients[lIdx][i];

                    // Bias correction
                    float mHat = biasMomentums[lIdx][i] / biasCorr1;
                    float vHat = biasVelocities[lIdx][i] / biasCorr2;

                    l.biases[i] -= lr * mHat / (std::sqrt(vHat) + epsilon);
                }
            }
        }
    };
}