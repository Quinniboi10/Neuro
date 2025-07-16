#pragma once

#include "network.h"

namespace optimizers {
    static void zeroGrad(Network& net) {
        for (Layer& l : net.layers) {
            for (vector<float>& row : l.weightGradients)
                for (float& grad : row)
                    grad = 0;

            for (float& grad : l.biasGradients)
                grad = 0;
        }
    }

    static void clipGrad(Network& net, float maxNorm) {
        // Compute total norm of all gradients (weights and biases) across all layers
        float totalNormSq = 0.0f;
        for (const auto& layer : net.layers) {
            // Weights gradients
            for (const auto& row : layer.weightGradients)
                for (float wg : row)
                    totalNormSq += wg * wg;
            // Bias gradients
            for (float bg : layer.biasGradients)
                totalNormSq += bg * bg;
        }
        float totalNorm = std::sqrt(totalNormSq);

        // Scale all gradients if needed
        if (totalNorm > maxNorm && totalNorm > 0.0f) {
            float scale = maxNorm / totalNorm;
            for (auto& layer : net.layers) {
                // Scale weight gradients
                for (auto& row : layer.weightGradients)
                    for (float& wg : row)
                        wg *= scale;
                // Scale bias gradients
                for (float& bg : layer.biasGradients)
                    bg *= scale;
            }
        }
    }

    struct Optimizer {
        Network& net;
        float lr;
        float momentum;

        Optimizer(Network& net, float lr, float momentum = 0.9f) : net(net), lr(lr), momentum(momentum) {}

        virtual void step() = 0;
    };

    struct SGD : Optimizer {
        using Optimizer::Optimizer;

        inline void step() override {
            for (Layer& l : net.layers) {
                // Update weights with momentum
                for (usize i = 0; i < l.weights.size(); i++) {
                    for (usize j = 0; j < l.weights[i].size(); j++) {
                        l.weightVelocities[i][j] = momentum * l.weightVelocities[i][j] - lr * l.weightGradients[i][j];
                        l.weights[i][j] += l.weightVelocities[i][j];
                    }
                }

                // Update biases with momentum
                for (usize i = 0; i < l.biases.size(); i++) {
                    l.biasVelocities[i] = momentum * l.biasVelocities[i] - lr * l.biasGradients[i];
                    l.biases[i] += l.biasVelocities[i];
                }
            }
        }
    };
}