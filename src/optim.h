#pragma once

#include "network.h"

namespace optimizers {
    struct SGD {
        Network& net;
        float lr;

        SGD(Network& net, float lr) : net(net) {
            this->lr = lr;
        }

		void zeroGrad() {
			for (Layer& l : net.layers) {
				for (vector<float>& row : l.weightGradients)
					for (float& grad : row)
						grad = 0;

				for (float& grad : l.biasGradients)
					grad = 0;
			}
		}

        void clipGrad(float maxNorm) {
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

        void step() {
            for (Layer& l : net.layers)
                l.learn(lr);
        }
    };
}