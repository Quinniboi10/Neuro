#pragma once

#include "network.h"

#include <fstream>

static inline void saveWeights(const string path, const Network& net) {
	std::ofstream file(path, std::ios::binary);

	const auto write = [&](const auto& val) {
		file.write(reinterpret_cast<const char*>(&val), sizeof(val));
	};

	write(net.layers.size());

	for (const Layer& l : net.layers) {
		write(l.size);
		write(l.activation);
		for (const vector<float>& weights : l.weights)
			for (const float f : weights)
				write(f);

		for (const float bias : l.biases)
			write(bias);
	}
}

static inline Network loadWeights(const string path) {
    std::vector<Layer> layers;
    std::ifstream file(path, std::ios::binary);

    if (!file)
        exitWithMsg("File not found " + path, -1);

    // Lambda for reading data, file by reference
    const auto read = [&file](auto* ptr, size_t size) {
        file.read(reinterpret_cast<char*>(ptr), size);
    };

    usize numLayers;
    read(&numLayers, sizeof(usize));

    for (usize i = 0; i < numLayers; ++i) {
        usize lSize;
        Activation lAct;
        read(&lSize, sizeof(usize));
        read(&lAct, sizeof(Activation));

        Layer layer(lSize, lAct);

        if (i > 0) {
            layer.init(layers[i - 1]);

            // Read weights
            for (auto& weights : layer.weights) {
                for (float& f : weights) {
                    read(&f, sizeof(float));
                }
            }

            // Read biases
            for (float& bias : layer.biases) {
                read(&bias, sizeof(float));
            }
        }

        layers.push_back(std::move(layer));
    }

    return Network(layers);
}