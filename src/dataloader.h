#pragma once

#include "layer.h"
#include "util.h"

#include <compare>
#include <filesystem>
#include <fstream>
#include <random>

struct DataPoint {
	InputLayer input;
	Target target;

	DataPoint(const InputLayer& input, const Target& target) {
		this->input = input;
		this->target = target;
	}
};

InputLayer loadGreyscaleImage(const std::string& path, usize w, usize h);

struct DataLoader {
    float trainSplit;
    u64 numSamples;
    vector<DataPoint> batchData;

    virtual void loadBatch(usize batchSize) = 0;
    virtual void loadTestSet() = 0;
    virtual bool hasNext() const = 0;
    virtual DataPoint next() = 0;
    virtual ~DataLoader() = default;
};

struct ImageDataLoader : DataLoader {
    string dataDir;
    vector<string> types;
    vector<u64> samplesPerType;
    std::mt19937 rng{ std::random_device{}() };

    usize width;
    usize height;

    ImageDataLoader(const string path, usize width, usize height, float trainSplit);

    void loadBatch(usize batchSize) override;

    void loadTestSet() override;

    bool hasNext() const override;

    DataPoint next() override;
};