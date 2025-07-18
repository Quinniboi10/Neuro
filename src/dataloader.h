#pragma once

#include "layer.h"
#include "util.h"

#include <filesystem>
#include <fstream>
#include <thread>
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
    u64 threads;
    u64 batchSize;
    float trainSplit;
    u64 numSamples;
    vector<DataPoint> batchData;

    DataLoader(u64 batchSize, float trainSplit, u64 threads) {
        this->batchSize = batchSize;
        this->trainSplit = trainSplit;

        if (threads == 0)
            threads = std::thread::hardware_concurrency();
        if (threads == 0) {
            cerr << "Failed to detect number of threads" << endl;
            threads = 1;
        }

        this->threads = threads;
    }

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

    ImageDataLoader(const string path, u64 batchSize, float trainSplit, u64 threads = 0, usize width = 0, usize height = 0);

    void loadBatch(usize batchSize) override;

    void loadTestSet() override;

    bool hasNext() const override;

    DataPoint next() override;
};