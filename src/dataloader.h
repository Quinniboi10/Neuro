#pragma once

#include "layer.h"
#include "util.h"

#include <filesystem>
#include <fstream>
#include <thread>
#include <future>
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

    usize currBatch;
    std::future<void> dataFuture;
    array<vector<DataPoint>, 2> data;

    DataLoader(u64 batchSize, float trainSplit, u64 threads) {
        this->batchSize = batchSize;
        this->trainSplit = trainSplit;
        this->currBatch = 0;

        data[0].reserve(batchSize);
        data[1].reserve(batchSize);

        this->threads = threads;
    }

    // Loads batch into other buffer
    virtual void loadBatch(usize batchSize, usize batchIdx) = 0;
    virtual void loadTestSet() = 0;
    virtual bool hasNext() const = 0;
    virtual DataPoint next() = 0;

    // Attempts to load data asynchronously if threads > 0
    virtual void asyncPreloadoadBatch(usize batchSize) {
        dataFuture = std::async(threads > 0 ? std::launch::async : std::launch::deferred, [this, batchSize]() { this->loadBatch(batchSize, currBatch ^ 1); });
    }

    virtual void waitForBatch() {
        if (dataFuture.valid())
            dataFuture.get();
    }

    virtual vector<DataPoint>& batchData() {
        return data[currBatch];
    }

    virtual void swapBuffers() {
        currBatch ^= 1;
    }

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

    void loadBatch(usize batchSize, usize batchIdx) override;

    void loadTestSet() override;

    bool hasNext() const override;

    DataPoint next() override;
};