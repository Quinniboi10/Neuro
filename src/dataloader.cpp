#include "dataloader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../external/stb_image.h"

#include <mutex>

InputLayer loadGreyscaleImage(const std::string& path, usize w, usize h) {
    int width, height, channels;
    unsigned char* data = stbi_load(path.data(), &width, &height, &channels, 1);
    if (!data)
        throw std::runtime_error("Failed to load image: " + path);

    InputLayer vec(width * height);

    if ((w == static_cast<usize>(width) || w == 0) && (h == static_cast<usize>(height) || h == 0)) {
        for (usize i = 0; i < width * height; i++)
            vec[i] = static_cast<float>(data[i]) / 255;
    }
    else {
        // Simple nearest-neighbor resize
        for (usize y = 0; y < h; ++y) {
            for (usize x = 0; x < w; ++x) {
                int sourceX = x * width / w;
                int sourceY = y * height / h;
                int sourceIdx = sourceY * width + sourceX;
                int destIdx = y * w + x;
                vec[destIdx] = static_cast<float>(data[sourceIdx]) / 255;
            }
        }
    }

    stbi_image_free(data);
    return vec;
}

ImageDataLoader::ImageDataLoader(const string path, u64 batchSize, float trainSplit, u64 threads, usize width, usize height)
        : DataLoader(batchSize, trainSplit, threads)
    {
    this->width = width;
    this->height = height;

    cout << "Attempting to open data dir: '" << path << "'" << endl;
    if (!std::filesystem::exists(path) || !std::filesystem::is_directory(path))
        throw std::runtime_error("Data directory does not exist or is not a directory: " + path);

    this->dataDir = path;

    for (const auto& entry : std::filesystem::directory_iterator(this->dataDir)) {
        if (entry.is_directory())
            types.push_back(entry.path().string());
    }

    cout << "Found " << types.size() << " types" << endl;

    samplesPerType.resize(types.size());

    numSamples = 0;
    for (usize typeIdx = 0; typeIdx < types.size(); typeIdx++) {
        for (const auto& entry : std::filesystem::directory_iterator(types[typeIdx])) {
            if (entry.is_regular_file()) {
                numSamples++;
                samplesPerType[typeIdx]++;
            }
        }
    }

    cout << "Using train to test ratio of " << trainSplit / (1 - trainSplit) << " with approximately " << formatNum(numSamples * trainSplit) << " train samples and " << formatNum(numSamples * (1 - trainSplit)) << " test samples" << endl;
}

void ImageDataLoader::loadBatch(usize batchSize, usize batchIdx) {
    data[batchIdx].clear();

    if (types.empty())
        throw std::runtime_error("No types found in data dir: " + dataDir);

    std::mutex dataMut;

    #pragma omp parallel for num_threads(threads)
    for (usize i = 0; i < batchSize; i++) {
        // Randomly pick a type
        std::uniform_int_distribution<usize> typeDist(0, types.size() - 1);
        usize typeIdx = typeDist(rng);
        const string& typeDir = types[typeIdx];

        // Gather image files in that directory
        std::vector<std::filesystem::path> imgs;
        for (const auto& entry : std::filesystem::directory_iterator(typeDir)) {
            if (entry.is_regular_file())
                imgs.push_back(entry.path());
        }

        if (imgs.empty())
            throw std::runtime_error("No images in type dir: " + typeDir);

        // Randomly pick an image
        std::uniform_int_distribution<usize> imgDist(0, imgs.size() * trainSplit - 1);
        usize imgIdx = imgDist(rng);

        InputLayer input = loadGreyscaleImage(imgs[imgIdx].string(), width, height);
        Target target(types.size());
        target[typeIdx] = 1;

        dataMut.lock();
        data[batchIdx].emplace_back(input, target);
        dataMut.unlock();
    }
}

void ImageDataLoader::loadTestSet() {
    data[currBatch].clear();

    if (types.empty())
        throw std::runtime_error("No types found in data dir: " + dataDir);

    for (usize typeIdx = 0; typeIdx < types.size(); typeIdx++) {
        u64 currIdx = 0;
        for (const auto& entry : std::filesystem::directory_iterator(types[typeIdx])) {
            if (entry.is_regular_file()) {
                if (currIdx < samplesPerType[typeIdx] * trainSplit - 1) {
                    currIdx++;
                    continue;
                }

                InputLayer input = loadGreyscaleImage(entry.path().string(), width, height);
                Target target(types.size());
                target[typeIdx] = 1;

                data[currBatch].emplace_back(input, target);
            }
        }
    }
}

bool ImageDataLoader::hasNext() const {
    return data[currBatch].size() > 0;
}

DataPoint ImageDataLoader::next() {
    assert(hasNext());
    const DataPoint dataPoint = data[currBatch].back();
    data[currBatch].pop_back();
    return dataPoint;
}