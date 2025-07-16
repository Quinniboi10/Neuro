#pragma once

#include <algorithm>
#include <iostream>
#include <cassert>
#include <numbers>
#include <cstdint>
#include <string>
#include <vector>
#include <array>


#define exit(msg, code) \
    std::cerr << msg << std::endl; \
    std::exit(code);

using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;
using u8 = uint8_t;

using i64 = int64_t;
using i32 = int32_t;
using i16 = int16_t;

#ifdef _MSC_VER
#include <__msvc_int128.hpp>
using u128 = std::_Unsigned128;
#else
using u128 = unsigned __int128;
#endif

using usize = size_t;

using std::vector;
using std::string;
using std::array;
using std::cerr;
using std::cout;
using std::endl;

enum Activation : i16 {
    TANH,
    RELU,
    CRELU,
    SCRELU,
    SQRELU,
    SIGMOID,
    SOFTMAX,
    FSIGMOID,
    SOFTPLUS,
    GAUSSIAN,
    NONE,
    NUM_ACTIVATIONS
};

inline array<string, NUM_ACTIVATIONS> activNames = {
    "TANH", "RELU", "CRELU", "SCRELU", "SQRELU",
    "SIGMOID", "SOFTMAX", "FSIGMOID", "SOFTPLUS", "GAUSSIAN", "NONE"
};

namespace activations {
    inline float tanh(float x) { return (std::pow(std::numbers::e, x) - std::pow(std::numbers::e, -x)) / (std::pow(std::numbers::e, x) + std::pow(std::numbers::e, -x)); }
    inline float ReLU(float x) { return std::max<float>(x, 0); }
    inline float CReLU(float x) { return std::clamp<float>(x, 0, 1); }
    inline float SCReLU(float x) { return std::pow<float>(CReLU(x), 2); }
    inline float SQReLU(float x) { return std::pow<float>(ReLU(x), 2); }
    inline float sigmoid(float x) { return 1 / (1 + std::pow(std::numbers::e, -x)); }
    inline float fsigmoid(float x) { return x / (1 + std::abs(x)); }
    inline float softplus(float x) { return std::log(1 + std::pow(std::numbers::e, x)); }
    inline float gaussian(float x) { return std::pow(std::numbers::e, -(x * x)); }


    inline float dtanh(float x) { return 1 - std::pow(tanh(x), 2); }
    inline float dReLU(float x) { return x == 0 ? 0 : 1; }
    inline float dCReLU(float x) { return (x == 0 || x == 1) ? 0 : 1; }
    inline float dSCReLU(float x) { return (x == 0 || x == 1) ? 0 : 2 * x; }
    inline float dSQReLU(float x) { return x == 0 ? 0 : 2 * x; }
    inline float dsigmoid(float x) { return sigmoid(x) * (1 - sigmoid(x)); }
    inline float dfsigmoid(float x) { return x == 0 ? 1 : (x > 0 ? 1 / std::pow(1 + x, 2) : (1 / std::pow(1 - x, 2))); }
    inline float dsoftplus(float x) { return sigmoid(x); }
    inline float dgaussian(float x) { return -2 * x * std::pow(std::numbers::e, -(x * x)); }

    // Performs a softmax on the given vector
    inline vector<float> softmax(vector<float> values) {
        assert(!values.empty());
        // Find the max value
        float maxIn = values[0];
        for (usize idx = 1; idx < values.size(); idx++)
            maxIn = std::max(maxIn, values[idx]);

        // Compute exponentials and sum
        float sum = 0;
        for (auto& score : values) {
            score = std::exp(score - maxIn);
            sum += score;
        }

        // Scale down by sum of exponents
        if (sum == 0) {
            // Set to uniform or error handle
            float uniform = 1.0f / values.size();
            for (auto& score : values)
                score = uniform;
        }
        else
            for (float& score : values)
                score /= sum;

        return values;
    }

    inline vector<float> activate(Activation act, const vector<float>& vec) {
        vector<float> out(vec.size());

        switch (act) {
        case SOFTMAX: return softmax(vec);
        default: break;
        }

        for (usize i = 0; i < vec.size(); ++i) {
            switch (act) {
            case TANH:     out[i] = tanh(vec[i]); break;
            case RELU:     out[i] = ReLU(vec[i]); break;
            case CRELU:    out[i] = CReLU(vec[i]); break;
            case SCRELU:   out[i] = SCReLU(vec[i]); break;
            case SQRELU:   out[i] = SQReLU(vec[i]); break;
            case SIGMOID:  out[i] = sigmoid(vec[i]); break;
            case FSIGMOID: out[i] = fsigmoid(vec[i]); break;
            case SOFTPLUS: out[i] = softplus(vec[i]); break;
            case GAUSSIAN: out[i] = gaussian(vec[i]); break;
            case NONE:     out[i] = vec[i]; break;
            default: break;
            }
        }
        return out;
    }

    inline float derivActivate(Activation act, float f) {
        switch (act) {
        case TANH:     return dtanh(f);
        case RELU:     return dReLU(f);
        case CRELU:    return dCReLU(f);
        case SCRELU:   return dSCReLU(f);
        case SQRELU:   return dSQReLU(f);
        case SIGMOID:  return dsigmoid(f);
        case FSIGMOID: return dfsigmoid(f);
        case SOFTPLUS: return dsoftplus(f);
        case GAUSSIAN: return dgaussian(f);
        case NONE:     return f;
        default: exit("Unsupported activation on non-output layer: " + activNames[act], -1);
        }
    }
}

namespace internal {
    template <typename T, usize kN, usize... kNs>
    struct MultiArrayImpl {
        using Type = array<typename MultiArrayImpl<T, kNs...>::Type, kN>;
    };

    template <typename T, usize kN>
    struct MultiArrayImpl<T, kN> {
        using Type = array<T, kN>;
    };

    template <typename T, usize D>
    struct MultiVectorImpl {
        using Type = vector<typename MultiVectorImpl<T, D - 1>::Type>;
    };

    template <typename T>
    struct MultiVectorImpl<T, 1> {
        using Type = vector<T>;
    };

}

template <typename T, usize... kNs>
using MultiArray = typename internal::MultiArrayImpl<T, kNs...>::Type;

template <typename T, std::size_t D>
using MultiVector = typename internal::MultiVectorImpl<T, D>::Type;

using InputLayer = vector<float>;
using Target = vector<float>;
struct Layer;
struct Network;