#pragma once

#include <iostream>

#include "immintrin.h"
#include "avxintrin.h"

class SIMDNumber {
public:
    SIMDNumber() {
    }

    SIMDNumber(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7) {
        const float floats[] = {f0, f1, f2, f3, f4, f5, f6, f7};
        data_.simdData_ = _mm256_load_ps(floats);
    }

private:
    union Data {
        __m256 simdData_;
        float floatData_[8];
    };

    Data data_;

    friend std::ostream& operator<<(std::ostream& os, const SIMDNumber& obj);
};

std::ostream& operator<<(std::ostream& os, const SIMDNumber& obj) {
    os << "[";
    for (size_t i = 0; i < 8; ++i) {
        if (i) {
            os << ", ";
        }
        os << obj.data_.floatData_[i];
    }
    os << "]";
    return os;
}
