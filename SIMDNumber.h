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

    SIMDNumber(const float* f) {
        init(f);
    }

    SIMDNumber(float value) {
        data_.simdData_ = _mm256_broadcast_ss(&value);
    }

    SIMDNumber(const __m256& data)
    {
        data_.simdData_ = data;
    }

    SIMDNumber(__m256&& data)
    {
        data_.simdData_ = std::move(data);
    }

    void init(const float* f) {
        data_.simdData_ = _mm256_load_ps(f);
    }

    float operator[](size_t index) const {
        return data_.floatData_[index];
    }

    SIMDNumber sqrt() const {
        return _mm256_sqrt_ps(data_.simdData_);
    }

    SIMDNumber& operator+=(const SIMDNumber& rhs) {
        data_.simdData_ = _mm256_add_ps(data_.simdData_, rhs.data_.simdData_);
        return *this;
    }

private:
    union Data {
        __m256 simdData_;
        float floatData_[8];
    };

    Data data_;

    friend std::ostream& operator<<(std::ostream& os, const SIMDNumber& obj);

    friend SIMDNumber operator*(const SIMDNumber& a, const SIMDNumber& b);
    friend SIMDNumber operator/(const SIMDNumber& a, const SIMDNumber& b);
    friend SIMDNumber operator+(const SIMDNumber& a, const SIMDNumber& b);
    friend SIMDNumber operator-(const SIMDNumber& a, const SIMDNumber& b);
};

SIMDNumber operator*(const SIMDNumber& a, const SIMDNumber& b) {
    return _mm256_mul_ps(a.data_.simdData_, b.data_.simdData_);
}

SIMDNumber operator/(const SIMDNumber& a, const SIMDNumber& b) {
    return _mm256_div_ps(a.data_.simdData_, b.data_.simdData_);
}

SIMDNumber operator+(const SIMDNumber& a, const SIMDNumber& b) {
    return _mm256_add_ps(a.data_.simdData_, b.data_.simdData_);
}

SIMDNumber operator-(const SIMDNumber& a, const SIMDNumber& b) {
    return _mm256_sub_ps(a.data_.simdData_, b.data_.simdData_);
}

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
