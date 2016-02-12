#pragma once

#include <iostream>
#include <cassert>

#include "immintrin.h"
#include "avxintrin.h"

class SIMDNumber {
public:
    SIMDNumber() {
        check();
    }

    SIMDNumber(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7) {
        check();
        const float floats[] = {f0, f1, f2, f3, f4, f5, f6, f7};
        data_.simdData_ = _mm256_load_ps(floats);
    }

    SIMDNumber(const float* f) {
        check();
        init(f);
    }

    SIMDNumber(float value) {
        check();
        init(value);
    }

    SIMDNumber(const __m256& data) {
        check();
        data_.simdData_ = data;
    }

    SIMDNumber(__m256&& data) {
        check();
        data_.simdData_ = std::move(data);
    }

    void init(float value) {
        data_.simdData_ = _mm256_set1_ps(value);
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

    static const SIMDNumber& zero() {
        static const SIMDNumber sZero(0.f);
        return sZero;
    }

    static const SIMDNumber& one() {
        static const SIMDNumber sOne(1.f);
        return sOne;
    }

    SIMDNumber cmpL(const SIMDNumber& rhs) const {
        return _mm256_cmp_ps(data_.simdData_, rhs.data_.simdData_, _CMP_LT_OQ);
    }

    static SIMDNumber select(const SIMDNumber& a, const SIMDNumber& b, const SIMDNumber& mask) {
        return _mm256_blendv_ps(a.data_.simdData_, b.data_.simdData_, mask.data_.simdData_);
    }

    SIMDNumber operator-() const {
        return zero() - *this;
    }

    SIMDNumber abs() const {
      SIMDNumber zSign = zero().cmpL(*this);
      SIMDNumber minusThis = -*this;
      return select(minusThis, *this, zSign);
    }

    static const size_t kSize = 8;

private:
    union Data {
        __m256 simdData_;
        float floatData_[kSize];
    }  __attribute__((aligned(64), packed));

    Data data_;

    friend std::ostream& operator<<(std::ostream& os, const SIMDNumber& obj);

    friend SIMDNumber operator*(const SIMDNumber& a, const SIMDNumber& b);
    friend SIMDNumber operator/(const SIMDNumber& a, const SIMDNumber& b);
    friend SIMDNumber operator+(const SIMDNumber& a, const SIMDNumber& b);
    friend SIMDNumber operator-(const SIMDNumber& a, const SIMDNumber& b);

    void check() const {
        assert(0 == reinterpret_cast<size_t>(&(data_.simdData_)) % 64);
    }
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
