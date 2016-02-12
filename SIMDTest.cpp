#include <cmath>

#include <iostream>

#include <vector>
#include <random>
#include <string>
#include <chrono>
#include <exception>

#include "SIMDNumber.h"

using namespace std;

struct Gradient {
    float g_;
    float z_;
    float n_;
    float weight_;

    void RandomFill() {
        static random_device rd;
        static mt19937 gen(rd());
        static uniform_real_distribution<float> dis(0.f, 1.f);
        g_ = dis(gen);
        z_ = dis(gen);
        n_ = dis(gen);
        weight_ = dis(gen);
    }
};

struct Config {
    float alpha_;
    float beta_;
    float lambda1_;
    float lambda2_;
};

struct ScopedTimer {
    string message_;
    chrono::high_resolution_clock::time_point begin_;

    ScopedTimer(const string& message)
        : message_(message)
        , begin_(chrono::high_resolution_clock::now())
    {
    }

    ~ScopedTimer() {
        auto timeSpan = chrono::duration<float>(chrono::high_resolution_clock::now() - begin_);
        cout << message_ << ": " << timeSpan.count() << endl;
    }
};

static inline int sign(double v) {
    return v < -1e-8 ? -1 : v > 1e-8;
}

inline void update(const float& g,
                   float& z,
                   float& n,
                   float& weight,
                   const Config& featureConfig) {
  double g2 = g * g;
  double sigma = (sqrt(n + g2) - sqrt(n)) / featureConfig.alpha_;
  z += g - sigma * weight;
  n += g2;
  const double newWeight =
      std::abs(z) <= featureConfig.lambda1_
          ? 0
          : (sign(z) * featureConfig.lambda1_ - z) /
                ((featureConfig.beta_ + sqrt(n)) / featureConfig.alpha_ +
                 featureConfig.lambda2_);
  if (std::isnan(newWeight) || std::isinf(newWeight)) {
    throw exception();
  } else {
    weight = newWeight;
  }
}

int main() {
    SIMDNumber x(0, 1, 2, 3, 4, 5, 6, 7);
    std::cout << x << std::endl;

    static const size_t N = 1024*1024;
    vector<Gradient> data(N);
    {
        ScopedTimer tFill("Fill");
        for (auto& g: data) {
            g.RandomFill();
        }
    }

    Config c;
    c.alpha_ = 0.01f;
    c.beta_ = 0.01f;
    c.lambda1_ = 0.02f;
    c.lambda2_ = 0.03f;

    static const size_t M = 100;

    {
        float sum = 0.f;
        vector<Gradient> copy(data);
        {
            ScopedTimer tOld("Old");
            for (size_t j = 0; j < M; ++j) {
                for (auto& g: copy) {
                    update(g.g_, g.z_, g.n_, g.weight_, c);
                    sum += g.weight_;
                }
            }
        }
        cout << "sum: " << sum << endl;
    }

    return 0;
}
