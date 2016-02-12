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

struct SIMDConfig {
    SIMDNumber alpha_;
    SIMDNumber beta_;
    SIMDNumber lambda1_;
    SIMDNumber lambda2_;

    SIMDConfig(const Config& c)
        : alpha_(c.alpha_)
        , beta_(c.beta_)
        , lambda1_(c.lambda1_)
        , lambda2_(c.lambda2_)
    {
    }
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

inline void update(const SIMDNumber& g,
                   SIMDNumber& z,
                   SIMDNumber& n,
                   SIMDNumber& weight,
                   const SIMDConfig& featureConfig) {
  SIMDNumber g2 = g * g;
  SIMDNumber sigma = ((n + g2).sqrt() - n.sqrt()) / featureConfig.alpha_;
  z += g - sigma * weight;
  n += g2;
  /*
  const SIMDNumber newWeight =
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
  */
}

int main() {
    SIMDNumber x(0, 1, 2, 3, 4, 5, 6, 7);
    std::cout << x << std::endl;
    SIMDNumber y(0, 1, 2, 3, 4, 5, 6, 7);


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
            ScopedTimer tOld("New");
            static const size_t N8 = N / 8;
            SIMDNumber g;
            SIMDNumber z;
            SIMDNumber n;
            SIMDNumber weight;
            float fg[8];
            float fz[8];
            float fn[8];
            float fweight[8];
            SIMDConfig sc(c);
            for (size_t j = 0; j < M; ++j) {
                for (size_t i = 0; i < N8; ++i) {
                    for (size_t k = 0; k < 8; ++k) {
                        const auto& gk = copy[8*i + k];
                        fg[k] = gk.g_;
                        fz[k] = gk.z_;
                        fn[k] = gk.n_;
                        fweight[k] = gk.weight_;
                    }
                    g.init(fg);
                    z.init(fz);
                    n.init(fn);
                    weight.init(fweight);
                    update(g, z, n, weight, sc);
                    for (size_t k = 0; k < 8; ++k) {
                        sum += weight[k];
                    }
                }
            }
        }
        cout << "sum: " << sum << endl;
    }

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
