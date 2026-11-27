#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

struct SamplerConfig {
    float temperature = 1.0f;

    // 0 表示禁用 Top-K
    std::size_t top_k = 0;

    // 1.0 表示禁用 Top-P
    float top_p = 1.0f;

    // 固定随机种子，方便测试复现
    std::uint32_t seed = 42;

    // true：贪心采样；false：概率随机采样
    bool greedy = false;
};

class Sampler {
  public:
    explicit Sampler(const SamplerConfig & config = {});

    // 根据配置选择下一个 Token ID
    std::size_t sample(const std::vector<float> & logits);

    // 直接选择 Logits 最大的 Token
    std::size_t greedy_sample(const std::vector<float> & logits) const;

    // 获取经过 Temperature、Top-K、Softmax、Top-P 处理后的概率
    std::vector<float> get_probabilities(const std::vector<float> & logits) const;

  private:
    SamplerConfig config_;
    std::mt19937  generator_;

    void validate_logits(const std::vector<float> & logits) const;

    std::vector<float> apply_temperature(const std::vector<float> & logits) const;

    std::vector<float> apply_top_k(const std::vector<float> & logits) const;

    std::vector<float> softmax(const std::vector<float> & logits) const;

    std::vector<float> apply_top_p(const std::vector<float> & probabilities) const;

    void normalize(std::vector<float> & probabilities) const;
};
