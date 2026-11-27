#include "sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

Sampler::Sampler(const SamplerConfig & config) : config_(config), generator_(config.seed) {
    if (!std::isfinite(config_.temperature) || config_.temperature <= 0.0f) {
        throw std::invalid_argument("Temperature must be finite and greater than 0");
    }

    if (!std::isfinite(config_.top_p) || config_.top_p <= 0.0f || config_.top_p > 1.0f) {
        throw std::invalid_argument("Top-P must be in the range (0, 1]");
    }
}

void Sampler::validate_logits(const std::vector<float> & logits) const {
    if (logits.empty()) {
        throw std::invalid_argument("Logits cannot be empty");
    }

    for (float value : logits) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Logits must contain only finite values");
        }
    }
}

std::vector<float> Sampler::apply_temperature(const std::vector<float> & logits) const {
    std::vector<float> result;
    result.reserve(logits.size());

    for (float value : logits) {
        result.push_back(value / config_.temperature);
    }

    return result;
}

std::vector<float> Sampler::apply_top_k(const std::vector<float> & logits) const {
    std::vector<float> result = logits;

    // top_k == 0 表示禁用过滤
    if (config_.top_k == 0 || config_.top_k >= logits.size()) {
        return result;
    }

    // 保存每个 Logit 对应的下标
    std::vector<std::size_t> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);

    // 按 Logit 从大到小排序
    std::partial_sort(indices.begin(), indices.begin() + config_.top_k, indices.end(),
                      [&logits](std::size_t a, std::size_t b) { return logits[a] > logits[b]; });

    // 标记需要保留的 Token
    std::vector<bool> keep(logits.size(), false);

    for (std::size_t i = 0; i < config_.top_k; ++i) {
        keep[indices[i]] = true;
    }

    // 其他 Token 设置为负无穷
    const float negative_infinity = -std::numeric_limits<float>::infinity();

    for (std::size_t i = 0; i < result.size(); ++i) {
        if (!keep[i]) {
            result[i] = negative_infinity;
        }
    }

    return result;
}

std::vector<float> Sampler::softmax(const std::vector<float> & logits) const {
    const float max_logit = *std::max_element(logits.begin(), logits.end());

    std::vector<float> probabilities;
    probabilities.reserve(logits.size());

    float sum = 0.0f;

    for (float value : logits) {
        if (std::isinf(value) && value < 0.0f) {
            probabilities.push_back(0.0f);
            continue;
        }

        const float exp_value = std::exp(value - max_logit);

        probabilities.push_back(exp_value);
        sum += exp_value;
    }

    if (!std::isfinite(sum) || sum <= 0.0f) {
        throw std::runtime_error("Softmax calculation failed");
    }

    for (float & probability : probabilities) {
        probability /= sum;
    }

    return probabilities;
}

std::vector<float> Sampler::apply_top_p(const std::vector<float> & probabilities) const {
    std::vector<float> result = probabilities;

    if (config_.top_p >= 1.0f) {
        return result;
    }

    // 保存 Token 下标
    std::vector<std::size_t> indices(probabilities.size());
    std::iota(indices.begin(), indices.end(), 0);

    // 按概率从大到小排序
    std::stable_sort(indices.begin(), indices.end(),
                     [&probabilities](std::size_t a, std::size_t b) { return probabilities[a] > probabilities[b]; });

    std::vector<bool> keep(probabilities.size(), false);

    float cumulative_probability = 0.0f;

    for (std::size_t index : indices) {
        keep[index] = true;
        cumulative_probability += probabilities[index];

        // 达到或超过 Top-P 阈值后停止
        if (cumulative_probability >= config_.top_p) {
            break;
        }
    }

    // 过滤掉不在候选集合中的 Token
    for (std::size_t i = 0; i < result.size(); ++i) {
        if (!keep[i]) {
            result[i] = 0.0f;
        }
    }

    return result;
}

void Sampler::normalize(std::vector<float> & probabilities) const {
    const float sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0f);

    if (!std::isfinite(sum) || sum <= 0.0f) {
        throw std::runtime_error("Probability normalization failed");
    }

    for (float & probability : probabilities) {
        probability /= sum;
    }
}

std::vector<float> Sampler::get_probabilities(const std::vector<float> & logits) const {
    validate_logits(logits);

    // 1. Temperature
    std::vector<float> processed = apply_temperature(logits);

    // 2. Top-K
    processed = apply_top_k(processed);

    // 3. Softmax
    std::vector<float> probabilities = softmax(processed);

    // 4. Top-P
    probabilities = apply_top_p(probabilities);

    // 5. 重新归一化
    normalize(probabilities);

    return probabilities;
}

std::size_t Sampler::greedy_sample(const std::vector<float> & logits) const {
    validate_logits(logits);

    return static_cast<std::size_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

std::size_t Sampler::sample(const std::vector<float> & logits) {
    validate_logits(logits);

    // Greedy 模式不进行随机采样
    if (config_.greedy) {
        return greedy_sample(logits);
    }

    const std::vector<float> probabilities = get_probabilities(logits);

    std::discrete_distribution<std::size_t> distribution(probabilities.begin(), probabilities.end());

    return distribution(generator_);
}
