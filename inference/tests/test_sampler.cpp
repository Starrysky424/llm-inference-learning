#include "sampler.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

int passed = 0;
int failed = 0;

void run_test(const std::string & name, const std::function<void()> & test) {
    try {
        test();
        ++passed;
        std::cout << "[PASS] " << name << '\n';
    } catch (const std::exception & e) {
        ++failed;
        std::cout << "[FAIL] " << name << " : " << e.what() << '\n';
    }
}

void expect_true(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_near(float actual, float expected, float epsilon = 1e-5f) {
    if (std::fabs(actual - expected) > epsilon) {
        throw std::runtime_error("Expected " + std::to_string(expected) + ", got " + std::to_string(actual));
    }
}

template <typename Exception, typename Function> void expect_throw(Function && function) {
    try {
        function();
    } catch (const Exception &) {
        return;
    }

    throw std::runtime_error("Expected exception was not thrown");
}

// 1. Greedy Sampling
void test_greedy_sampling() {
    SamplerConfig config;
    config.greedy = true;

    Sampler sampler(config);

    const std::vector<float> logits = { 1.2f, 3.5f, 0.8f, 2.1f };

    expect_true(sampler.sample(logits) == 1, "Greedy should select index 1");
}

// 2. Greedy 遇到相同最大值时，返回第一个
void test_greedy_duplicate_maximum() {
    Sampler sampler;

    const std::vector<float> logits = { 3.0f, 5.0f, 2.0f, 5.0f };

    expect_true(sampler.greedy_sample(logits) == 1, "Should select the first maximum");
}

// 3. Softmax 概率总和为 1
void test_softmax_probability_sum() {
    Sampler sampler;

    const auto probabilities = sampler.get_probabilities({ 2.0f, 1.0f, 0.5f });

    float sum = 0.0f;

    for (float p : probabilities) {
        sum += p;
    }

    expect_near(sum, 1.0f);
}

// 4. Softmax 概率范围
void test_softmax_probability_range() {
    Sampler sampler;

    const auto probabilities = sampler.get_probabilities({ 2.0f, 1.0f, 0.5f });

    for (float p : probabilities) {
        expect_true(p >= 0.0f && p <= 1.0f, "Probability must be in [0, 1]");
    }
}

// 5. Temperature 必须大于 0
void test_temperature_zero() {
    SamplerConfig config;
    config.temperature = 0.0f;

    expect_throw<std::invalid_argument>([&]() { Sampler sampler(config); });
}

// 6. Temperature 不能为负数
void test_temperature_negative() {
    SamplerConfig config;
    config.temperature = -1.0f;

    expect_throw<std::invalid_argument>([&]() { Sampler sampler(config); });
}

// 7. Temperature 不能是无穷大
void test_temperature_infinity() {
    SamplerConfig config;
    config.temperature = std::numeric_limits<float>::infinity();

    expect_throw<std::invalid_argument>([&]() { Sampler sampler(config); });
}

// 8. Temperature 越低，最高概率越集中
void test_temperature_effect() {
    SamplerConfig normal_config;
    normal_config.temperature = 1.0f;

    SamplerConfig low_config;
    low_config.temperature = 0.5f;

    Sampler normal_sampler(normal_config);
    Sampler low_sampler(low_config);

    const std::vector<float> logits = { 2.0f, 1.0f, 0.5f };

    const auto normal = normal_sampler.get_probabilities(logits);

    const auto low = low_sampler.get_probabilities(logits);

    expect_true(low[0] > normal[0], "Lower temperature should increase max probability");
}

// 9. Top-K 保留最大的两个候选
void test_top_k() {
    SamplerConfig config;
    config.top_k = 2;

    Sampler sampler(config);

    const auto probabilities = sampler.get_probabilities({ 2.0f, 5.0f, 1.0f, 4.0f, 3.0f });

    expect_near(probabilities[0], 0.0f);
    expect_true(probabilities[1] > 0.0f, "Token 1 should remain");
    expect_near(probabilities[2], 0.0f);
    expect_true(probabilities[3] > 0.0f, "Token 3 should remain");
    expect_near(probabilities[4], 0.0f);
}

// 10. Top-K 为 0 时禁用
void test_top_k_disabled() {
    SamplerConfig config;
    config.top_k = 0;

    Sampler sampler(config);

    const auto probabilities = sampler.get_probabilities({ 3.0f, 2.0f, 1.0f });

    for (float p : probabilities) {
        expect_true(p > 0.0f, "All candidates should remain");
    }
}

// 11. Top-K 大于词表大小时不进行过滤
void test_top_k_larger_than_vocabulary() {
    SamplerConfig config;
    config.top_k = 100;

    Sampler sampler(config);

    const auto probabilities = sampler.get_probabilities({ 3.0f, 2.0f, 1.0f });

    for (float p : probabilities) {
        expect_true(p > 0.0f, "All candidates should remain");
    }
}

// 12. Top-P 保留累计概率达到阈值的最小集合
void test_top_p() {
    SamplerConfig config;
    config.top_p = 0.7f;

    Sampler sampler(config);

    const auto probabilities = sampler.get_probabilities({ 0.0f, 0.0f, 0.0f });

    // 三个 Logit 相同，概率各约为 1/3。
    // Top-P=0.7，需要保留三个中的三个？
    // 前两个累计约 0.6667，第三个加入后达到 1.0。
    expect_true(probabilities[0] > 0.0f, "Token 0 should remain");
    expect_true(probabilities[1] > 0.0f, "Token 1 should remain");
    expect_true(probabilities[2] > 0.0f, "Token 2 should remain");
}

// 13. Top-P 过滤后重新归一化
void test_top_p_normalization() {
    SamplerConfig config;
    config.top_p = 0.7f;

    Sampler sampler(config);

    const auto probabilities = sampler.get_probabilities({ 10.0f, 9.0f, 0.0f });

    float sum = 0.0f;

    for (float p : probabilities) {
        sum += p;
    }

    expect_near(sum, 1.0f);
}

// 14. Top-P 为 1 时禁用过滤
void test_top_p_disabled() {
    SamplerConfig config;
    config.top_p = 1.0f;

    Sampler sampler(config);

    const auto probabilities = sampler.get_probabilities({ 3.0f, 2.0f, 1.0f });

    for (float p : probabilities) {
        expect_true(p > 0.0f, "All candidates should remain");
    }
}

// 15. Top-P 必须大于 0
void test_top_p_zero() {
    SamplerConfig config;
    config.top_p = 0.0f;

    expect_throw<std::invalid_argument>([&]() { Sampler sampler(config); });
}

// 16. Top-P 不能大于 1
void test_top_p_greater_than_one() {
    SamplerConfig config;
    config.top_p = 1.1f;

    expect_throw<std::invalid_argument>([&]() { Sampler sampler(config); });
}

// 17. 空 Logits
void test_empty_logits() {
    Sampler sampler;

    expect_throw<std::invalid_argument>([&]() { sampler.sample({}); });
}

// 18. NaN Logits
void test_nan_logits() {
    Sampler sampler;

    const float nan = std::numeric_limits<float>::quiet_NaN();

    expect_throw<std::invalid_argument>([&]() { sampler.sample({ 1.0f, nan, 2.0f }); });
}

// 19. 无穷大 Logits
void test_infinity_logits() {
    Sampler sampler;

    const float infinity = std::numeric_limits<float>::infinity();

    expect_throw<std::invalid_argument>([&]() { sampler.sample({ 1.0f, infinity, 2.0f }); });
}

// 20. 随机采样返回合法 Token ID
void test_random_sampling_valid_id() {
    Sampler sampler;

    const std::vector<float> logits = { 2.0f, 1.0f, 0.5f };

    const std::size_t token_id = sampler.sample(logits);

    expect_true(token_id < logits.size(), "Sampled Token ID is out of range");
}

// 21. 相同 Seed 产生相同采样序列
void test_same_seed_reproducibility() {
    SamplerConfig config1;
    config1.seed = 42;

    SamplerConfig config2;
    config2.seed = 42;

    Sampler sampler1(config1);
    Sampler sampler2(config2);

    const std::vector<float> logits = { 2.0f, 1.0f, 0.5f };

    for (int i = 0; i < 100; ++i) {
        const auto token1 = sampler1.sample(logits);
        const auto token2 = sampler2.sample(logits);

        expect_true(token1 == token2, "Same seed should produce the same sequence");
    }
}

// 22. Top-K 和 Top-P 同时启用
void test_top_k_and_top_p() {
    SamplerConfig config;
    config.top_k = 3;
    config.top_p = 0.8f;

    Sampler sampler(config);

    const auto probabilities = sampler.get_probabilities({ 5.0f, 4.0f, 3.0f, 2.0f, 1.0f });

    float sum = 0.0f;

    for (float p : probabilities) {
        sum += p;
    }

    expect_near(sum, 1.0f);

    expect_near(probabilities[3], 0.0f);
    expect_near(probabilities[4], 0.0f);
}

int main() {
    run_test("greedy_sampling", test_greedy_sampling);
    run_test("greedy_duplicate_maximum", test_greedy_duplicate_maximum);
    run_test("softmax_probability_sum", test_softmax_probability_sum);
    run_test("softmax_probability_range", test_softmax_probability_range);
    run_test("temperature_zero", test_temperature_zero);
    run_test("temperature_negative", test_temperature_negative);
    run_test("temperature_infinity", test_temperature_infinity);
    run_test("temperature_effect", test_temperature_effect);
    run_test("top_k", test_top_k);
    run_test("top_k_disabled", test_top_k_disabled);
    run_test("top_k_larger_than_vocabulary", test_top_k_larger_than_vocabulary);
    run_test("top_p", test_top_p);
    run_test("top_p_normalization", test_top_p_normalization);
    run_test("top_p_disabled", test_top_p_disabled);
    run_test("top_p_zero", test_top_p_zero);
    run_test("top_p_greater_than_one", test_top_p_greater_than_one);
    run_test("empty_logits", test_empty_logits);
    run_test("nan_logits", test_nan_logits);
    run_test("infinity_logits", test_infinity_logits);
    run_test("random_sampling_valid_id", test_random_sampling_valid_id);
    run_test("same_seed_reproducibility", test_same_seed_reproducibility);
    run_test("top_k_and_top_p", test_top_k_and_top_p);

    std::cout << "\nPassed: " << passed << '\n';
    std::cout << "Failed: " << failed << '\n';

    return failed == 0 ? 0 : 1;
}
