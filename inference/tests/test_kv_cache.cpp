
#include "kv_cache.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int passed = 0;
int failed = 0;

void check(bool condition, const std::string & name) {
    if (condition) {
        ++passed;
        std::cout << "[PASS] " << name << '\n';
    } else {
        ++failed;
        std::cout << "[FAIL] " << name << '\n';
    }
}

template <typename Exception, typename Function> void check_throws(Function && function, const std::string & name) {
    try {
        function();
        check(false, name);
    } catch (const Exception &) {
        check(true, name);
    } catch (...) {
        check(false, name + " (wrong exception type)");
    }
}

bool nearly_equal(float a, float b) {
    return std::fabs(a - b) < 1e-5f;
}

// 1. 构造函数测试
void test_constructor() {
    KVCache cache(4);

    check(cache.get_capacity() == 4, "test_constructor_capacity");

    check(cache.get_current_length() == 0, "test_constructor_initial_length");
}

// 2. 容量为 0
void test_zero_capacity() {
    check_throws<std::invalid_argument>([] { KVCache cache(0); }, "test_zero_capacity");
}

// 3. 正常追加 K/V
void test_append() {
    KVCache cache(4);

    std::vector<float> key   = { 1.0f, 2.0f, 3.0f };
    std::vector<float> value = { 4.0f, 5.0f, 6.0f };

    cache.append(key, value);

    check(cache.get_current_length() == 1, "test_append_length");

    check(cache.get_key(0) == key, "test_append_key");

    check(cache.get_value(0) == value, "test_append_value");
}

// 4. 追加多个 Token
void test_append_multiple() {
    KVCache cache(4);

    cache.append({ 1.0f, 2.0f }, { 3.0f, 4.0f });
    cache.append({ 5.0f, 6.0f }, { 7.0f, 8.0f });
    cache.append({ 9.0f, 10.0f }, { 11.0f, 12.0f });

    check(cache.get_current_length() == 3, "test_append_multiple_length");

    check(cache.get_key(1) == std::vector<float>({ 5.0f, 6.0f }), "test_append_multiple_key");

    check(cache.get_value(2) == std::vector<float>({ 11.0f, 12.0f }), "test_append_multiple_value");
}

// 5. 读取不存在的位置
void test_out_of_range() {
    KVCache cache(4);

    cache.append({ 1.0f, 2.0f }, { 3.0f, 4.0f });

    check_throws<std::out_of_range>([&cache] { cache.get_key(1); }, "test_get_key_out_of_range");

    check_throws<std::out_of_range>([&cache] { cache.get_value(1); }, "test_get_value_out_of_range");
}

// 6. 缓存满后继续追加
void test_cache_overflow() {
    KVCache cache(2);

    cache.append({ 1.0f, 2.0f }, { 3.0f, 4.0f });
    cache.append({ 5.0f, 6.0f }, { 7.0f, 8.0f });

    check(cache.get_current_length() == 2, "test_cache_full");

    check_throws<std::overflow_error>([&cache] { cache.append({ 9.0f, 10.0f }, { 11.0f, 12.0f }); },
                                      "test_cache_overflow");

    // 溢出后，原来的数据不能被覆盖
    check(cache.get_key(0) == std::vector<float>({ 1.0f, 2.0f }), "test_overflow_preserves_old_key");

    check(cache.get_value(1) == std::vector<float>({ 7.0f, 8.0f }), "test_overflow_preserves_old_value");

    check(cache.get_current_length() == 2, "test_overflow_preserves_length");
}

// 7. 空 K/V 输入
void test_empty_input() {
    KVCache cache(4);

    check_throws<std::invalid_argument>([&cache] { cache.append({}, { 1.0f, 2.0f }); }, "test_empty_key");

    check_throws<std::invalid_argument>([&cache] { cache.append({ 1.0f, 2.0f }, {}); }, "test_empty_value");

    check(cache.get_current_length() == 0, "test_empty_input_preserves_length");
}

// 8. K 维度不一致
void test_key_dimension_mismatch() {
    KVCache cache(4);

    cache.append({ 1.0f, 2.0f }, { 3.0f, 4.0f });

    check_throws<std::invalid_argument>([&cache] { cache.append({ 5.0f, 6.0f, 7.0f }, { 8.0f, 9.0f }); },
                                        "test_key_dimension_mismatch");

    check(cache.get_current_length() == 1, "test_key_mismatch_preserves_length");
}

// 9. V 维度不一致
void test_value_dimension_mismatch() {
    KVCache cache(4);

    cache.append({ 1.0f, 2.0f }, { 3.0f, 4.0f });

    check_throws<std::invalid_argument>([&cache] { cache.append({ 5.0f, 6.0f }, { 7.0f, 8.0f, 9.0f }); },
                                        "test_value_dimension_mismatch");

    check(cache.get_current_length() == 1, "test_value_mismatch_preserves_length");
}

}  // namespace

int main() {
    test_constructor();
    test_zero_capacity();
    test_append();
    test_append_multiple();
    test_out_of_range();
    test_cache_overflow();
    test_empty_input();
    test_key_dimension_mismatch();
    test_value_dimension_mismatch();

    std::cout << "\n============================\n";
    std::cout << "Passed: " << passed << '\n';
    std::cout << "Failed: " << failed << '\n';
    std::cout << "============================\n";

    return failed == 0 ? 0 : 1;
}
