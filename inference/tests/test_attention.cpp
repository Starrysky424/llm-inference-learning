
#include "attention.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

bool close(float a, float b, float eps = 1e-3f) {
    return std::fabs(a - b) < eps;
}

void test_numerical_stability() {
    Tensor Q({ 1, 2 });
    Q.at({ 0, 0 }) = 1000.0f;
    Q.at({ 0, 1 }) = 0.0f;

    Tensor K({ 2, 2 });
    K.at({ 0, 0 }) = 1000.0f;
    K.at({ 0, 1 }) = 0.0f;
    K.at({ 1, 0 }) = 0.0f;
    K.at({ 1, 1 }) = 0.0f;

    Tensor V({ 2, 2 });
    V.at({ 0, 0 }) = 7.0f;
    V.at({ 0, 1 }) = 11.0f;
    V.at({ 1, 0 }) = 13.0f;
    V.at({ 1, 1 }) = 17.0f;

    Tensor output = attention(Q, K, V);

    if (!close(output.at({ 0, 0 }), 7.0f) || !close(output.at({ 0, 1 }), 11.0f)) {
        throw std::runtime_error("Numerical stability test failed");
    }

    std::cout << "test_numerical_stability passed\n";
}
void test_nontrivial_shape() {
    Tensor Q({ 2, 3 });
    Tensor K({ 4, 3 });
    Tensor V({ 4, 5 });

    Tensor output = attention(Q, K, V);

    if (output.shape() != std::vector<std::size_t>{ 2, 5 }) {
        throw std::runtime_error("Nontrivial shape test failed");
    }

    std::cout << "test_nontrivial_shape passed\n";
}
void test_qk_shape_mismatch() {
    Tensor Q({ 2, 3 });
    Tensor K({ 4, 5 });
    Tensor V({ 4, 2 });

    bool caught = false;

    try {
        attention(Q, K, V);
    } catch (const std::invalid_argument &) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("QK shape mismatch test failed");
    }

    std::cout << "test_qk_shape_mismatch passed\n";
}

void test_invalid_dimension() {
    Tensor Q({ 3 });     // 一维
    Tensor K({ 2, 3 });  // 二维
    Tensor V({ 2, 4 });  // 二维

    try {
        attention(Q, K, V);
    } catch (const std::invalid_argument &) {
        std::cout << "test_invalid_dimension passed\n";
        return;
    }

    throw std::runtime_error("Expected invalid_argument for non-2D input");
}
void test_single_query_single_key() {
    Tensor Q({ 1, 2 });
    Q.at({ 0, 0 }) = 1.0f;
    Q.at({ 0, 1 }) = 0.0f;

    Tensor K({ 1, 2 });
    K.at({ 0, 0 }) = 1.0f;
    K.at({ 0, 1 }) = 0.0f;

    Tensor V({ 1, 3 });
    V.at({ 0, 0 }) = 10.0f;
    V.at({ 0, 1 }) = 20.0f;
    V.at({ 0, 2 }) = 30.0f;

    Tensor output = attention(Q, K, V);

    if (output.shape() != std::vector<std::size_t>{ 1, 3 }) {
        throw std::runtime_error("Single query output shape failed");
    }

    if (!close(output.at({ 0, 0 }), 10.0f) || !close(output.at({ 0, 1 }), 20.0f) ||
        !close(output.at({ 0, 2 }), 30.0f)) {
        throw std::runtime_error("Single query output value failed");
    }

    std::cout << "test_single_query_single_key passed\n";
}
void test_kv_shape_mismatch() {
    Tensor Q({ 2, 3 });
    Tensor K({ 4, 3 });
    Tensor V({ 5, 2 });

    bool caught = false;

    try {
        attention(Q, K, V);
    } catch (const std::invalid_argument &) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("KV shape mismatch test failed");
    }

    std::cout << "test_kv_shape_mismatch passed\n";
}
int main() {
    // Q: {1, 2}
    Tensor Q({ 1, 2 });
    Q.at({ 0, 0 }) = 1.0f;
    Q.at({ 0, 1 }) = 0.0f;

    // K: {2, 2}
    Tensor K({ 2, 2 });
    K.at({ 0, 0 }) = 1.0f;
    K.at({ 0, 1 }) = 0.0f;
    K.at({ 1, 0 }) = 0.0f;
    K.at({ 1, 1 }) = 1.0f;

    // V: {2, 2}
    Tensor V({ 2, 2 });
    V.at({ 0, 0 }) = 10.0f;
    V.at({ 0, 1 }) = 0.0f;
    V.at({ 1, 0 }) = 0.0f;
    V.at({ 1, 1 }) = 20.0f;

    // 执行 Attention
    Tensor output = attention(Q, K, V);

    // 检查 Shape
    if (output.shape() != std::vector<std::size_t>{ 1, 2 }) {
        throw std::runtime_error("Output shape test failed");
    }

    // 检查数值
    if (!close(output.at({ 0, 0 }), 6.6976f) || !close(output.at({ 0, 1 }), 6.6048f)) {
        throw std::runtime_error("Attention result test failed");
    }

    std::cout << "Output: " << output.at({ 0, 0 }) << ", " << output.at({ 0, 1 }) << '\n';

    std::cout << "Attention basic test passed!\n";
    test_qk_shape_mismatch();
    test_kv_shape_mismatch();
    test_single_query_single_key();
    test_invalid_dimension();
    test_nontrivial_shape();
    test_numerical_stability();
    return 0;
}
