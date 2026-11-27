
#include "attention.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

// 二维矩阵转置
Tensor transpose(const Tensor & input) {
    const auto & shape = input.shape();

    if (shape.size() != 2) {
        throw std::invalid_argument("Transpose requires a 2D Tensor");
    }

    const std::size_t rows = shape[0];
    const std::size_t cols = shape[1];

    Tensor result({ cols, rows });

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            result.at({ j, i }) = input.at({ i, j });
        }
    }

    return result;
}

// 二维矩阵乘法
Tensor matmul(const Tensor & A, const Tensor & B) {
    const auto & shapeA = A.shape();
    const auto & shapeB = B.shape();

    if (shapeA.size() != 2 || shapeB.size() != 2) {
        throw std::invalid_argument("Matmul requires 2D Tensors");
    }

    const std::size_t M = shapeA[0];
    const std::size_t N = shapeA[1];

    const std::size_t N2 = shapeB[0];
    const std::size_t P  = shapeB[1];

    // 检查矩阵乘法维度
    if (N != N2) {
        throw std::invalid_argument("Matmul shape mismatch");
    }

    Tensor result({ M, P });

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < P; ++j) {
            float sum = 0.0f;

            for (std::size_t k = 0; k < N; ++k) {
                sum += A.at({ i, k }) * B.at({ k, j });
            }

            result.at({ i, j }) = sum;
        }
    }

    return result;
}

// 对二维矩阵的每一行执行 Softmax
Tensor softmax(const Tensor & input) {
    const auto & shape = input.shape();

    if (shape.size() != 2) {
        throw std::invalid_argument("Softmax requires a 2D Tensor");
    }

    const std::size_t rows = shape[0];
    const std::size_t cols = shape[1];

    Tensor result({ rows, cols });

    for (std::size_t i = 0; i < rows; ++i) {
        // 找到当前行的最大值
        float maxValue = input.at({ i, 0 });

        for (std::size_t j = 1; j < cols; ++j) {
            maxValue = std::max(maxValue, input.at({ i, j }));
        }

        // 计算 exp(x - maxValue)
        float sum = 0.0f;

        for (std::size_t j = 0; j < cols; ++j) {
            float value = std::exp(input.at({ i, j }) - maxValue);

            result.at({ i, j }) = value;
            sum += value;
        }

        // 归一化
        for (std::size_t j = 0; j < cols; ++j) {
            result.at({ i, j }) /= sum;
        }
    }

    return result;
}

}  // anonymous namespace

// Attention 主函数
Tensor attention(const Tensor & Q, const Tensor & K, const Tensor & V) {
    const auto & qShape = Q.shape();
    const auto & kShape = K.shape();
    const auto & vShape = V.shape();

    // 检查输入必须是二维 Tensor
    if (qShape.size() != 2 || kShape.size() != 2 || vShape.size() != 2) {
        throw std::invalid_argument("Attention requires 2D Q, K and V");
    }

    // 检查 Q 和 K 的特征维度
    if (qShape[1] != kShape[1]) {
        throw std::invalid_argument("Q and K feature dimensions do not match");
    }

    // 检查 K 和 V 的序列长度
    if (kShape[0] != vShape[0]) {
        throw std::invalid_argument("K and V sequence lengths do not match");
    }

    // 1. K 转置
    Tensor K_T = transpose(K);

    // 2. 计算 QK^T
    Tensor scores = matmul(Q, K_T);

    // 3. Scale
    const std::size_t d_k = qShape[1];

    const float scale = std::sqrt(static_cast<float>(d_k));

    for (std::size_t i = 0; i < scores.shape()[0]; ++i) {
        for (std::size_t j = 0; j < scores.shape()[1]; ++j) {
            scores.at({ i, j }) /= scale;
        }
    }

    // 4. Softmax
    Tensor weights = softmax(scores);

    // 5. Attention 输出
    Tensor output = matmul(weights, V);

    return output;
}
