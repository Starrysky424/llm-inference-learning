
#include "tensor.hpp"

#include <limits>
#include <stdexcept>

// 构造函数
Tensor::Tensor(const std::vector<std::size_t> & shape) : shape_(shape) {
    // 不允许空 Shape
    if (shape_.empty()) {
        throw std::invalid_argument("Tensor shape cannot be empty");
    }

    // 初始化 Strides
    strides_.resize(shape_.size());

    std::size_t total = 1;

    // 从最后一个维度向前计算 Strides
    for (std::size_t i = shape_.size(); i-- > 0;) {
        // 不允许维度长度为 0
        if (shape_[i] == 0) {
            throw std::invalid_argument("Tensor dimension cannot be zero");
        }

        strides_[i] = total;

        // 检查乘法是否会发生整数溢出
        if (total > std::numeric_limits<std::size_t>::max() / shape_[i]) {
            throw std::length_error("Tensor size overflow");
        }

        total *= shape_[i];
    }

    // 分配数据空间，所有元素初始化为 0
    data_.resize(total, 0.0f);
}

// 返回元素总数
std::size_t Tensor::numel() const {
    return data_.size();
}

// 返回 Shape
const std::vector<std::size_t> & Tensor::shape() const {
    return shape_;
}

// 返回 Strides
const std::vector<std::size_t> & Tensor::strides() const {
    return strides_;
}

// 计算多维索引对应的一维偏移量
std::size_t Tensor::calculate_offset(const std::vector<std::size_t> & indices) const {
    // 检查索引维度数量
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Index dimension does not match Tensor dimension");
    }

    std::size_t offset = 0;

    for (std::size_t i = 0; i < indices.size(); ++i) {
        // 检查当前维度是否越界
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Tensor index out of range");
        }

        offset += indices[i] * strides_[i];
    }

    return offset;
}

// 可读可写的 at
float & Tensor::at(const std::vector<std::size_t> & indices) {
    return data_[calculate_offset(indices)];
}

// 只读的 at
const float & Tensor::at(const std::vector<std::size_t> & indices) const {
    return data_[calculate_offset(indices)];
}

// Tensor 逐元素加法
Tensor Tensor::add(const Tensor & other) const {
    // 检查两个 Tensor 的 Shape 是否一致
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes do not match for addition");
    }

    Tensor result(shape_);

    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }

    return result;
}

// Tensor 逐元素乘法
Tensor Tensor::multiply(const Tensor & other) const {
    // 检查两个 Tensor 的 Shape 是否一致
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes do not match for multiplication");
    }

    Tensor result(shape_);

    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }

    return result;
}
