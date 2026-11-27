
#pragma once

#include <cstddef>
#include <vector>

class Tensor {
  public:
    // 根据 Shape 创建 Tensor
    explicit Tensor(const std::vector<std::size_t> & shape);

    // 返回元素总数
    std::size_t numel() const;

    // 返回 Shape
    const std::vector<std::size_t> & shape() const;

    // 返回 Strides
    const std::vector<std::size_t> & strides() const;

    // 可读可写的多维索引
    float & at(const std::vector<std::size_t> & indices);

    // 只读的多维索引
    const float & at(const std::vector<std::size_t> & indices) const;

    // Tensor 逐元素加法
    Tensor add(const Tensor & other) const;

    // Tensor 逐元素乘法
    Tensor multiply(const Tensor & other) const;

  private:
    // 每个维度的长度
    std::vector<std::size_t> shape_;

    // 每个维度对应的步长
    std::vector<std::size_t> strides_;

    // 实际存储 Tensor 数据
    std::vector<float> data_;

    // 将多维索引转换成一维偏移量
    std::size_t calculate_offset(const std::vector<std::size_t> & indices) const;
};
