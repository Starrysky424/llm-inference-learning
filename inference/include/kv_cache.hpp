
#pragma once

#include <cstddef>
#include <vector>

class KVCache {
  public:
    explicit KVCache(std::size_t capacity);

    // 追加一个 Token 对应的 K 和 V
    void append(const std::vector<float> & key, const std::vector<float> & value);

    // 根据位置读取 K
    const std::vector<float> & get_key(std::size_t position) const;

    // 根据位置读取 V
    const std::vector<float> & get_value(std::size_t position) const;

    // 获取最大缓存容量
    std::size_t get_capacity() const noexcept;

    // 获取当前已缓存的 Token 数量
    std::size_t get_current_length() const noexcept;

  private:
    // 每个内部 vector 保存一个 Token 的 K/V 向量
    std::vector<std::vector<float>> key_cache_;
    std::vector<std::vector<float>> value_cache_;

    // 最大容量
    std::size_t capacity_;

    // 当前已经缓存的 Token 数量
    std::size_t current_length_;
};
