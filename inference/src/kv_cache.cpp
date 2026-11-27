
#include "kv_cache.hpp"

#include <stdexcept>
#include <utility>

KVCache::KVCache(std::size_t capacity) : capacity_(capacity), current_length_(0) {
    if (capacity_ == 0) {
        throw std::invalid_argument("KVCache capacity cannot be zero");
    }

    // 预留外层 vector 的空间，避免追加时反复扩容
    key_cache_.reserve(capacity_);
    value_cache_.reserve(capacity_);
}

void KVCache::append(const std::vector<float> & key, const std::vector<float> & value) {
    // 1. 检查缓存是否已满
    if (current_length_ >= capacity_) {
        throw std::overflow_error("KVCache is full");
    }

    // 2. 检查 K/V 是否为空
    if (key.empty() || value.empty()) {
        throw std::invalid_argument("Key and Value cannot be empty");
    }

    // 3. 检查后续 Token 的 K/V 维度是否一致
    if (current_length_ > 0) {
        if (key.size() != key_cache_[0].size()) {
            throw std::invalid_argument("Key dimension mismatch");
        }

        if (value.size() != value_cache_[0].size()) {
            throw std::invalid_argument("Value dimension mismatch");
        }
    }

    // 4. 先追加 K，再追加 V
    // 如果 V 追加失败，回滚刚刚追加的 K
    key_cache_.push_back(key);

    try {
        value_cache_.push_back(value);
    } catch (...) {
        key_cache_.pop_back();
        throw;
    }

    // 5. K 和 V 都追加成功后，再更新长度
    ++current_length_;
}

const std::vector<float> & KVCache::get_key(std::size_t position) const {
    if (position >= current_length_) {
        throw std::out_of_range("Key position out of range");
    }

    return key_cache_[position];
}

const std::vector<float> & KVCache::get_value(std::size_t position) const {
    if (position >= current_length_) {
        throw std::out_of_range("Value position out of range");
    }

    return value_cache_[position];
}

std::size_t KVCache::get_capacity() const noexcept {
    return capacity_;
}

std::size_t KVCache::get_current_length() const noexcept {
    return current_length_;
}
