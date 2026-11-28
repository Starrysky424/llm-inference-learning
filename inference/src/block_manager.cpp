#include "block_manager.hpp"

#include <stdexcept>

BlockManager::BlockManager(std::size_t total_blocks, std::size_t block_size) :
    total_blocks_(total_blocks),
    block_size_(block_size),
    allocated_(total_blocks, false) {
    if (total_blocks == 0) {
        throw std::invalid_argument("BlockManager total_blocks cannot be zero");
    }

    if (block_size == 0) {
        throw std::invalid_argument("BlockManager block_size cannot be zero");
    }

    // 初始化空闲 Block 池
    // 例如 total_blocks = 5：
    // free_blocks_ = [0, 1, 2, 3, 4]
    for (std::size_t i = 0; i < total_blocks_; ++i) {
        free_blocks_.push_back(i);
    }
}

bool BlockManager::allocate_block(std::size_t & block_id) {
    // 没有空闲 Block
    if (free_blocks_.empty()) {
        return false;
    }

    // 从空闲池中取出一个物理 Block
    block_id = free_blocks_.back();
    free_blocks_.pop_back();

    // 标记为已分配
    allocated_[block_id] = true;

    return true;
}

bool BlockManager::release_block(std::size_t block_id) {
    // 检查物理 Block ID 是否合法
    if (block_id >= total_blocks_) {
        return false;
    }

    // 防止重复释放
    if (!allocated_[block_id]) {
        return false;
    }

    // 重新放回空闲池
    free_blocks_.push_back(block_id);

    // 标记为空闲
    allocated_[block_id] = false;

    return true;
}

std::size_t BlockManager::total_blocks() const noexcept {
    return total_blocks_;
}

std::size_t BlockManager::block_size() const noexcept {
    return block_size_;
}

std::size_t BlockManager::free_block_count() const noexcept {
    return free_blocks_.size();
}

std::size_t BlockManager::allocated_block_count() const noexcept {
    return total_blocks_ - free_blocks_.size();
}

bool BlockManager::is_allocated(std::size_t block_id) const noexcept {
    if (block_id >= total_blocks_) {
        return false;
    }

    return allocated_[block_id];
}
