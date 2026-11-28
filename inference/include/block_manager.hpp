#pragma once

#include <cstddef>
#include <vector>

class BlockTable;

class BlockManager {
  public:
    // total_blocks：物理 Block 总数
    // block_size：每个 Block 可以容纳的 Token 数量
    BlockManager(std::size_t total_blocks, std::size_t block_size);

    // 禁止拷贝，避免两个 Manager 管理同一批物理 Block
    BlockManager(const BlockManager &)             = delete;
    BlockManager & operator=(const BlockManager &) = delete;

    // 分配一个物理 Block
    // 成功返回 true，并通过 block_id 返回物理 Block ID
    // 失败返回 false
    bool allocate_block(std::size_t & block_id);

    // 释放一个物理 Block
    // 非法 ID、未分配的 Block 都返回 false
    bool release_block(std::size_t block_id);

    // 获取物理 Block 总数
    std::size_t total_blocks() const noexcept;

    // 获取每个 Block 的 Token 容量
    std::size_t block_size() const noexcept;

    // 获取当前空闲 Block 数量
    std::size_t free_block_count() const noexcept;

    // 获取当前已分配 Block 数量
    std::size_t allocated_block_count() const noexcept;

    // 判断某个物理 Block 是否已经分配
    bool is_allocated(std::size_t block_id) const noexcept;

  private:
    friend class BlockTable;

    // 物理 Block 总数
    std::size_t total_blocks_;

    // 每个 Block 的 Token 容量
    std::size_t block_size_;

    // 空闲物理 Block ID
    std::vector<std::size_t> free_blocks_;

    // allocated_[id] == true 表示该 Block 已经分配
    std::vector<bool> allocated_;
};
