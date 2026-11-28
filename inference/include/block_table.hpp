#pragma once

#include "block_manager.hpp"

#include <cstddef>
#include <optional>
#include <vector>

// Token 在物理 Block 中的位置
struct TokenLocation {
    std::size_t physical_block_id;
    std::size_t block_offset;
};

class BlockTable {
  public:
    explicit BlockTable(BlockManager & manager);

    // 一个请求对应一个 BlockTable
    // 禁止复制，防止两个对象重复释放相同物理 Block
    BlockTable(const BlockTable &)             = delete;
    BlockTable & operator=(const BlockTable &) = delete;

    // 当前版本不支持移动
    BlockTable(BlockTable &&)             = delete;
    BlockTable & operator=(BlockTable &&) = delete;

    // 析构时自动归还自己持有的 Block
    ~BlockTable();

    // 为当前请求分配一个新的物理 Block
    bool allocate_next_block();

    // 根据 Token 逻辑位置，查询物理位置
    // 如果对应的逻辑 Block 尚未分配，返回 nullopt
    std::optional<TokenLocation> map_token(std::size_t token_position) const;

    // 释放当前请求持有的全部 Block
    // 成功返回 true
    bool release_all();

    // 获取当前请求持有的 Block 数量
    std::size_t block_count() const noexcept;

    // 获取 Block Table
    const std::vector<std::size_t> & block_ids() const noexcept;

  private:
    // BlockManager 必须比 BlockTable 活得更久
    BlockManager & manager_;

    // 下标代表逻辑 Block ID
    // 元素值代表物理 Block ID
    std::vector<std::size_t> block_table_;
};
