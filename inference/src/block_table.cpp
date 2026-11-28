#include "block_table.hpp"

#include <algorithm>

BlockTable::BlockTable(BlockManager & manager) : manager_(manager) {}

BlockTable::~BlockTable() {
    release_all();
}

bool BlockTable::allocate_next_block() {
    std::size_t physical_block_id = 0;

    // 向 BlockManager 申请物理 Block
    if (!manager_.allocate_block(physical_block_id)) {
        return false;
    }

    // 将物理 Block ID 记录到当前请求的 Block Table
    block_table_.push_back(physical_block_id);

    return true;
}

std::optional<TokenLocation> BlockTable::map_token(std::size_t token_position) const {
    const std::size_t block_size = manager_.block_size();

    // 逻辑 Block ID
    const std::size_t logical_block_id = token_position / block_size;

    // Block 内部偏移
    const std::size_t block_offset = token_position % block_size;

    // 当前请求还没有分配到对应的逻辑 Block
    if (logical_block_id >= block_table_.size()) {
        return std::nullopt;
    }

    // 查找逻辑 Block 对应的物理 Block
    const std::size_t physical_block_id = block_table_[logical_block_id];

    return TokenLocation{ physical_block_id, block_offset };
}

bool BlockTable::release_all() {
    // 逐个归还当前请求持有的物理 Block
    // 只有成功释放的 ID 才从 Block Table 中删除
    bool all_released = true;

    auto it = block_table_.begin();

    while (it != block_table_.end()) {
        if (manager_.release_block(*it)) {
            it = block_table_.erase(it);
        } else {
            all_released = false;
            ++it;
        }
    }

    return all_released;
}

std::size_t BlockTable::block_count() const noexcept {
    return block_table_.size();
}

const std::vector<std::size_t> & BlockTable::block_ids() const noexcept {
    return block_table_;
}
