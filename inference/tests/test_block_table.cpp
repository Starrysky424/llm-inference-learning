#include "block_table.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int passed = 0;
int failed = 0;

void check(bool condition, const std::string & name) {
    if (condition) {
        ++passed;
        std::cout << "[PASS] " << name << '\n';
    } else {
        ++failed;
        std::cout << "[FAIL] " << name << '\n';
    }
}

template <typename Exception, typename Function> void check_throws(Function && function, const std::string & name) {
    try {
        function();
        check(false, name);
    } catch (const Exception &) {
        check(true, name);
    } catch (...) {
        check(false, name + " (wrong exception type)");
    }
}

// 正常分配测试
void test_allocate() {
    BlockManager manager(5, 4);

    BlockTable request(manager);

    check(request.allocate_next_block(), "allocate_first_block");
    check(request.allocate_next_block(), "allocate_second_block");

    check(request.block_count() == 2, "block_table_size");

    check(manager.allocated_block_count() == 2, "manager_allocated_count");
}

// Token 位置映射测试
void test_map_token() {
    BlockManager manager(10, 4);

    BlockTable request(manager);

    request.allocate_next_block();
    request.allocate_next_block();
    request.allocate_next_block();

    auto location = request.map_token(5);

    check(location.has_value(), "map_token_success");

    if (location.has_value()) {
        check(location->physical_block_id == request.block_ids()[1], "map_token_physical_block");

        check(location->block_offset == 1, "map_token_offset");
    }
}

// Block 边界测试
void test_block_boundary() {
    BlockManager manager(10, 4);

    BlockTable request(manager);

    request.allocate_next_block();
    request.allocate_next_block();

    auto location3 = request.map_token(3);
    auto location4 = request.map_token(4);
    auto location7 = request.map_token(7);

    check(location3.has_value(), "position_3_exists");
    check(location4.has_value(), "position_4_exists");
    check(location7.has_value(), "position_7_exists");

    if (location3 && location4 && location7) {
        check(location3->physical_block_id == request.block_ids()[0], "position_3_in_first_block");

        check(location3->block_offset == 3, "position_3_offset");

        check(location4->physical_block_id == request.block_ids()[1], "position_4_in_second_block");

        check(location4->block_offset == 0, "position_4_offset");

        check(location7->block_offset == 3, "position_7_offset");
    }
}

// 查询尚未分配的逻辑 Block
void test_unallocated_logical_block() {
    BlockManager manager(5, 4);

    BlockTable request(manager);

    request.allocate_next_block();

    auto location = request.map_token(4);

    check(!location.has_value(), "unallocated_logical_block_returns_empty");
}

// Block 池耗尽测试
void test_pool_exhausted() {
    BlockManager manager(2, 4);

    BlockTable request(manager);

    check(request.allocate_next_block(), "first_block_success");
    check(request.allocate_next_block(), "second_block_success");

    check(!request.allocate_next_block(), "third_block_fails");

    check(request.block_count() == 2, "table_count_after_exhaustion");
}

// 两个请求之间的隔离测试
void test_two_requests() {
    BlockManager manager(10, 4);

    BlockTable request_a(manager);
    BlockTable request_b(manager);

    request_a.allocate_next_block();
    request_b.allocate_next_block();

    check(request_a.block_ids()[0] != request_b.block_ids()[0], "two_requests_use_different_physical_blocks");
}

// 主动释放测试
void test_release_all() {
    BlockManager manager(5, 4);

    BlockTable request(manager);

    request.allocate_next_block();
    request.allocate_next_block();

    check(manager.allocated_block_count() == 2, "allocated_before_release");

    check(request.release_all(), "release_all_success");

    check(request.block_count() == 0, "table_empty_after_release");

    check(manager.allocated_block_count() == 0, "manager_empty_after_release");

    check(manager.free_block_count() == 5, "all_blocks_returned");
}

// 释放后重新分配测试
void test_reallocate_after_release() {
    BlockManager manager(2, 4);

    BlockTable request_a(manager);

    request_a.allocate_next_block();
    request_a.allocate_next_block();

    request_a.release_all();

    BlockTable request_b(manager);

    check(request_b.allocate_next_block(), "reallocate_after_release");
    check(manager.allocated_block_count() == 1, "allocated_count_after_reuse");
}

// 析构自动释放测试
void test_destructor_release() {
    BlockManager manager(3, 4);

    {
        BlockTable request(manager);

        request.allocate_next_block();
        request.allocate_next_block();

        check(manager.allocated_block_count() == 2, "allocated_before_destructor");
    }

    check(manager.allocated_block_count() == 0, "destructor_releases_blocks");
}

}  // namespace

int main() {
    test_allocate();
    test_map_token();
    test_block_boundary();
    test_unallocated_logical_block();
    test_pool_exhausted();
    test_two_requests();
    test_release_all();
    test_reallocate_after_release();
    test_destructor_release();

    std::cout << "\n============================\n";
    std::cout << "Passed: " << passed << '\n';
    std::cout << "Failed: " << failed << '\n';
    std::cout << "============================\n";

    return failed == 0 ? 0 : 1;
}
