#include "block_manager.hpp"

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

// 构造函数测试
void test_constructor() {
    BlockManager manager(5, 4);

    check(manager.total_blocks() == 5, "total_blocks");
    check(manager.block_size() == 4, "block_size");
    check(manager.free_block_count() == 5, "initial_free_blocks");
    check(manager.allocated_block_count() == 0, "initial_allocated_blocks");
}

// 非法参数测试
void test_invalid_constructor() {
    check_throws<std::invalid_argument>([] { BlockManager manager(0, 4); }, "zero_total_blocks");

    check_throws<std::invalid_argument>([] { BlockManager manager(5, 0); }, "zero_block_size");
}

// 正常分配测试
void test_allocate() {
    BlockManager manager(3, 4);

    std::size_t id1 = 0;
    std::size_t id2 = 0;

    bool result1 = manager.allocate_block(id1);
    bool result2 = manager.allocate_block(id2);

    check(result1, "allocate_first");
    check(result2, "allocate_second");
    check(id1 != id2, "allocated_ids_are_different");

    check(manager.is_allocated(id1), "first_block_allocated");
    check(manager.is_allocated(id2), "second_block_allocated");

    check(manager.free_block_count() == 1, "free_count_after_allocate");
}

// 空闲池耗尽测试
void test_pool_exhausted() {
    BlockManager manager(2, 4);

    std::size_t id1 = 0;
    std::size_t id2 = 0;
    std::size_t id3 = 0;

    manager.allocate_block(id1);
    manager.allocate_block(id2);

    check(!manager.allocate_block(id3), "allocation_fails_when_pool_exhausted");

    check(manager.allocated_block_count() == 2, "all_blocks_allocated");
}

// 正常释放测试
void test_release() {
    BlockManager manager(2, 4);

    std::size_t id = 0;
    manager.allocate_block(id);

    check(manager.release_block(id), "release_allocated_block");
    check(!manager.is_allocated(id), "released_block_is_free");
    check(manager.free_block_count() == 2, "free_count_after_release");
}

// 重复释放测试
void test_double_release() {
    BlockManager manager(2, 4);

    std::size_t id = 0;
    manager.allocate_block(id);

    manager.release_block(id);

    check(!manager.release_block(id), "double_release_is_rejected");

    check(manager.free_block_count() == 2, "double_release_does_not_duplicate");
}

// 非法 ID 测试
void test_invalid_id() {
    BlockManager manager(2, 4);

    check(!manager.release_block(100), "invalid_id_release_rejected");

    check(!manager.is_allocated(100), "invalid_id_is_not_allocated");
}

}  // namespace

int main() {
    test_constructor();
    test_invalid_constructor();
    test_allocate();
    test_pool_exhausted();
    test_release();
    test_double_release();
    test_invalid_id();

    std::cout << "\n============================\n";
    std::cout << "Passed: " << passed << '\n';
    std::cout << "Failed: " << failed << '\n';
    std::cout << "============================\n";

    return failed == 0 ? 0 : 1;
}
