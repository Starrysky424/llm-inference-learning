cmake_minimum_required(VERSION 3.14)

project(inference_learning LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 开启警告
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

# 头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 核心模块
add_library(inference_core
    src/tensor.cpp
    src/attention.cpp
    src/kv_cache.cpp
    src/block_manager.cpp
    src/block_table.cpp
    src/sampler.cpp
)

# 测试
enable_testing()

add_executable(test_attention
    tests/test_attention.cpp
)
target_link_libraries(test_attention PRIVATE inference_core)
add_test(NAME test_attention COMMAND test_attention)

add_executable(test_kv_cache
    tests/test_kv_cache.cpp
)
target_link_libraries(test_kv_cache PRIVATE inference_core)
add_test(NAME test_kv_cache COMMAND test_kv_cache)

add_executable(test_block_manager
    tests/test_block_manager.cpp
)
target_link_libraries(test_block_manager PRIVATE inference_core)
add_test(NAME test_block_manager COMMAND test_block_manager)

add_executable(test_block_table
    tests/test_block_table.cpp
)
target_link_libraries(test_block_table PRIVATE inference_core)
add_test(NAME test_block_table COMMAND test_block_table)

add_executable(test_sampler
    tests/test_sampler.cpp
)
target_link_libraries(test_sampler PRIVATE inference_core)
add_test(NAME test_sampler COMMAND test_sampler)