
#pragma once

#include "tensor.hpp"

// 计算教学级单头 Attention
Tensor attention(const Tensor & Q, const Tensor & K, const Tensor & V);
