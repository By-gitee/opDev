/**
 * Copyright (C)  2020-2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_OP_SPARSE_MAT_MUL_BASE_H
#define GE_OP_SPARSE_MAT_MUL_BASE_H
#include "graph/operator_reg.h"
namespace ge {

REG_OP(SparseMatMulBase)
    .INPUT(sparse_matrix, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(weight, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output, TensorType({DT_FLOAT16,DT_FLOAT}))
    .ATTR(block_x_id, Int, 0)
    .ATTR(block_y_id, Int, 0)
    .ATTR(block_x_dim, Int, 1)
    .ATTR(block_y_dim, Int, 1)
    .ATTR(block_size, Int, 64)
    .OP_END_FACTORY_REG(SparseMatMulBase)
}
#endif //GE_OP_SPARSE_MAT_MUL_BASE_H
