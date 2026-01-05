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

#ifndef GE_OP_SPARSE_MAT_MUL_GATHER_COO_H
#define GE_OP_SPARSE_MAT_MUL_GATHER_COO_H
#include "graph/operator_reg.h"
namespace ge {

REG_OP(SparseMatMulGatherCOO)
    .INPUT(indices_x, TensorType({DT_INT32}))
    .INPUT(indices_y, TensorType({DT_INT32}))
    .INPUT(sparse_matrix, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(weight, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(output, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(SparseMatMulGatherCOO)
}
#endif //GE_OP_SPARSE_MAT_MUL_GATHER_COO_H
