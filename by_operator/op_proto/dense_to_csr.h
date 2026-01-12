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

#ifndef GE_OP_DENSE_TO_CSR_H
#define GE_OP_DENSE_TO_CSR_H
#include "graph/operator_reg.h"
namespace ge {

REG_OP(DenseToCSR)
    .INPUT(dense_matrix, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(threshold, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(matrix_shape, TensorType({DT_INT32}))
    .OUTPUT(row_ptr, TensorType({DT_INT32}))
    .OUTPUT(col_indices, TensorType({DT_INT32}))
    .OUTPUT(values, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OP_END_FACTORY_REG(DenseToCSR)
}
#endif //GE_OP_DENSE_TO_CSR_H
