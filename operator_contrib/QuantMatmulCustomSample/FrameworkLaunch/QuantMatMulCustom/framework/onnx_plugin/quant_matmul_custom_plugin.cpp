/* Copyright (C) 2020-2021. Huawei Technologies Co., Ltd. All
rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "register/register.h"

namespace domi {

// register QuantMatmulCustom op info to GE
REGISTER_CUSTOM_OP("QuantMatmulCustom")     // Set the registration name of operator
    .FrameworkType(ONNX)   // Operator name with the original framework
    .OriginOpType("QuantMatmulCustom")      // Set the original frame type of the operator
    .ParseParamsByOperatorFn(AutoMappingByOpFn); // Registering the callback function for parsing operator parameters
}  // namespace domi
