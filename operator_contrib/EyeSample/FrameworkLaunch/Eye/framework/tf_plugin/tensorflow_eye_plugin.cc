/* 版权所有 (c) 华为技术有限公司 2020-2021
 * 注册 Eye 操作信息到 GE
 */
#include "register/register.h"

namespace domi {
// register op info to GE
REGISTER_CUSTOM_OP("Eye")
    .FrameworkType(TENSORFLOW)   // type: CAFFE, TENSORFLOW
    .OriginOpType("Eye")      // name in tf module
    .ParseParamsByOperatorFn(AutoMappingByOpFn);
}  // namespace domi
