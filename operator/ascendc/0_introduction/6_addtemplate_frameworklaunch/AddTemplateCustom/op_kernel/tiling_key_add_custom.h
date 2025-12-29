/**
 * @file tiling_key_add_custom.h
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef TILING_KEY_ADD_CUSTOM_H
#define TILING_KEY_ADD_CUSTOM_H
#include "ascendc/host_api/tiling/template_argument.h"

#define ADD_TPL_FP16 10
#define ADD_TPL_FP32 20

#define ADD_TPL_ND 2 //数据格式定义
#define ADD_TPL_NZ 29 //数据格式定义
/**
ASCENDC_TPL_ARGS_DECL(args0, ...):算子的模板参数定义, args0表示算子唯一标识, 建议与opType保持一致,后续为若干个DTYPE、FORMAT、UINT、BOOL的模板参数定义
    ASCENDC_TPL_DTYPE_DECL(args0, ...): DTYPE类型的模板参数定义,args0表示参数名,后续若干个参数为穷举的DTYPE枚举值
    ASCENDC_TPL_FORMAT_DECL(args0, ...): FORMAT类型的模板参数定义,args0表示参数名,后续若干个参数为穷举的FORMAT枚举值
    ASCENDC_TPL_UINT_DECL(args0, args1, args2, ...): UINT类型的模板参数定义
        args0: 参数名
        args1: 比特位,表示该UINT定义范围内数据的个数能用多少个比特位表示
        args2: 后续...参数表示类型,支持的表示类型为 ASCENDC_TPL_UI_RANGE,ASCENDC_TPL_UI_LIST,ASCENDC_TPL_UI_MIX
               ASCENDC_TPL_UI_RANGE:范围模式,如果设置该模式,后续紧跟着第一个值表示范围个数,后续的每个数值为一组表示该范围的起、终位置;注意定义的范围个数要和后续的组数保持一致
               ASCENDC_TPL_UI_LIST:穷举模式,如果设置该模式,则表示后续将穷举出所有的值;注意穷举的位置决定了后续tilingkey编码值,数值大小不会影响编码值
               ASCENDC_TPL_UI_MIX:混合模式,如果设置该模式,则表示前n个数值为范围模式的参数定义,后m个数值为穷举模式的参数定义
          实例说明1 ASCENDC_TPL_UI_RANGE,2, 0, 2, 3, 5:则表示2组范围参数,这2组范围为{0, 2}, {3, 5};因此该参数定义的UINT参数合法值为{0, 1, 2, 3, 4, 5},并且后续编码将以这个定义的顺序编码
          实例说明2 ASCENDC_TPL_UI_LIST,10, 12, 13, 9, 8:则表示1组穷举参数,[10, 12, 13, 9, 8]为穷举值,因此该参数定义的UINT参数合法值为{10, 12, 13, 9, 8},并且后续编码将以这个定义的顺序编码
          实例说明3 ASCENDC_TPL_UI_MIX,2, 0, 2, 3, 5, 10, 12, 13, 9, 8:则表示2组穷举参数,这2组范围为{0, 2}, {3, 5};[10, 12, 13, 9, 8]为穷举值,因此该参数定义的UINT参数合法值为{0, 1, 2, 3, 4, 5, 10, 12, 13, 9, 8},并且后续编码将以这个定义的顺序编码
        【使用注意事项】:
            1. tilingkey编码是以UNINT定义的位置进行编码,在更新ASCENDC_TPL_UINT_DECL参数定义是,一定要注意不影响原有的参数位置!
            2. ASCENDC_TPL_UINT_DECL中的比特位一经定义后续不可更改(会影响其他参数的编码位置偏移,引起tilingkey兼容问题),所以用户在首次定义的比特位的时候需要考虑到后续的参数范围扩展
    ASCENDC_TPL_BOOL_DECL(args0, ...): bool类型的模板参数定义,args0表示参数名,后续可自定义true、false中的一个或两个值
*/
ASCENDC_TPL_ARGS_DECL(AddTemplateCustom, //唯一标识,建议与OpType保持一致
ASCENDC_TPL_DTYPE_DECL(D_T_X, ADD_TPL_FP16, ADD_TPL_FP32), //编码位宽为8比特
ASCENDC_TPL_DTYPE_DECL(D_T_Y, ADD_TPL_FP16, ADD_TPL_FP32),
ASCENDC_TPL_DTYPE_DECL(D_T_Z, ADD_TPL_FP16, ADD_TPL_FP32),
ASCENDC_TPL_UINT_DECL(TILE_NUM, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_MIX, 2, 0, 2, 3, 5, 10, 12, 13, 9, 8),//编码位宽位为用户定义的ASCENDC_TPL_8_BW比特,注意用户指定的位宽需要大于定义的参数个数
ASCENDC_TPL_BOOL_DECL(IS_SPLIT, 0, 1), //编码位宽为1比特
);
/**
ASCENDC_TPL_SEL(...):算子的模板参数整体组合,可设置多个模板参数组合
    ASCENDC_TPL_ARGS_DSEL(...):算子的模板参数组合,可设置多个参数组合
        ASCENDC_TPL_DTYPE_SEL(args0, ...): DTYPE类型的模板参数组合,args0表示参数名,后续若干个参数为对应的ASCENDC_TPL_DTYPE_DECL定义的参数范围子集
        ASCENDC_TPL_FORMAT_SEL(args0, ...): FORMAT类型的模板参数组合,args0表示参数名,后续若干个参数为对应的ASCENDC_TPL_FORMAT_DECL定义的参数范围子集
        ASCENDC_TPL_UINT_SEL(args0, args1, args2, ...): UINT类型的模板参数定义,args0表示参数名,args1是参数的表示类型,支持的表示类型为ASCENDC_TPL_UI_RANGE,ASCENDC_TPL_UI_LIST,ASCENDC_TPL_UI_MIX,后续的数值定义参考ASCENDC_TPL_UINT_DECL的规则
        ASCENDC_TPL_BOOL_SEL(args0, ...): bool类型的模板参数定义,args0表示参数名,后续若干个参数为对应的ASCENDC_TPL_BOOL_DECL定义的参数范围子集
        ASCENDC_TPL_TILING_STRUCT_SEL(args0): 此模板参数组合对应的自定义tiling结构体,此处需要和kernel侧的判断逻辑保持一致，否则会有oom问题,args0表示tiling结构体名
*/
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_DTYPE_SEL(D_T_X, ADD_TPL_FP16),
    ASCENDC_TPL_DTYPE_SEL(D_T_Y, ADD_TPL_FP16),
    ASCENDC_TPL_DTYPE_SEL(D_T_Z, ADD_TPL_FP16),
    ASCENDC_TPL_UINT_SEL(TILE_NUM, ASCENDC_TPL_UI_LIST, 1, 8),
    ASCENDC_TPL_BOOL_SEL(IS_SPLIT, 0, 1),
    ASCENDC_TPL_TILING_STRUCT_SEL(optiling::TilingDataFp16)
    ),
    ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_DTYPE_SEL(D_T_X, ADD_TPL_FP32),
    ASCENDC_TPL_DTYPE_SEL(D_T_Y, ADD_TPL_FP32),
    ASCENDC_TPL_DTYPE_SEL(D_T_Z, ADD_TPL_FP32),
    ASCENDC_TPL_UINT_SEL(TILE_NUM, ASCENDC_TPL_UI_LIST, 1, 8),
    ASCENDC_TPL_BOOL_SEL(IS_SPLIT, 0, 1),
    ASCENDC_TPL_TILING_STRUCT_SEL(optiling::TilingDataFp)
    )
);
#endif  // TILING_KEY_ADD_CUSTOM_H