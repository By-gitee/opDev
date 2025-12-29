### 1. tiling使用说明 
  请参考[tiling_key_add_custom.h](op_kernel/tiling_key_add_custom.h)进行tiling配置。

### 2. 算子实现描述
- kernel侧

  引入[tiling的声明文件](op_kernel/tiling_key_add_custom.h)，kernel入口支持template&lt;typename args...&gt;函数模板的声明调用，算子实现时，args参数可以替换tilingkey，作为不同代码分支的判断条件，以便于在开发过程中无需关注tilingkey。
    ```
    #include "tiling_key_add_custom.h"
    ……
    template<int D_T_X, int D_T_Y, int D_T_Z, int TILE_NUM, int IS_SPLIT>
     __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
    {
         …… 
    }
    ```
- host侧
  
  引入[tiling的声明文件](op_kernel/tiling_key_add_custom.h)，tilingkey通过GET_TPL_TILING_KEY函数及模板参数进行生成，调用SetTilingKey函数设置tilingkey，tilingkey用于代码编译，同时将模板参数传入kernel侧。
    ```
    #include "tiling_key_add_custom.h"
    ……
    namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        ……
        const uint64_t tilingKey = GET_TPL_TILING_KEY(D_T_X, D_T_Y, D_T_Z, TILE_NUM, IS_SPLIT); // 模板参数tilingkey配置
        context->SetTilingKey(tilingKey);
        ……
    }
    ```