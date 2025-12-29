## 概述
本样例基于QuantMatmulCustom算子工程，介绍了单算子调用的调用方式。
## 目录结构介绍
``` 
├── FrameworkLaunch                   //使用框架调用的方式调用QuantMatmulCustom算子
│   ├── CannMobileInvocation          // 通过CannMobile调用的方式调用QuantMatmulCustom算子
│   ├── QuantMatmulCustom             // QuantMatmulCustom算子工程
│   └── QuantMatmulCustom.json        // QuantMatmulCustom算子的原型定义json文件
``` 
## 算子工程介绍
算子工程目录QuantMatmulCustom包含算子实现的模板文件、编译脚本等，如下所示:
``` 
├── QuantMatmulCustom               //QuantMatmulCustom自定义算子工程
│   ├── cmake
│   ├── framework           // 算子插件实现文件目录，单算子模型文件的生成不依赖算子适配插件，无需关注
│   ├── op_host             // host侧实现文件
│   ├── op_kernel           // kernel侧实现文件
│   ├── scripts             // 自定义算子工程打包相关脚本所在目录
│   ├── build.sh            // 编译入口脚本
│   ├── build_devices.sh    // 编译devices侧交付件脚本
│   ├── CMakeLists.txt      // 算子工程的CMakeLists.txt
│   └── CMakePresets.json   // 编译配置项
``` 
CannMobile 软件包中提供了工程创建工具msopgen，QuantMatmulCustom算子工程可通过QuantMatmulCustom.json自动创建。
创建前需要先准备好环境请参考[准备：环境安装](../README.md#envready)。
执行如下命令创建QuantMatmulCustom算子工程：
```
msopgen gen -i ./QuantMatMulCustom.json -c ai_core-kirin9020 -f ONNX -out ./QuantMatMulCustom
```

## 编译运行样例算子
针对自定义算子工程，编译运行包含如下步骤：
- 编译自定义算子工程生成算子安装包；
- 安装自定义算子到算子库中；
- 调用执行自定义算子；

详细操作如下所示。
### 1.&nbsp;获取源码包
编译此样例前，请参考[准备：获取样例代码](../README.md#codeready)获取源码包。
### 2.&nbsp;编译算子工程<a name="operatorcompile"></a>
  编译自定义算子工程，构建生成自定义算子包。

  - 执行如下命令，切换到算子工程QuantMatmulCustom目录。

    ```bash
    cd ${git_clone_path}/samples/operator_contrib/QuantMatmulCustomSample/FrameworkLaunch/QuantMatmulCustom
    ```
  - 在算子工程QuantMatmulCustom目录下执行如下命令，进行算子工程编译。

    ```bash
    ./build.sh
    ```
  编译成功后，日志会显示"Install the project..."
### 3.&nbsp;调用执行算子工程
- [CannMobile调用QuantMatmulCustom算子工程](./CannMobileInvocation/README.md)

## 更新说明
  | 时间         | 更新事项 |
|------------|------|
| 2024/12/10 | 新版readme更新 |
