## 目录结构介绍
``` 
├── CannMobileInvocation                                 //通过cann mobile调用的方式调用QuantMatMulCustom算子
│   └── create_quantmatmul_onnx.py                       //生成onnx格式QuantMatMulCustom自定义算子
```
## 运行样例算子
### 1.&nbsp;编译算子工程
运行此样例前，请参考[编译算子工程](../README.md#operatorcompile)完成前期准备。
### 2.&nbsp;Cann mobile调用样例运行

  - 进入到样例目录

    ```bash
    cd ${git_clone_path}/samples/operator_contrib/QuantMatMulCustomSample/FrameworkLaunch/CannMobileInvocation
    ```
  - 生成onnx格式QuantMatMulCustom自定义算子(要求环境中安装有torch和onnx，或者使用 pip install torch onnx 安装)

    ```
    python  create_quantmatmul_onnx.py
    ```
  - 下载[CANN Mobile软件包](https://contentcenter-vali-drcn.dbankcdn.cn/pvt_2/DeveloperAlliance_package_901_9/24/v3/-oSN_kh6Tba4GDB0EMlkMg/DDK_tools_5.0.2.0.zip?HW-CC-KV=V1&HW-CC-Date=20241226T031910Z&HW-CC-Expire=315360000&HW-CC-Sign=B06D348B1E3B988F9259B7EAEAFF56E3618890A10EC4FF2C2DF4A69139E51EA3)，解压后，将上一步生成的QuantMatMulCustom.onnx拷贝到ddk_external/tools/tools_omg目录下，执行如下命令，生成cann mobile编译后QuantMatMulCustom自定义算子

    ```
    ./omg --model=./QuantMatMulCustom.onnx --framework=5 --output=./QuantMatMulCustom --target=omc
    ```
  - 下载[CANN Mobile软件包](https://contentcenter-vali-drcn.dbankcdn.cn/pvt_2/DeveloperAlliance_package_901_9/24/v3/-oSN_kh6Tba4GDB0EMlkMg/DDK_tools_5.0.2.0.zip?HW-CC-KV=V1&HW-CC-Date=20241226T031910Z&HW-CC-Expire=315360000&HW-CC-Sign=B06D348B1E3B988F9259B7EAEAFF56E3618890A10EC4FF2C2DF4A69139E51EA3)，解压后，将ddk_external/tools/tools_sysdbg目录下的所有文件和编译后算子QuantMatMulCustom.omc，PUSH到手机环境中，放到"/data/local/tmp"目录下（必须放到"/data/local/tmp"或其子目录下）
  - 添加动态链接库路径，与步骤1中存放路径保持一致。执行命令：export LD_LIBRARY_PATH={步骤1的so存放路径}。
  - 添加model_run_tool的执行权限。执行命令：chmod +x model_run_tool。
  - 执行model_run_tool

    ```
    ./model_run_tool --model=QuantMatMulCustom.omc
    ```    
    运行成功时，日志显示“running model successded”
  - 精度和性能调试，参见[Cann Mobile调试工具使用说明](https://developer.huawei.com/consumer/cn/doc/hiai-Guides/system-debug-tool-instructions-0000001053005651)
## 更新说明
  | 时间         | 更新事项 |
|------------|------|
| 2024/12/10 | 新增本readme |