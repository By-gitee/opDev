## Directory Structure Introduction
``` 
├── CannMobileInvocation                            // Invoking the QuantMatMulCustom operator using the cann mobile method
│   └── create_quantmatmul_onnx.py                  // Generate QuantMatMulCustom in onnx format
``` 

## Running the Example Operator
### 1. Compile the Operator Project
Before running this example, please refer to [Compile the Operator Project](../README.en.md#operatorcompile) to complete the preparation.

### 2. Running the cann mobile Example

  - Enter the example directory

    ```bash
    cd ${git_clone_path}/samples/operator_contrib/QuantMatMulCustomSample/FrameworkLaunch/CannMobileInvocation
    ```

  - Generate QuantMatMulCustom in onnx format(Require torch and onnx to be installed in the environment, or use pip install torch onnx to install)

    ```
    python  create_quantmatmul_onnx.py
    ```
  - Download [CANN Mobile software package](https://contentcenter-vali-drcn.dbankcdn.cn/pvt_2/DeveloperAlliance_package_901_9/24/v3/-oSN_kh6Tba4GDB0EMlkMg/DDK_tools_5.0.2.0.zip?HW-CC-KV=V1&HW-CC-Date=20241226T031910Z&HW-CC-Expire=315360000&HW-CC-Sign=B06D348B1E3B988F9259B7EAEAFF56E3618890A10EC4FF2C2DF4A69139E51EA3) and decompress the package. copy QuantMatMulCustom.onnx generated in the previous step to the ddk_external/tools/tools_omg directory, Generate QuantMatMulCustom in cann mobile format

    ```
    ./omg --model=./QuantMatMulCustom.onnx --framework=5 --output=./QuantMatMulCustom --target=omc
    ```

  - Download [CANN Mobile software package](https://contentcenter-vali-drcn.dbankcdn.cn/pvt_2/DeveloperAlliance_package_901_9/24/v3/-oSN_kh6Tba4GDB0EMlkMg/DDK_tools_5.0.2.0.zip?HW-CC-KV=V1&HW-CC-Date=20241226T031910Z&HW-CC-Expire=315360000&HW-CC-Sign=B06D348B1E3B988F9259B7EAEAFF56E3618890A10EC4FF2C2DF4A69139E51EA3) and decompress the package. Push all the files in ddk_external/tools/tools_sysdbg and QuantMatMulCustom.omc to /data/local/tmp on the mobile phone environment. (The files must be placed in /data/local/tmp or its subdirectories.)
  - Add the path of dynamic link libraries, which must be the same as the path specified in Step 1. Run the export LD_LIBRARY_PATH={Path for storing the .so files in Step 1}.
  - Grant the execute permission on model_run_tool. Run the chmod +x model_run_tool command.
  - Run model_run_tool
    ```
    ./model_run_tool --model=QuantMatMulCustom.omc
    ```
    When running successfully, the log displays“running model succeeded”
  - Precision and performance debugging, refer to the [Cann Mobile Debugging Tool User Manual](https://developer.huawei.com/consumer/cn/doc/hiai-Guides/system-debug-tool-instructions-0000001053005651)

## Update Log
  | Date       | Update Item |
|------------|------|
| 2024/12/10 | Added this readme |