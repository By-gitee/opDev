## Directory Structure Introduction
``` 
├── AclNNInvocationSample             // Invokes the AddCustom operator using the aclnn method
│   ├── gen_data.py             // Script for generating input data and ground truth data
│   ├── verify_result.py        // Ground truth comparison file
│   ├── main.cpp                // Entry point for the single operator invocation application
│   └── run.sh                  // Execution command script
``` 

## Code Implementation Introduction
After developing and deploying the custom operator, you can verify the functionality of the single operator by invoking it. The code in src/main.cpp is the execution method for the single operator API. Single operator API execution is based on the C language API to execute the operator, without the need to provide a single operator description file for offline model conversion, directly calling the single operator API interface.

After the custom operator is compiled and deployed, the single operator API is automatically generated and can be directly called in the application. The operator API is generally defined as a "two-stage interface", such as:
   ```cpp    
   aclnnStatus aclnnAddCustomGetWorkspaceSize(const aclTensor *x, const aclTensor *y, const alcTensor *out, uint64_t workspaceSize, aclOpExecutor **executor);
   aclnnStatus aclnnAddCustom(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream);
   ```
Where aclnnAddCustomGetWorkspaceSize is the first stage interface, mainly used to calculate how much workspace memory is needed during this API call. After obtaining the required workspace size for this API calculation, allocate Device-side memory according to the workspaceSize, and then call the second stage interface aclnnAddCustom to perform the calculation. For specific reference, see the [AscendCL Single Operator Invocation](https://hiascend.com/document/redirect/CannCommunityAscendCInVorkSingleOp) > Single Operator API Execution section.

## Running the Example Operator
### 1. Compiling the Operator Project
Before running this example, please refer to [Compiling the Operator Project](../README.en.md#operatorcompile) to complete the preliminary preparation.

### 2. Running the aclnn Example

  - Enter the example directory

    ```bash
    cd ${git_clone_path}/samples/operator_contrib/AddCustomSample/FrameworkLaunch/AclNNInvocation
    ```
  - Modify compile file of the sample.

    In the CMakeLists.txt file, update the values for INC_PATH, LIB_PATH, and LIB_PATH1 to reflect the actual installation path of the CANN software package, replacing the default setting "/usr/local/Ascend/ascend-toolkit/latest"
    eg:/home/HwHiAiUser/Ascend/ascend-toolkit/latest

  - Environment variable configuration

    Set the environment variable NPU_HOST_LIB, in an x86 environment using the following command:
    ```bash
    export NPU_HOST_LIB=/home/HwHiAiUser/Ascend/ascend-toolkit/latest/x86_64-linux/lib64
    ```    
  - Example execution    

    During the example execution, test data will be automatically generated, then the aclnn example will be compiled and run, and finally, the results will be verified. The specific process can be found in the run.sh script.

    ```bash
    bash run.sh
    ```

## Update Log
  | Date | Update Items |
|----|------|
| 2024/09/09 | Update README |