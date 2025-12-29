## Directory Structure Introduction
```
├── AclNNInvocation         // Invoking the BallQuery operator using the aclnn method
│   ├── inc                     // Header file directory
│   │   ├── common.h            // Declares common method classes, used for reading binary files
│   │   ├── op_runner.h         // Operator description declaration file, including operator inputs/outputs, operator type, and input/output descriptions
│   │   └── operator_desc.h     // Operator runtime information declaration file, including the number of operator inputs/outputs, input/output sizes, etc.
│   ├── scripts
│   │   ├── acl.json            // acl configuration file
│   │   ├── gen_data.py         // Script for generating input data and ground truth data
│   │   └── verify_result.py    // Ground truth comparison file
│   ├── src
│   │   ├── CMakeLists.txt      // Compilation rules file
│   │   ├── common.cpp          // Common functions, implementation file for reading binary files
│   │   ├── main.cpp            // Entry point for the single operator invocation application
│   │   ├── op_runner.cpp       // Main flow implementation file for single operator invocation
│   │   └── operator_desc.cpp   // Constructs the inputs and outputs descriptions for the operator
│   └── run.sh                  // Execution command script
```
## Code Implementation Introduction
After completing the development and deployment of the custom operator, you can verify the functionality of the single operator by invoking it. The code in src/main.cpp is the execution method for the single operator API. Single operator API execution is based on the C language API to execute the operator, without the need for a single operator description file for offline model conversion, directly calling the single operator API interface.

After the custom operator is compiled and deployed, the single operator API is automatically generated and can be directly called in the application. The operator API is generally defined in the form of a "two-stage interface", such as:
   ```cpp    
aclnnStatus aclnnBallQueryGetWorkspaceSize(
    const aclTensor *xyz,
    const aclTensor *centerXyz,
    const aclTensor *xyzBatchCntOptional,
    const aclTensor *centerXyzBatchCntOptional,
    double minRadius,
    double maxRadius,
    int64_t sampleNum,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnBallQuery(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);
   ```
The aclnnBallQueryGetWorkspaceSize is the first stage interface, mainly used to calculate how much workspace memory is needed for this API call. After obtaining the workspace size required for this API calculation, allocate Device-side memory according to the workspaceSize, and then call the second stage interface aclnnBallQuery to perform the calculation. For specific reference, see the [AscendCL Single Operator Invocation](https://hiascend.com/document/redirect/CannCommunityAscendCInVorkSingleOp) > Single Operator API Execution section.

## Running the Example Operator
### 1. Compile the Operator Project
Before running this example, please refer to [Compile the Operator Project](../README.en.md#operatorcompile) to complete the preparation.

### 2. Running the aclnn Example

  - Enter the example directory

    ```bash
    cd ${git_clone_path}/samples/operator_contrib/BallQuerySample/FrameworkLaunch/AclNNInvocation
    ```

  - Example execution    

    During the example execution, test data will be automatically generated, then the aclnn example will be compiled and run, and finally, the running results will be verified. The specific process can be seen in the run.sh script.

    ```bash
    bash run.sh
    ```

## Update Log
  | Date | Update Item |
|----|------|
| 2024/10/23 | add this readme |
