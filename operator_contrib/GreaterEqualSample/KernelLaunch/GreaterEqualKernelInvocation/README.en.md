## Directory Structure Introduction
``` 
├── GreaterEqualKernelInvocation
│   ├── cmake                   // Compilation project files
│   ├── input                   // Directory for storing script-generated input data
│   ├── output                  // Directory for storing operator runtime output data and ground truth data
│   ├── scripts
│   │   ├── acl.json            // acl configuration file
│   │   ├── gen_data.py         // Script for generating input data and ground truth data
│   │   └── verify_result.py    // Ground truth comparison file
│   ├── greater_equal.cpp       // Operator kernel implementation
│   ├── greater_equal_tiling.h  // Operator tiling implementation
│   ├── CMakeLists.txt          // Compilation project file
│   ├── data_utils.h            // Data input and output functions
│   ├── main.cpp                // Main function, application invoking the operator, including CPU and NPU domain calls
│   └── run.sh                  // Script for compiling and running the operator
``` 

## Code Implementation Introduction
This invocation example implements the generalized shape GreaterEqual operator.
- Kernel implementation   
  The mathematical expression for the GreaterEqual operator is:  
  ```
  y = x1 >= x2
  ```
  The calculation logic is: The vector calculation interfaces provided by Ascend C operate on elements of LocalTensor. Input data needs to be first moved into on-chip storage, then the calculation interfaces are used to compare the two input parameters x1 and x2, and the final result y is obtained and moved out to external storage.   
   
  The implementation process of the GreaterEqual operator is divided into three basic tasks: CopyIn, Compute, and CopyOut. The CopyIn task is responsible for moving the input Tensor xGm, yGm, and zGm from Global Memory to Local Memory, stored in x1Local and x2Local, respectively. The Compute task is responsible for performing operations on x1Local and x2Local, and storing the results in yLocal. The CopyOut task is responsible for moving the output data from yLocal to the output Tensor outGm in Global Memory. For details, please refer to [greater_equal.cpp](./greater_equal.cpp).

- Invocation implementation  
  1. CPU-side runtime verification is mainly accomplished through interfaces provided by the CPU debugging library such as ICPU_RUN_KF CPU debugging macros;  
  2. NPU-side runtime verification is mainly accomplished through the use of the <<<>>> kernel invocation operator.    
  The application distinguishes between code logic running on the CPU side and the NPU side through the ASCENDC_CPU_DEBUG macro.

## Running the Example Operator
  - Enter the example directory

    ```bash
    cd ${git_clone_path}/samples/operator_contrib/GreaterEqualSample/KernelLaunch/GreaterEqualKernelInvocation
    ```
  - Configure environment variables

    Please select the corresponding command to configure environment variables according to the [installation method](https://hiascend.com/document/redirect/CannCommunityInstSoftware) of the CANN development toolkit package on the current environment.
    - Default path, root user installs CANN software package
      ```bash
      export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
      ```
    - Default path, non-root user installs CANN software package
      ```bash
      export ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
      ```
    - Specified path install_path, installs CANN software package
      ```bash
      export ASCEND_INSTALL_PATH=${install_path}/ascend-toolkit/latest
      ```
    
    Configure the simulation mode log file directory, default is sim_log.
    ```bash
    export CAMODEL_LOG_PATH=./sim_log
    ```

  - Example execution

    ```bash
    bash run.sh -r [RUN_MODE] -v  [SOC_VERSION] 
    ```
    - RUN_MODE: Compilation method, can choose CPU debugging, NPU simulation, NPU on-board. Supported parameters are [cpu / sim / npu], default is cpu.
    - SOC_VERSION: Ascend AI processor model. If you cannot determine the specific [SOC_VERSION], execute the npu-smi info command on the server with the Ascend AI processor to query it. Add Ascend information before the "Name" value, for example, if "Name" corresponds to the value xxxyy, the actual configured [SOC_VERSION] value is Ascendxxxyy. Supported product models:
      - Atlas inference series products AI Core
      - Atlas training series products
      - Atlas A2 training series products

    Note: For Atlas training series products using NPU simulation debugging, there may be precision issues. You can choose other chips for NPU simulation debugging.

    Example:
    ```bash
    bash run.sh -r cpu -v Ascendxxxyy
    ```   

## Update Log
  | Date | Update Items |
|----|------|
| 2023/5/24 | Added this readme |
| 2024/07/24 | Modified readme format |