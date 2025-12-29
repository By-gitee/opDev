## Overview
This example is based on the ScatterMax operator project and introduces the method of single operator invocation.

## Directory Structure
``` 
├── FrameworkLaunch           // Project for invoking the ScatterMax operator using a framework
│   ├── AclNNInvocation       // Invocation of the ScatterMax operator via aclnn
│   ├── ScatterMax            // ScatterMax operator project
│   └── ScatterMax.json       // Prototype definition JSON file for the ScatterMax operator
``` 

## Operator Project Description
The ScatterMax directory contains the template files for the operator implementation, compilation scripts, and more for the operator implementation:
``` 
├── ScatterMax              // ScatterMax custom operator project
│   ├── cmake
│   ├── framework           // Directory for operator plugin implementation files; single operator model files do not depend on operator adaptation plugins, there is no need to pay attention to it.
│   ├── op_host             // Host-side implementation files
│   ├── op_kernel           // Kernel-side implementation files
│   ├── scripts             // Directory containing scripts related to the packaging of the custom operator project
│   ├── build.sh            // Entry script for compilation
│   ├── CMakeLists.txt      // CMakeLists.txt for the operator project
│   └── CMakePresets.json   // Compilation configuration items
``` 

The CANN software package provides the project creation tool msopgen, and the ScatterMax operator project can be automatically created through ScatterMax.json. For details, please refer to the [Ascend C Operator Development](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC) > Operator Development > Operator Development Project > Operator Development Based on Custom Operator Project > Creating Operator Project section.

## Compile and Run the Example Operator

The steps for compiling and running the custom operator project include:
- Compiling the custom operator project to generate the operator installation package.
- Installing the custom operator into the operator library.
- Invoking the custom operator.

Detailed operations are as follows.

### 1. Obtain the Source Code
Before compiling and running this sample, please refer to [Preparation: Obtain Sample Code](../README.en.md#codeready) for source code package.

### 2. Compile the Operator Project<a name="operatorcompile"></a>
Compile the custom operator project to build the custom operator package.

- Execute the following command to switch to the ScatterMax project directory.

    ```bash
    cd ${git_clone_path}/samples/operator_contrib/ScatterMaxSample/FrameworkLaunch/ScatterMax
    ```

- Modify the ASCEND_CANN_PACKAGE_PATH in CMakePresets.json to the actual installation path of the CANN software package.

    ```json
    {
        ……
        "configurePresets": [
            {
                    ……
                    "ASCEND_CANN_PACKAGE_PATH": {
                        "type": "PATH",
                        "value": "/usr/local/Ascend/ascend-toolkit/latest"   // Replace with the actual installation path of the CANN software package. For example: /home/HwHiAiUser/Ascend/ascend-toolkit/latest
                    },
                    ……
            }
        ]
    }
    ```

- Execute the following command in the ScatterMax operator project directory to compile the operator project.

    ```bash
    ./build.sh
    ```

After successful compilation, the build_out directory will be created in the current directory, and the custom operator installation package custom_opp_<target os>_<target architecture>.run will be generated in the build_out directory, for example, "custom_opp_ubuntu_x86_64.run".

**Note:** If you want to use the dump debugging feature, you need to remove the configurations for Atlas training series products and Atlas 200/500 A2 inference products from op_host and CMakeLists.txt.

### 3. Deploy the Operator Package

Run the following commands in the directory where the custom operator installation package is located to install the custom operator package.
```bash
cd build_out
./custom_opp_<target os>_<target architecture>.run
```

After successful execution, the relevant files in the custom operator package will be deployed to the `vendors/customize` directory of the OPP operator library in the current environment.

### 4. Configure Environment Variables

Please select the corresponding command to configure environment variables based on the [installation method](https://hiascend.com/document/redirect/CannCommunityInstSoftware) of the CANN development toolkit package on the current environment.

- Default path, root user installation of the CANN software package:
    ```bash
    export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    ```
- Default path, non-root user installation of the CANN software package:
    ```bash
    export ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    ```
- Specified path `install_path` for CANN software package installation:
    ```bash
    export ASCEND_INSTALL_PATH=${install_path}/ascend-toolkit/latest
    ```

### 5. Invoke and Execute the Operator Project
- [Aclnn invocation of the ScatterMax operator project](./AclNNInvocation/README.en.md)

## Update Log
| Date       | Updates                     |
|------------|------------------------------|
| 2024/05/27 | Updated README to new version |
| 2024/07/22 | Modified environment configuration for different user environments |
| 2024/07/24 | Modified README format         |