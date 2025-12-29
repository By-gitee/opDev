## Overview
This sample is based on the LpNormV2Custom operator project and introduces the invocation method for single operator calls.

## Directory Structure Overview
```
├── FrameworkLaunch           // Project for invoking the Add operator using a framework
│   ├── AclNNInvocation       // Invocation of the LpNormV2Custom operator via aclnn
│   ├── LpNormV2Custom        // LpNormV2Custom operator project
│   └── LpNormV2Custom.json   // JSON file defining the prototype of LpNormV2Custom operator
```

## Operator Project Description
The LpNormV2Custom directory contains template files, compilation scripts, and more for the operator implementation:
```
├── LpNormV2Custom            // LpNormV2Custom custom operator project
│   ├── cmake
│   ├── framework             // Directory for operator plugin implementation files; single operator model files do not depend on operator adaptation plugins, there is no need to pay attention to it.
│   ├── op_host               // Host-side implementation files
│   ├── op_kernel             // Kernel-side implementation files
│   ├── scripts               // Directory for packaging-related scripts
│   ├── build.sh              // Compilation entry script
│   ├── CMakeLists.txt        // CMakeLists.txt for the operator project
│   └── CMakePresets.json     // Compilation configuration items
```
The CANN software package provides the project creation tool msopgen, and the LpNormV2Custom operator project can be automatically created through LpNormV2Custom.json. For details, please refer to the [Ascend C Operator Development](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC) > Operator Development > Operator Development Project > Operator Development Based on Custom Operator Project > Creating Operator Project section.

## Compile and Run Sample Operator
The steps for compiling and running the custom operator project include:
- Compiling the custom operator project to generate the operator installation package.
- Installing the custom operator into the operator library.
- Invoking the custom operator.

Detailed operations are as follows.

### 1. Obtain the Source Package
Before compiling and running this sample, please refer to [Preparation: Obtain Sample Code](../README.en.md#codeready) for source code package.

### 2. Compile the Operator Project<a name="operatorcompile"></a>
Compile the custom operator project to build the custom operator package.

- Execute the following command to switch to the LpNormV2Custom project directory.
  ```bash
  cd ${git_clone_path}/samples/operator_contrib/LpNormV2CustomSample/FrameworkLaunch/LpNormV2Custom
  ```

- Modify the `CMakePresets.json` file to set `ASCEND_CANN_PACKAGE_PATH` to the actual installation path of the CANN software package.
  ```json
  {
      ……
      "configurePresets": [
          {
              …… 
              "ASCEND_CANN_PACKAGE_PATH": {
                  "type": "PATH",
                  "value": "/usr/local/Ascend/ascend-toolkit/latest" // Replace with the actual path, e.g., /home/HwHiAiUser/Ascend/ascend-toolkit/latest
              },
              …… 
          }
      ]
  }
  ```

- Run the following command in the LpNormV2Custom project directory to compile the operator project.
  ```bash
  ./build.sh
  ```
After a successful compilation, a `build_out` directory will be created in the current directory, containing the custom operator installation package, named `custom_opp_<target os>_<target architecture>.run`, e.g., “custom_opp_ubuntu_x86_64.run”.

**Note:** If you want to use the dump debugging feature, you need to remove the configurations for Atlas training series products and Atlas 200/500 A2 inference products from op_host and CMakeLists.txt.

### 3. Deploy the Operator Package
Run the following command to install the custom operator package from its directory.
```bash
cd build_out
./custom_opp_<target os>_<target architecture>.run
```
After successful execution, the related files from the custom operator package will be deployed to the `vendors/customize` directory of the OPP operator library in the current environment.

### 4. Configure Environment Variables
Select the corresponding command to configure environment variables based on the installation method of the CANN development suite:
- Default path, installed by root user:
  ```bash
  export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
  ```
- Default path, installed by non-root user:
  ```bash
  export ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
  ```
- Specified path `install_path`, where the CANN software package is installed:
  ```bash
  export ASCEND_INSTALL_PATH=${install_path}/ascend-toolkit/latest
  ```

### 5. Invoke and Execute the Operator Project
- [Invoke LpNormV2Custom Operator Project using aclnn](./AclNNInvocation/README.en.md)

## Update Notes
| Date       | Updates                         |
|------------|---------------------------------|
| 2024/05/24 | Updated README version          |
| 2024/07/22 | Modified environment configuration for different users |
| 2024/07/24 | Modified README format          |