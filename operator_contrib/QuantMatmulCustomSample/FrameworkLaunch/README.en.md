## Overview
This example is based on the QuantMatMulCustom operator project and introduces the invocation method for single operator calls.

## Directory Structure Introduction
``` 
├── FrameworkLaunch                     // Invoking the QuantMatMulCustom operator using the framework
│   ├── CannMobileInvocation            // Invoking the QuantMatMulCustom operator using the CannMobile method
│   ├── QuantMatMulCustom               // QuantMatMulCustom operator project
│   └── QuantMatMulCustom.json          // Prototype definition json file for the QuantMatMulCustom operator
``` 

## Operator Project Introduction
The operator project directory QuantMatMulCustom contains template files for operator implementation, compilation scripts, etc., as shown below:
``` 
├── QuantMatMulCustom               // QuantMatMulCustom custom operator project
│   ├── cmake
│   ├── framework           // Directory for operator plugin implementation files, not required for single operator model file generation
│   ├── op_host             // Host-side implementation files
│   ├── op_kernel           // Kernel-side implementation files
│   ├── scripts             // Directory for custom operator project packaging scripts
│   ├── build.sh            // Compilation entry script
│   ├── build_devices       // Compilation script of devices
│   ├── CMakeLists.txt      // CMakeLists.txt for the operator project
│   └── CMakePresets.json   // Compilation configuration items
``` 
The CANN software package provides the project creation tool msopgen. The QuantMatMulCustom operator project can be automatically created via QuantMatMulCustom.json. 

Before creating, it is necessary to prepare the environment. Please refer to [Preparation: Environment Installation](../README.en.md#envready)。
Execute the following command to create a QuantMatmulCustom operator project:
```
msopgen gen -i ./QuantMatMulCustom.json -c ai_core-kirin9020 -f ONNX -out ./QuantMatMulCustom
```

## Compiling and Running the Example Operator
For custom operator projects, the compilation and running process includes the following steps:
- Compiling the custom operator project to generate the operator installation package;
- Installing the custom operator into the operator library;
- Invoking and executing the custom operator;

Detailed operations are as follows.

### 1. Obtain the Source Package
Before compiling this example, please refer to [Preparation: Obtain the Example Code](../README.en.md#codeready) to obtain the source package.

### 2. Compile the Operator Project<a name="operatorcompile"></a>
Compile the custom operator project to build and generate the custom operator package.

- Execute the following command to switch to the QuantMatMulCustom operator project directory.

  ```bash
  cd ${git_clone_path}/samples/operator_contrib/QuantMatMulCustomSample/FrameworkLaunch/QuantMatMulCustom
  ```

- Execute the following command in the QuantMatMulCustom operator project directory to compile the operator project.

  ```bash
  ./build.sh
  ```

After successful compilation, log will display "Install the project..."

### 5. Invoke and Execute the Operator Project
- [Invoking the QuantMatMulCustom Operator Project using CannMobile](./CannMobileInvocation/README.en.md)

## Update Log
  | Date       | Update Item |
|------------|------|
| 2024/12/10 | New version of readme updated |