## GreaterEqual Custom Operator Example Explanation
This example implements the GreaterEqual operator using the Ascend C programming language and provides end-to-end implementations for different operator invocation methods.

- [FrameworkLaunch](./FrameworkLaunch/README.en.md): Invokes the GreaterEqual custom operator using the framework.
  The operator development is completed following the process of project creation -> operator implementation -> compilation and deployment -> operator invocation. The entire process relies on the operator project: the operator kernel function and Tiling implementation are completed based on the project code framework, the operator is compiled and deployed through the project compilation script, and then the operator is invoked either as a single operator or within a third-party framework.
- [KernelLaunch](./KernelLaunch/README.en.md): Invokes the GreaterEqual custom operator using direct kernel function calls.
  The basic kernel function invocation (Kernel Launch) method allows developers to complete the development and Tiling implementation of the operator kernel function, and then invoke the operator through the AscendCL runtime interface.

This example includes the following invocation methods:
<table>
    <th>Invocation Method</th><th>Directory</th><th>Description</th>
    <tr>
        <!-- Column occupies 1 cell -->
        <td rowspan='1'><a href="./FrameworkLaunch/README.en.md"> FrameworkLaunch</td><td><a href="./FrameworkLaunch/AclNNInvocation/README.en.md"> AclNNInvocation</td><td>Invokes the GreaterEqualCustom operator using the aclnn method.</td>
    </tr>
    <tr>
        <!-- Column occupies 1 cell -->
        <td rowspan='1'><a href="./KernelLaunch/README.en.md"> KernelLaunch</td><td><a href="./KernelLaunch/GreaterEqualKernelInvocation/README.en.md"> GreaterEqualKernelInvocation</td><td>Host-side kernel function invocation program, including CPU-side and NPU-side verification methods.</td>
    </tr>
</table>

## Operator Description
The GreaterEqual operator implements the functionality where the corresponding position of vector y outputs true if vector x1 is greater than or equal to x2, otherwise false.
The corresponding mathematical expression is:
```
y = x1 >= x2
```

## Operator Specification Description
<table>  
<tr><th align="center">Operator Type (OpType)</th><th colspan="4" align="center">GreaterEqual</th></tr>  
<tr><td rowspan="3" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td align="center">x1</td><td align="center">-</td><td align="center">float32, float16, int32</td><td align="center">ND</td></tr>  
<tr><td align="center">x2</td><td align="center">-</td><td align="center">float32, float16, int32</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">Operator Output</td><td align="center">y</td><td align="center">-</td><td align="center">float32, float16, int32</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">GreaterEqual</td></tr>
</table>

## Supported Product Models
This example supports the following product models:
- Atlas 200/500 A2 Inference Products
- Atlas A2 Training Series Products / Atlas 800I A2 Inference Products
## Directory Structure Introduction
```
├── FrameworkLaunch    // Project for invoking the GreaterEqual custom operator using the framework.
└── KernelLaunch       // Project for invoking the GreaterEqual custom operator using direct kernel function calls.
```

## Environment Requirements
Before compiling and running this example, please refer to [《CANN Software Installation Guide》](https://hiascend.com/document/redirect/CannCommunityInstSoftware) to deploy the development and runtime environment.

## Compiling and Running the Example Operator

### 1. Preparation: Obtain the Example Code<a name="codeready"></a>

You can download the source code using one of the following two methods. Please choose one.

- Command line method (long download time, simple steps).

  ```bash
  # In the development environment, execute the following command as a non-root user to download the source repository. git_clone_path is a directory created by the user.
  cd ${git_clone_path}
  git clone https://gitee.com/ascend/samples.git
  ```
  **Note: If you need to switch to another tag version, for example, v0.5.0, you can execute the following command.**
  ```bash
  git checkout v0.5.0
  ```
- Zip package method (short download time, slightly more complex steps).

  **Note: If you need to download the code for another version, please first switch the samples repository branch according to the preconditions.**
  ```bash
  # 1. In the samples repository, select the 【Clone/Download】 dropdown and choose 【Download ZIP】.
  # 2. Upload the ZIP package to a directory of a normal user in the development environment, for example, ${git_clone_path}/ascend-samples-master.zip.
  # 3. In the development environment, execute the following command to unzip the zip package.
  cd ${git_clone_path}
  unzip ascend-samples-master.zip
  ```

### 2. Compile and Run the Example Project
- If using the framework invocation method, please refer to [FrameworkLaunch](./FrameworkLaunch/README.en.md) for compilation and running operations.
- If using the direct kernel function invocation method, please refer to [KernelLaunch](./KernelLaunch/README.en.md) for compilation and running operations.

## Update Log
  | Date | Update Items |
|----|------|
| 2024/05/24 | New version of readme updated |
| 2024/07/22 | Modified to clone to any directory |
| 2024/07/24 | Modified readme format |