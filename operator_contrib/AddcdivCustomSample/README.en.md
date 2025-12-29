## Addcdiv Custom Operator Example Explanation
This example implements the Addcdiv operator using the Ascend C programming language and provides end-to-end implementations for different operator invocation methods.

- [FrameworkLaunch](./FrameworkLaunch/README.en.md): Invokes the Addcdiv custom operator using the framework.  
  Follows the process of project creation -> operator implementation -> compilation and deployment -> operator invocation to complete the operator development. The entire process relies on the operator project: develop the operator kernel function and Tiling implementation based on the project code framework, compile and deploy the operator using the project's compilation script, and then achieve single operator invocation or operator invocation within third-party frameworks.

- [KernelLaunch](./KernelLaunch/README.en.md): Invokes the Addcdiv custom operator directly using the kernel function.  
  The basic kernel function invocation (Kernel Launch) method. After developers complete the development of the operator kernel function and Tiling implementation, they can use the AscendCL runtime interface to complete the operator invocation.

This example includes the following invocation methods:
<table>
    <th>Invocation Method</th><th>Directory</th><th>Description</th>
    <tr>
        <!-- Column occupies 1 cell -->
        <td rowspan='1'><a href="./FrameworkLaunch/README.en.md"> FrameworkLaunch</td><td><a href="./FrameworkLaunch/AclNNInvocation/README.en.md"> AclNNInvocation</td><td>Invokes the AddcdivCustom operator using the aclnn method.</td>
    </tr>
    <tr>
        <!-- Column occupies 1 cell -->
        <td rowspan='1'><a href="./KernelLaunch/README.en.md"> KernelLaunch</td><td><a href="./KernelLaunch/AddcdivKernelInvocation/README.en.md"> AddcdivKernelInvocation</td><td>Kernel function invoking program on the host side, including the running verification methods on the CPU side and NPU side.</td>
    </tr>
</table>

## Operator Description
The Addcdiv operator implements the functionality of dividing vector y by vector z, multiplying the result by attribute value, adding vector x, and returning the computed result.

The corresponding mathematical expression is:  
```
out = (x + (y / z) * value)
```

## Operator Specification Description
<table>  
<tr><th align="center">Operator Type (OpType)</th><th colspan="5" align="center">Addcdiv</th></tr>  
<tr><td rowspan="4" align="center">Operator Inputs</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">default</td></tr>  
<tr><td align="center">x</td><td align="center">-</td><td align="center">float16</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td align="center">y</td><td align="center">-</td><td align="center">float16</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td align="center">z</td><td align="center">-</td><td align="center">float16</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td rowspan="1" align="center">Operator Output</td><td align="center">out</td><td align="center">-</td><td align="center">float16</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td align="center">attr Attributes</td><td align="center">value</td><td align="center">\</td><td align="center">float16</td><td align="center">\</td><td align="center">1.0</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">addcdiv_custom</td></tr>  
</table>

## Supported Product Models
This example supports the following product models:
- Atlas 200/500 A2 inference products

## Directory Structure Introduction
```
├── FrameworkLaunch    // Project for invoking the Addcdiv custom operator using the framework.
└── KernelLaunch       // Project for invoking the Addcdiv custom operator directly using the kernel function.
```

## Environment Requirements
Before compiling and running this example, please refer to the [CANN Software Installation Guide](https://hiascend.com/document/redirect/CannCommunityInstSoftware) to deploy the development and runtime environment.

## Compiling and Running the Example Operator

### 1. Preparation: Obtain the Example Code<a name="codeready"></a>

You can download the source code using one of the following two methods. Please choose one.

- Command line download (takes longer but is simpler).

  ```bash
  # In the development environment, execute the following command as a non-root user to download the source code repository. git_clone_path is a directory created by the user.
  cd ${git_clone_path}
  git clone https://gitee.com/ascend/samples.git
  ```
  **Note: If you need to switch to another tag version, for example, v0.5.0, you can execute the following command.**
  ```bash
  git checkout v0.5.0
  ```

- Zip file download (takes shorter time but is slightly more complex).

  **Note: If you need to download the code for another version, please first switch the samples repository branch according to the prerequisites.**
  ```bash
  # 1. In the samples repository, click the 【Clone/Download】 dropdown and select 【Download ZIP】.
  # 2. Upload the ZIP file to a directory in the development environment, for example, ${git_clone_path}/ascend-samples-master.zip.
  # 3. In the development environment, execute the following command to unzip the zip file.
  cd ${git_clone_path}
  unzip ascend-samples-master.zip
  ```

### 2. Compile and Run the Example Project
- If using the framework invocation method, please refer to [FrameworkLaunch](./FrameworkLaunch/README.en.md) for compilation and running instructions.    
- If using the kernel function direct invocation method, please refer to [KernelLaunch](./KernelLaunch/README.en.md) for compilation and running instructions.

## Update Log
| Date       | Update Items                       |
| ---------- | ---------------------------------- |
| 2024/05/24 | New readme update                  |
| 2024/07/22 | Modified to clone to any directory |