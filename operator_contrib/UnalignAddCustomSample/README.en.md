## UnalignAdd Custom Operator Example Explanation
This example implements the Add operator(Unaligned) using the Ascend C programming language and provides end-to-end implementations for different operator invocation methods.

- [FrameworkLaunch](./FrameworkLaunch/README.en.md): Invokes the Add custom operator using the framework.
  The operator development is completed following the process of project creation -> operator implementation -> compilation and deployment -> operator invocation. The entire process relies on the operator project: the operator kernel function and Tiling implementation are completed based on the project code framework, the operator is compiled and deployed through the project compilation script, and then the operator is invoked either as a single operator or within a third-party framework.

This example includes the following invocation methods:
<table>
    <th>Invocation Method</th><th>Directory</th><th>Description</th>
    <tr>
        <!-- Column occupies 7 cell -->
        <td rowspan='2'><a href="./FrameworkLaunch/README.en.md"> FrameworkLaunch</td><td><a href="./FrameworkLaunch/AclNNInvocation/README.en.md"> AclNNInvocation</td><td>Invokes the Add operator using the aclnn method.</td>
    </tr>
    <tr>
        <td><a href="./FrameworkLaunch/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>Invokes the AddCustom operator using the aclnn method, with a simplified compilation script.</td>
    </tr>
    <tr>    
</table>

## Operator Description
The Add operator implements the functionality of adding two vectors together and returning the sum. This operator has achieved the functionality for non-32B aligned shapes, thereby accomplishing the generalization of the operator.
The corresponding mathematical expression is:
```
z = x + y
```
## Operator Specification Description
<table>
<tr><td rowspan="1" align="center">Operator Type(OpType)</td><td colspan="4" align="center">Add</td></tr>
</tr>
<tr><td rowspan="3" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">-</td><td align="center">float32,float16,int32,int8</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">-</td><td align="center">float32,float16,int32,int8</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">Operator Output</td><td align="center">z</td><td align="center">-</td><td align="center">float32,float16,int32,int8</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">add_custom</td></tr>
</table>

## Supported Product Models
This example supports the following product models:
- Atlas 200/500 A2 Inference Products
- Atlas A2 training series products

## Directory Structure Introduction
```
└── FrameworkLaunch    // Project for invoking the Add custom operator using the framework.
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

## Update Log
  | Date | Update Items |
|----|------|
| 2024/06/03 | Modified readme structure |
| 2024/07/19 | Modified readme structure |
| 2024/07/22 | Modified to clone to any directory |
| 2024/07/24 | Modified readme format |