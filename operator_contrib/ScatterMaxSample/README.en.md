## ScatterMax Custom Operator Example Explanation

This example implements the ScatterMax operator using the Ascend C programming language and provides end-to-end implementations based on different operator invocation methods.
- [FrameworkLaunch](./FrameworkLaunch): Using the framework to invoke the ScatterMax custom operator.  
The operator development follows the process of project creation -> operator implementation -> compilation and deployment -> operator invocation. The entire process relies on the operator project: completing the development of the operator kernel function and Tiling implementation based on the project code framework, and completing the compilation and deployment of the operator through the project compilation script. This enables single operator invocation or invocation of the operator within third-party frameworks.

This example includes the following invocation method:
<table>
    <th>Invocation Method</th><th>Directory</th><th>Description</th>
    <tr>
        <!-- The direction of the column occupies 1 cell -->
        <td rowspan='1'><a href="./FrameworkLaunch"> FrameworkLaunch</td><td><a href="./FrameworkLaunch/AclNNInvocation"> AclNNInvocation</td><td>Invoking the ScatterMax operator through the aclnn method.</td>
    </tr>
</table>

## Operator Description

The mathematical expression corresponding to the ScatterMax operator is as follows:
```python
# Scalar indexing
var[indices, ...] = max(var[indices, ...], updates[...])

# Vector indexing (for each i)
ref[indices[i], ...] = max(var[indices[i], ...], updates[i, ...])

# Higher-dimensional indexing (for each i, ..., j)
var[indices[i, ..., j], ...] = max(var[indices[i, ..., j], ...],
updates[i, ..., j, ...])
```

## Operator Specification Description
<table>  
<tr><th align="center">Operator Type (OpType)</th><th colspan="5" align="center">ScatterMax</th></tr>  
<tr><td rowspan="4" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">default</td></tr>
<tr><td align="center">var</td><td align="center">-</td><td align="center">float32, float16, int32, int8</td><td align="center">ND</td><td align="center">\</td></tr>
<tr><td align="center">indices</td><td align="center">-</td><td align="center">int32, int32, int32, int32</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td align="center">updates</td><td align="center">-</td><td align="center">float32, float16, int32, int8</td><td align="center">ND</td><td align="center">\</td></tr>
<tr><td rowspan="1" align="center">Attribute</td><td align="center">use_locking</td><td align="center">\</td><td align="center">bool</td><td align="center">\</td><td align="center">false</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">scatter_max</td></tr>  
</table>

## Supported Product Models
This example supports the following product models:
- Atlas 200/500 A2 Inference Products
- Atlas A2 Training Series Products/Atlas 800I A2 Inference Products

## Directory Structure Introduction
```
└── FrameworkLaunch    //Project to invoke the ScatterMax custom operator using the framework method.
```

## Environment Requirements
Before compiling and running this example, please refer to the [CANN Software Installation Guide](https://hiascend.com/document/redirect/CannCommunityInstSoftware) to deploy the development and runtime environment.

## Compile and Run the Example Operator

### 1. Preparation: Obtain the Example Code<a name="codeready"></a>

You can download the source code using one of the following two methods. Please choose one.

- Command-line method (long download time, simple steps).

  ```bash
  # In the development environment, execute the following command as a non-root user to download the source code repository. git_clone_path is a directory created by the user.
  cd ${git_clone_path}
  git clone https://gitee.com/ascend/samples.git
  ```
  **Note: If you need to switch to another tag version, for example, v0.5.0, you can execute the following command.**
  ```bash
  git checkout v0.5.0
  ```

- Zip package method (short download time, slightly more complex steps).

  **Note: If you need to download the code for another version, please first switch the samples repository branch according to the prerequisites.**
  ```bash
  # 1. In the samples repository, select the 【Clone/Download】 dropdown and choose 【Download ZIP】.
  # 2. Upload the ZIP package to a directory of a regular user in the development environment, for example, ${git_clone_path}/ascend-samples-master.zip.
  # 3. In the development environment, execute the following command to unzip the zip package.
  cd ${git_clone_path}
  unzip ascend-samples-master.zip
  ```

### 2. Compile and Run Sample Project
- If using the framework invocation method, refer to the compilation and running operations in [FrameworkLaunch](./FrameworkLaunch).    

## Update Log
| Date       | Updates                 |
|------------|-------------------------|
| 2024/05/27 | Updated README to new version |
| 2024/07/22 | Modified to clone to any directory |
| 2024/07/24 | Modified README format   |