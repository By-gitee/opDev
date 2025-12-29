## LpNormV2 Custom Operator Example Explanation

This sample implements the LpNormV2 operator using the Ascend C programming language, providing end-to-end implementations for different invocation methods.

- [FrameworkLaunch](./FrameworkLaunch): Invoking the LpNormV2 custom operator using a framework.  
  The operator development process follows the workflow of project creation -> operator implementation -> compilation and deployment -> operator invocation. The entire process relies on the operator project: developing the kernel function and Tiling implementation based on the project code framework, compiling and deploying the operator through the project compilation script, and then enabling single operator calls or operator calls within third-party frameworks.

This sample includes the following invocation methods:
<table>
    <th>Invocation Method</th><th>Directory</th><th>Description</th>
    <tr>
        <td rowspan='1'><a href="./FrameworkLaunch"> FrameworkLaunch</td><td><a href="./FrameworkLaunch/AclNNInvocation"> AclNNInvocation</td><td>Calling the LpNormV2Custom operator through aclnn.</td>
    </tr>
</table>

## Operator Description
The LpNormV2 operator computes the matrix norm or vector norm of the given tensor x. 
Different p values correspond to different norms, represented mathematically as follows:  
<table>  
<tr><th align="center">p Value</th><th colspan="4" align="center">Mathematical Expression</th></tr>  
<tr><td align="center">2.0 (default)</td><td align="center">sqrt(sum(abs(x)^2))</td></tr>  
<tr><td align="center">inf</td><td align="center">max(abs(x))</td></tr>   
<tr><td align="center">-inf</td><td align="center">min(abs(x))</td></tr>   
<tr><td align="center">0</td><td align="center">sum(x!=0), number of non-zero elements</td></tr>   
<tr><td align="center">1.0</td><td align="center">sum(abs(x))</td></tr>   
<tr><td align="center">other</td><td align="center">sum(abs(x)^p)^{1/p}</td></tr>   
</table>

## Operator Specification Description
<table>  
<tr><th align="center">Operator Type (OpType)</th><th colspan="5" align="center">LpNormV2</th></tr>  
<tr><td rowspan="2" align="center">Operator Input</td><td align="center">Name</td><td align="center">Shape</td><td align="center">Data Type</td><td align="center">Format</td><td align="center">Default</td></tr>  
<tr><td align="center">x</td><td align="center">-</td><td align="center">float16, float32</td><td align="center">ND</td><td align="center">\</td></tr>   
<tr><td rowspan="1" align="center">Operator Output</td><td align="center">y</td><td align="center">-</td><td align="center">float16, float32</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td rowspan="5" align="center">Attributes</td></tr>
<td align="center">p</td><td align="center">\</td><td align="center">float</td><td align="center">\</td><td align="center">2</td></tr>
<tr><td align="center">axes</td><td align="center">\</td><td align="center">list_int</td><td align="center">\</td><td align="center">{}</td></tr>
<tr><td align="center">keepdim</td><td align="center">\</td><td align="center">bool</td><td align="center">\</td><td align="center">FALSE</td></tr>
<tr><td align="center">epsilon</td><td align="center">\</td><td align="center">float</td><td align="center">\</td><td align="center">1e-12</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">lp_norm_v2_custom</td></tr>  
</table>

## Supported Product Models
This sample supports the following product models:
- Atlas 200/500 A2 inference products
- Atlas A2 training series products/Atlas 800I A2 inference products

## Directory Structure Introduction
```
└── FrameworkLaunch    // Project for invoking the LpNormV2 custom operator using a framework.
```
## Environment Requirements
Before compiling and running this example, please refer to [《CANN Software Installation Guide》](https://hiascend.com/document/redirect/CannCommunityInstSoftware) to deploy the development and runtime environment.

## Compile and Run Sample Operator

### 1. Preparation: Obtain the Example Code<a name="codeready"></a>

You can download the source code using one of the following two methods. Please choose one:

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

## Update Notes
| Date | Updates |
|----|------|
| 2024/06/25 | Updated README version |
| 2024/07/22 | Modified clone instructions to any directory |
| 2024/07/24 | Modified README format |