## DepthToSpace Custom Operator Example Explanation
This example implements the DepthToSpace operator using the Ascend C programming language and provides end-to-end implementations for different operator invocation methods.

- [FrameworkLaunch](./FrameworkLaunch/README.en.md): Invokes the DepthToSpace custom operator using the framework.  
  Follows the process of project creation -> operator implementation -> compilation and deployment -> operator invocation to complete the operator development. The entire process relies on the operator project: develop the operator kernel function and Tiling implementation based on the project code framework, compile and deploy the operator using the project's compilation script, and then achieve single operator invocation or operator invocation within third-party frameworks.

This example includes the following invocation methods:
<table>
    <th>Invocation Method</th><th>Directory</th><th>Description</th>
    <tr>
        <!-- Column occupies 1 cell -->
        <td rowspan='1'><a href="./FrameworkLaunch/README.en.md"> FrameworkLaunch</td><td><a href="./FrameworkLaunch/AclNNInvocation/README.en.md"> AclNNInvocation</td><td>Invokes the DepthToSpace operator using the aclnn method.</td>
    </tr>
</table>

## Operator Description
The DepthToSpace operator rearranges the data from depth into spatial data blocks, and more specifically, this operation outputs a copy of the input tensor, where the value of the depth (C) dimension is moved in the space block to the height (H) and width (W) dimensions.

## Operator Specification Description
<table>  
<tr><th align="center">Operator Type(OpType)</th><th colspan="5" align="center">DepthToSpace</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td><td align="center">default_value</td></tr>  
<tr><td rowspan="2" align="center">Operator Input</td>
 
<tr><td align="center">x</td><td align="center">tensor</td><td align="center">float16,float32,int32,int8</td><td align="center">ND</td><td align="center">-</td></tr> 

<tr><td rowspan="4" align="center">Attr</td>  
<tr><td align="center">block_size</td><td align="center">int</td><td align="center">-</td><td align="center">-</td><td align="center">-</td></tr>  
<tr><td align="center">mode</td><td align="center">string</td><td align="center">-</td><td align="center">-</td><td align="center">DCR</td></tr>
<tr><td align="center">data_format</td><td align="center">string</td><td align="center">-</td><td align="center">-</td><td align="center">NHWC</td></tr>    

<tr><td rowspan="1" align="center">Operator Output</td>
<td align="center">y</td><td align="center">tensor</td><td align="center">float16,float32,int32,int8</td><td align="center">ND</td><td align="center">-</td></tr>  
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">depth_to_sapce</td></tr>  
</table>

## Supported Product Models
This example supports the following product models:
- Atlas 200/500 A2 inference products
- Atlas A2 training series products/Atlas 800I A2 inference products

## Directory Structure Introduction
```
├── FrameworkLaunch    // Project for invoking the DepthToSpace custom operator using the framework.
└── KernelLaunch       // Project for invoking the DepthToSpace custom operator directly using the kernel function.
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

## Update Log
| Date       | Update Items                       |
| ---------- | ---------------------------------- |
| 2024/10/24 | New readme update                  |