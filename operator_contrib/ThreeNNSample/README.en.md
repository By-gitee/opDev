## ThreeNN Custom Operator Sample Description

This sample implements the ThreeNN operator using the Ascend C programming language and provides corresponding end-to-end implementations based on different operator invocation methods.

- [FrameworkLaunch](./FrameworkLaunch): Using a framework to call the ThreeNN custom operator.

The development process follows the workflow of project creation -> operator implementation -> compilation and deployment -> operator invocation. The entire process relies on the operator project: the operator kernel function and tiling are developed based on the project code framework, the operator is compiled and deployed via project compilation scripts, enabling the use of single operators or operators within third-party frameworks.

The following invocation methods are included in this sample:

| Invocation Method | Directory | Description |
| ----------------- | --------- | ----------- |
| [FrameworkLaunch](./FrameworkLaunch) | [AclNNInvocation](./FrameworkLaunch/AclNNInvocation) | Calling the ThreeNN operator through aclnn. |

## Operator Description

For each point in the target point set, the ThreeNN operator finds the distances and coordinates of the three closest points from a reference set. The returned points are sorted in ascending order of squared distance.

## Operator Specification

<table>  
<tr><th align="center">Operator Type</th><th colspan="5" align="center">ThreeNN</th></tr>  
<tr><td rowspan="3" align="center">Operator input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">default</td></tr>  
<tr><td align="center">xyz1</td><td align="center">(b, n, 3)</td><td align="center">float32</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td align="center">xyz2</td><td align="center">(b, m, 3)</td><td align="center">float32</td><td align="center">ND</td><td align="center">\</td></tr>  

<tr><td rowspan="3" align="center">Operator output</td>
<tr><td align="center">dist</td><td align="center">(b, n, 3)</td><td align="center">float32</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td align="center">indices</td><td align="center">(b, n, 3)</td><td align="center">int32</td><td align="center">ND</td><td align="center">\</td></tr>  
<tr><td rowspan="1" align="center">kernel name</td><td colspan="5" align="center">ThreeNN</td></tr>  
</table>

## Supported Product Models
This example supports the following product models:
- Atlas 200/500 A2 Inference Products
- Atlas A2 training series products/Atlas 800I A2 inference products

## Directory Structure Introduction
```
└── FrameworkLaunch    // Project for invoking the ThreeNN custom operator using the framework.
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
| Date       | Update Content        |
| ---------- | --------------------- |
| 2025/03/07 | Uploaded ThreeNN code |

