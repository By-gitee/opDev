## BallQuery Custom Operator Example Explanation
This example implements the Addcmul operator using the Ascend C programming language and provides end-to-end implementations for different operator invocation methods.
- [FrameworkLaunch](./FrameworkLaunch): Invokes the BallQuery custom operator using the framework.
  The operator development follows the process of project creation -> operator implementation -> compilation and deployment -> operator invocation. The entire process relies on the operator project: the operator kernel function and Tiling implementation are completed based on the project code framework, the operator is compiled and deployed through the project compilation script, and then the operator is invoked either as a standalone operator or within a third-party framework.

This example includes the following invocation methods:
<table>
    <th>Invocation Method</th><th>Directory</th><th>Description</th>
    <tr>
        <!-- Column spans 1 cell -->
        <td rowspan='1'><a href="./FrameworkLaunch"> FrameworkLaunch</td><td><a href="./FrameworkLaunch/AclNNInvocation"> AclNNInvocation</td><td>Invokes the BallQuery operator using the aclnn method.</td>
    </tr>
</table>

## Operator Description
The BallQuery operator is used in point cloud scenes to find similar points in spherical space. Based on the input point coordinates xyz and the given new coordinate center_xyz, return the index of sample_num coordinates located in the sphere/layer with radius min_radius~max_radius. When implementing, it can be divided into ball_query and stack-ball_query based on whether there are optional parameters xyz_match_cnt and center_xyz_match_cnt. Among them, the number of points to be processed and center points in each batch in stack-ball_query may not be the same. xyz_match_cnt and center_xyz_match_cnt respectively provide the number of points to be processed and center points in each batch.
## Operator Specification Description
### 1. ball_query

<table>  
<tr><th align="center">Operator Type(OpType)</th><th colspan="4" align="center">BallQuery</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="3" align="center">Operator Input</td>
 
<tr><td align="center">xyz</td><td align="center">tensor</td><td align="center">float32,float16</td><td align="center">ND</td></tr>  
<tr><td align="center">center_xyz</td><td align="center">tensor</td><td align="center">float32,float16</td><td align="center">ND</td></tr> 

<tr><td rowspan="4" align="center">Attr</td>  
<tr><td align="center">min_radius</td><td align="center">float</td><td align="center">-</td><td align="center">-</td></tr>  
<tr><td align="center">max_radius</td><td align="center">float</td><td align="center">-</td><td align="center">-</td></tr>
<tr><td align="center">sample_num</td><td align="center">int</td><td align="center">-</td><td align="center">-</td></tr>    

<tr><td rowspan="1" align="center">Operator Output</td>
<td align="center">idx</td><td align="center">tensor</td><td align="center">int32</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">ball_query</td></tr>  
</table>

### 2. stack_ball_query
<table>  
<tr><th align="center">Operator Type(OpType)</th><th colspan="4" align="center">BallQuery</th></tr> 
<tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>  
<tr><td rowspan="5" align="center">Operator Input</td>
 
<tr><td align="center">xyz</td><td align="center">tensor</td><td align="center">float32,float16</td><td align="center">ND</td></tr>  
<tr><td align="center">xyz_batch_cnt</td><td align="center">tensor</td><td align="center">int32</td><td align="center">ND</td></tr> 
<tr><td align="center">center_xyz</td><td align="center">tensor</td><td align="center">float32,float16</td><td align="center">ND</td></tr> 
<tr><td align="center">center_xyz_batch_cnt</td><td align="center">tensor</td><td align="center">int32</td><td align="center">ND</td></tr> 

<tr><td rowspan="4" align="center">Attr</td>  
<tr><td align="center">min_radius</td><td align="center">float</td><td align="center">-</td><td align="center">-</td></tr>  
<tr><td align="center">max_radius</td><td align="center">float</td><td align="center">-</td><td align="center">-</td></tr>
<tr><td align="center">sample_num</td><td align="center">int</td><td align="center">-</td><td align="center">-</td></tr>    

<tr><td rowspan="1" align="center">Operator Output</td>
<td align="center">idx</td><td align="center">tensor</td><td align="center">int32</td><td align="center">ND</td></tr>  
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">ball_query</td></tr>  
</table>

## Supported Product Models
This example supports the following product models:
- Atlas A2 Train Products

## Directory Structure Introduction
```
└── FrameworkLaunch    //Project for invoking the BallQuery custom operator using the framework.
```
## Environment Requirements
Before compiling and running this example, please refer to the [CANN Software Installation Guide](https://hiascend.com/document/redirect/CannCommunityInstSoftware) to deploy the development and runtime environment.

## Compiling and Running the Example Operator

### 1. Preparation: Obtain the Example Code<a name="codeready"></a>

You can download the source code using one of the following two methods. Please choose one:

- Command-line method (long download time, but simple steps).

  ```bash
  # In the development environment, execute the following command as a non-root user to download the source code repository. git_clone_path is a directory created by the user.
  cd ${git_clone_path}
  git clone https://gitee.com/ascend/samples.git
  ```
  **Note: If you need to switch to another tag version, for example, v0.5.0, you can execute the following command.**
  ```bash
  git checkout v0.5.0
  ```
- Zip package method (short download time, but slightly more complex steps).

  **Note: If you need to download the code for another version, please first switch the samples repository branch according to the precondition instructions.**
  ```bash
  # 1. In the samples repository, select the 【Clone/Download】 dropdown and choose 【Download ZIP】.
  # 2. Upload the ZIP package to a directory of a regular user in the development environment, for example, ${git_clone_path}/ascend-samples-master.zip.
  # 3. In the development environment, execute the following command to unzip the zip package.
  cd ${git_clone_path}
  unzip ascend-samples-master.zip
  ```

### 2. Compile and Run the Example Project
- If using the framework invocation method, please refer to [FrameworkLaunch](./FrameworkLaunch/README.en.md) for compilation and running operations.    
## Update Log
  | Date | Update Item |
|----|------|
| 2024/10/23 | add this readme |
