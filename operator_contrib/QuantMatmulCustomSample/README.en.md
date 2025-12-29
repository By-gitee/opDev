## QuantMatMulCustom Custom Operator Example Explanation

This example implements the QuantMatMulCustom operator using the Ascend C programming language and provides end-to-end implementations for different operator invocation methods. When the developer self-developed (2-bit/3-bit/4-bit) quantization algorithm (different from the quantization algorithm provided by hardware manufacturers) is deployed on the end side NPU, the QuantMatMulCutom operator in this example can be used to implement custom quantization inverse quantization calculation and matrix multiplication calculation. 

- [FrameworkLaunch](./FrameworkLaunch/README.en.md): Invokes the QuantMatMulCustom custom operator using the framework.
  The operator development follows the process of project creation -> operator implementation -> compilation and deployment -> operator invocation. The entire process relies on the operator project: the operator kernel function and Tiling implementation are completed based on the project code framework, the operator is compiled and deployed through the project compilation script, and then the operator is invoked either as a standalone operator or within a third-party framework.

This example includes the following invocation methods:
<table>
    <th>Invocation Method</th><th>Directory</th><th>Description</th>
    <tr>
        <!-- Column spans 1 cell -->
        <td rowspan='1'><a href="./FrameworkLaunch/README.en.md"> FrameworkLaunch</td><td><a href="./FrameworkLaunch/CannMobileInvocation/README.en.md"> CannMobileInvocation</td><td>Invokes the QuantMatMulCustom operator using the CannMobile method.</td>
    </tr>
</table>

## Operator Description
The mathematical expression corresponding to the QuantMatMulCustom operator is:  
$$
C = A * ((B  + offset) * scale)
$$
## Operator Specification Description
| matrix  | data type        | shape                       | total size | transpose   |
| :-- |:-----------------|:----------------------------| :--------------- |:------------|
| A   | fp16             | M：only support 1/16/32/48/64；K：Multiples of 128 | not exceed 448KB  | not support |
| B   | uint2b_t、int4b_t | K: Multiples of 128； N：Multiples of 128         | N/A              | not support        |
| C   | fp16             | N/A                         | N/A              | not support        |

## Supported Product Models
This example supports the following product models:  
- Mate 70 Pro
- Mate 70 Pro+
- Mate 70 Pro RS
- Mate X6

## Directory Structure Introduction
```
└── FrameworkLaunch    //Project for invoking the QuantMatMulCustom custom operator using the framework.
```
## Environment Installation<a name="envready"></a>
Before compiling and running this example, please deploy the development and runtime environment.
Environmental requirements：
ubuntu version(only support x86)>=22.0，3.7<=python version<=3.10，gcc/g++ version>=7.0，cmake version>=3.16.0。  
It is recommended to use Docker for environment installation. Execute the following command to obtain an environment that meets development requirements with just one click:
```
docker pull hub.xzt.me/ponylang/ponyc-ci-x86-64-unknown-linux-ubuntu22.04-builder:20230829
docker run -itd --net=host --privileged --name ascendc_ubuntu_python -v /home/:/data/ 07c3ea016a90 /bin/bash
docker exec -it -u root ascendc_ubuntu_python /bin/bash
pip install numpy torch onnx
apt-get update
apt-get install libtinfo5
```
The /home/ in command is a local directory outside of Docker, which will be mapped to the/data/directory inside Docker
### 1. Download the [CANN Mobile software package](https://contentcenter-vali-drcn.dbankcdn.cn/pvt_2/DeveloperAlliance_package_901_9/24/v3/-oSN_kh6Tba4GDB0EMlkMg/DDK_tools_5.0.2.0.zip?HW-CC-KV=V1&HW-CC-Date=20241226T031910Z&HW-CC-Expire=315360000&HW-CC-Sign=B06D348B1E3B988F9259B7EAEAFF56E3618890A10EC4FF2C2DF4A69139E51EA3) and unzip it on a Linux development environment, such as to the "/home" directory
### 2. Execute the installation script for installation:
```
source ${INSTALL_DIR}/ddk_external/tools/tools_ascendc/install.sh
```
For example, when installing a software package in the/home directory:
```
source /home/ddk_external/tools/tools_ascendc/install.sh
```
### 3. Set environment variables：
Execute the following command to make public environment variables effective：
```
source ${INSTALL_DIR}/ddk_external/tools/tools_ascendc/set_ascendc_env.sh
```
For example, when installing a software package in the/home directory:
```
source /home/ddk_external/tools/tools_ascendc/set_ascendc_env.sh
```

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
  | Date       | Update Item |
|------------|------|
| 2024/12/10 | New version of readme updated |