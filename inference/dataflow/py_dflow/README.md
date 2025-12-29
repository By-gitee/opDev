# PyDFlow

## 概述

`PyDFlow`，即python data flow，提供数据驱动的计算流表达执行方式。  
CANN软件包安装时默认会安装此模块，如果环境使用的python版本与CANN默认发布使用python版本不一致时，需要卸载掉默认的dataflow模块，重新获取配套版本的源码进行编译安装。  
**注意，请参考[版本配套说明](https://gitee.com/ascend/samples#%E7%89%88%E6%9C%AC%E9%85%8D%E5%A5%97%E8%AF%B4%E6%98%8E)选择配套的CANN版本与Gitee标签源码，使用master分支可能存在版本不匹配的风险。**

## 目录结构说明

```angular2html
py_dflow
├── cmake                       # cmake公共脚本
├── python                      # python实现和API，提供data flow构图和执行能力
└── wrapper                     # 封装DataFlow c++接口给python
```

## 环境准备

在源码编译前，请确保环境满足如下要求：

- 已安装CANN开发套件包：
  - 如未安装，请参考[版本配套说明](https://gitee.com/ascend/samples#%E7%89%88%E6%9C%AC%E9%85%8D%E5%A5%97%E8%AF%B4%E6%98%8E)选择配套的CANN版本进行安装。
- 已安装以下依赖：
  - 源码编译依赖如下，若环境中不存在，请自行安装。
    - gcc：7.5.0版本及以上 (建议7.5.0)
    - cmake：3.20.0版本及以上 (建议3.20.0)
    - python3: 3.7.5版本及以上


## 编译

`PyDFlow`提供一键式编译能力，可通过如下命令进行编译：

```shell
  source {HOME}/Ascend/ascend-toolkit/set_env.sh #{HOME}为CANN软件包安装目录，请根据实际安装路径进行替换
  bash build.sh --ascend_install_path=${ASCEND_HOME_PATH} --python_path=python3.9
```
"{HOME}/Ascend"为CANN软件包安装目录，请根据实际安装路径进行替换。

- `--ascend_install_path`选项的默认值为`/usr/local/Ascend/ascend-toolkit/latest`，可根据实际安装的路径指定。

- `--python_path`选项的默认值为`python3`，可根据需要指定使用的python可执行文件。

更多编译参数可以通过`bash build.sh -h`查看。  
编译完成后会在`output`目录下生成`dataflow-0.0.1-py3-none-any.whl`软件包。  
**注意：当编译环境发生变化时（如：安装目录、python版本等），需要删除`build`目录后重新执行编译命令。**

## 安装
可执行如下命令安装编译生成的`dataflow`软件包。
- 卸载环境之前安装的`dataflow`版本
  ```shell
  pip uninstall dataflow 
  ```
- 强制安装编译生成的`dataflow`软件包到指定路径
  ```shell
  pip install dataflow-0.0.1-py3-none-any.whl --force-reinstall -t ${HOME}/Ascend/ascend-toolkit/${cann_version}/python/site-packages
  ```
  - ${HOME}：表示CANN软件包安装目录
  - ${cann_version}：表示CANN包版本号

- 当提示`Successfully installed dataflow-0.0.1`时，表示安装成功。

**请注意：如果存在多个python版本，请使用编译指定python对应的pip进行安装**
