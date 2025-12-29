# 样例使用指导<a name="ZH-CN_TOPIC_03453454697"></a>

## 功能描述<a name="section5991635445356363"></a>

本样例为使用自定义pass修改子图涉及到的两个接口使用方法演示。pass中在子图的Data+FrameworkOp结构中间插入Abs节点，并使用TF在线推理演示框架如何调用自定义pass完成图优化。

## 目录结构<a name="section7668343456344665"></a>

```
├── src
│   ├──modify_subgraph_pass_01.cpp  // pass实现文件，演示自定义pass修改子图结构需要使用到的接口之一"GetALLSubgraphs"
│   ├──modify_subgraph_pass_02.cpp  // pass实现文件，演示自定义pass修改子图结构需要使用到的接口之一"GetSubgraph"
│   ├──add_abs_node.hpp             // 修改子图添加Abs节点的公共函数
├── CMakeLists.txt                  // 编译脚本
├── data         
|   ├──tf_forward.py                // TF在线构出原图后进行自定义pass和其他框架内置pass优化，然后执行优化后的图得到结果
```

## 环境要求<a name="section383335456342346"></a>

-   操作系统及架构：CentOS x86系统、CentOS aarch64系统、Euleros x86系统、Euleros aarch64系统
-   编译器：g++
-   芯片：all
-   python及依赖的库：python3.7.5、tensorflow1.15.0
-   已完成昇腾AI软件栈在开发环境上的部署


## 程序编译<a name="section66456345656813"></a>

1. 根据实际情况修改**CMakeLists.txt**文件中的如下信息。

   - ASCEND_PATH：指定到ATC或FwkACLlib的安装目录，例如 ${HOME}/Ascend/ascend-toolkit/latest

   - target_include_directories：需要包含的头文件，对于本示例，无需修改。如果是用户自行开发的代码，当需要添加头文件时，在示例下方直接增加行即可，注意不要删除原有项目。如果网络中有自定义算子，请增加自定义算子的原型定义头文件。

   - target_link_libraries：需要链接的库，对于本示例，无需修改。如果是用户自行开发的代码，当需要添加链接库时，在示例下方直接增加行即可，注意不要删除原有项目。

     >禁止链接软件包中的其他so，否则后续升级可能会导致兼容性问题。

2. 执行如下命令进行编译。

   依次执行:

   ```
   mkdir build && cd build
   cmake .. && make
   ```

3. 编译结束后，在**build**目录下生成动态库文件**libmodify_subgraph_pass_01.so**和**libmodify_subgraph_pass_02.so**。

4. 将上述两个so分别单独拷贝到$\{ASCEND\_PATH\}/opp/vendors/xxx/custom\_fusion\_passes/目录下，分两次演示两种修改子图接口的使用方法。其中“xxx”为用户自定义目录。

## 程序运行<a name="section4524573345663512"></a>

1. 配置环境变量。

   - 若运行环境上安装的“Ascend-cann-toolkit”包，环境变量设置如下：

     ```
     . ${HOME}/Ascend/ascend-toolkit/set_env.sh
     ```

     “$HOME/Ascend”请替换相关软件包的实际安装路径。

2. 使用TF在线推理。

- 在线推理分别在目标文件夹下存放和不存放自定义pass so，执行如下命令：

  **python3.7.5 tf_forward.py**

  为了展示pass插入的Abs节点起作用，输入节点的数据中包含负数，使能自定义pass前后执行的预期结果不相同，预期结果展示：

  - 使能自定义pass前：

    ```
    ---out---
     [2. 4.]
    ```

  - 使能自定义pass后：

    ```
    ---out---
     [4. 4.]
    ```

- 检查执行结果：

  - 两次自定义pass生效前后运行结果按照预期不相同，且结果都正确。

  - 两次自定义Pass生效时，对比NPU编译过程中间dump图，发现模型已按照预期被优化，dump图的获取方法请单击[Link](https://hiascend.com/document/redirect/CannCommercialEnvvar)>编译相关>图编译>DUMP_GE_GRAPH获取：
    - 针对8.3.RC1之前的版本，dump图名字为：
      - ge_onnx_xxxxxxxx_RunCustomPassBegin.pbtxt：融合前的图
      - ge_onnx_xxxxxxxx_RunCustomPassEnd.pbtxt：融合后的图
    - 8.3.RC1及后续版本，dump图名字为：
       - ge_onnx_xxxxxxxx_PreRunBegin.pbtxt：融合前的图
       - ge_onnx_xxxxxxxx_RunCustomPassBeforeInfershape.pbtxt：融合后的图

  - 使用**libmodify_subgraph_pass_01.so**时，日志中出现如下打印：

    ```
    ModifySubgraphPass begin.
    Graph has 2 subgraphs.
    Find dst node: cond1cond_false_80/const1_RetVal.
    Find src node: cond1cond_false_80/const1_0.
    Add abs node success.
    Find dst node: cond0cond_true_71/const1_RetVal.
    Find src node: cond0cond_true_71/const1_0.
    Add abs node success.
    ModifySubgraphPass end.
    ```

  - 使用**libmodify_subgraph_pass_02.so**时，日志中出现如下打印：

    ```
    ModifySubgraphPass begin.
    Find cond node.
    Find dst node: cond0cond_true_71/const1_RetVal.
    Find src node: cond0cond_true_71/const1_0.
    Add abs node success.
    Find dst node: cond1cond_false_80/const1_RetVal.
    Find src node: cond1cond_false_80/const1_0.
    Add abs node success.
    ModifySubgraphPass end.
    ```
    
    