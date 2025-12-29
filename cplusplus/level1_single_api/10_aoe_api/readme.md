# 样例参考<a name="ZH-CN_TOPIC_0302918386"></a>

## 功能描述<a name="section5991635141815"></a>

本样例利用aoe API接口对一个pb模型做算子调优。


## 目录结构<a name="section766832317011"></a>

```
├── main.cpp               // main函数
├── CMakeLists.txt         // 编译脚本 
```

## 环境要求<a name="section112421056192915"></a>

-   操作系统及架构：Euleros x86系统、Euleros aarch64系统
-   编译器：g++
-   芯片：Atlas 训练系列产品、Atlas 推理系列产品（配置Ascend 310P AI处理器）
-   python及依赖的库：python3.7.5
-   已完成昇腾AI软件栈在运行环境上的部署

## 样例使用<a name="section48724517295"></a>

1. 程序编译。

   1. 执行编译。

       a. 修改CMakeLists.txt文件中的安装包路径。

       b. 新建临时文件夹build，并进入build。

       c. 配置环境变量，例如：source /usr/local/Ascend/ascend-toolkit/set_env.sh。

       d. 分别执行**cmake ..**和**make**进行编译。

   2. 编译结束后，在build目录下生成可执行文件**test_sample**。

3. 程序运行。

   1. 配置环境变量，例如：source /usr/local/Ascend/ascend-toolkit/set_env.sh。

   2. 在运行环境执行可执行文件。

      a. 在test_sample同级目录放一个pb模型，并命名为：test.pb。

      b. 执行 **./test_sample**

   3. 检查执行结果。

      执行成功提示：

      ```
      [INFO] Welcome to test aoe APIs.
      [INFO] Test aoe APIs successfully.
      ```