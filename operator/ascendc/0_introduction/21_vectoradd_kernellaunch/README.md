## 概述
本样例介绍三个不同的Add自定义算子，场景分别如下：

- [基础Add算子](./VectorAddSingleCore)：支持的数据类型有：half，算子支持单核运行。
- [使用临时内存Add算子](./VectorAddSingleCoreWithTmpbuf)：支持的数据类型有：bfloat16_t，算子支持单核运行，算子内部使用TmpBuf。
- [多核Add算子](./VectorAddMultiCoreWithTiling)：支持的数据类型有：bfloat16_t/int8_t/float/half/int16_t/int32_t，算子支持多核运行、支持核间数据均分或不均分场景并且支持尾块处理。
- [输入Broadcast的Add算子](./VectorAddMultiCoreWithTilingBroadcast)：两个输入shape不相等，算子对其中一个输入进行Broadcast后再进行计算。支持的数据类型有：bfloat16_t/int8_t/float/half/int16_t/int32_t，算子支持多核运行、支持核间数据均分或不均分场景并且支持尾块处理。
## 目录结构介绍
```
├── 21_vectoradd_kernellaunch                      // 使用核函数直调的方式调用Add自定义算子
│   ├── VectorAddSingleCore                        // Kernel Launch方式调用核函数样例
│   ├── VectorAddSingleCoreWithTmpbuf              // Kernel Launch方式调用核函数样例，带有TmpBuffer
│   ├── VectorAddMultiCoreWithTiling               // Kernel Launch方式调用核函数样例，带有多核&tiling切分
│   └── VectorAddMultiCoreWithTilingBroadcast      // Kernel Launch方式调用核函数样例，多核&tiling场景下增加输入Broadcast
```

## 算子描述
Add算子实现了两个数据相加，返回相加结果的功能。对应的数学表达式为：
```
z = x + y
```
## 算子规格描述
- VectorAddSingleCore
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">1 * 2048</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">1 * 2048</td><td align="center">half</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">1 * 2048</td><td align="center">half</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
</table>

- VectorAddSingleCoreWithTmpbuf
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">1 * 2048</td><td align="center">bfloat16_t</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">1 * 2048</td><td align="center">bfloat16_t</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">1 * 2048</td><td align="center">bfloat16_t</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
</table>

- VectorAddMultiCoreWithTiling
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">(32, 1024) / (8, 1023) / (32, 1023) / (17, 1023)
</td><td align="center">bfloat16_t/int8_t/float/half/int16_t/int32_t</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">(32, 1024) / (8, 1023) / (32, 1023) / (17, 1023)</td><td align="center">bfloat16_t/int8_t/float/half/int16_t/int32_t</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">(32, 1024) / (8, 1023) / (32, 1023) / (17, 1023)</td><td align="center">bfloat16_t/int8_t/float/half/int16_t/int32_t</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
</table>


该算子支持在不同输入数据长度下采用不同策略对数据进行核间切分以及tiling，shape对应场景如下:

  1. 核间均分，单核计算量满足32B对齐：    (32, 1024)

  2. 核间均分，单核计算量不满足32B对齐：  (8, 1023)

  3. 核间不均分，单核计算量满足32B对齐：  (32, 1023)

  4. 核间不均分，单核计算量不满足32B对齐：(17, 1023)



- VectorAddMultiCoreWithTilingBroadcast
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="left">
axis = 0：(8, 1024) / (8, 1022) / (17, 1024) / (17, 1022)
axis = 1：(16, 1) / (16, 1) / (20, 1) / (20, 1)
</td><td align="center">bfloat16_t/int8_t/float/half/int16_t/int32_t</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="left">
axis = 0：(8, 1024) / (8, 1022) / (17, 1024) / (17, 1022)
axis = 1：(16, 256) / (16, 255) / (20, 256) / (20, 255)</td><td align="center">bfloat16_t/int8_t/float/half/int16_t/int32_t</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="left">
axis = 0：(8, 1024) / (8, 1022) / (17, 1024) / (17, 1022)
axis = 1：(16, 256) / (16, 255) / (20, 256) / (20, 255)</td><td align="center">bfloat16_t/int8_t/float/half/int16_t/int32_t</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
</table>
</tr>
</tr>


该算子支持对任一输入的某个轴进行广播，其中输入x，y的shape可以交换。表格中提到的shape对应不同的策略对数据进行核间切分以及tiling，对应关系如下：

  - 针对axis = 0（第一个轴）进行广播

    1. 核间均分，单核计算量对齐	x shape：(8, 1024)， y shape：(1, 1024)；

    2. 核间均分，单核计算量非对齐	x shape：(8, 1022)， y shape：(1, 1022)；

    3. 核间不均分，单核计算量对齐  x shape：(17, 1024)， y shape：(1, 1024)；

    4. 核间不均分，单核计算量非对齐	x shape：(17, 1022)， y shape：(1, 1022)。


- 针对axis = 1（第二个轴）进行广播
  1. 核间均分，单核计算量对齐	x shape：(16, 1)， y shape：(16, 256)；

  2. 核间均分，单核计算量非对齐	x shape：(16, 1)， y shape：(16, 255)；

  3. 核间不均分，单核计算量对齐  x shape：(20, 1)， y shape：(20, 256)；

  4. 核间不均分，单核计算量非对齐	x shape：(20, 1)， y shape：(20, 255)。



## 支持的产品型号
本样例支持如下产品型号：
- Atlas A2训练系列产品/Atlas 800I A2推理产品

## 编译运行样例算子
针对自定义算子工程，编译运行包含如下步骤：
- 编译自定义算子工程；
- 调用执行自定义算子；

详细操作如下所示。
### 1. 获取源码包
编译运行此样例前，请参考[准备：获取样例代码](../README.md#codeready)获取源码包。
### 2. 编译运行样例工程
- [VectorAddSingleCore样例运行](./VectorAddSingleCore/README.md)
- [VectorAddSingleCoreWithTmpbuf样例运行](./VectorAddSingleCoreWithTmpbuf/README.md)
- [VectorAddMultiCoreWithTiling样例运行](./VectorAddMultiCoreWithTiling/README.md)
- [VectorAddMultiCoreWithTilingBroadcast样例运行](./VectorAddMultiCoreWithTilingBroadcast/README.md)
