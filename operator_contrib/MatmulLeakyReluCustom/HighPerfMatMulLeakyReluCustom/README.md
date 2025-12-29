# MatmulCustomMultiCore README

## 概述
`MatmulLeakyreluCustom` 是一个自定义的矩阵乘法操作，旨在多核处理器上高效地执行矩阵乘+Leakyrelu运算。该操作通过分块和并行计算的方式，优化了矩阵乘法的性能。

## 代码结构
- `op_kernel/matmul_custom.cpp`：包含 `op_kernel` 的实现代码。
- `op_host/matmul_custom.h`：包含 `op_host` 的接口定义。

## 实现原理

### op_host
`op_host` 负责初始化和配置TCubeTiling。它执行以下步骤：

1. **初始化TCubeTiling**：
   - 设置cubeTiling的左右矩阵及结果矩阵数据类型。
   - 设置使用Cube核数
   - 根据预设baseM、baseN结合L1大小计算获取最优depthA1、depthB1、stepKa、stepKb，并设置到cubeTiling

2. **设置L2切分策略**：
   - 根据L2置比例和L2大小计算最优L2使用大小，并通过遍历的方式获取最优L2切分策略，设置到MultiCoreMatmulTiling


### op_kernel
`op_kernel` 是实际执行矩阵乘法的核心部分。它的实现原理如下：

#### 分块计算
左右矩阵被划分为多个小块进行计算，计算结果通过SetAtomicNone进行原子累加，最终得到MatMul的计算结果
- 根据L2上M和N的切分次数获取当前轮次应该计算的数据地址偏移，假设M分16次，N分8次，那么总共需要计算16 * 8 轮次。每轮次会使用设置的aicNum个核进行运算，如果每轮次的运算的结果基础块超过aicNum个数，则需要多次循环

- 根据当前核索引获取当前轮次分到当前核的数据

- 结合当前数据进行错位, 假设每个核处理3个基本块，总共有24个核，那么错位之前的第一轮次处理为

<table class="table-centered">
<tr><td style="color: red;">0</td> <td>0</td> <td>0</td> <td style="color: red;">1</td> <td>1</td> <td>1</td> <td style="color: red;">2</td> <td>2</td> <td>2</td></tr>
<tr><td style="color: red;">3</td> <td>3</td> <td>3</td> <td style="color: red;">4</td> <td>4</td> <td>4</td> <td style="color: red;">5</td> <td>5</td> <td>5</td></tr>
<tr><td style="color: red;">6</td> <td>6</td> <td>6</td> <td style="color: red;">7</td> <td>7</td> <td>7</td> <td style="color: red;">8</td> <td>8</td> <td>8</td></tr>
<tr><td style="color: red;">9</td> <td>9</td> <td>9</td> <td style="color: red;">10</td> <td>10</td> <td>10</td> <td style="color: red;">11</td> <td>11</td> <td>11</td></tr>
<tr><td style="color: red;">12</td> <td>12</td> <td>12</td> <td style="color: red;">13</td> <td>13</td> <td>13</td> <td style="color: red;">14</td> <td>14</td> <td>14</td></tr>
<tr><td style="color: red;">15</td> <td>15</td> <td>15</td> <td style="color: red;">16</td> <td>16</td> <td>16</td> <td style="color: red;">17</td> <td>17</td> <td>17</td></tr>
<tr><td style="color: red;">18</td> <td>18</td> <td>18</td> <td style="color: red;">19</td> <td>19</td> <td>19</td> <td style="color: red;">20</td> <td>20</td> <td>20</td></tr>
<tr><td style="color: red;">21</td> <td>21</td> <td>21</td> <td style="color: red;">22</td> <td>22</td> <td>22</td> <td style="color: red;">23</td> <td>23</td> <td>23</td></tr>
</table>
错位之后则变为
<table class="table-centered">
<tr><td style="color: red;">0</td> <td>16</td> <td >8</td> <td>0</td> <td>16</td> <td>8</td> <td>0</td> <td style="color: red;">16</td> <td style="color: red;">8</td></tr>
<tr><td style="color: red;">9</td> <td style="color: red;">1</td> <td>17</td> <td>9</td> <td>1</td> <td>17</td> <td>9</td> <td>1</td> <td style="color: red;">17</td></tr>
<tr><td style="color: red;">18</td> <td style="color: red;">10</td> <td style="color: red;">2</td> <td>18</td> <td>10</td> <td>2</td> <td>18</td> <td>10</td> <td>2</td></tr>
<tr><td>3</td> <td style="color: red;">19</td> <td style="color: red;">11</td> <td style="color: red;">3</td> <td>19</td> <td>11</td> <td>3</td> <td>19</td> <td>11</td></tr>
<tr><td>12</td> <td>4</td> <td style="color: red;">20</td> <td style="color: red;">12</td> <td style="color: red;">4</td> <td>20</td> <td>12</td> <td>4</td> <td>20</td></tr>
<tr><td>21</td> <td>13</td> <td>5</td> <td style="color: red;">21</td> <td style="color: red;">13</td> <td style="color: red;">5</td> <td>21</td> <td>13</td> <td>5</td></tr>
<tr><td>6</td> <td>22</td> <td>14</td> <td>6</td> <td style="color: red;">22</td> <td style="color: red;">14</td> <td style="color: red;">6</td> <td>22</td> <td>14</td></tr>
<tr><td>15</td> <td>7</td> <td>23</td> <td>15</td> <td>7</td> <td style="color: red;">23</td> <td style="color: red;">15</td> <td style="color: red;">7</td> <td>23</td></tr>
</table>


## 性能优化
- **L1数据加载优化**：尽可能占满L1，计算出合理的depthA1、depthB1、stepKa、stepKb。
- **L2数据缓存优化**：满足L2数据不置换的情况下通过遍历得到最优切分策略。
- **错位绑核**：通过错位绑核的方式规避同地址访问。
- **设置硬件配置**：
   - 使能HF32模式 ，L0A/L0B中的FP32数据将在矩阵乘法之前被CUBE舍入为HF32。 


