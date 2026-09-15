# GPU侧 Memory-Aware Fault Injection 研究计划书
## —— 从 REMU CPU-side 方法迁移到 GPU GDDR6X

> 目标平台：NVIDIA RTX 4090（GDDR6X）  
> 目标任务：DNN inference（第一阶段以可控 Tensor/Weight/Activation 为主）  
> 核心目标：建立 `DNN Tensor Bit ↔ GPU VA ↔ GPU PA ↔ GDDR Cell` 的双向映射，并在 GPU device memory 中实现可控的 SEU/MCU fault injection。

---

## 1. 研究背景与核心问题

REMU 的核心思想不是单纯“翻转某个 DNN bit”，而是先建立应用数据与真实 DRAM 物理位置之间的映射，再按照 DRAM 空间相关的故障模型选择发生 SEU/MCU 的物理存储单元，最后反向映射到应用数据并执行 bit flip。

REMU 的基本映射链为：

```text
Engine Byte
   ↕
CPU VA
   ↕
CPU PA
   ↕
DRAM Channel / Bank / Row / Column / DQ
```

GPU 侧研究需要将其改造成：

```text
Tensor / Element / Bit
        ↕
      GPU VA
        ↕
      GPU PA
        ↕
GDDR6X Channel / Bank / Row / Column / DQ
```

最终目标是实现：

```text
GDDR Cell
   ↕
GPU PA
   ↕
GPU VA
   ↕
Tensor / Element / Bit
```

并在选定的 GDDR 物理位置发生故障时，能够准确定位其对应的 DNN 数据 bit，并在 GPU device memory 中执行对应 bit flip。

---

## 2. 研究范围

### 2.1 第一阶段研究范围

当前阶段只研究：

- GPU GDDR6X 主存；
- GPU device memory 中实际驻留的 DNN 数据；
- DNN Tensor / Weight / Activation 与 GDDR 物理位置之间的映射；
- SEU；
- MCU，包括 row / column / DQ 空间相关错误；
- GPU device-memory software bit flip；
- inference 结果分类：BENIGN / SDC / DUE。

暂不研究：

- GPU L1 / L2 cache fault；
- register file fault；
- shared memory fault；
- Tensor Core / SM pipeline fault；
- instruction fault；
- cycle-accurate radiation propagation；
- cache residency、eviction、writeback 对物理 GDDR upset 的动态传播。

因此第一阶段应明确定位为：

> **GDDR-location-aware GPU device-memory fault injection**

而不是：

> **cycle-accurate physical GDDR-cell radiation propagation simulation**

---

## 3. REMU 中可直接继承的部分

### 3.1 Dual Addressing 思路

保留：

```text
正向：
DNN Data → VA → PA → Memory Cell

反向：
Memory Cell → PA → VA → DNN Data
```

但地址空间从 CPU/LPDDR 改成 GPU/GDDR6X。

### 3.2 Fault Model

可以继续沿用：

- BER；
- SEU；
- MCU multiplicity；
- row-adjacent MCU；
- column-adjacent MCU；
- DQ-adjacent MCU；
- fault count；
- fault probability；
- fault spatial correlation。

后续根据 GDDR6X 的实验数据或公开 radiation characterization 文献重新配置概率参数。

### 3.3 Bitmap Tree / Physical Adjacency Search

REMU 的 bitmap-tree 思想仍然适用：

```text
有效 GPU physical address
        ↓
转换为 GDDR coordinate
        ↓
建立有效 GDDR cell 集合
        ↓
在有效区域中搜索 SEU / MCU
```

目的仍然是避免在整个 VRAM 地址空间中随机选点后大量命中无效区域。

### 3.4 Fault Statistics

保留：

- injection ID；
- target tensor；
- physical location；
- bit index；
- BER；
- error pattern；
- inference result；
- BENIGN；
- SDC；
- DUE；
- retry；
- reproducibility seed。

---

## 4. 必须重做的部分

## 4.1 Module A：DNN Semantic Mapper

### 目标

建立：

```text
Tensor / Element / Bit
        ↕
      GPU VA
```

### 需要记录

每个目标 Tensor 至少记录：

```text
tensor_name
tensor_type
shape
dtype
layout
allocation_base_gpu_va
allocation_size
element_offset
bit_offset
lifetime
```

### 第一阶段建议

不要一开始覆盖 TensorRT 内部所有 opaque allocation。

先从一个完全可控的 CUDA allocation 开始，例如：

```text
某一层 weight tensor
或
某一层 activation buffer
```

要求能做到：

```text
layer3.conv1.weight
element = 12345
bit = 5

↕

GPU VA = 0xXXXXXXXX
```

### 验收标准

- 给定 Tensor element / bit，可计算 GPU VA；
- 给定 GPU VA，可反查 Tensor / element / bit；
- 结果可通过人工构造 pattern 验证。

---

## 4.2 Module B：GPU VA → GPU PA

### 目标

建立：

```text
GPU Virtual Address
        ↕
GPU Physical / VRAM Address
```

这是 GPU 版 REMU 的第一个核心难点。

### REMU 原方法

CPU：

```text
VA
 ↓
/proc/<pid>/pagemap
 ↓
PA
```

### GPU 侧问题

RTX 4090 上：

```text
cudaMalloc()
 ↓
GPU VA
 ↓
GPU MMU
 ↓
GPU PA
```

普通 CUDA API 不直接提供 GPU physical address。

因此需要单独研究 GPU VA→PA 获取方式。

### 研究要求

优先级如下：

1. 优先寻找无需修改 NVIDIA kernel module 的方法；
2. 优先使用已有可验证工具 / driver interface / GPU memory introspection；
3. 如只能通过 driver instrumentation 实现，必须与主实验代码隔离；
4. 不允许把 CPU `/proc/pid/pagemap` 结果误认为 GPU VRAM PA；
5. 每次 allocation 后重新获取 mapping，不允许跨 run 直接复用 Tensor→PA。

### 验收标准

同一次 allocation：

```text
GPU VA A ↔ GPU PA X
```

应稳定可重复查询。

重新 allocation 后：

```text
GPU VA / GPU PA
```

允许变化，但必须重新建立本轮 mapping snapshot。

---

## 4.3 Module C：GPU PA → GDDR6X Coordinate

### 目标

恢复：

```text
GPU PA
 ↓
RTX 4090 Memory Controller Mapping
 ↓
Channel / Bank Group / Bank / Row / Column / DQ
```

即：

\[
GDDRCoord = f_{4090MC}(GPU\ PA)
\]

这是 GPU 版 REMU 的第二个核心难点，也是整个项目最重要的物理映射问题。

### 原则

不能直接把 REMU 的：

```text
LPDDR4 default Ramulator mapping
```

当成 RTX 4090 的真实映射。

必须区分：

```text
模拟 DRAM mapping
```

和：

```text
真实 RTX4090 GDDR6X mapping
```

### 研究内容

需要恢复或验证：

- channel selection；
- bank-group selection；
- bank selection；
- row bits；
- column bits；
- burst offset；
- interleaving；
- XOR / hash function；
- DQ / bit-lane organization（若可观测）。

### 验收标准

至少能够验证：

- same-bank relationship；
- different-bank relationship；
- row-conflict relationship；
- row/column adjacency consistency；
- 多次实验结果稳定；
- 同型号卡之间 mapping 是否一致，单独做 cross-GPU validation，不能默认可迁移。

---

## 4.4 Module D：GDDR Reverse Index

当 Module B 和 Module C 完成后，为每次 workload 建立：

```text
Tensor Bit
   ↕
GPU VA
   ↕
GPU PA
   ↕
GDDR Coordinate
```

建议额外构建两个索引：

```text
GPU_PA → GDDR_COORD
GDDR_COORD → GPU_PA
```

以及：

```text
GPU_VA → TensorBit
TensorBit → GPU_VA
```

最终得到：

```text
GDDR_COORD → GPU_PA → GPU_VA → TensorBit
```

### 重要原则

这里保存的是：

> **当前 allocation / 当前 run 的 mapping snapshot**

不是永久 Tensor→GDDR mapping。

真正可能跨 run 保持稳定的是：

```text
GPU PA → GDDR Coordinate
```

这一硬件 Memory Controller mapping rule。

---

## 4.5 Module E：GPU-side Fault Injector

### REMU 原方法

CPU：

```cpp
*byteAddress ^= mask;
```

### GPU 版

改为 GPU device-memory 修改：

```text
target GPU VA
     ↓
CUDA fault kernel
     ↓
load
     ↓
XOR mask
     ↓
store
```

建议第一版采用：

```text
正常完成 Tensor allocation / initialization
        ↓
建立 mapping
        ↓
选择 fault
        ↓
GPU kernel XOR
        ↓
cudaDeviceSynchronize()
        ↓
DNN inference
```

### 验收标准

- injection 前 dump target value；
- injection 后 dump target value；
- 只允许指定 bit 变化；
- 对照无故障 inference；
- repeated run 可复现。

---

## 5. 完整 GPU-side REMU 工作流

```text
[1] Load DNN / Prepare Tensor
        ↓
[2] GPU allocation 完成
        ↓
[3] 记录 Tensor ↔ GPU VA
        ↓
[4] 获取 GPU VA ↔ GPU PA
        ↓
[5] GPU PA → GDDR6X coordinate
        ↓
[6] 建立本轮 TensorBit ↔ GDDRCell mapping
        ↓
[7] 建立 bitmap tree / reverse index
        ↓
[8] 根据 BER / SEU / MCU model 选择 GDDR cells
        ↓
[9] GDDR cell → GPU PA → GPU VA → Tensor bit
        ↓
[10] GPU-side XOR bit flip
        ↓
[11] cudaDeviceSynchronize()
        ↓
[12] Run DNN inference
        ↓
[13] BENIGN / SDC / DUE classification
        ↓
[14] 保存 mapping + fault + result log
```

---

## 6. 推荐分阶段实施

### G1：Semantic Mapping

目标：

```text
Tensor Bit ↔ GPU VA
```

任务：

- 建立 Tensor metadata；
- 获取 GPU pointer；
- 实现 element offset；
- 实现 bit offset；
- reverse lookup。

验收：

```text
Tensor element / bit
↔
GPU VA
```

完全可验证。

---

### G2：GPU Virtual-to-Physical Mapping

Status（2026-09-15）：**PASS**（三卡全部通过 scratch device/VMM、VMM alias
double-mapping、TensorRT 全工作负载三层验证；`gpu_va_pa_map.csv` 已生成）。
详见 `docs/G2_VALIDATION.md`。

目标：

```text
GPU VA ↔ GPU PA
```

任务：

- 调研 NVIDIA GPU MMU；
- 查找可用 driver/API/tool；
- 验证 RTX4090 device allocation；
- 验证 mapping temporal stability；
- 验证 re-allocation 后 mapping 是否变化。

输出：

```text
gpu_va_pa_map.csv
```

---

### G3：GPU PA-to-GDDR Mapping

目标：

```text
GPU PA ↔ GDDR6X coordinate
```

建议拆分：

#### G3-P1：GDDR6X topology
确认：

- memory size；
- channel；
- memory-chip organization；
- bank group；
- bank；
- row；
- column；
- burst。

#### G3-P2：Address-bit characterization
研究：

```text
GPU PA bit
→
channel/bank/row/column
```

#### G3-P3：Bank mapping
恢复：

```text
PA → Bank / Bank Group
```

#### G3-P4：Row mapping
恢复：

```text
PA → Row
```

#### G3-P5：Column / offset mapping
恢复：

```text
PA → Column / Burst / DQ
```

#### G3-P6：Cross-card validation
在 3 张同型号 RTX4090 上验证：

```text
f_MC_card0
f_MC_card1
f_MC_card2
```

不能预设完全一致。

---

### G4：Dual Addressing Integration

将：

```text
TensorBit ↔ GPU VA
GPU VA ↔ GPU PA
GPU PA ↔ GDDR
```

组合成：

```text
TensorBit ↔ GDDRCell
```

建立：

- mapping snapshot；
- reverse index；
- bitmap tree；
- mapping checksum；
- reproducibility log。

---

### G5：GPU Fault Injection

第一阶段只做：

- SEU；
- 2-bit MCU；
- 3-bit MCU；
- row-adjacent；
- column-adjacent；
- DQ-adjacent（若映射可验证）。

执行：

```text
fault select
→ reverse map
→ GPU XOR
→ inference
```

---

### G6：DNN Reliability Evaluation

统计：

```text
BER
fault count
fault location
tensor
layer
element
bit
SEU / MCU
BENIGN
SDC
DUE
accuracy drop
retry behavior
```

最终形成：

```text
GDDR physical fault
→ DNN semantic impact
```

的实验数据库。

---

## 7. 每次实验的 Mapping Snapshot

因为 GPU VA / GPU PA 会随 allocation 变化，每次 run 必须重新建表：

```text
RUN N
 ├── Tensor metadata
 ├── GPU VA map
 ├── GPU PA map
 ├── GDDR coordinate map
 ├── reverse index
 ├── selected faults
 └── inference result
```

禁止直接复用：

```text
Tensor → GPU PA
```

的历史映射。

可长期复用的只有经验证稳定的：

```text
GPU PA → GDDR coordinate
```

Memory Controller mapping rule。

---

## 8. Cache 处理原则

第一阶段不研究 GPU cache fault。

明确采用如下抽象：

```text
GDDR fault model
      ↓
选择物理 GDDR location
      ↓
映射到 GPU device-memory bit
      ↓
software XOR
      ↓
inference
```

因此第一阶段论文/报告中应明确声明：

> GPU cache hierarchy is outside the current fault model.  
> GDDR faults are materialized as application-visible device-memory corruptions.

后续可以单独扩展：

```text
GPU L2 / L1 cache fault injection
```

不能将第一阶段结果描述为完整的：

```text
physical GDDR capacitor upset
→ cache
→ SM
```

动态传播模型。

---

## 9. 实验正确性验证

必须分别验证四段，不允许只验证最终 inference。

### V1：Tensor ↔ GPU VA

使用已知 pattern：

```text
0x00
0x55
0xAA
0xFF
```

检查 element / bit 定位。

### V2：GPU VA ↔ GPU PA

检查：

- 同 allocation 重复查询一致；
- page boundary；
- 多 allocation；
- free/realloc；
- temporal stability。

### V3：GPU PA ↔ GDDR

检查：

- same-bank；
- different-bank；
- row conflict；
- row adjacency；
- 多次测量一致性。

### V4：Fault Injection

检查：

```text
before
after
xor_mask
expected
```

要求：

```text
after = before XOR mask
```

且非目标数据不应变化。

### V5：End-to-End

最终：

```text
GDDR Cell
→ Tensor Bit
→ bit flip
→ inference result
```

必须可重复。

---

## 10. 主要风险

### R1：GPU VA→PA 无公开接口

这是当前第一高风险。

策略：

- 优先查找已有研究工具；
- 优先 driver-side read-only introspection；
- 将 kernel modification 作为最后方案；
- mapping 模块独立，不污染主 fault framework。

### R2：RTX4090 Memory Controller mapping 非公开

这是第二高风险。

策略：

- 独立做 mapping characterization；
- 不使用假设 mapping 作为最终实验结论；
- 所有 mapping rule 必须有实测证据。

### R3：GDDR6X DQ 级位置难以完全恢复

策略：

第一阶段可以先做到：

```text
Channel / Bank / Row / Column
```

若 DQ mapping 无法严谨验证，则：

- SEU 可以继续；
- row/column MCU 可以继续；
- DQ MCU 标记为 unsupported / unverified；
- 不伪造 DQ 物理坐标。

### R4：TensorRT 内部 allocation 不透明

策略：

先使用可控 CUDA tensor / PyTorch CUDA tensor 建立完整链路。

确认全流程正确后，再扩展 TensorRT runtime allocation。

---

## 11. 第一阶段最小可行目标（MVP）

第一阶段不要追求完整 GPU memory system。

MVP：

```text
一个明确 Tensor
     ↓
GPU VA
     ↓
GPU PA
     ↓
GDDR Bank/Row/Column
     ↓
选择一个 SEU
     ↓
反向映射
     ↓
定位 Tensor element/bit
     ↓
GPU XOR
     ↓
验证 bit flip
     ↓
运行 inference
```

MVP 验收必须回答：

> **“RTX4090 GDDR 中这个物理位置发生一个 bit flip，对应 DNN 中哪个 Tensor、哪个 element、哪个 bit？”**

并能实际翻转这一位并观察 inference 结果。

---

## 12. 最终目标架构

```text
                    GPU-side REMU
┌──────────────────────────────────────────────────┐
│                                                  │
│   DNN Semantic Mapper                            │
│   Tensor / Element / Bit                         │
│              ↕                                   │
│          GPU Virtual Address                     │
│              ↕                                   │
│   GPU VA→PA Mapper                               │
│              ↕                                   │
│          GPU Physical Address                    │
│              ↕                                   │
│   RTX4090 GDDR Address Decoder                   │
│              ↕                                   │
│   Ch / BG / Bank / Row / Column / DQ             │
│              ↕                                   │
│   Bitmap Tree / Reverse Index                    │
│              ↕                                   │
│   SEU / MCU Fault Generator                      │
│              ↓                                   │
│   GPU Device-Memory Fault Injector               │
│              ↓                                   │
│   DNN Inference                                  │
│              ↓                                   │
│   BENIGN / SDC / DUE                             │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## 13. 核心研究结论定位

本项目不应简单描述为：

> “在 GPU 上随机翻 DNN bit。”

而应定义为：

> **基于 GPU 真实地址映射和 GDDR 物理组织的 memory-aware fault injection framework。**

核心贡献链应是：

```text
DNN semantic bit
↔ GPU virtual address
↔ GPU physical address
↔ GDDR physical location
```

然后基于真实 GDDR spatial correlation 实现：

```text
SEU / MCU
→ DNN fault
→ inference impact
```

最终目标是将 REMU 的：

```text
CPU VA ↔ CPU PA ↔ LPDDR
```

系统性替换为：

```text
GPU VA ↔ GPU PA ↔ GDDR6X
```

并保留其：

```text
Dual Addressing
+
Memory-aware Error Model
+
Reverse Mapping
+
Large-scale Reliability Evaluation
```

这一整体方法学框架。
