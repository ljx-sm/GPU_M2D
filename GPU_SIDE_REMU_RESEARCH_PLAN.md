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

Status（2026-09-15）：S0 尽调完成——参考工具（GPUHammer/GDDRHammer，仅研读
不拷贝）与 AD102/GDDR6X 已验证平台事实见 `docs/G3_SURVEY.md`。
Status（2026-09-16）：S1 标定 **PASS**（GPU0）——GDDR6X row-conflict 延迟差
≈39 ns（98 cyc @2520 MHz），冲突簇紧致（4.4 cyc），与 0 冲突的页内偏移与
GA102/A6000 公开结果一致；工具 `tools/g3_probe/`。映射规则求解尚未开始。
Status（2026-09-16）：S2 **PASS**（GPU0）——PA 标注计时池：64×8MiB 门控分配，
256 页全部 valid/VIDEO 且 PA 连续成 512MiB 整块（0x1ee00000..0x3ec00000），
chunk 内 PA 连续、按分配序每 chunk 步进 8MiB，但 VA 序乱序（观察 PTE 不可
用 VA 算术替代）；计时 floor/baseline/conflict=1023/1014/1140 cyc（≈47 ns），
跨页对 1038–1098 cyc 分布于中间态。工具 `tools/g3_probe/` +
`tools/g2_observer/run_g3_pool_probe.py`（`--api g3pool`）。S3 系统采集待启。
Status（2026-09-16）：S3 工具链就绪——`--work-mode bit-scan`（校准三元组 +
每 PA 位一组单 bit 差分对，页内 4 基页投票、页级按池覆盖到 bit 31）与
`analyze_bit_scan.py`（low/mid/conflict 三态分类 + 逐位投票表 + constraints.csv），
自测通过并在 S2 真实池图上干跑验证（512MiB 池 599 条查询，bit 0..28 全覆盖）。
Status（2026-09-16）：S3 采集完成（GPU0，4GiB 池 791 条查询，788 条可用约束，
0 非对称 0 完整性错误，幅度 124 cyc≈46 ns）。发现：页内 bit 0–9 恒 low（列/
burst 区候选）；bit 10–20 跨基页 SPLIT 且 bit 10/11/16/19/20 出硬冲突票；
页级 bit 21–31 每位三态混合（冲突占 13–28%，远超均匀 32 bank 的 ~3%）→ 线性
bit-slice 映射被排除，bank 选择为非线性（行参与）哈希，且哈希支撑集覆盖整个
可测 PA 范围。不同池区域基线差达 ~30 cyc（疑 L2 slice 邻近效应），分类余量足够。
S3b 工具链就绪——`--work-mode pair-scan`（以 S1 冲突锚 0xd0100 做锚定单 bit
与两位联合探测，区分列位与 bank 位；含锚有效性扫描与页级三联探测，
真实 4GiB 池图干跑 1587 条）与 `analyze_pair_scan.py`（线性玩具解码器自测
验证 bank_kept/bank_changed/两位抵消/锚扫描全部解释路径）。
Status（2026-09-16）：S3b 采集完成（GPU0，4GiB 池 1587 条查询，0 非对称 0 丢事件）。
锚 0xd0100 的保 bank 性随位置变化：页内 4 基页仅 1 有效（S1 页族 1144/1144），
扫页 35/128 有效（32 mid/61 low）→ 固定支撑集线性哈希被直接测量排除。页内位
在有效基页切分：列位 {0–7,9,11}、哈希位 {8,12–14,16–18,20}（8/16/17/18 掉至
1022–1023 最深）、居中 {10,15,19}。两位联合 840 对 66 冲突呈“列位×行载体”形态
且载体随基页轮换（10@b2/11@b0/16@b3/19@b2），基页 0 内存有成对抵消簇 → 局部
低次、系数种子化。页级位裁决：行位 25/27/28/29，bank 位 22/26，列样折叠 21，
混合 23/30/31；扫页覆盖到 PA bit 32。S4 求解器输入就绪（constraints.csv 788 +
pair_constraints.csv 1587，mid 作软约束）。
Status（2026-09-16）：S4 求解器就绪——`solve_mapping.py`（冲突→同 bank 齐次
GF(2) 方程，low∧xor 触及行位→分离约束；bank 为度≤2 种子化多项式族，按冲突
签名精确枚举到权重 3 + beam XOR 生长；行支撑按单调子句+单元重解并计违例；
两阶段交替；mid 只评分。留出集 ≥95% gate（S5 预演）；合成种子化真值自测：
度 2 零残差零违例、留出集 ≥99%（真实泛函恰在权重 2–3 被恢复），度 1 必须
报不足。`--predict` 用保存的模型对任意 PA 对分类）。
Status（2026-09-16）：S4 采集侧求解完成，**模型类不足——gate FAIL（诚实负结果）**。
真实 S3+S3b 约束（2375 对）上：度 1 退化到 low 基率 0.782；度 2+线性行支撑
0.774；度 2+双族种子化（bank 与行均为度≤2 GF(2) 泛函族，交替求解，停滞时
锚定-补全搜索到权重 4，节俭门拒绝窄覆盖）0.758——均远低于 95%。诊断链：
(1) 619/1424（43%）low 落在冲突特征张成空间内（其中 216 个恰为锚 mask
{8,16,18,19} 位），线性「xor 触及固定行支撑」模型被线性代数直接排除，行折叠
必须种子化；(2) 标签可复现（S3b 新鲜票与 S3 一致 5/5、6/6），low 延迟无随
PA 距离的干净漂移——不是噪声；(3) e16/e18/e19 各点燃 ~130 锚定冲突但被 ~190
锚无效 low 阻塞，行位 25/27/28/29 各 ~17 冲突被 23–38 个 low 阻塞——分离它们
的种子结构需要权重 >4 或度 ≥3 项；(4) 556 mid（23%）被排除，肩带吞掉了
恰好在阈值附近的信息对。求解器本身在合成真值上精确恢复（含权重-4 线性核
bank 位与纯二次行折叠项）。S4b 方向：池区域局部标定三元组（收窄 mid 带）
+ 锚无效页上的阻塞对定向探测（裁决「行折叠度 ≥3」vs「bank 覆盖缺口”）；
S5 预测验证阻塞在过 gate 的模型上。
Status（2026-09-16）：S4b-0 完成（纯离线分析，零新采集）——新增
`analyze_local_recal.py`：(1) 页级延迟指纹 λ 实测真实（1011–1063，
~50 cyc≈19 ns；页内复测中位差 7–9 ≪ 全局散布；无单一 PA 位效应 → 哈希状
位置性，与通道/L2 slice 路径差一致；对值只有一个标量，无法离线分辨 ~12 个
离散带）；(2) 局部重标定为不对称规则（low/mid 边界随页 λ 局部化，conflict
门保持全局——对称版会把 134/139 个 S3 冲突降级为 mid，且实测
conflict−amp 落在 981–1001、低于 λ 域 1011–1063，即冲突惩罚不骑在 low 路径
基线上）：mid 556→257（−54%），395 个冲突全保留（其中 134+139 个低于自身
局部冲突阈值，标记为 S4b-1 定向复测候选），11 个边缘 low 修正为 mid；
(3) 残余 257 个 mid 保存为同通道候选（same_channel_candidates.csv）；同
通道图（冲突+残余 mid 边）只有 147 个小分量（最大 22 页）——与 ~12 通道不
矛盾，但边密度不足以合并每通道 ~170 页，通道划分需 S4b-1 逐页普查裁决；
(4) 求解器在重标定标签上 holdout 0.758→0.809——真实提升但仍未过 0.95
gate：标签质量是瓶颈之一，度 2/权 4 模型类不足仍是主瓶颈（与 S4 诊断一
致）。S4b-1（通道普查 + 阻塞对定向复测）待 GPU0 空闲窗口执行。
Status（2026-09-16）：S4b-1 完成（GPU0，`--work-mode census`，3164 条查询，
0 丢事件；4GiB PA 洞第三次复现，273 个复测对 + 4 个试点页全部解析、0 丢弃）
——**旧 conflict 类被证明是双峰的，这是 S4 求解失败的具体原因之一**：
(1) 逐页单访问 λ 普查（2048 页干净自对）：λ 真实且空间性（self 段内部平坦，
无时间趋势），但**无 ~12 离散带**（gap-4 仅 2 簇 83%/17%），两簇均为细粒度
逐页布局（512 chunk 中 249 个快慢混页、无 PA 位效应 >0.02）→ λ 不是通道
观测量，λ 带通道假设被实测证伪（S4b-0 的 pair-λ 与普查 λ 相关性仅 r=0.11，
旧估计是噪声上界）；(2) 运行内晚期台阶：self 段之后的所有段落统一低 ~18 cyc
（repeat 块实测；self 段平坦 → 是台阶不是漂移），分析器对 self_second/
reprobe/pilot 加性校正；(3) 273 个可疑冲突复测：228 个落入**可复现浅带**
（校正后 p10–p90=1104–1120 ≈ 0.70–0.85 幅度），36 个降为 low，9 个 mid，
**0 个深冲突**；用 0.90 门回溯旧标签：S3 的 139 个 conflict **全部**是浅带，
S3b 185 浅 + 71 深（≥1139）——双带分类器把"部分惩罚区制"与全行冲突混为
一类，S4 的同 bank GF(2) 约束正是这个混合物；附带发现：≥1200 深尾（22 对）
聚在页号 ≡7 mod 16 的页（32MiB 周期超级冲突，S4b-2 素材）；(4) 行等价类
试点（仅深冲突并 bank、bank 内 low 并 row，0 矛盾）：2 个深锚页的传递类恰为
{0,0x200}/行 与 {M,M|0x200}/行——bit 9 在硬件级确认为列位，锚翻转行，
非锚哈希位离开 bank 组；2 个浅页的锚对只付肩带 → S3b 的"锚有效性 27%"实为
深/浅拆分而非线性哈希性质；(5) 更新物理模型（S4b-2 求解器输入）：low =
异通道或同行；shoulder ~1114 = 同通道异 bank/bank-group；deep ≥1139 =
同 bank 异行。肩带给同通道候选一个大的正样本集（228 复测 + 185 S3b 浅 +
残余 mid）：通道划分现在是在肩带边上的图论问题，不再是 λ 问题。
Status（2026-09-16）：S4b-2 完成（纯离线分析，零新采集）——新增
`analyze_three_band.py`（用普查 λ 对 S3/S3b 做三带重标定，PA 洞一致故页表
1:1 对接，缺页即拒绝）与 `analyze_channel_graph.py`（肩带+深带跨页边建
同通道页图）。(1) 回溯拆分与普查复测估计完全一致：S3 的 139 个 conflict
全部为浅带、S3b 185 浅 + 70 深；普查 λ 几乎抹掉旧 mid 带（S3 241→5、
S3b 315→124）——旧 mid 多为 λ 涂抹，现正确归 low（2375 对中 1851 low）；
(2) 通道划分信号存在但欠定：380 条跨页同通道边（379 肩 + 1 深，含普查
复测肩带）只触 250 页 → 82 个分量（最大 13 页 5%），边一致性良好（跨页
low 落入分量仅 28/845=3.3% 矛盾率，且是单条坏边非规则失效），但远不够
合并每通道 ~170/2048 页；最大两个分量带强高位签名（bit 27/28 100%、
bit 31 92%、bit 24–26 互斥），分量内普查 λ 混合 → 划分是地址结构性而非
λ 结构性，闭环 S4b-1 的 λ 结论；(3) 干净架构事实：70 个深冲突 69 个
in-page（bank 哈希吃 PA bit < 21），页级位翻转只出肩带/low → 通道选择
在高位、bank 哈希在页内；(4) 求解器在三带标签上仍无法拟合深集——度 2
的 18 个 bank 泛函全部在训练冲突上消没，但 ~57 个训练冲突 47 个无行泛函
可解释，holdout 冲突召回 0/14；数值上 0.9557 会"过"95% 门，但那只是
all-low 基率（冲突仅占硬对 3.6%）——`solve_mapping.py` 已改为打印逐类
召回并标记 DEGENERATE PASS，S5 依旧阻塞。结论：干净标签下 S4 的判决
成立——度 2/权 4 模型类不足以表达种子化行折叠，70 个结构相近的深冲突
撑不开它。下一步：池内经验 (bank,row) 类表作为 G4/G5 的兜底交付 +
通道/bank 定向加密采样（S4b-3）。
Status（2026-09-17）：**方向调整（pivot）——放弃闭式哈希逆向，改为
GeForge 式经验映射表（EMT）**。决策依据：(1) 我们的 S4/S4b-2 负结果与
GeForge（S&P'26，仓库无 license，仅方法学参考，见 `docs/G3_SURVEY.md`
§1）脚注 1 互相印证——PA→bank 函数"高度非线性且混合（几乎）全部地址
位"，他们也没有逆向闭式解，而是离线 per-model 建表（每 bank 一文件，
行→PA 块归属表）+ 行条带单调行号模型（App. B 假设，间接验证）+ 同型号
复用；(2) 我们的结构性优势：G2 观察器每 run 直接给出池页真实 PA，
GeForge 为无特权攻击者设计的 page anchoring（L2 指纹对齐）问题对我们
不存在。S4/S4b-2 求解器线封存为诚实负结果。新阶段 S5 = EMT：
S5-T0 种子表（纯离线）→ S5-T1 锚扩充采集（`--work-mode table-build`）
→ S5-T2 验证门（传递一致性/类基数/复现/跨卡 G3-P6/端到端预测——原
被 gate 阻塞的 S5 预测验证以表形态复活）→ S5-T3 查询 API + G4 集成。
诚实边界：行 ID 线性为继承假设（timing 无行距信息，与文献同等地位）；
表覆盖 = 池 PA 范围（G5 注入只发生在池内，闭环成立）；驱动升级可致
PA 漂移 → 表构建吞吐须支持按需重建；DQ 维持 unsupported（R3 不变）。
S5-T0 完成（零新采集）——新增 `tools/g3_probe/build_bank_table.py`：
以 reprobe > pilot > S3b > S3 优先级去重合并三源边（2455 条：54 深/
320 肩/1932 low；mid 永不作类证据），bank 类 = 深冲突并查、row 类 =
bank 类内 low 并查、通道分量图重算并与 S4b-2 产物交叉核对；一致性门
C1 肩带入 bank 类 / C2 深入 row 类 / C3 通道分量内跨页 low / C4 bank
类内跨页 low / C5 跨源标签分歧。两处对 S4b-2 记录的**修正**：(a) 深
冲突"70"实为 71 行 47 个不同 PA 对（S3b 多探测类型重复发同一对）+
pilot 3 确认 − 1 被 reprobe 降级 = 54；(b) 通道分量应为 **74 个/220
页**而非 82/250——S4b-2 的图只追加了 reprobe 肩带边、未用复测结果
降级 bands 肩带标签，29 条被复测推翻的跨页肩带边在此被剔除，C3 矛盾
率降至 20/807（2.5%，修正使划分更干净）。结果（`artifacts/g3/table_v0/`：
gddr_seed_table.csv 2048 页 + bank_classes.csv + table_report.txt）：
40 个 ≥2 节点 bank 类（最大 12，pilot 页为骨架；38 个二元对）、4 个
row 对、**0 个 bank 类跨页**（S4b-2 唯一跨页深边即被降级那条）、
C1/C2/C4 全零矛盾；覆盖诚实稀疏——通道分量 220/2048 页、分类节点
38/2048 页、单例 2022/2114。此即 S5-T1 的工作量基线。

Status（2026-09-17）：**S5-T1 完成——全池采集一次跑通，且实测数据
修正了物理模型**。新增 `--work-mode table-build` 六段式负载
（calibration 3 / self 2048 / anchor_sweep 49152=24 掩码×2048 页 /
classify 151478=2048 页×74 种子代表 / bank_map 392=8 页×25 偏移格点
双探测 / repeat 64，共 203137 查询、工作段 19 s ≈10.8k q/s，0 失败；
锚掩码与代表页取自 `--seed-table artifacts/g3/table_v0`）+
`tools/g3_probe/analyze_table_build.py`（逐行 PA 校验 fail-closed，
产出 table_build_edges.csv / anchor_validity.csv / bank_map_pages.csv /
channel_partition.csv）+ `build_bank_table.py --t1`（T1 边为最高优先
源）与 `--channel-deep-only`（页图仅按跨页深边并查）。

**模型修正（本次实测，取代 S4b-1 跨页三带说）**：每对值以
max(λ_a,λ_b) 为参考而非全局基线。页间 λ 展宽 ~120 周期 > 冲突幅度
（110），任何全局门都落进 λ 展宽内，把慢页错造成"肩带"。λ 参考下
classify 段双峰、+30..+80 谷为空：low 在 d≈0（96%）、deep 在
d≥+80（0.26%≈1/384，与 AD102 先验 24 通道×16 bank 吻合）。历史上
的跨页"浅带"（S3b 185、census 复测 228）在 λ 参考下肩带与 low 判决
的 d 分布相同——选择偏置而非物理区制。推论：跨页 low 与"同通道
不同 bank"相容，旧 C3 规则失效；深冲突是唯一跨页类证据。门限：
low < 0.35·amp、deep ≥ 0.60·amp（页内谷在 +45..+70）。

结果（run_pool_gpu0_1789656551689364094）：201022 对分类
{深 11932 / low 187761 / mid 1329}；同 bank 页图 387 条跨页深边 →
60 个分量、370/2048 页（最大 11），分量内跨页 low（坏深边标记）
**0**；锚有效性 **2048/2048 页全有有效锚**——0x1fdc80 与 0x1f9dc0
为全页通用掩码（bank 哈希的核掩码），0x119980/0x11e300/0xd0100
部分有效（818/650/484 页），修正 S3b"27% 锚有效率、位置相关"为
掩码相关；bank 图谱健康页呈干净的 base-low/anchor-deep 行拆分
（0x2ae00000 9/25：0x200→row0、0xd0300/0xd3880/0xd7b00→rowM；
0x42e00000 6/25），≡7 mod 16 超冲突页（0x1ee00000/0x32e00000/
0x3ce00000）对全部 24 个扫描候选读深、25/25 格点同 bank——整页
配对基线抬高（sweep d p50≈+85 vs 典型 +15）是超冲突结构而非污染
（0x200 列探测仍读 low，λ 无误）；跨页同 bank 集合 Jaccard p50
0.36——通用核 + 每页种子结构，与 S4 判决一致。表 v1
（`artifacts/g3/table_v1/`，`--t1` + `--channel-deep-only`）：
200183 条去重边（T1 占 198129 为最高优先源；C5 记录 321 处分歧，
以 low↔shoulder 翻转为主——预期的 λ 修正）、39059 个 bank 类
（1741 个 ≥2 节点，最大 125、60 个跨页）、13444 个 row 类、
**C1/C2/C3/C4 全零矛盾**；分类节点覆盖 **2048/2048 页**（T0 为
38/2048），同 bank 页分量 370/2048 页——其余为诚实余量（1/384
命中率下多数页本就不与任何代表页共享 bank）。下一步 S5-T2 验证门
（R-a 传递一致性 / R-b 类基数 vs 先验 / R-c 复现 / R-d 跨卡 /
R-e 端到端预测）。

Status（2026-09-17）：**S5-T2 完成——五门全过（5/5 PASS），EMT 表
获得 G4/G5 消费资格；行类为概率性（~1% 误差）**。新增
`tools/g3_probe/validate_table.py`（自测固定各门机制）+ 编排器
`--work-mode predict-check --pairs-csv`（R-e 落地负载：calibration 3 /
self 触摸页 / predict 逐对 / repeat 校漂移；离开池的对丢弃并报告）。
各门（bar 与理由均写入模块 docstring）：(1) **R-a PASS** a1–a4 全 0
（含新计数 a3：row 类跨两个 2 MiB 页物理不可能——页内列字段吸收不了
≥21 的 PA 位）；(2) **R-b PASS** 无偏跨页深率 774/302956=0.2555% vs
先验 1/384=0.2604%（z=−0.53），隐含 bank 数 391（2σ 365..422）⊇ 384；
均匀哈希蒙特卡洛零假设降级为诊断——代表页按 v0 通道分量每分量取一、
λ 与 bank 相关 → 代表页 bank 聚簇（rep-rep 深边 64 vs 零假设中位 6），
分量/覆盖低于零假设区间是选取偏置而非哈希违例（下轮建表应均匀取代表
页）；(3) **R-c PASS** 同卡复跑（run_pool_gpu0_1789662064404501578）：
200830 公共对一致率 99.45%，硬翻转（deep↔low）**0**、deep|mid 区间
召回 10936/10936、d 漂移中位 4 周期、锚硬翻转 0、同 bank 划分共宿
Jaccard 1.000——deep↔mid 计为门限摆动（每次 run 的门骑自己的单查询
标定幅度 110 vs 121），跨空谷的 deep↔low 才证伪；(4) **R-d PASS**
跨卡（GPU1 run_pool_gpu1_1789663447672819650：UUID 不同、池 PA 布局
完全相同 first_pa 0x1ee00000）：一致率 99.36%、硬翻转率 0/200830、
Jaccard 1.000、387 条深边→同样 60 分量/370 页、bank 图谱偏移集与
void 页相同、0x1f9dc0 双卡通用（0x1fdc80 在 GPU1 为 2047/2048）；
λ 绝对漂移中位 25 周期为信息项不计门（跨卡时序整体偏移：amp 110 vs
111、标定 1032/1038/1149）——这正是表存"类"不存"周期数"的原因；
表按**型号**成立，GeForge 同型号复用主张获得实测证据；(5) **R-e
PASS** 端到端预测（215 对未测对：128 深 + 87 low，跨页优先、行关系
未知者排除）：深硬翻转 0/128=100%、low 硬翻转 1/87=98.85%；一条硬
证伪 0x2aed3880/0x2aed7b00——两地址对 0x2ae00000 的 M 均可复现读
锚 low（推得同 row），互测却 deep +97：双探测行推断非普遍成立，行类
为概率性 ~1%，bank 级结论稳固（各门深硬翻转 0）。表 v2
（`artifacts/g3/table_v2/`，双 T1 run 合并 + --channel-deep-only）为
G4/G5 消费版本：bank 类按实测事实消费，row 类碰撞按强证据处理（未来
建表轮对用于 row 合并的 low 边做二次读）。下一步 S5-T3（查询 API +
G4 集成）。

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
