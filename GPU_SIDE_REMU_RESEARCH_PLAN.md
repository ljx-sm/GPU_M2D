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

Status（2026-09-17）：**用户批准执行 T3.0 大池全覆盖建表——三卡顺序：
GPU0 大池 → GPU1/GPU2 复制 → 用户确认 → 才进入 S5-T3 集成**。决策
依据：表是"绝对 PA→关系"的实测快照而非 offset 公式（S4 逆向失败 +
页内格点每页种子 Jaccard p50 0.36 证明页基混入哈希），新 PA 只能测不
能推；与其实验中途撞上缺页，不如一次付清覆盖成本。工程参数：池
704×32 MiB ≈ 22 GiB / 11264 页（全卡 24 GiB 留余量）；代表页改为
**均匀采样**（新 `--rep-uniform N`，取代 v0 种子表的一分量一代表——
后者经 R-b 诊断证实 bank 聚簇），数量按覆盖数学定：384 bank 下 R 个
均匀代表页把 1−(1−1/384)^R 比例的页连到同 bank 代表（R=384→63%、
R=1024→93%、R=1536→98%），取 **R=1024**：classify 11264×1024≈11.5M
查询、全负载 ~11.8M ≈ 18 分钟工作段（此前口头估的"384 代表 7 分钟"
按此数学修正——63% 覆盖不值，93% 才是合理点位）。一次性窗口要求：
卡空闲 + 持有整池 22 GiB；被他人占用的页进不了表，事后可增量补测
（多 `--t1` 合并，旧条目永有效——硬件映射不变）。预期收益：此后
任何 ≤22 GiB 的实验池开工前查覆盖（fail-closed 覆盖检查进 T3 API），
命中即开跑，"中途建表"被结构性排除；且 R-d 已证同型号映射逐卡一致，
一张大表三张 4090 通用。

Status（2026-09-17）：**T3.0 GPU0 大池建表完成——表 v3 覆盖 96%，五门
（R-a/R-b/R-c/R-d/R-e）全过**。大池 run
`run_pool_gpu0_1789670148751001580`：704×32 MiB=11264 页、11,815,371 查询
513 s（23,028 q/s）、0 失败、amp 117；PA 洞稳定（first_pa 0x1ee00000，
整 22 GiB 连续，旧 2048 页为严格子集）。表 v3（`artifacts/g3/table_v3/`）：
宇宙 11264 页（2048 census + 9216 仅 T1 池）、1139 万去重边、C1–C4 全零
矛盾、同 bank 页分量 372 个覆盖 10852/11264 页（**96.2%**，此前 370/2048
=18%）、分类节点 11264/11264 页全覆盖。锚有效性 11264/11264
（0x1f9dc0/0x1fdc80 近全页通用）；≡7 mod 16 超冲突三页 25/25 格点与 T1
完全一致；三个 bank-map void 页（0x22e/0x2ce/0x48e）查明为 S3b 挖掘在
λ 参考模型下的陈旧标签（三 run 皆 low），非复现失败。两处**证据驱动的
门修正**（理由写入 `validate_table.py` 常量注释与 README T3.0 节）：
(1) **R-b 改为有效类数带宽 [368,400]**——11.5M 对测得跨页深率稳定低于
1/384 2%（隐含有效类 ≈392，2σ 387..396，z=−3.4；三次 run 一致），而
低于 1/384 对任何 ≤384 桶的固定分布不可能（非均匀只会升高碰撞率），
方向上排除了门要抓的"合并/变少 bank"腐坏；逐页度分解证明 σ 未被低估
（非 rep 页度欠散 2.20 vs 2.67、无页超零假设最大度——表面 27× 过散只是
"普通页 ~2.7 / rep 页 ~30"两个人群）。精确均匀 384 零假设 z 检验与 MC
零假设保留为诊断输出。(2) **R-d Jaccard 仅对同形态 run 把门**——不同
池/代表集的共宿比较的是对覆盖度而非结构（大池 vs 旧 2048 页：Jaccard
0.199 但双向硬翻转 0），形态不匹配时打印为信息项，硬翻转率/锚翻转/区间
召回照常把门；T3.0 协议的三卡同形态大池比较仍按 ≥0.99 把门。
R-c（vs 2048 页复跑）：硬翻转 0、区间召回 10521/10521、d 漂移中位 5、
锚硬翻转 0；R-e：deep 硬翻转 0/128、low 86/87（唯一硬证伪仍是 T2 记录的
0x2aed3880/0x2aed7b00 行推断局限，可复现）。`build_bank_table.py` 同步
修正完整性宇宙为逐 T1 run 的 pool_map（大池边跨 11264 页，2048 页 census
宇宙是错误参照——v3 首次构建即在此 fail-closed）并诚实留空 T1-only 页的
λ 列。下一步：GPU1/GPU2 同形态大池建表 + big-vs-big R-d，全部通过后
向用户汇报并等待确认进入 S5-T3。

Status（2026-09-17）：**T3.0 三卡完成——结构逐卡一致，big-vs-big R-d
三对全过；表 v4（五源合并）为 G4/G5 规范表**。GPU1
`run_pool_gpu1_1789673269815802288`、GPU2
`run_pool_gpu2_1789673819559585717` 与 GPU0 同形态（704×32 MiB、
`--rep-uniform 1024 --rep-seed 7`）：跨页深边 29441/29439/29441、分量
尺寸头 [42,41,40,38,…] 相同、bank-map 格点表与三个 void 页一致、
0x1f9dc0 三卡近全页通用；标定幅度逐卡不同（117/121/101 cyc）。三对
big-vs-big R-d：硬翻转 1/43/635 每对 ≤0.0054%（bar 0.1%）、deep|mid
区间召回 100%/99.95%/99.32%、同 bank 页共宿 Jaccard 全部 1.000
（156286 对双方同宿、0 对单宿）——**同型号映射一致的主张获得三卡
完备矩阵证据**。两处合并/门修正（均为证据驱动，理由写入代码常量与
README）：(1) **R-d 锚转移门改消费层判据**——anchor sweep 的 mid 谷
有人口（每卡 3.5–7% mid 格，远高于 classify 的 0.08%），逐卡门位置
（幅度 117/121/101）把谷带格子两次软摆动复合成逐格 deep↔low：实测
633 格 GPU2=deep/GPU1=low、其中 601 格 GPU0=mid（单向、209 页），
而结构门全部 1.000、**11264/11264 页每对都保有公共锚**（每页锚集合
Jaccard p50 1.000）——故逐格锚硬翻转降为信息项，门改为"无公共锚页
率 ≤0.1%"（真不同的 bank 哈希会在几乎所有页上失败）；(2) **合并规则
补全**——mid 永不推翻已决标签（深边实测一次即同 bank 事实），跨 T1
run 的 deep↔low 竞争按多数表决（平局→mid 排除，不让单卡带边深读
变成错误同 bank 边；三卡合并 656 竞争：622 平局、34 low 多数）。
另修正一处归因：旧记"GPU2 mid 杀掉 26555 深边致覆盖降"实为 GPU2
文件内 sweep/classify 重复行的 C5 记账，真正的覆盖差是 v4 首建漏折
旧 2048 页 T1 run（大池 rep 集未测过其对，未被推翻的独立证据）——
补齐后覆盖恢复。表 v4（`artifacts/g3/table_v4/`，五个 T1 源按时间序
合并）：1139 万去重边（深 91911）、C1–C4 全零、**372 个同 bank 页分
量覆盖 10852/11264 页（96%）**、分类节点 11264/11264 页全覆盖。诚实
余量：412 页未连（1/384 命中率下与 1024 均匀 rep 无同 bank）、行类
~1% 概率性（0x2aed3880/0x2aed7b00 可复现证伪对）、三个 bank-map
void 页（S3b 陈旧标签）、λ 列仅 census 页。**下一步：等用户确认后进
入 S5-T3（查询 API + G4 集成，含 fail-closed 覆盖检查）。**

Status（2026-09-17）：**用户确认三项边界决策后 S5-T3 完成——EMT 查询
API 落地（`tools/g3_probe/query_table.py`，纯离线、自测锁定），G3 阶段
闭合**。用户批准的决策：同/异 bank、同/异行按实测交付；行邻接借鉴
GeForge（假设+显式标注，不复杂化）；相邻列放弃（timing 原理上不可测，
G5 故障模型简化：列相邻折叠为"同行随机列"，DQ 维持 unsupported）。
API 语义：(1) **四级 provenance**——measured（直接实测：锚扫描格、行类
low 边、完备 classify 闭包）/ transitive（等价类闭包，R-e 已验证
0/128）/ assumed（行邻接：GeForge App. B 单调行条带先验，timing 无行距
信息）/ unknown（不猜测）；(2) **四条可靠性论证写入 docstring**——页面
分量互异⇒异 bank（classify 对页×代表完备，同 bank 页必同分量）；
跨页同分量⇒异行（a3 物理论证）；**页内节点 bank 类互异⇒未知而非异
bank**（页内稀疏采样不享完备性论证）；异行只由直接 deep 证据或跨页物
理论证宣布（行不等式不可经未测对传递）；(3) **fail-closed 覆盖检查**——
宇宙外 PA 拒绝（exit 2），未连页默认警告、`--require-linked` 升级为拒
绝；(4) **两个模型拒绝**——dq-adjacent（R3）、column-adjacent（折叠，
附原理说明）；(5) **证伪对守卫**——0x2aed3880/0x2aed7b00 查询打
`measured-contradicted` 标签、选择器剔除；(6) `--annotate-pool` 把
run 的 pool_map（VA 页↔PA 页）与 GDDR 类拼接为 G4 消费的快照 CSV
（G3 腿）；(7) `--build-anchors` 把三个大池 run 的锚扫描格折叠为逐页
严格多数共识（per-cell 翻转维持信息级——正是 R-d 记录的谷带摆动）。
表 v4 上实测：锚共识 270,336 格、**11264/11264 页保有有效锚**、通用
掩码 0x1f9dc0+0x1fdc80；大池覆盖检查 11264/11264 PASS（412 未连页警
告）；实测同行点位 = 9 个 row 类上 113 对（114 减剔除的证伪对——同行
故障的诚实选址上限）；S3b run 标注 2048/2048 页、bank 已知 1992/2048、
5 页含同行点位。**下一步 G4**：以 annotate-pool 快照为 G3 腿，组合
G1（Tensor↔VA）与 G2（VA↔PA）建立 TensorBit↔GDDR 双向链与
reverse index。

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

Status（2026-09-20）：**G4-T1 完成（纯离线，零新采集）——三腿拼接工具落地，
三卡真实快照建成**。新增 `tools/g4_dualaddr/build_snapshot.py`：G1.5
AllocationRegistry（allocation_id↔VA↔语义标签）× G2 逐 run
`gpu_va_pa_map.csv`（join 键 = allocation_id + VA 页，非 VA 算术）× G3 表 v4
（PA 页→bank 组件/row 类/锚）→ 每 run 双向寻址快照（snapshot_pages.csv +
manifest.json 含 mapping checksum=快照 sha256）。fail-closed：ACTIVE 分配的
每 VA 页必须有 pte_valid+VIDEO+COMPLETE 观察行；驻留 PA 页必须在表宇宙内；
跨 VA 页 PA 别名拒绝；同页字节驻留不交叠（TRT 实测 5 个小分配共享一页）；
map 悬挂 allocation 拒绝；未连页/无锚页如实标注+警告。自测锁定全部路径
（含 gap/outside/invalid-PTE/alias/overlap/dangling 六种拒绝）。一个数据
教训：顶层 `artifacts/g1_5/` 与聚合 map 来自不同执行（g1_5_run_id 是逻辑
标签），必须用 run 目录内的配对 `g1_5_allocations.csv`。三卡实测（各自
最新 TRT run）：各 18 行/14 PA 页/26.4 MiB 驻留/7 分配，全部在宇宙内；
bank 已知 14/14（GPU0/GPU1）、12/14（GPU2，2 页未连警告）；锚 14/14 三卡
全有；`--lookup-va/--lookup-pa` 双向查询通过。两个塑造 G4-T2 的发现：
(1) 三卡 PA 页集互不相同（0x1f0../0x205../0x26e.. 段）——共驻显存状态改变
分配落点，per-run 快照 + 宇宙 fail-closed 按设计吸收（用户批准的共驻结论
获得实测印证：观察与注入不受他人进程影响，仅显存压力可能把分配推出宇宙
而拒绝 run）；(2) 负载驻留页与 5 个实测同行类页交集为 0——G5 同行 MCU
需放置规划（受测数据导向 row 类页）或 row-mining 定向补测，G4-T2 出决断
数据。
Status（2026-09-20）：**G4-T2 完成——三卡实机门控 run 全部 PASS，双地址链
闭环（GDDR 侧选点 → VA → TensorRT 字节 → XOR 位翻转 → 恢复 → 零残留）**。
交付：`tools/g4_dualaddr/run_g4_t2_injection.py`（编排器 + 离线自测）+
runner 增量 T2 模式（`--injection-work/--injection-release`，legacy 路径不变
且回归通过）+ `scripts/run_g2_observer_probe.sh --api g4t2`（唯一 sudo 面；
无 `--device` 时顺序循环三卡，满足串行 eBPF 约束）+ `docs/G4_VALIDATION.md`。
流程：观察器先挂 → runner 建全部分配+干净推理 → 写 gate 时刻注册表并阻塞 →
编排器**在线**建 PTE 台账 + gate map + 快照（复用 T1 `run_build` 同一代码路
径）→ 从快照**反向链选点**（t1 binding：优先 bank 已连页再选 data 输入
binding；t2 internal：优先 bank 已连页再选最大 TRT 内部分配；字节=驻留区间
中点；位=固定策略常数）→ 工作文件释放 → runner 对每个目标做活体注册表 VA
一致性检查、整分配快照、XOR、`after==before^mask`、非目标字节不变、反向
映射，随后带故障推理（非法输出记 DUE 而非工具失败）、逐字节恢复+整分配比
对、恢复后 sanity 推理必须复现干净结果；收尾严格台账 + 注入窗口内存活检查
+ gate/final 注册表一致性 + gate/final map 差分（中途重映射检测器）+ 编排器
独立复验每一行。共驻策略生效：三卡 run 时均有 3 个他人进程（共驻显存
10.2/4.7/8.8 GiB），记录不拒绝；三卡 PA 段完全不同（0x29a../0x142../0x242..）
但全在 22 GiB 宇宙内——共驻压力被 per-run 快照+宇宙 fail-closed 按设计吸
收。实测（各 7 分配/18 map 行/119 eBPF 事件/0 丢失）：t1=data binding 字节
301056 bit5（50→18），t2=trt-internal-0（bit2），全部 XOR 验证/守卫字节不
变/反向映射/恢复/sanity 通过；**六次注入推理全部 SDC_NUMERIC**——故障是真
实语义效应而非仅内存演示；G3 通用锚 0x1f9dc0/0x1fdc80 在三卡选中页上全部
出现。G4 闭合。**下一步 G5**：GPU 故障注入（SEU/2-bit/3-bit MCU + 行/列
空间相关；同行 MCU 需先决断放置策略——负载驻留页 ∩ 5 个实测 row 类页 =
0）。
Status（2026-09-21）：**G5-T0（用户文献调研）+ G5-T1（故障模型参数表）完成
——`docs/G5_FAULT_MODEL.md` 定稿并经用户确认**。T0：用户 9.21 完成调研，定比
SBU 60% / MCU 40%（事件占比），MCU 内 2-bit:3-bit = 1:1（事件数），空间形态
取文献图示（2-bit 同行/同列相邻；3-bit 横三连/竖三连/4×L，类内均匀），并
决定**不再强求紧相邻**——每个形态落在可实测的最强关系上。T1 冻结参数：横向
= 同行（同 256B 块内随机字节；S3/S3b 实测页内 bit 0–7 恒列位 ⇒ 同块即同行
同 bank）；纵向 = 同 bank 异行（表 v4 逐页锚共识核掩码 XOR，实测，掩码 <2MiB
必落同页）；L 四形塌缩为同一采样分布、朝向标签保留；"assumed"层（GeForge
行条带先验）不被 G5 消费——G5 注入的每条关系均为实测；放置规划/row-mining
双双退役（row 类页只服务远距同行对），G5 零新 GDDR 采集。BER 语义 = 单次
trial（1000 图带故障推理一遍）的驻留位翻转比例，R = 26,428,428×8 =
211,427,424 bit，五档 1e-8/5e-8/1e-7/5e-7/1e-6 → B = 2/11/21/106/211，
(s,d,t) 穷举最小二乘冻结（L1 = 2×SBU 纯单比特点——B=2 无法表达 60/40，用户
确认）；每档 1 campaign（1 进程 1 bootstrap）× 100 trial，档内 B 与构成冻结、
仅位置随机（SBU 字节/bit、MCU 基址/锚选/块内偏移/朝向标签）；分类 DUE >
SDC_TOP1 > SDC_NUMERIC > BENIGN 逐图比对 clean pass；文献出处节留待用户补。
**下一步 G5-T3**：campaign 实现（采样器 + runner campaign 模式 + 编排器，
复用 T2 门控/恢复/收尾骨架）。
Status（2026-09-21）：**G5-T3 campaign 实现完成，L3×2 trial 冒烟
G5_CAMPAIGN_VERIFIED**。三件套：`tools/g5_faultinj/fault_model.py`（冻结
档位表 + 站点采样器，纯 stdlib，重推导自校验）；runner campaign 模式
（`apps/resnet50_int8_g1_5.cpp`：门控前 CPU 预处理全部 1000 图 → 严格
clean pass 逐图记录 → 每 trial 全站点同时翻转（逐点 after==before^mask、
反向链、分配级 guard=pristine^masks）→ 带故障保持的 1000 图推理（输入
故障随每图重打、输出故障随每次 enqueue 重打、TRT-internal 不重打=权重
持久/scratch 软翻转）→ 逐图对 clean 记录分类（首 DUE 中止本 trial 余图）
→ 逆序恢复 + 按类恢复验证（输入绑定对最后评估图精确比对 fail-closed；
输出绑定 skipped:engine-owned-output；TRT-internal 信息性 mismatch:N）→
单图 sanity 推理须复现 clean）；`tools/g5_faultinj/run_g5_campaign.py`
编排器（门控窗口内 snapshot→驻留字节数必须等于冻结 R=26,428,428 否则
拒跑→采样→构成/PA-byte-bit 去重复核→work.csv；收尾 ledger/map diff/
registry 稳定/事件流逐条对齐 work/四张结果 CSV 独立复核（含分类从记录
数值重推导、DUE 块形、事件↔CSV 一致））。一次真实的验证器抓序 bug：
SITE_RESTORED 逆序（T2 约定）与校验器正向预期不符 → 修校验器并让自测
独立构造事件流。入口 `scripts/run_g2_observer_probe.sh --api g5campaign
--level L1..L5 [--trials N] [--seed N]`。冒烟：2 trial/21 站点/图，
全链 VERIFIED，trial≈0.7 s → 100-trial 档 ≈2 min。遗留：5×100 全量
campaign 待用户放行后执行。
Status（2026-09-21）：**全量 campaign 执行完成——5 档 × 3 卡 = 15 个
campaign、1500 trial、1,500,000 次带故障图像评估全部
G5_CAMPAIGN_VERIFIED（0 fail-closed、0 丢失 BPF 事件、1500/1500 sanity
复现 clean）**。用户在 tmux 串行执行（观察器单活约束）；分析器
`tools/g5_faultinj/analyze_campaign.py`（纯 stdlib，独立于 runner 从四张
CSV 重算全部指标）。pooled top-1 改变率（±95% CI，二项）：
L1 0.140%±0.013 / L2 0.242%±0.018 / L3 0.324%±0.020 / L4 0.773%±0.031 /
L5 1.030%±0.036；数值 SDC 率 63.1%→98.3%→99.9%→~100%；**DUE=0（全部
1500 trial、105,300 站点）**——构造性解释：输出 binding 仅 ~8 B/26.4 MiB
驻留（3e-7 占比），均匀采样 105,300 站点从未命中，restore skipped=0 印证；
精度（clean→L5）95.30%→95.00%；P(trial 含 ≥1 top-1) 39.0%→80.3%→88.3%→
100%→100%；r2w:w2r ≈ 2:1（量化权重翻转损伤对称偏正确→错误）。三卡一致
性：L1 逐位相同（确定性分配布局 → 权重站点相同；仅 5 个输入 SBU 位不同
且输出零效应——INT8 量化吸收单输入元素翻转），L2–L5 卡间差 0.014–0.334
pp，由重尾 trial（单 trial 最高损坏 163/1000 图）的过散解释。采样特性
（如实记录）：SBU 站点在驻留字节上均匀（输入占比 2.18% ≈ 驻留占比
2.28%），但 V/L 图样的锚 mate 落在稀疏驻留页（输入页 602 KiB/2 MiB）外
时重试 → 输入页在 V/L 站点中占比降至 ~0.5-0.7%——冻结模型的确定性采样
性质，非偏差 bug。数据：`artifacts/g5/campaign/`（15 formal + 2 smoke run
目录 + 每档 console log）。**G5 数据采集闭合，进入 G6 分析阶段。**
Status（2026-09-21）：**扩展档决策与实现——L6–L9（5e-6/1e-5/5e-5/1e-4）
加入档位表（用户选定四档方案）**。依据：L1–L5 五点幂律拟合（top-1 ~ B^0.44）
预测只补 5e-6/1e-5 会停在 ~94.7/94.5%，曲线仍近直线；四档覆盖到外推
−2.2pp（~93.1%），且 L9 处首 DUE 出现概率 ~50%（输出 binding 命中期望
0.64/campaign）。同一推导规则：构成 (397,132,132)/(794,264,264)/
(3964,1322,1321)/(7928,2643,2643)，字面值由穷举求解器产出；B>3000 时
求解器改加窗（每次 campaign 启动的重推导断言不一致即 fail-closed 拒跑，
自测交叉验证窗口=穷举，启动开销 ~52s→~1s）。执行：单卡（依据 L1–L5
三卡一致性结论），seed 7、100 trial、R 守卫与全部校验不变，L1–L5 冻结
档原样保留。V/L 稀疏驻留页采样特性补入 docs/G5_FAULT_MODEL.md §4，
扩展档理由与预测记录在 §5。待用户在 tmux 执行 L6–L9 四个 campaign。
Status（2026-09-22）：**扩展 campaign 收官——L6–L9 全部
G5_CAMPAIGN_VERIFIED，九档曲线闭合**。L6/L7/L8（GPU1）一次通过；L9
两次被超时杀死（69/100、88/100 trial），时间戳取证定位为编排器排水
循环对累积缓冲的逐 64KiB 块全量重扫（O(n²) 读侧 → 管道背压 → runner
逐 trial 打印阻塞、单 trial 耗时随序号线性增长 24→298 s）——修复为
只扫新尾部窗口（commit d817c70，等价性自测 + g3/g4/g5 自测全过），非
实验本身问题；两次失败 run 目录留档（无 summary.json，分析器自动跳过）。
L9 复跑（GPU2）：100/100 trial、2,114,300 站点、0 丢失事件，工作段
1524 s（25.4 min，修复前 88 trial >4 h）。九档 pooled 曲线（clean
95.30%）：L1–L5 精度 95.25/95.22/95.21/95.10/95.00%（平台），L6
91.88%、L7 90.68%、L8 67.84%、**L9 36.09%**；top-1 改变率 0.14%→
1.03%（L1–L5）→ 5.08/6.72/31.24/63.58%；DUE 九档全 0（L9 输出
binding 命中期望 λ=0.64，P(0)=e^-0.64≈53%——与构造性模型一致）；
r2w:w2r 从 2:1 恶化至 ~28–32:1。**跑前幂律外推（94.7/94.5/93.7/
93.1%）全部被向上击穿：拐点在 BER ≈1e-6..5e-6 之间，其上损伤强超
线性**——比"补两条点"更有价值的论文发现；预测 vs 实测对照已回写
docs/G5_FAULT_MODEL.md §5。G5 全阶段（9 档、19 campaign、1900
trial、190 万次带故障评估）数据采集完成。

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

Status（2026-09-22）：**G6-T0 完成（归因分析，零新采集）——工具
`tools/g5_faultinj/analyze_g6_attribution.py`（纯 stdlib，全部数字从 19 个
VERIFIED campaign 的 CSV 现场重算），结论全文见 `docs/G6_ANALYSIS.md`**。
四个问题（用户 9.22 提出）的回答：(1) **平台期不是漏翻转**——每档站点数
= B×trial 严格相等、0 校验违例；且 L1 档 63.1% 图像输出已数值偏移（L3 起
~100%）——翻转真实穿过全部层到达 logit，只是 |ΔP| 中位数（L5: 0.008）比
决策边界（clean 置信度 p10/p50/p90 = 0.55/0.72/0.81）小两个数量级，被边界
余量吸收；半数 top-1 改变还翻回正确（r2w:w2r 2:1）。(2) **翻转 91.8% 落在
INT8 权重 blob**（trt-internal-0，24.0 MB ≈ 25.6M 参数，驻留占比 90.79% ≈
站点占比 → 采样无偏）；权重区 330 万站点恢复全 exact = 引擎从不改写 → 一个
trial 内 1000 图用同一套腐蚀权重推理（系统性损伤载体）；scratch 区
（internal-3）翻转被引擎改写 = 软翻转瞬态；输入绑定受 V/L 锚定效应欠采样
（1.29% vs 2.28%）且被 INT8 量化吸收；输出绑定（8 B）从未命中。(3) **knee
机制 = 扰动分布越过边界分布**：权重腐蚀 L5 193 字节（0.0008%）→ L9 19,362
（0.081%），|ΔP| 中位数 0.008→0.26；低置信图（<0.5 占 7.4%）先失守，
r2w:w2r 2:1→28:1 单向化，L9 时扰动与边界同量级 → 63.6% 图翻转、精度 36%；
每字节损伤效率 0.0016→0.003 pp（逐层累积的第二重超线性）。(4) **DUE=0 是
结构性**：能致 DUE 的目标仅 8 B/26.4 MiB（L9 期望命中 0.64、P(0)≈53%，
实测 0 一致）；INT8 饱和定点算术无 NaN/Inf 传播 = 量化充当 DUE 防火墙；
XOR 故障通道本身无崩溃路径（ECC 级 DUE 在故障模型外，§8）。**总结论：
高容忍地板 + 悬崖式失效，无线性缓降带——低 BER 行为不可外推（本次跑前
幂律外推低估 L9 损失 57 pp）。**后续可选：G6-T1 站点级归因（按层/字节段
分解 top-1 改变，需 work_detail.json 逐站点 join 逐图明细）。

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

### G7：多模型 / ImageNet-1K 扩展 campaign

Status（2026-09-23）：**G7-T0 准备启动——用户五项决策已冻结**。(1) 六模型
变体：ResNet-50 / MobileNetV3-Large / EfficientNet-B0 / ViT-B/16 /
DeiT-S / Swin-T，全部 INT8 PTQ（timm ImageNet-1k 预训练权重）；(2) 校准集
= val 内 1000 类 × 1 张 = 1000 图；(3) 评测集 = 1000 类 × 10 张 = 10000 图，
与校准集不交叉（已 fail-closed 校验）；(4) 执行严格单模型串行——先
ResNet-50/Imagenet 跑出满意结果，再逐个上其余五个；但基础工作（下载 +
量化 + FP32/INT8 clean）一次性全做；(5) 六个模型 FP32 与 INT8 各跑一遍
clean（同一 10K 评测集），量化损失先行量化。**冻结不变**：故障模型
（SBU 60%/MCU 40%、空间形态、构成规则）、BER 语义（B = round(BER×R_bits)，
R 改为 per-workload 重推导，L1–L9 九档 BER 值跨模型一致以便对比）、门控/
快照/驻留守卫/翻转/恢复/分类/校验全套协议、seed 7、100 trial。已就绪：
ImageNet val（本机 /data1/luojx/datasets/imagenet1k，50 000 图全部与官方
val_map 交叉核验 0 错）；split 构建器
`tools/g7_prep/build_imagenet_splits.py`（seed 7：calib 1000 + eval 10000，
不交叉，manifest 落盘）；六模型下载器
`tools/g7_prep/download_models.py`（含 default_cfg 预处理元数据，runner
预处理将据此参数化，杜绝第二事实源）。构建链已找到并落地：vit_fault 环境
内装有 `tensorrt_bindings` 8.6.1（此前"本机无 TensorRT"结论有误——只搜了
`tensorrt` 模块名），配合本机 `/data1/luojx/REMU/.local/deps/` 下 tensorrt
8.6.1 运行库 + cuDNN 8.9.7（stage13 同款 LD_LIBRARY_PATH 接线，包装在
`build_g7_engines.sh`/`eval_g7_clean.sh`）。新工具：`export_g7_onnx.py`
（三 binding `data/prob/index` 契约、opset 17、INT64_MAX slice 尾哨改写，
per-model 事实全部来自 model_meta.json）、`build_g7_int8_engine.py`
（`IInt8EntropyCalibrator2` batch=1、G7 千图校准集、校准缓存按
onnx+calib+mean/std+插值 哈希、构建后零输入冒烟；stage13 先验：当年七个
模型 INT8 PTQ 全部健康，vit_b16 95.6% 无 Transformer 塌陷）、
`eval_g7_clean.py`（FP32 torch 与 INT8 TRT 同一预处理同一 10K 集，
INT8 双遍逐位一致验收，量化损失先行量化）。待办：其余五模型
（下载完成后 ONNX 导出→INT8 构建→clean 评测一次跑完）、G5 runner 的
per-workload 参数化（预处理 mean/std、engine 路径、评测 split、R 表）。**实验前预设假设（实验裁决）**：① 各架构
"耐受地板"（knee 位置）与塌方斜率不同——大稠密 GEMM 权重块的 Transformer
同 BER 下单字节腐蚀占比更小，但 LayerNorm/位置编码小参数区可能脆弱；
② MobileNetV3 depthwise 层每 kernel 字节极少，单字节腐蚀相对影响更大；
③ INT8 饱和算术的 DUE 防火墙在 Transformer 上同样成立（预期 DUE 全 0）；
④ r2w:w2r 不对称演化跨架构一致或分化。产出：六条精度-BER 曲线一张图
（各自 clean 归一 + 绝对值两版）、knee 对比表、第二横轴（权重字节腐蚀
比例）。

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
