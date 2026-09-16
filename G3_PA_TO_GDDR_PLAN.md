# 快速实现 RTX 4090 PA-to-GDDR 映射行动计划书 (G3 Phase)

> **目标**：利用现有时序侧信道（Timing Side-Channel）开源工具，逆向 RTX 4090 的 Memory Controller (MC) 寻址哈希函数，实现 `GPU PA → GDDR6X Coordinate (Channel/Bank/Row/Column)` 的精准映射。
> **前提**：已实现 `Bit ↔ VA ↔ PA` 的映射，测试程序能够自由获取并控制目标数据的物理地址 (PA)。

---

## 阶段一：工具白嫖与环境适配 (Tooling & Setup)
**核心思想：绝不自己手写底层测速代码，直接复用安全界已经趟过坑的微基准测试（Microbenchmarking）工具。**

1. **提取核心测速 Kernel**
   * 从开源项目 `heelsec/GDDRHammer` 或类似项目（如 `Fractional-GPUs`）中，剥离其用于 DRAM 时序分析的 CUDA Kernel。
   * **重点保留**：使用了 `clock64()` 或 `globaltimer` 的高精度延迟测量代码。
   * **重点保留**：绕过 L1/L2 Cache 的访存指令（通常通过特殊的 PTX 汇编指令如 `ld.global.cg` 或特定的步长冲刷策略，确保每次访问直接落到 GDDR）。

2. **改造数据输入接口**
   * 修改开源工具的输入接口，使其能够直接接收你已知的 PA 列表，而不是让工具自己去随机生成地址。

---

## 阶段二：时序数据采集 (Data Collection)
**核心思想：利用 Row Conflict（行冲突）会带来显著延迟的物理特性，找出哪些 PA 被映射到了同一个 Bank。**

1. **构建探测地址池**
   * 固定一个基准 PA（Base_PA）。
   * 按位翻转（Bit-flip）生成测试 PA 列表：例如，依次翻转 Base_PA 的第 10 位到第 25 位，生成一系列待测 PA。
   
2. **执行“交替乒乓”测速**
   * 运行第一阶段提取的 Kernel，让 GPU 不断交替读取 `Base_PA` 和 `测试_PA`。
   * 记录每次交替读取的平均时钟周期延迟（Latency）。

3. **数据打标分类**
   * **Row Hit（低延迟）**：说明 `测试_PA` 和 `Base_PA` 映射到了 **同一 Bank 的同一 Row**。
   * **Bank 并行（中延迟）**：说明 `测试_PA` 和 `Base_PA` 映射到了 **不同 Bank**。
   * **Row Conflict（高延迟）**：说明 `测试_PA` 和 `Base_PA` 映射到了 **同一 Bank 的不同 Row**。
   * **输出产物**：一个包含数万组 PA 对及其对应物理关系（同 Bank / 不同 Bank）的数据集。

---

## 阶段三：哈希规则求解 (Reverse Engineering)
**核心思想：将搜集到的同 Bank 地址规律，转化为线性 XOR 方程组并求解。**

1. **分离线性位（Row / Column）**
   * 观察未引发 Bank 冲突的连续地址段。低位（通常是 Bit 0-5）通常是 Byte/Burst 偏移；极高位（不参与 Bank 交织的连续位）通常直接是 Row 地址。
   * 确定 Row 和 Column 占用的具体比特位范围。

2. **推导 Bank 映射 XOR 哈希函数**
   * 现代 GPU 的 Bank 位通常由 `低位段 XOR 高位段` 组成（例如 `Bank_Bit_0 = PA[8] ^ PA[16] ^ PA[18]`）。
   * 编写一个 Python 脚本，使用 **Z3 约束求解器 (Z3 Theorem Prover)** 或者简单的线性代数求解模块（如 Gaussian elimination for GF(2)）。
   * 将阶段二收集到的“同 Bank PA 对”（意味着它们经过哈希函数计算后得出相同的 Bank ID）输入求解器，自动解出 RTX 4090 的 XOR 映射函数。

---

## 阶段四：验证与固化 (Verification & Integration)
**核心思想：通过预测来验证规则的绝对正确性，并将其封装为项目 API。**

1. **预测验证模型**
   * 在 Python 端根据你解出的规则，随机生成 100 对“理论上”会产生 Row Conflict 的 PA 对，以及 100 对不会冲突的 PA 对。
   * 将这 200 对地址送入 CUDA Kernel 实测。如果实测延迟的“高/低”分布与你的理论预测达到 100% 吻合，说明映射规则破解成功。

2. **跨卡稳定性确认 (按原计划书要求)**
   * 在实验室的 3 张不同 RTX 4090 上运行同一套测试脚本，确认同型号架构的哈希规则是否完全一致（大概率一致，但必须验证）。

3. **固化 API 输出**
   * 用 C++ 或 Python 封装成标准映射函数，集成到你的故障注入框架中：
   ```python
   def get_gddr_coordinate(gpu_pa):
       # 依据求解出的规则进行位运算
       channel = f_channel(gpu_pa)
       bank = f_bank(gpu_pa)
       row = f_row(gpu_pa)
       col = f_col(gpu_pa)
       return (channel, bank, row, col)