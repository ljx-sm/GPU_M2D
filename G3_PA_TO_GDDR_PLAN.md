# Rapid RTX 4090 PA-to-GDDR Mapping Action Plan (G3 Phase)

> **Outcome (added 2026-09-29): this plan is historical.** G3 closed on
> 2026-09-17 by a different route than stage three below. What was
> actually done, stage by stage:
>
> - **Stage one.** The reference tools (GPUHammer / GDDRHammer / GeForge)
>   carry no license. They were studied only; no code was copied, and the
>   project wrote its own PA-annotated timing probe (`tools/g3_probe/`;
>   survey in `docs/G3_SURVEY.md`).
> - **Stage two.** Ran as planned (S1–S4b).
> - **Stage three: the closed-form XOR/GF(2) hash solve FAILED.** The
>   PA→bank function is non-linear and mixes nearly all address bits,
>   which GeForge footnote 1 independently confirms. The project pivoted
>   to a GeForge-style **empirical mapping table (EMT)** built from
>   measured pairs.
> - **Stage four.** Realized as the EMT validation gates R-a…R-e (all
>   PASS). The structure was confirmed identical on all three cards
>   (table v4), and the "API" is the query tool `tools/g3_probe/query_table.py`
>   with provenance labels rather than a `get_gddr_coordinate()` formula.
>   Column and DQ coordinates are not recoverable by timing: column
>   adjacency is folded into "same row, random column", and DQ is
>   unsupported.
>
> Full record: the G3 status log in `GPU_SIDE_REMU_RESEARCH_PLAN.md` and
> `tools/g3_probe/README.md`. The original plan follows, translated from
> Chinese.

> **Goal**: use existing open-source timing side-channel tools to
> reverse-engineer the RTX 4090 Memory Controller (MC) addressing hash
> function, giving a precise `GPU PA → GDDR6X Coordinate
> (Channel/Bank/Row/Column)` mapping.
> **Precondition**: the `Bit ↔ VA ↔ PA` mapping already exists, and the
> test program can freely obtain and control the physical address (PA) of
> the target data.

---

## Stage one: reuse the tooling and adapt the environment (Tooling & Setup)

**Core idea: never hand-write low-level timing code. Reuse the
microbenchmarking tools the security community has already debugged.**

1. **Extract the core timing kernel**
   * Strip the CUDA kernel used for DRAM timing analysis out of an
     open-source project such as `heelsec/GDDRHammer` (or similar projects,
     e.g. `Fractional-GPUs`).
   * **Keep**: the high-precision latency measurement based on `clock64()`
     or `globaltimer`.
   * **Keep**: the memory-access instructions that bypass L1/L2 cache
     (usually special PTX instructions such as `ld.global.cg`, or a
     specific stride-flush strategy, so each access really reaches GDDR).

2. **Rework the data-input interface**
   * Change the tool's input interface so it takes our known PA list
     directly instead of generating random addresses itself.

---

## Stage two: timing data collection (Data Collection)

**Core idea: exploit the physical fact that a row conflict adds
significant latency to find which PAs map to the same bank.**

1. **Build the probe address pool**
   * Fix a base PA (Base_PA).
   * Generate test PAs by flipping bits: e.g. flip bits 10 through 25 of
     Base_PA one at a time.

2. **Run alternating "ping-pong" timing**
   * Run the stage-one kernel so the GPU alternately reads `Base_PA` and
     `Test_PA`.
   * Record the mean clock-cycle latency of each alternating read.

3. **Label the data**
   * **Row hit (low latency)**: `Test_PA` and `Base_PA` map to the **same
     row of the same bank**.
   * **Bank parallel (medium latency)**: `Test_PA` and `Base_PA` map to
     **different banks**.
   * **Row conflict (high latency)**: `Test_PA` and `Base_PA` map to
     **different rows of the same bank**.
   * **Output**: a dataset of tens of thousands of PA pairs with their
     physical relation (same bank / different bank).

---

## Stage three: solve the hash rule (Reverse Engineering)

**Core idea: turn the collected same-bank address regularities into a
linear XOR system and solve it.**

1. **Separate the linear bits (row / column)**
   * Look at contiguous address ranges that cause no bank conflict. Low
     bits (usually bits 0–5) are typically the byte/burst offset; very high
     bits (contiguous bits that do not take part in bank interleaving) are
     typically the row address directly.
   * Pin down exactly which bit ranges row and column occupy.

2. **Derive the bank-mapping XOR hash**
   * Modern GPU bank bits are usually `low-bit field XOR high-bit field`
     (e.g. `Bank_Bit_0 = PA[8] ^ PA[16] ^ PA[18]`).
   * Write a Python script using the **Z3 theorem prover**, or a simple
     linear-algebra solver (Gaussian elimination over GF(2)).
   * Feed the stage-two "same-bank PA pairs" (pairs whose hashed bank IDs
     are equal) into the solver to recover the RTX 4090 XOR mapping
     function automatically.

---

## Stage four: verification and consolidation (Verification & Integration)

**Core idea: prove the rule by prediction, then wrap it as a project API.**

1. **Predictive verification**
   * From the solved rule, generate in Python 100 random PA pairs that
     "should" row-conflict and 100 pairs that should not.
   * Measure these 200 pairs with the CUDA kernel. If the measured
     high/low latency split matches the prediction 100%, the mapping rule
     is cracked.

2. **Cross-card stability (as the main plan requires)**
   * Run the same test scripts on the lab's three RTX 4090 cards and
     confirm the hash rule is identical across cards of the same model
     (very likely, but it must be verified).

3. **Consolidate the API**
   * Wrap the rule as a standard mapping function in C++ or Python and
     integrate it into the fault-injection framework:
   ```python
   def get_gddr_coordinate(gpu_pa):
       # bit operations according to the solved rule
       channel = f_channel(gpu_pa)
       bank = f_bank(gpu_pa)
       row = f_row(gpu_pa)
       col = f_col(gpu_pa)
       return (channel, bank, row, col)
   ```
