# GPU accelerated-radiation (beam) tests reporting per-bit SRAM and DRAM/GDDR upset rates (for rho = SRAM/DRAM per-bit rate)

Scope note: the goal is rho = per-bit upset rate of on-chip SRAM (ideally L2) / per-bit upset rate of GPU DRAM/GDDR, for an RTX 4090 (AD102, TSMC 4N, 72 MiB L2, Micron GDDR6X, ECC off). Bottom line up front: **no published GPU beam campaign that I could find reports absolute per-bit cross sections for both an on-chip SRAM structure and the DRAM/GDDR of the same device.** The main GPU beam groups (Rech / UFRGS / NVIDIA co-authors) deliberately keep DRAM out of the beam or neutralize it (ECC on or TMR). They report SRAM numbers only *normalized* to hide NVIDIA-confidential absolute values. The only GPU-DRAM beam paper (NVIDIA, MICRO'21, HBM2) reports relative rates and error patterns only. The only same-device SRAM-vs-DRAM split I found is **field** data (ORNL Titan, K20X, ECC on), not beam data. Any rho estimate therefore has to combine sources, and every number below is labelled measured, normalized, or field.

---

## Q1. Which GPU beam tests measured both SRAM structures and DRAM/GDDR per bit?

### Takeaway
None that I could find. Rech/NVIDIA GPU beam campaigns (Fermi C2050, Kepler K20/K40, Volta Titan V/V100) measured the register file (RF), L2, and L1/shared memory per bit with micro-benchmarks, ECC off. They **explicitly excluded DRAM** by aiming the beam spot only at the die, or by using ECC/TMR on HBM2. NVIDIA's HBM2 beam test (Sullivan et al., MICRO'21) characterized only DRAM. The only same-device SRAM-vs-DRAM per-structure breakdown I found is field data from the Titan supercomputer (Tiwari et al., HPCA'15). The same paper also reports SRAM-only beam data.

### Cited Findings

**Paper catalog (per-paper records)**

1. **Rech, Carro, Wang, Tsai, Hari, Keckler, "Measuring the Radiation Reliability of SRAM Structures in GPUs Designed for HPC," IEEE SELSE 2014** (UFRGS + NVIDIA) — [PDF](https://www.cs.utexas.edu/~skeckler/pubs/SELSE_2014_Reliability.pdf)
   - Devices: NVIDIA Tesla C2050 (Fermi, 40 nm; 768 kB L2, 896 kB L1 total, 1.75 MB RF) and K20 (Kepler GK110, 28 nm; 1.25 MB L2 per the paper, 832 kB L1, 3.25 MB RF); boards carry GDDR5.
   - Facility: LANSCE (Los Alamos) high-energy neutrons, flux ~5×10^5 n/(cm²·s) above 10 MeV. Beam spot 2 in. plus 1 in. penumbra, "without directly affecting nearby board power control circuitry and **DDR memories**". DRAM was not irradiated.
   - ECC turned **OFF** "to access the raw radiation sensitivity of GPU SRAM structures".
   - Per-bit cross sections, **normalized** to the K20 L2 with the FF pattern (absolute values not disclosed), ±10% (K20) and ±12% (C2050):
     - K20 L2: 00 = 1.44, FF = 1.00, AA = 1.36
     - C2050 L2: 00 = 3.09, FF = 2.29
     - K20 RF: 00 = 1.17, FF = 1.13, AA = 1.11
     - C2050 RF: 00 = 2.86, FF = 2.64
   - MBU (multiple bits in the same row/word) share of RF errors: K20 4% (00), 6% (FF), 1% (AA); C2050 1% (00 and FF).
   - MCU (multiple cells, different rows) share of RF errors: K20 19% (00), 18% (FF), 17% (AA); C2050 not statistically significant.
   - No MBU of more than 2 bits per row was observed.
   - L2 bits holding 0 are more sensitive than bits holding 1.
   - No DRAM numbers.
2. **Tiwari, Gupta, Rogers, Maxwell, Rech, Vazhkudai, Oliveira, Londo, DeBardeleben, Navaux, Carro, Bland, "Understanding GPU Errors on Large-scale HPC Systems and the Implications for System Design and Operation," IEEE HPCA 2015** (ORNL, LANL, UFRGS, Cray) — [OSTI PDF](https://www.osti.gov/servlets/purl/1185857)
   - **Field part:** Titan supercomputer, 18,688 K20X GPUs (28 nm; 1.5 MB L2; 6 GB GDDR5). ECC (SECDED) is **ON** on device memory, L2, L1, RF, shared memory, and instruction cache. Single-bit errors (SBEs) were collected via nvidia-smi from Feb 2013 to Aug 2014.
     - Only 899 of 18,688 cards ever logged an SBE.
     - About 98% of all SBEs came from 10 cards.
     - Across all cards, 98% of SBEs were in L2 and about 2% in device memory (Fig. 12). Top-10 cards: 99% in L2.
     - Excluding the top-10 cards, "the device memory is the structure where most of the SBEs occur (96% of all SBEs)" per the text. Fig. 12 shows 94% / 6% for this group.
     - The authors state that SBE fractions are "not proportional to the respective structure sizes".
     - **Internal inconsistency:** "Observation 6" says the cards with the most SBEs have them in device memory, which contradicts their own Fig. 12 and text (top-10 cards: 99% in L2).
   - **Beam part:** the same K20 and C2050 L2/RF per-bit data as SELSE'14 (Fig. 14, normalized to K20 L2 1s pattern, CI below 12%). K20 DBE share is about 6%, with no triple-bit corruptions.
3. **Fernandes dos Santos, "Understanding and Improving GPUs' Reliability Combining Beam Experiments with Fault Simulation," PhD thesis, UFRGS 2022** — [PDF](https://lume.ufrgs.br/bitstream/handle/10183/234971/001136966.pdf). Related papers: **Fernandes dos Santos, Hari, Basso, Carro, Rech, "Demystifying GPU Reliability: Comparing and Combining Beam Experiments, Fault Simulation, and Profiling," IEEE IPDPS 2021** — [ResearchGate](https://www.researchgate.net/publication/352810052_Demystifying_GPU_Reliability_Comparing_and_Combining_Beam_Experiments_Fault_Simulation_and_Profiling); and an IEEE journal version — [IEEE Xplore 10174206](https://ieeexplore.ieee.org/document/10174206).
   - Devices:
     - Tesla K40c (Kepler, TSMC 28 nm planar; 1.5 MB L2; 6 GB GDDR5; SECDED on RF, shared memory, and caches), tested with ECC ON and OFF.
     - Titan V and Tesla V100 (Volta, "TSMC FinFET 12nm"; 12 GB and 16 GB HBM2). Only the V100 has SECDED on main memory.
   - Facilities: ChipIR (RAL, UK) at about 3.5×10^6 n/(cm²·s), and LANSCE.
   - Micro-benchmarks:
     - **RF** and **SHARED** write a known pattern (255 registers per thread; 48 KB static shared memory), wait about 1 s, then read back. The thesis says L1 "has the same technology as Shared Memory".
     - **LDST** exercises global memory with ECC enabled.
   - FIT rates are **normalized by an undisclosed constant β** "not to reveal business-sensitive data".
   - RF is "the most critical memory resource at the SM level".
   - MBU rate: under 2% for RF and under 0.9% for SHARED.
   - Kepler RF (28 nm planar) has "approximately an order of magnitude higher error rate than Volta RF (16nm FinFET)". The thesis calls Volta both 12 nm and 16 nm FinFET; TSMC 12FFN is a 16 nm-class derivative.
   - **DRAM excluded:** "this evaluation considers errors occurring in the GPU core, not in the main memory… For Kepler, the beam spot is sufficiently small (2cm of diameter) to not hit the onboard DDR… For Volta… all the global memory accesses are made through Triple Modular Redundancy". Main-memory reliability was "already… extensively studied".
4. **Fernandes dos Santos & Rech, "Can GPU performance increase faster than the code error rate?," J. Supercomputing 2024**, doi:10.1007/s11227-024-06119-4 — [PDF](https://d-nb.info/1334458731/34)
   - Devices: GTX480 (Fermi, 40 nm, GDDR5), K20 (Kepler, 28 nm, GDDR5), and a Volta (12 nm, HBM2).
   - Facilities: ChipIR (about 10^6 n/(cm²·s)) and LANSCE.
   - Only relative FIT rates are reported, "not showing the absolute FIT rate of GPUs". GPU error rate "ranges between 10s to 1000s of FITs" depending on device and code.
   - Again "Only the GPU core is irradiated, that is, DDR… is outside of the beam spot… For Volta GPUs… we either turn the ECC ON or triplicate the data in the main memory."
5. **Sullivan, Saxena, O'Connor, Lee, Racunas, Hukerikar, Tsai, Hari, Keckler (NVIDIA), "Characterizing and Mitigating Soft Errors in GPU DRAM," MICRO-54, 2021**, doi:10.1145/3466752.3480111 — [ACM](https://dl.acm.org/doi/10.1145/3466752.3480111), [Semantic Scholar](https://www.semanticscholar.org/paper/3e6495ddcb72c1a8803177d3395a39289218877d)
   - Device: HBM2 on a "compute-class GPU" (model not confirmed from accessible text).
   - Particle: high-energy neutron beam.
   - Method: a CUDA micro-benchmark writes a known pattern to every entry and reads back repeatedly.
   - Found beam-induced **intermittent errors from cell damage** that must be filtered from beam data.
   - Reports **relative** error rates and patterns. Multi-bit errors show locality attributed to the HBM2 structure.
   - Proposed ECC cuts SDC risk by up to 5 orders of magnitude versus SEC-DED and DUEs by up to 7.87×.
   - The full text was paywalled (HTTP 403), so I could not verify per-bit numbers.
   - Search-indexed summaries attribute "approximately 31.5% of SEUs in device memory affect multiple bits in at least one word" and "most multi-bit errors (~75%) being byte-aligned" to this work — [academia.edu listing](https://www.academia.edu/70184963/Characterizing_and_Mitigating_Soft_Errors_in_GPU_DRAM), [NVIDIA patent US 12,149,259](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12149259). I treat these as plausible but **unverified against the paper text**.
   - **Caution:** a search summary also attributed "up to 79% difference in bit error rate across HBM2 channels" to this work. That figure appears to come from the HBM2 *RowHammer* study (Olgun et al.) — [arXiv 2305.17918](https://arxiv.org/pdf/2305.17918) — not from neutron data. Do not use it.
6. **Rech et al. / Hari et al., "Estimating Silent Data Corruption Rates Using a Two-Level Model," arXiv 2005.01445 (NVIDIA + UFRGS)** — [arXiv](https://arxiv.org/pdf/2005.01445)
   - Device: Kepler GPU with 30 Mbit RF, 7.86 Mbit L1/shared, and 12 Mbit L2 (per the paper). RF, shared memory, caches, and DRAM were **ECC-protected** during beam.
   - Only **relative** FIT rates, normalized to the IADD micro-benchmark. No per-bit memory numbers.
7. **de Oliveira, Pilla, Santini, Rech, "Evaluation and Mitigation of Radiation-Induced Soft Errors in Graphics Processing Units," IEEE Trans. Computers 65(3):791–804, 2016**, doi:10.1109/TC.2015.2444855 — [IEEE Xplore](https://ieeexplore.ieee.org/document/7122902/)
   - Per the abstract, it evaluates "neutron sensitivity of modern GPUs memory structures, highlighting pattern dependence and multiple error occurrences". The K20A tested has no ECC (28 nm) — [ResearchGate](https://www.researchgate.net/publication/282621498_Evaluation_and_Mitigation_of_Radiation-Induced_Soft_Errors_in_Graphics_Processing_Units).
   - Full text paywalled, so per-bit numbers are not verified.
8. **Sabena, Sonza Reorda, Sterpone, Rech, Carro, "Evaluating the radiation sensitivity of GPGPU caches: New algorithms and experimental results," Microelectronics Reliability 2014** — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0026271414001516)
   - Cross section and FIT results for shared memory, L1, and L2 of a commercial (Fermi-era) GPU under neutrons, per the indexed abstract.
   - Full text not accessible (HTTP 403), so the numbers are unverified.
9. **Oliveira et al., "High-Energy vs. Thermal Neutron Contribution to Processor and Memory Error Rates," IEEE TNS 2020** — [IEEE Xplore 8975983](https://ieeexplore.ieee.org/document/8975983/), [ResearchGate](https://www.researchgate.net/publication/338927054_High-Energy_vs_Thermal_Neutron_Contribution_to_Processor_and_Memory_Error_Rates)
   - Devices: AMD APU, three NVIDIA GPUs (K20, Titan X, Titan V per indexed summaries), Intel Xeon Phi, Zynq FPGA, and **DDR3 and DDR4 DIMMs** (not GPU GDDR), tested separately under thermal and high-energy neutrons at ISIS ChipIR and ROTAX.
   - GPU results are application-level error rates, not per-bit per-structure. The DDR parts are stand-alone memories. Full text not accessed.
10. **Bustos et al., "Response of HPC hardware to neutron radiation at the dawn of exascale," J. Supercomputing 79(12), 2023**, doi:10.1007/s11227-023-05199-y (CC-BY, ORNL co-authors) — [Springer](https://link.springer.com/article/10.1007/s11227-023-05199-y), [ORNL](https://impact.ornl.gov/en/publications/response-of-hpc-hardware-to-neutron-radiation-at-the-dawn-of-exas/)
    - Devices: A100 40 GB (Ampere), V100 16 GB (Volta), and T4 (Turing), plus Xeon and EPYC CPUs. Sources: ²⁵²Cf and ²⁴¹Am-Be radioactive neutron sources (not a spallation beam).
    - Per the indexed summaries: A100 is most sensitive, then T4, then V100. On the A100 only correctable errors (CE) and single-event indications were seen. CE MTBF was about 20–30 min close to the Cf source.
    - Device-level only; no per-bit SRAM-vs-HBM split found (full text download was blocked by a redirect).
11. **León, Belloch, Entrena et al., "Analyzing the Influence of Memory and Workload on the Reliability of GPUs Under Neutron Radiation," IEEE TNS, Aug 2024** — [IEEE Xplore 10496949](https://ieeexplore.ieee.org/document/10496949/), [ResearchGate](https://www.researchgate.net/publication/379775253_Analysing_the_influence_of_memory_and_workload_on_the_reliability_of_GPUs_under_neutron_radiation)
    - Device: Jetson Nano (Tegra X1, Maxwell).
    - Compares micro-benchmark variants using different CUDA memory types. Per the abstract, memory type "has a significant influence".
    - No per-bit SRAM/DRAM cross sections found in accessible text.

### Inferences
- The absence of same-campaign SRAM+DRAM per-bit GPU beam data is structural, not an oversight. The two groups that produce most GPU beam data (UFRGS/Rech and NVIDIA) exclude DRAM to isolate the GPU core, and they normalize SRAM numbers to protect NVIDIA's confidential absolute rates. Any rho for an RTX 4090 will have to combine (a) GPU-SRAM beam data or vendor-neutral SRAM-per-node data with (b) stand-alone DRAM beam or field data.
- The Titan field split (Tiwari et al.) is the only same-device L2-vs-GDDR5 per-bit comparison I found. It mixes radiation with weak or faulty cells, has ECC on, and is dominated by a few outlier cards, so it gives bounds rather than a radiation-only rho (arithmetic in Q2).

### Gaps
- No accessible full text for Sullivan MICRO'21 (per-bit HBM2 numbers, if any), Oliveira TC'16, or Sabena MR'14. Absolute per-bit SRAM numbers from those papers, if they exist, could not be verified.
- No LANL (Quinn / Blanchard / DeBardeleben) or Northeastern (Previlon / Kaeli) GPU beam paper with per-bit memory numbers was found within the search budget.

---

## Q2. Measured per-bit cross sections for L2/L1/RF vs DRAM/GDDR, and the resulting SRAM:DRAM ratios

### Takeaway
GPU SRAM per-bit data exist only in normalized form. On K20 (28 nm), L2 and RF per-bit sensitivities are within about 10–20% of each other, and Fermi (40 nm) cells are about 2.1–2.4× more sensitive per bit than K20. No beam-measured GPU-DRAM per-bit value exists to divide by. The only same-device ratio I can compute comes from Titan field SBE logs (K20X, ECC on): rho(L2/GDDR5 per bit) ≈ 2×10^5 if outlier cards are included, and **≤ about 170–260 if the 10 outlier cards are excluded**. Neither number is a clean radiation-only ratio.

### Cited Findings
- **K20 L2 vs RF per bit (beam, ECC off, LANSCE):** L2 = 1.44 / 1.00 / 1.36 and RF = 1.17 / 1.13 / 1.11 for patterns 00 / FF / AA, in units of the K20 L2 FF cross section — [Rech et al. SELSE'14](https://www.cs.utexas.edu/~skeckler/pubs/SELSE_2014_Reliability.pdf)
- **Fermi C2050 vs Kepler K20 per bit:** C2050 L2 = 3.09 (00) and 2.29 (FF); C2050 RF = 2.86 (00) and 2.64 (FF), in the same units — [Rech et al. SELSE'14](https://www.cs.utexas.edu/~skeckler/pubs/SELSE_2014_Reliability.pdf)
- **Kepler K40 RF vs Volta RF:** Kepler RF has about an order of magnitude higher per-byte error rate than Volta RF (planar vs FinFET). RF is the most sensitive SM-level memory, above shared memory/L1 per byte (normalized values only) — [Fernandes dos Santos thesis, UFRGS 2022](https://lume.ufrgs.br/bitstream/handle/10183/234971/001136966.pdf)
- **Titan field SBE split (K20X, ECC on, 1.5 MB L2 vs 6 GB GDDR5):**
  - All cards: L2 98%, device memory about 2%.
  - Excluding the top-10 cards: device memory 96% (text) or 94% (figure).
  - [Tiwari et al. HPCA'15](https://www.osti.gov/servlets/purl/1185857)
- **Absolute GPU error rates (for scale only):** GPU code-level FIT ranges from "10s to 1000s of FITs" depending on device and code — [Fernandes dos Santos & Rech 2024](https://d-nb.info/1334458731/34)

**Arithmetic: ratios from Rech et al. SELSE'14 (K20, 28 nm, beam, ECC off)**
- RF/L2 per bit:
  - FF: 1.13/1.00 = **1.13**
  - 00: 1.17/1.44 = **0.81**
  - AA: 1.11/1.36 = **0.82**
  - Pattern-averaged: mean RF (1.17+1.13+1.11)/3 = 1.137; mean L2 (1.44+1.00+1.36)/3 = 1.267; ratio 1.137/1.267 = **0.90**
- C2050/K20 per bit:
  - L2: 3.09/1.44 = **2.15** (00) and 2.29/1.00 = **2.29** (FF)
  - RF: 2.86/1.17 = **2.44** (00) and 2.64/1.13 = **2.34** (FF)
  - The paper's text says "approximately 3 times", but its own tables give about 2.1–2.4×. **Flagged inconsistency.**

**Arithmetic: L2/GDDR5 per-bit ratio from Titan field data (Tiwari et al. HPCA'15)**
- Bits: L2 = 1.5 MiB × 8 = 12,582,912 bits (12.58 Mbit). GDDR5 = 6 GiB × 8 = 51,539,607,552 bits (51,540 Mbit). Capacity ratio DRAM/L2 = **4096**.
- Formula: rho = (f_L2 / N_L2) / (f_DRAM / N_DRAM) = (f_L2 / f_DRAM) × 4096.
- (a) All cards (f_L2 = 0.98, f_DRAM = 0.02): rho = 49 × 4096 = **≈ 2.0×10^5**. This is dominated by 10 cards, which points to weak or defective L2 cells re-reporting, not independent particle strikes. **Not usable as a radiation rho.**
- (b) Excluding the top-10 cards:
  - Figure values (f_DRAM = 0.94, everything else 0.06, all assigned to L2): rho ≤ (0.06/0.94) × 4096 = **≤ 261**.
  - Text value (f_DRAM = 0.96, f_L2 ≤ 0.04): rho ≤ (0.04/0.96) × 4096 = **≤ 171**.
  - These are upper bounds: the non-DRAM remainder also includes L1, RF, and texture SBEs.
  - Caveats: ECC is on, so these are corrected SBE counts. Logging is via nvidia-smi snapshots, which the paper says under-reports some events. Field DRAM SBEs include hard and intermittent faults.

### Inferences
- If the Titan field bound (rho ≲ 170–260 at 28 nm, L2 vs GDDR5) holds even roughly, a modern GPU's L2 would be about two orders of magnitude more upset-prone per bit than its GDDR. That fits the general trend that DRAM per-bit SER has fallen generation over generation while SRAM per-bit SER fell more slowly. This is inference; it is not supported by any same-campaign beam measurement.
- SRAM per-bit sensitivity drops strongly with node and transistor type: about 2.2× from 40 nm to 28 nm planar, and about 10× from 28 nm planar to 16/12 nm FinFET RF. Ada AD102 (TSMC 4N FinFET) per-bit SRAM sensitivity is therefore likely below Volta's. No GPU beam data exist at 4N, so any rho for an RTX 4090 extrapolates the SRAM side by at least 2–3 nodes.
- On K20, L2 and RF have nearly equal per-bit sensitivity (ratio about 0.8–1.1). If an RF or shared-memory number is available but an L2 number is not, using it as an L2 proxy at the same node introduces only about ±20% error, at least at 28 nm.

### Gaps
- No absolute per-bit cross section (cm²/bit) or FIT/Mbit for any NVIDIA GPU SRAM structure appears in the accessible literature. All are normalized (SELSE'14, HPCA'15, thesis, IPDPS'21).
- No beam-measured per-bit GDDR5, GDDR6, GDDR6X, or HBM2 cross section from a GPU board was verified. Sullivan MICRO'21 may contain relative HBM2 rates but is paywalled.
- No L1-specific beam number separate from shared memory (the thesis treats L1 as the same technology as shared memory).

---

## Q3. Tests isolating DRAM (memory-only micro-benchmarks) vs caches (cache-resident micro-benchmarks) on the same GPU

### Takeaway
Cache-, RF-, and shared-memory-resident micro-benchmarks are well established (Rech/NVIDIA, Kepler and Volta). A DRAM-only micro-benchmark exists in NVIDIA's HBM2 test (Sullivan MICRO'21). They were never combined on the same device with both structures irradiated and unprotected.

### Cited Findings
- SRAM micro-benchmarks:
  - Each thread fills a pre-assigned portion of the L2 or RF with a pattern (00 / FF / AA), errors accumulate under beam, and the data are read back. Patterns and timing were chosen to detect MBU/MCU.
  - The DRAM was outside the 2-inch beam spot.
  - [Rech et al. SELSE'14](https://www.cs.utexas.edu/~skeckler/pubs/SELSE_2014_Reliability.pdf)
- RF, SHARED, and LDST micro-benchmarks:
  - RF and SHARED: write a pattern, expose for about 1 s, read back. The minimum number of threads is used to avoid errors in other resources.
  - LDST (global-memory load/store, 2 GB) runs **with ECC enabled**, so it does not measure raw DRAM upsets.
  - [Fernandes dos Santos thesis](https://lume.ufrgs.br/bitstream/handle/10183/234971/001136966.pdf)
- DRAM micro-benchmark: a CUDA kernel "writes a known pattern to every memory entry and reads back the memory repeatedly, recording any mismatches" on HBM2. Beam-induced cell damage produces intermittent errors that must be filtered — [Sullivan et al. MICRO'21](https://dl.acm.org/doi/10.1145/3466752.3480111)
- Two-level SDC model: per-instruction micro-benchmarks with memories ECC-protected (RF 30 Mbit, L1/shared 7.86 Mbit, L2 12 Mbit on the Kepler tested) — [Hari/Rech et al. arXiv 2005.01445](https://arxiv.org/pdf/2005.01445)

### Inferences
- A natural design for an RTX 4090 (ECC off) is to run an L2-resident pattern kernel and a GDDR-resident pattern kernel in the same session. That would be the first same-device SRAM:DRAM per-bit ratio, but it needs a beam spot covering both die and GDDR, or two exposures. Sullivan et al.'s warning about beam-induced intermittent DRAM cell damage implies the DRAM-side result needs filtering for repeat-address errors.

### Gaps
- No published same-device "cache-resident vs DRAM-resident" beam campaign was found.

---

## Q4. Data on GDDR6/GDDR6X specifically, or on Ada/Ampere consumer GPUs

### Takeaway
I found no published beam data for GDDR6 or GDDR6X, and no beam test of any Ada (RTX 40xx) GPU. The nearest modern data are:
- **Device-level** radioactive-source tests of A100, V100, and T4, with no per-bit split.
- **HBM2** beam patterns from NVIDIA (relative only).
- Ampere and Hopper field studies (outside this beam-focused scope).

### Cited Findings
- A100 (Ampere), V100 (Volta), and T4 (Turing) were irradiated with ²⁵²Cf and Am-Be neutron sources.
  - Per the indexed summary, A100 was most sensitive, then T4, then V100.
  - Only correctable errors were logged on A100; CE MTBF was about 20–30 min near the Cf source.
  - No per-structure per-bit numbers.
  - [Bustos et al. J. Supercomputing 2023](https://link.springer.com/article/10.1007/s11227-023-05199-y)
- Targeted searches for RTX 4090, RTX 3080, or Ada neutron beam tests returned no publications. Searches for GDDR5/GDDR6 SEE test reports (NASA NEPP compendia etc.) returned no GDDR-specific data — [NASA NEPP compendium example](https://ntrs.nasa.gov/api/citations/20160009072/downloads/20160009072.pdf)
- Consumer-GPU GDDR field data (not beam): Haque & Pande's MemtestG80 study on more than 20,000 G80/GT200 GPUs (Folding@home) found that "two-thirds of tested GPUs exhibit a detectable, pattern-sensitive rate of memory soft errors". These depend strongly on the board; they are not shown to be radiation-induced — [Haque & Pande, arXiv 0910.0505](https://arxiv.org/pdf/0910.0505)

### Inferences
- For an RTX 4090 with Micron GDDR6X, the DRAM-side per-bit rate would have to come from stand-alone DRAM beam or vendor data (DDR4/DDR5/LPDDR-class at a similar node) or from field studies. No GPU-board GDDR6X beam measurement exists to cite.

### Gaps
- No GDDR6 or GDDR6X neutron/proton SEU cross sections found, and no Ada-generation GPU beam results found.

---

## Q5. MCU/MBU fractions in GPU SRAM vs DRAM, and thermal vs high-energy neutrons

### Takeaway
GPU SRAM (ECC off, beam) shows low intra-word MBU fractions:
- K20 RF: about 1–6%.
- Kepler/Volta RF: under 2%; shared memory: under 0.9%.

Multi-cell upsets across rows are more common (K20 RF 17–19%). GPU HBM2 shows a much larger multi-bit fraction: about 31.5% of SEUs affect multiple bits in at least one word, with about 75% of multi-bit errors byte-aligned (indexed summaries; unverified). Thermal neutrons are usually a minor (around 1% to over 10%) contributor but can reach 59% of the total error rate for some device/code pairs. DDR3/DDR4 are thermal-sensitive.

### Cited Findings
- K20 RF MBU: 4% / 6% / 1% (00 / FF / AA); C2050 RF MBU: 1%. K20 RF MCU: 19% / 18% / 17%. No more than 2 bits per row upset — [Rech et al. SELSE'14](https://www.cs.utexas.edu/~skeckler/pubs/SELSE_2014_Reliability.pdf)
- K20 DBE share about 6%, with no triple-bit corruption, so SECDED suffices for SRAM — [Tiwari et al. HPCA'15](https://www.osti.gov/servlets/purl/1185857)
- RF MBU under 2% and SHARED MBU under 0.9% (Kepler and Volta, ChipIR/LANSCE) — [Fernandes dos Santos thesis](https://lume.ufrgs.br/bitstream/handle/10183/234971/001136966.pdf)
- HBM2 (beam):
  - About 31.5% of device-memory SEUs affect multiple bits in at least one word, and about 75% of multi-bit errors are byte-aligned. Source: search-indexed summaries of [Sullivan et al. MICRO'21](https://dl.acm.org/doi/10.1145/3466752.3480111) and the related [NVIDIA patent US 12,149,259](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12149259). **Not verified against full text.**
  - Multi-bit errors show locality attributed to the HBM2 structure, and beam exposure causes intermittent errors from cell damage — [ResearchGate abstract](https://www.researchgate.net/publication/355362966_Characterizing_and_Mitigating_Soft_Errors_in_GPU_DRAM)
- Thermal vs high-energy neutrons:
  - Devices tested (K20, Titan X, Titan V GPUs, APU, Xeon Phi, FPGA, DDR3/DDR4) at ISIS ChipIR (high-energy) and ROTAX (thermal).
  - Thermal neutrons can account for up to 59% of the total error rate (MTBF) for some applications on some devices. For most devices the thermal contribution is as low as about 1%; one device exceeded 10%.
  - DDR3/DDR4 show thermal-neutron sensitivity.
  - [Oliveira et al. IEEE TNS 2020](https://ieeexplore.ieee.org/document/8975983/), [ResearchGate](https://www.researchgate.net/publication/338927054_High-Energy_vs_Thermal_Neutron_Contribution_to_Processor_and_Memory_Error_Rates). Summary-level only; full text not accessed.

### Inferences
- For fault injection on an RTX 4090 with ECC off:
  - SRAM (L2) flips can reasonably be modeled as mostly single-bit per word. Beam data on older nodes show about 1–6% intra-word MBU and roughly 20% physically adjacent MCU; SRAM interleaving spreads MCUs across words.
  - DRAM flips should include a substantial multi-bit-per-word fraction (HBM2: about 31.5%). That may not transfer to GDDR6X, whose organization differs (non-stacked, PAM4 I/O).
- Thermal-neutron sensitivity depends on ¹⁰B. If rho is meant for ground-level environments with significant thermal flux (concrete or water-cooled machine rooms), the thermal contribution could move the DRAM and SRAM rates differently. No GPU-level per-structure thermal data were found to quantify this.

### Gaps
- No MBU/MCU fractions for GPU L2 specifically (only RF and shared memory).
- No GDDR5/GDDR6/GDDR6X MBU fractions from beam.
- The 31.5% HBM2 figure needs confirmation from the MICRO'21 full text.
- No per-structure thermal-neutron cross sections for GPU SRAM vs GPU DRAM.
