# Fault-injection / reliability-modeling methodology: how GPU & DNN-accelerator studies set SRAM/cache vs DRAM error rates and convert BER/FIT into injected-fault counts

Scope note: this file covers *methodology* papers (AVF/ACE, fault-injection frameworks, DNN-accelerator FIT models). Beam/field measurements are only included where a methodology paper uses them as its rate source. Primary PDFs were downloaded and grepped for every number marked "verified"; anything only recalled from background knowledge is put in Gaps.

## Q1. How do AVF / FI studies assign FIT per Mbit to SRAM vs DRAM? (incl. exact values in Li et al. SC17)

### Takeaway
The commonly assumed "Li et al. SC17 gives separate SRAM and DRAM FIT rates" is **not correct**. Li et al. SC17 uses **one raw rate, 20.49 FIT/Mb, for both latches and on-chip SRAM buffers**, extrapolated from a 28 nm SRAM beam measurement (Neale & Sachdev, IEEE TNS 2016). It **explicitly excludes DRAM/main memory** from its fault model. Later GPU microarchitectural FI work (gpuFI-4, ISPASS 2022) follows the same pattern: **one per-bit raw FIT applied to every on-chip SRAM structure (RF, shared, L1D/L1T, L2)**, taken from Li SC17 and Neale. Two cited sources trace back to these numbers: (a) the Neale & Sachdev 28 nm SRAM beam data, and (b) a 600 FIT/MB flip-flop rate (Jagannathan et al.) used by FIdelity. None of the FI frameworks examined models DRAM and L2 with separate per-bit rates in a single campaign.

### Cited Findings
**Li, Hari, Sullivan, Tsai, Pattabiraman, Emer, Keckler, "Understanding Error Propagation in Deep Learning Neural Network (DNN) Accelerators and Applications," SC17, DOI 10.1145/3126908.3126964** — [PDF (NVIDIA Research)](https://research.nvidia.com/sites/default/files/pubs/2017-11_Understanding-Error-Propagation/SC17_DNN_Resilience.pdf)
- Structures modeled: datapath latches, plus storage elements: Eyeriss Global Buffer (SRAM), Filter SRAM, Img REG, PSum REG. In the slide version, the "Storage" fault model is listed as "buffer SRAM, scratch pad, REG". — [SC17 PDF](https://research.nvidia.com/sites/default/files/pubs/2017-11_Understanding-Error-Propagation/SC17_DNN_Resilience.pdf); [Pattabiraman CMC workshop slides](https://www.cmc.ca/wp-content/uploads/2020/12/Karthik_CMC-Workshop-presentation.pdf)
- DRAM is excluded. Quote: "because our focus is on DNN accelerators, we do not consider faults in the CPU, main memory, or the memory/data buses." Control logic is also excluded. — [SC17 PDF §4](https://research.nvidia.com/sites/default/files/pubs/2017-11_Understanding-Error-Propagation/SC17_DNN_Resilience.pdf)
- The raw rate (Sec. 4.7) is "Rraw is the raw FIT rate (estimated as 20.49 FIT/Mb by extrapolating the results of Neale et al. [39]. The original measurement for a 28nm process is 157.62 FIT/MB in the paper. We project this for a 16nm process by applying the trend shown in Figure 1 of the Neale paper)". — [SC17 PDF](https://research.nvidia.com/sites/default/files/pubs/2017-11_Understanding-Error-Propagation/SC17_DNN_Resilience.pdf)
- Conversion formula (Eq. 1): FIT = Σ_component R_raw × S_component × SDC_component. Here S is the component size and SDC is the per-component SDC probability measured by FI. They inject 3,000 random single-bit faults per latch or buffer, one per DNN execution. — [SC17 PDF](https://research.nvidia.com/sites/default/files/pubs/2017-11_Understanding-Error-Propagation/SC17_DNN_Resilience.pdf)
- Primary source of the raw rate: A. Neale and M. Sachdev, "Neutron Radiation Induced Soft Error Rates for an Adjacent-ECC Protected SRAM in 28 nm CMOS," IEEE TNS 63(3):1912–1917, 2016, DOI 10.1109/TNS.2016.2547963. This is a **measured** neutron-beam SRAM result. The 16 nm value is an **assumed projection**. The slides add that "All raw FIT rates are projected based on the FIT at 28nm [Neale, IEEE TNS]", with a "Scaling factor = 2 by each tech. generation" applied to structure sizes in Table 7. — [SC17 PDF refs & Table 7](https://research.nvidia.com/sites/default/files/pubs/2017-11_Understanding-Error-Propagation/SC17_DNN_Resilience.pdf); [slides p.21](https://www.cmc.ca/wp-content/uploads/2020/12/Karthik_CMC-Workshop-presentation.pdf)
- Results: buffer FIT (e.g., Global Buffer 87.47 and Filter SRAM 62.74 for ConvNet; Filter SRAM 3.9 for NiN) is "usually a few orders of magnitude higher than datapath FIT". Filter SRAM FIT in NiN is about 1000× the whole datapath (0.004). They attribute this to (1) size and (2) reuse: "the same fault can be read multiple times". Img/PSum REGs have low FIT partly because of "a short time window for reuse". — [SC17 PDF §5.2.1](https://research.nvidia.com/sites/default/files/pubs/2017-11_Understanding-Error-Propagation/SC17_DNN_Resilience.pdf)

**Sartzetakis, Papadimitriou, Gizopoulos, "gpuFI-4: A Microarchitecture-Level Framework for Assessing the Cross-Layer Resilience of Nvidia GPUs," ISPASS 2022** — [PDF](https://www.ceid.upatras.gr/webpages/faculty/gpapad/assets/papers/ispass2022_sartzetakis.pdf); [IEEE Xplore](https://ieeexplore.ieee.org/document/9804675/); [code](https://github.com/caldi-uoa/gpuFI-4)
- Built on GPGPU-Sim 4.0. Target structures: register file, shared memory, L1 data cache, L1 texture cache and **L2 cache**, with single- and triple-bit faults. DRAM is not a target. — [PDF](https://www.ceid.upatras.gr/webpages/faculty/gpapad/assets/papers/ispass2022_sartzetakis.pdf)
- Formula: "FITstruct = AVFstruct × raw FITbit × #Bitsstruct". The GPU FIT is the sum over structures. — [PDF §F](https://www.ceid.upatras.gr/webpages/faculty/gpapad/assets/papers/ispass2022_sartzetakis.pdf)
- Values: "the raw FIR [sic] rate for RTX 2060 and Quadro GV100 (which are fabricated at 12nm) is 1.8 x 10^-6 and for GTX Titan (… 28nm) is 1.2 x 10^-5" (FIT per bit). The **same per-bit value is used for RF, shared, L1 and L2**. Stated sources are [21] Chatzidimitriou et al. DSN 2019 (ARM CPU beam vs FI), [30] Li et al. SC17 and [31] Neale & Sachdev TNS 2016. — [PDF](https://www.ceid.upatras.gr/webpages/faculty/gpapad/assets/papers/ispass2022_sartzetakis.pdf)

**He, Balaprakash, Li, "FIdelity: Efficient Resilience Analysis Framework for Deep Learning Accelerators," MICRO 2020** — [PDF](https://microarch.org/micro53/papers/738300a270.pdf)
- Models flip-flops (datapath, local control and global control FFs) in NVDLA. SRAM is not the focus.
- Rate: "The raw FF FIT rate used to derive our results is 600/MB for soft errors [7]". [7] is Jagannathan et al., "Frequency dependence of alpha-particle induced soft error rates of flip-flops…" (measured alpha data). The paper notes other raw rates "can be used". — [PDF §IV](https://microarch.org/micro53/papers/738300a270.pdf)

**Hari, Rech, Tsai, Stephenson, Zulfiqar, Sullivan, Shirvani, Racunas, Emer, Keckler, "Estimating Silent Data Corruption Rates Using a Two-Level Model," arXiv:2005.01445 (2020)** — [arXiv PDF](https://arxiv.org/pdf/2005.01445)
- Uses measured raw rates per architectural manifestation from neutron beam tests of microbenchmarks on a K40 (28 nm). The K40 has 30 Mbit RF, 7.86 Mbit L1/shared and 12 Mbit L2. The model is "beam-derived FIT × software-level propagation probability".
- These tests ran **with ECC on** ("register file, L1 and L2 caches, shared memory, and DRAM are protected from single-bit flips. We record but ignore uncorrected ECC errors"). The method therefore does **not** produce a raw L2 or DRAM rate. — [arXiv PDF](https://arxiv.org/pdf/2005.01445)

**Hari, Tsai, Stephenson, Keckler, Emer, "SASSIFI: An Architecture-level Fault Injection Tool for GPU Application Resilience Evaluation," ISPASS 2017** — [PDF](https://research.nvidia.com/sites/default/files/pubs/2017-04_SASSIFI:-An-Architecture-level/2017-ISPASS-SASSIFI.pdf)
- A grep of the paper found no per-bit FIT/Mbit constants. It reports derated vulnerabilities (e.g., the benefit of RF ECC) rather than absolute SRAM or DRAM rates. — [PDF](https://research.nvidia.com/sites/default/files/pubs/2017-04_SASSIFI:-An-Architecture-level/2017-ISPASS-SASSIFI.pdf)

### Inferences
- Li SC17 states "157.62 FIT/MB" at 28 nm. Converting MB to Mb gives 157.62/8 ≈ 19.7 FIT/Mb, very close to the 20.49 FIT/Mb they report as the "16 nm projection". The capitalization may be inconsistent, so the reported "projection" may effectively be about the original 28 nm per-bit rate. This should be checked against Neale & Sachdev's Fig. 1 before the value is reused.
- gpuFI-4's 28 nm value (1.2e-5 FIT/bit ≈ 12.6 FIT/Mbit) and 12 nm value (1.8e-6 FIT/bit ≈ 1.9 FIT/Mbit) are the same order as the Li/Neale ~20 FIT/Mb. The paper does not show the exact derivation.
- The methodology norm for on-chip structures is therefore: **a single technology-node SRAM per-bit FIT (order 10 FIT/Mbit at 16–28 nm, beam-measured) × bits × AVF or SDC probability.** DRAM is either excluded (Li SC17, gpuFI-4, FIdelity) or assumed ECC-protected (two-level model). A campaign that wants an L2 BER from a DRAM BER cannot take the ratio from these FI papers. The ratio has to come from separate beam data for each technology.

### Gaps
- No examined FI/AVF paper gives an explicit "SRAM per-bit FIT = X × DRAM per-bit FIT" assumption.
- Could not access Neale & Sachdev (IEEE TNS 2016) full text to confirm the 157.62 figure's unit and the Fig. 1 trend.
- Not verified in this session: Mukherjee et al. MICRO 2003 AVF paper and Mukherjee's textbook "Architecture Design for Soft Errors" (raw FIT/bit assumptions); GPU-Qin (Fang et al., ISPASS 2014); Tan & Fu (IISWC 2011, GPGPU AVF); Previlon et al.; PyTorchFI (Mahmoud et al., DSN-W 2020); NVBitFI (Tsai et al., DSN 2021); Ares (Reagen et al., DAC 2018, [ACM](https://dl.acm.org/doi/pdf/10.1145/3195970.3195997)).
  - From background knowledge only (not confirmed against PDFs), the software-level tools (GPU-Qin, SASSIFI, NVBitFI, PyTorchFI) inject a fixed number of faults per run and report probabilities, with no SRAM/DRAM FIT table. Ares parameterizes a per-bit fault probability on stored weights and activations, not a FIT/Mbit.
- The MICRO 2021 paper by Sullivan et al., "Characterizing and Mitigating Soft Errors in GPU DRAM" ([ACM](https://dl.acm.org/doi/10.1145/3466752.3480111)), holds the measured HBM2 raw rate. ACM full text was blocked (cookie wall), so no FIT/Mb value is recorded here.

## Q2. Do any works model cache errors with time-dependent residency (per-bit rate × resident-time exposure)?

### Takeaway
Yes. The canonical method is **ACE lifetime analysis for caches** (Biswas, Racunas, Cheveresan, Emer, Mukherjee, Rangan, ISCA 2005). It splits each cache bit's lifetime into intervals such as fill-to-read, read-to-read, write-to-read (ACE) versus idle, read-to-evict, fill-to-evict (un-ACE). AVF is the ACE fraction of bit-time. Cache FIT is then raw FIT/bit × #bits × AVF.

This is a time-averaged resident-bits model: expected faults = λ_bit × Σ_bits (ACE residency time). NVIDIA's two-level GPU model names such residency-dependent bits "V-bits" (data caches, load/store buffers, DRAM buffers). Li SC17 notes the reuse time-window effect qualitatively for buffers.

### Cited Findings
- **Biswas et al., "Computing Architectural Vulnerability Factors for Address-Based Structures," ISCA 2005, DOI 10.1145/1080695.1070014**:
  - "AVF is the fraction of the bit's lifetime during which the bit contained ACE state". Lifetime components are "idle, fill-to-read, read-to-write, write-to-write, write-to-read, read-to-evict, etc." (Table 1).
  - For a write-through cache, ACE = fill-to-read, read-to-read, write-to-read. For write-back, write-to-evict and write-to-end are also ACE.
  - Best-estimate data-array AVFs: 6% (data cache), 36% (DTLB), 4% (store buffer).
  - They also discuss residency times explicitly ("residency times of entries in the store buffer are much shorter than … in the data cache"). Flushing and scrubbing reduce AVF by converting ACE lifetime to un-ACE.
  - Sources: [PDF (Princeton)](https://liberty.princeton.edu/Publications/isca32_avf.pdf); [ACM](https://dl.acm.org/doi/10.1145/1080695.1070014)
- **Two-level model (Hari et al. 2020)** splits bits into F-bits ("FIT rate … scales linearly with instruction issue rate", e.g., flip-flops and pipeline SRAM buffers) and V-bits. V-bits are those "whose vulnerability varies with different microarchitecture-level buffer occupancies… Examples of V-bits include unprotected SRAM bits in data caches, load/store buffers, and DRAM buffers. Since data can reside in these structures for a variable amount of time, the vulnerability of such bits will also vary." — [arXiv PDF](https://arxiv.org/pdf/2005.01445)
- **Li SC17**: small register buffers have low FIT because of "a short time window for reuse". Large reused SRAM buffers have high FIT because a fault "can be read multiple times". The FIT formula itself uses SDC probability from random-bit injection, not an explicit residency integral. — [SC17 PDF](https://research.nvidia.com/sites/default/files/pubs/2017-11_Understanding-Error-Propagation/SC17_DNN_Resilience.pdf)
- **gpuFI-4** injects into a bit chosen within "the total size of the L2 cache" and measures AVF. Residency is therefore captured statistically: a flip in an invalid or dead line is masked, rather than modeled as an explicit time integral. — [PDF](https://www.ceid.upatras.gr/webpages/faculty/gpapad/assets/papers/ispass2022_sartzetakis.pdf)

### Inferences
- A GPU DNN campaign could derive L2 fault counts as N_L2 = λ_SRAM_bit × Σ_lines (bits_line × ACE-resident time). Here λ_SRAM_bit comes from beam data for that node, not scaled from DRAM. This is the Biswas lifetime model.
  - Equivalent shortcut: N_L2 = λ_SRAM_bit × (time-averaged resident bits of the tensor) × T_inference.
  - Random-time, random-bit injection, weighted by the fraction of L2 holding live tensor data, is the statistical-FI equivalent (gpuFI-4 style).
- For DRAM, the same logic gives N_DRAM = λ_DRAM_bit × bits_resident × T. The L2:DRAM fault ratio is then (λ_SRAM/λ_DRAM) × (resident L2 bit-time / resident DRAM bit-time). The per-bit ratio must come from beam/field sources (see Q3).

### Gaps
- No GPU-DNN-specific paper was found that computes L2 fault counts from a per-bit rate × measured L2 residency on real hardware. Biswas is CPU-simulated; gpuFI-4 is GPGPU-Sim on generic benchmarks.
- Not verified this session: Tan & Fu IISWC 2011 and Farazmand et al. SELSE 2012 GPU AVF studies. These likely include ACE-based L1/shared AVF but were not checked.

## Q3. Commonly cited SRAM and DRAM FIT/Mbit values, traced to primary sources

### Takeaway
The folklore "~1000 FIT/Mbit" originates largely from early-2000s vendor and trade-press figures compiled in the Tezzaron white paper (2004). That paper lists SRAM and DRAM in the **same 1,000–2,000 FIT/Mbit band** (e.g., Micron technote DT28: "SRAM and DRAM 1–2E-12 upset/bit-hour"), and concludes "1000 to 5000 FIT per Mbit seems to be a reasonable SER for modern memory devices". Field DRAM rates (Schroeder et al. 2009) are much higher, 25,000–75,000 FIT/Mbit, but count all correctable errors, including hard faults. Modern-node SRAM beam values used by FI papers are ~10–20 FIT/Mbit (Neale 28 nm, via Li SC17).

The literature gives no single defensible "SRAM = k × DRAM per bit" constant. The ratio depends strongly on node, vendor, and whether field (hard + soft) or beam (soft only) data is used.

### Cited Findings
- **Tezzaron Semiconductor, "Soft Errors in Electronic Memory – A White Paper" (2004)**: compiled table (all values per Mbit). Sources:
  - "SRAM (quoted by vendors) 200 to 2,000 FIT"; "'typical' 1,000 FIT"
  - "SRAM and DRAM 1–2E-12 upset/bit-hour → 1,000–2,000 FIT/Mbit [23 = Micron Technical Note DT28 'DRAM Soft Error Rate Calculations']"
  - "SRAM 1,000 FIT/Mbit [13 = Harling, Integrated System Design 2001]"
  - "1 Gbit of DRAM (Nite Hawk) 2,300"; "160 Gbits of DRAM (Fermilab) 700"; "32 Gbits DRAM (CRAY YMP-8) 600"; "~8.2 Gbits SRAM (CRAY YMP-8) 1,300"
  - "MoSys 1T-SRAM (no ECC) 500"; "Micron estimate 256 MB 120–240"
  - Conclusion: "1000 to 5000 FIT per Mbit seems to be a reasonable SER"
  - Bullet: "SRAM has become more susceptible to soft errors than DRAM"
  - Conversion note: 1 FIT/Mb = 1E-15 upset/bit-hour
  - — [Tezzaron PDF](https://tezzaron.com/media/soft_errors_1_1_secure.pdf)
- **Schroeder, Pinheiro, Weber, "DRAM Errors in the Wild: A Large-Scale Field Study," SIGMETRICS 2009**: "Li et al cite error rates in the 200–5000 FIT per Mbit range from previous lab studies, and themselves found error rates of < 1 FIT per Mbit. In comparison, we observe mean correctable error rates of 2000–6000 per GB per year, which translate to 25,000–75,000 FIT per Mbit". Abstract: "25,000 to 70,000" FIT/Mbit. Field-measured; includes hard faults. — [PDF](https://www.cs.toronto.edu/~bianca/papers/sigmetrics09.pdf)
- **DeBardeleben, Blanchard, Sridharan, Gurumurthi, Stearley, Ferreira, "Extra Bits on SRAM and DRAM Errors – More Data from the Field," SELSE 2014 (SAND2014-0408C)**:
  - DRAM fault rates are given per device, e.g., permanent faults "24.2 FIT to 10.7 FIT per DRAM chip" across vendors.
  - SRAM (L2/L3) fault rates appear only in **arbitrary units**. Cielo (Los Alamos) shows a 2.3× (L2) and 3.4× (L3) higher SRAM transient rate than Jaguar, versus a 4.39× flux ratio.
  - Field per-bit SRAM rates showed "good correlation" with static accelerated beam testing, with the field somewhat lower.
  - No absolute SRAM-vs-DRAM per-bit ratio is published. — [OSTI PDF](https://www.osti.gov/servlets/purl/1140778)
- **Neale & Sachdev (IEEE TNS 2016)**: 28 nm SRAM, 157.62 "FIT/MB" (as quoted by Li SC17). This is the root of the ~20 FIT/Mb SRAM value used in DNN/GPU FI. — [via SC17 PDF](https://research.nvidia.com/sites/default/files/pubs/2017-11_Understanding-Error-Propagation/SC17_DNN_Resilience.pdf)
- **Jagannathan et al. (alpha-particle flip-flop SER)**: 600 FIT/MB for flip-flops, as used by FIdelity. — [FIdelity PDF](https://microarch.org/micro53/papers/738300a270.pdf)

### Inferences
- Older-node sources put SRAM and DRAM per-bit SER in roughly the same order of magnitude (~10^3 FIT/Mbit, 1990s–2000s). Per-bit DRAM soft-error rates fell with scaling, while SRAM per-bit rates saturated.
  - The commonly stated "DRAM per bit is 10–100× lower than SRAM" is plausible for modern nodes, but no primary-source confirmation was obtained in this session.
  - Treat any fixed SRAM/DRAM multiplier as an **assumption to sweep**: e.g., ratio ∈ {1, 10, 100}, anchored on ~20 FIT/Mbit SRAM (Neale, 28 nm) and the measured DRAM BER.

### Gaps
- Baumann reviews (e.g., IEEE TDMR 2005) were not retrieved; their per-bit SRAM/DRAM trend figures were not verified here.
- HBM2/GDDR per-bit FIT from Sullivan et al. MICRO 2021 was not retrieved (ACM blocked).
- No primary source for an explicit "SRAM ≈ X × DRAM per bit" constant was found.

## Q4. Memory-aware / "physical" fault injection for DNNs on GPUs (REMU DAC 2025 and others) — do they include cache faults?

### Takeaway
REMU (DAC 2025) is DRAM-centric: it maps radiation-induced **DRAM** errors through virtual→physical→DRAM addressing into runtime DNN inference. The abstract does not indicate L2 or cache modeling. Beam-based DNN-on-GPU studies (dos Santos et al., IEEE TNS) irradiate the whole GPU, including L2, and pair it with NVBitFI-based injection. They compare ECC on/off rather than using separate per-structure per-bit rates. gpuFI-4 is the main tool that injects directly into the GPU L2, but in a simulator.

### Cited Findings
- **Xu, Wang, Qiu, Li et al., "REMU: Memory-Aware Radiation Emulation via Dual Addressing for In-Orbit Deep Learning System," DAC 2025, DOI 10.1109/DAC63849.2025.11132935**: "dual addressing mechanism across virtual, physical, and DRAM memory spaces, enabling precise mapping and efficient injection of radiation-induced errors from DRAM to runtime DNN inference". Evaluated on 10 DNNs and 2 in-orbit tasks on COTS GPUs. — [ACM](https://dl.acm.org/doi/10.1109/DAC63849.2025.11132935); [ResearchGate](https://www.researchgate.net/publication/395609269_REMU_Memory-aware_Radiation_Emulation_via_Dual_Addressing_for_In-orbit_Deep_Learning_System)
- **Fernandes dos Santos, Kritikakou, Rodriguez Condia, Guerrero-Balaguera, Sonza Reorda, Sentieys, Rech, "Characterizing a Neutron-Induced Fault Model for Deep Neural Networks," IEEE TNS (arXiv:2211.13094, 2022)**: the setup figure shows GPU SMs, L2 cache and DRAM, with "Modified NVBitFi injection". Beam results are reported with "ECC OFF / ECC ON". Volta was tested "only with ECC enabled". — [arXiv PDF](https://arxiv.org/pdf/2211.13094)
- **gpuFI-4** injects single- and triple-bit faults into the L2, L1D/L1T, shared memory and RF of simulated RTX 2060, GV100 and GTX Titan. — [PDF](https://www.ceid.upatras.gr/webpages/faculty/gpapad/assets/papers/ispass2022_sartzetakis.pdf)
- **Yang, Papadimitriou, Gizopoulos et al., "GPU Reliability Assessment: Insights Across the Abstraction Layers," IEEE Cluster 2024**: compares microarchitecture-level (GPGPU-Sim; RF, shared, L1D, L1T, L2) and software-level (V100) injection. It finds "hardening introduces extra vulnerabilities in L2 caches" and that more SDCs originate from L2 faults in some cases. — [PDF](https://www.ceid.upatras.gr/webpages/faculty/gpapad/assets/papers/cluster2024_yang.pdf)
- **Chai et al., "Analysis of LLM Vulnerability to GPU Soft Errors: An Instruction-Level Fault Injection Study" (arXiv:2601.19912, 2026)**: instruction-level injection with a configurable number of faults per run. A grep found no per-bit FIT/BER-to-count model or cache-specific rates. — [arXiv PDF](https://arxiv.org/pdf/2601.19912)

### Inferences
- In the works examined, no DNN-on-GPU physical/memory-aware injector models both DRAM and L2 with physically derived, separate per-bit rates and residency. A campaign doing this would combine:
  - REMU-style DRAM address mapping
  - Biswas-style residency/ACE accounting for L2
  - gpuFI-4/Li-style FIT = λ_bit × bits × AVF bookkeeping
  - an explicitly stated, swept SRAM/DRAM per-bit ratio

### Gaps
- The REMU full text was not accessed. Its exact BER source, flips-per-inference conversion, and any treatment of caches are unverified beyond the abstract.
- The dos Santos et al. TNS paper's per-structure cross-sections (if any, e.g., for L2) were not extracted. Only the ECC on/off framing was confirmed.
