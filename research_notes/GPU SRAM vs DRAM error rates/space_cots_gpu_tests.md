# Space-radiation (proton / heavy-ion / in-orbit) tests of COTS GPUs and GPU SoCs: SRAM vs DRAM SEU cross sections

Scope note: these notes cover only proton, heavy-ion and in-orbit data. Neutron/terrestrial GPU beam data (Rech et al. on K40/Volta, field data from HPC GPUs) is outside this assignment and is listed only as a pointer under Gaps. Full texts were read for the NASA TX2 report, the BSC/ESA Xavier and Orin papers, the JPL Snapdragon slides, the CNES/Alter DDR4 compendium, the NASA GSFC 2023 compendium, the Alicante Jetson Nano TNS paper, the REMU paper (local PDF in repo) and the Tsinghua arXiv predecessor of REMU. Other entries come from abstracts or metadata only and are marked as such.

## Q1. Which proton or heavy-ion campaigns on COTS GPUs/GPU SoCs give SRAM vs DRAM cross sections?

### Takeaway
No published campaign on a COTS GPU or GPU SoC reports per-bit cross sections for both on-chip SRAM (GPU L1/L2/register file/shared memory) and external DRAM (LPDDR4/LPDDR5/GDDR) together. The best structure-resolved data are (a) per-bit L2/L3 **CPU-complex** cache data on Jetson Xavier, taken with ARM RAS logging (no GPU errors seen), (b) a Snapdragon 835 L2 per-bit heavy-ion value, and (c) **per-device-region** (not per-bit) cross sections for the LPDDR5 packages on a Jetson Orin NX module. GPU-internal SRAM per-bit cross sections under protons or heavy ions were not found. Most NASA NEPP Jetson reports are black-box SEFI/"SEU" per-device numbers.

### Cited Findings

**A. BSC/UPC + ESA: Rodriguez-Ferrandez, Tali, Kosmidis, Rovituso, Steenari, "Sources of Single Event Effects in the NVIDIA Xavier SoC Family under Proton Irradiation", IEEE IOLTS 2022, Torino, DOI 10.1109/IOLTS56730.2022.9897236** — [BSC page](https://www.bsc.es/research-and-development/publications/sources-single-event-effects-the-nvidia-xavier-soc-family); [open PDF (UPCommons)](https://upcommons.upc.edu/server/api/core/bitstreams/80d67a8c-34a3-408b-aaf1-5bc5bc48b1c1/content)
- Devices: Jetson Xavier NX and Xavier Industrial (AGX Xavier Industrial). The SoC is 12 nm FinFET, 350 mm². CPU: 64 KB L1D, 2 MB L2 per 2-core cluster, 4 MB shared L3. Memory is 8 GB (NX) or 32 GB (Industrial) LPDDR4. The Industrial part has ECC/SECDED "in all memories including DRAM". — [same PDF](https://upcommons.upc.edu/server/api/core/bitstreams/80d67a8c-34a3-408b-aaf1-5bc5bc48b1c1/content)
- Particle: 200 MeV protons at Holland PTC (Delft), 9 cm² beam. — [same](https://upcommons.upc.edu/server/api/core/bitstreams/80d67a8c-34a3-408b-aaf1-5bc5bc48b1c1/content)
- Method: ARM RAS logs classify events. RAS-correctable errors count as SEU; RAS-uncorrectable errors (which force a kernel panic and reboot) count as SEFI. "No errors were reported from the GPU complex." — [same](https://upcommons.upc.edu/server/api/core/bitstreams/80d67a8c-34a3-408b-aaf1-5bc5bc48b1c1/content)
- Per-device cross sections: SEU 3.61e-8 cm² (Industrial) and 2.91e-8 cm² (NX). SEFI 1.52e-8 cm² (Industrial) and 1.30e-8 cm² (NX). SEFIs were 11% of SEEs. "The largest contribution to the overall SEE cross-section was the L2 and L3 SEU events." — [same](https://upcommons.upc.edu/server/api/core/bitstreams/80d67a8c-34a3-408b-aaf1-5bc5bc48b1c1/content)
- Per-configuration cross sections (Industrial, Table II) range from 1.19e-8 to 9.51e-8 cm². Runs with the GPU workload active gave 1.29e-8 to 2.67e-8 cm². These are whole-SoC events: the GPU workload did not produce GPU-attributed errors. — [same](https://upcommons.upc.edu/server/api/core/bitstreams/80d67a8c-34a3-408b-aaf1-5bc5bc48b1c1/content)
- Structure attribution: correctable events were "L3 Correctable ECC Error", "L2 MLC Correctable Error" (L2 data array) and "SCF to L2 Correctable ECC" (coherence interface). Every RAS SEFI involved cache tags: L3 tag parity, L2 MLT tag parity, multi-hit tag, plus one DVMU interface timeout. The paper reports a per-bit L3 cross section only graphically (Fig. 8). — [same](https://upcommons.upc.edu/server/api/core/bitstreams/80d67a8c-34a3-408b-aaf1-5bc5bc48b1c1/content)
- No DRAM (LPDDR4) cross section is reported.

**B. BSC/UPC + ESA + TRIUMF: Rodriguez-Ferrandez, Kosmidis, Tali, Steenari, Hands, Bélanger-Champagne, "Proton Evaluation of Single Event Effects in the NVIDIA GPU Orin SoM: Understanding Radiation Vulnerabilities Beyond the SoC", IEEE IOLTS 2024, DOI 10.1109/IOLTS60994.2024.10616076** — [IEEE](https://ieeexplore.ieee.org/document/10616076/); [open PDF (UPCommons)](https://upcommons.upc.edu/server/api/core/bitstreams/ef814395-1619-43ee-ba00-6a89f0c16d83/content)
- Device: Jetson Orin NX SoM. TSMC 7 nm SoC with A78AE CPUs and an Ampere GPU with 8 SMs. Memory is 16 GB Micron LPDDR5. — [PDF](https://upcommons.upc.edu/server/api/core/bitstreams/ef814395-1619-43ee-ba00-6a89f0c16d83/content)
- Particle: 480 MeV protons at TRIUMF BL1B. Day 1 used a 3×3 cm beam on the SoC; day 2 used a 1.5×3 cm beam on individual SoM sub-components (the RAM packages, I/O and power). — [PDF](https://upcommons.upc.edu/server/api/core/bitstreams/ef814395-1619-43ee-ba00-6a89f0c16d83/content)
- SoC SEU cross section by power mode: 3.90e-9 cm² (15 W), 4.43e-9 cm² (10 W), 2.50e-9 cm² (sub-10 W). SoC SEFI: 1.59e-9, 1.51e-9 and 6.81e-10 cm². — [PDF](https://upcommons.upc.edu/server/api/core/bitstreams/ef814395-1619-43ee-ba00-6a89f0c16d83/content)
- **GPU vs CPU** (fluence 1.99e10 p/cm²): GPU SEU 3.52e-10 cm² (CI 1.42e-10 to 7.26e-10) and GPU SEFI 6.54e-10 cm². CPU SEU 3.22e-9 cm² and CPU SEFI 6.04e-10 cm². Here a GPU "SEU" is an error corrected in-subsystem; a GPU "SEFI" requires a kernel restart. GPU error dumps most often name an SM or the GPU MMU. The authors say these are the first GPU-attributed SEEs in the literature. They observed no wrong outputs: "The SoC either detects and corrects the errors, or stop working with a proper error log." — [PDF](https://upcommons.upc.edu/server/api/core/bitstreams/ef814395-1619-43ee-ba00-6a89f0c16d83/content)
- **LPDDR5 region (per device, not per bit)**: RAM 1 SEU 2.03e-10 cm² and SEFI 3.04e-10 cm². RAM 2 SEU 2.55e-10 cm² and SEFI 3.83e-10 cm². When the DRAMs were irradiated, the main effects were "errors in the communication of the DRAMs" that triggered kernel panics. "No discernible memory data corruption was detected from our verification procedure in the running code on the GPU when this area was irradiated." — [PDF](https://upcommons.upc.edu/server/api/core/bitstreams/ef814395-1619-43ee-ba00-6a89f0c16d83/content)
- SoC SEFI breakdown: aborting core 37.5%, L3 directory parity 18.75%, I/O error 18.75%, and L2 directory parity, I$ tag parity, control backbone and memory abort at 6.25% each. — [PDF](https://upcommons.upc.edu/server/api/core/bitstreams/ef814395-1619-43ee-ba00-6a89f0c16d83/content)
- Related (abstract only, numbers not read): "Characterizing Single Event Functional Interrupts in Jetson Orin Nano SoC Under Proton Irradiation" (Springer chapter). — [Springer](https://link.springer.com/chapter/10.1007/978-3-032-35506-5_9)

**C. NASA GSFC NEPP: E. J. Wyrwas, "Proton Testing of nVidia Jetson TX2", NASA GSFC test report GSFC-E-DAA-TN72754 (test 2 June 2019, report 22 July 2019)** — [NTRS 20190031856](https://ntrs.nasa.gov/citations/20190031856); [PDF](https://ntrs.nasa.gov/api/citations/20190031856/downloads/20190031856.pdf)
- 200 MeV protons at MGH Francis H. Burr Proton Therapy Center. Flux 1e7 to 1e8 p/cm²/s. Device is listed as 20 nm. Workload: CUDA samples (particles, fluidsGL) plus CPU/GPU matrix compare. — [PDF](https://ntrs.nasa.gov/api/citations/20190031856/downloads/20190031856.pdf)
- Per-device "SEU cross section" (in practice SEFI-dominated): average 1.02e-9 cm², range 2.50e-10 to 4.17e-9 cm². The same table gives the predecessor Jetson TX1 an average of 6.22e-8 cm² (range 2.65e-9 to 5.05e-7 cm²). — [PDF](https://ntrs.nasa.gov/api/citations/20190031856/downloads/20190031856.pdf)
- "SEFI occurred during all runs… All SEFIs required the system to be reset through a power cycle." Also: "we lack insight into which element within the device experienced the upset. Further testing may include memory mapping vectors in SRAM and DRAM." This means **no SRAM/DRAM separation**. — [PDF](https://ntrs.nasa.gov/api/citations/20190031856/downloads/20190031856.pdf)
- Inconsistency: the report's front matter says testing was on 2 June 2019, but the body says "June 2, 2018". — [PDF](https://ntrs.nasa.gov/api/citations/20190031856/downloads/20190031856.pdf)

**D. NASA GSFC compendium (O'Bryan, Wilcox, …, Wyrwas, …, Pellish), NSREC REDW 2023** — [NTRS PDF](https://ntrs.nasa.gov/api/citations/20230009904/downloads/2023-Compendium-Paper-NSREC-v7.pdf)
- AMD Radeon e9173 PCIe GPU, heavy ions at LBNL (Aug 2022): SEL LETth > 16 MeV·cm²/mg, plus unstable behaviour above 60 °C. No per-structure data. Full report: Wyrwas, "AMD Radeon e9173 Low Power PCIE GPU SEE Test Report" (NEPP 2022). — [same](https://ntrs.nasa.gov/api/citations/20230009904/downloads/2023-Compendium-Paper-NSREC-v7.pdf)
- Mercury Systems DDR4 (4N1G72T-24BM), protons at MGH (Dec 2022): stuck bits seen at 60 and 200 MeV. No per-bit SEU value is given in the compendium. — [same](https://ntrs.nasa.gov/api/citations/20230009904/downloads/2023-Compendium-Paper-NSREC-v7.pdf)
- Correction: a search-engine summary attributed "1.41e-17 cm²/bit at 200 MeV" to this DDR4 part. In the compendium that value belongs to a **Micron 3D NAND flash (SLC)**, not to DDR4 DRAM. — [same](https://ntrs.nasa.gov/api/citations/20230009904/downloads/2023-Compendium-Paper-NSREC-v7.pdf)
- Also cited there: Wyrwas, "Proton Testing of AMD v1202b SoC" (NEPP 2022), and Casey et al., "SEE on COTS Edge-Processing AI ASICs", IEEE TNS 2023, DOI 10.1109/TNS.2023.3286728. — [same](https://ntrs.nasa.gov/api/citations/20230009904/downloads/2023-Compendium-Paper-NSREC-v7.pdf)

**E. JPL (NEPP): S. M. Guertin, "ARM Radiation Testing & Collaborations", NEPP ETW 2020 (Snapdragon 835/845 heavy ion)** — [NEPP ETW 2020 slides](https://nepp.nasa.gov/docs/etw/2020/17-JUN-WED/1130-Guertin-NEPP-ETW-CL20-2459-ARM-v5.pdf)
- Snapdragon 835 (10 nm, Adreno 540 GPU) and 845 (10 nm, Adreno 630 GPU), Android, video-playback and graphics-benchmark workloads. — [slides](https://nepp.nasa.gov/docs/etw/2020/17-JUN-WED/1130-Guertin-NEPP-ETW-CL20-2459-ARM-v5.pdf)
- **835 L2 cache: about 1e-11 cm²/bit** up to LET ≈ 2 MeV·cm²/mg, "not significantly change[d] up to LET 6.9". — [slides](https://nepp.nasa.gov/docs/etw/2020/17-JUN-WED/1130-Guertin-NEPP-ETW-CL20-2459-ARM-v5.pdf)
- 835 SEFI: 1e-4 cm² for LET < 2, rising to 3e-4 cm² at LET ≈ 6.9. 845 SEFI: 2e-4 cm² at LET 6.9. The 845 showed about 4 L1/L2/L3 bit errors per SEFI, giving a device-level bit-error cross section of about 8e-4 cm². "Both 835 and 845 SEFI cross sections are much higher than the DDR2 SEFI cross section — test ions had to traverse both the DDR4 device and the processor" (package-on-package). — [slides](https://nepp.nasa.gov/docs/etw/2020/17-JUN-WED/1130-Guertin-NEPP-ETW-CL20-2459-ARM-v5.pdf)
- No Adreno-GPU-internal or DRAM per-bit value is given. The slides list a Snapdragon 855 (Adreno 640) test as planned for FY20/21. — [slides](https://nepp.nasa.gov/docs/etw/2020/17-JUN-WED/1130-Guertin-NEPP-ETW-CL20-2459-ARM-v5.pdf)

**F. MDA / U. Saskatchewan (Hiemstra, L. Chen et al.)** — abstract/metadata only; per-bit numbers not accessible (IEEE/ResearchGate blocked)
- H. Wang, Q. Chen, L. Chen, D. M. Hiemstra, V. Kirischian, "Single Event Upset Characterization of the Tegra K1 Mobile Processor Using Proton Irradiation", IEEE REDW 2017, DOI 10.1109/NSREC.2017.8115446. Proton-induced SEU cross sections (from cache bit flips) and estimated space upset rates. Particle listed as p+/120 MeV in a later survey. — [ResearchGate](https://www.researchgate.net/publication/321257405_Single_Event_Upset_Characterization_of_the_Tegra_K1_Mobile_Processor_Using_Proton_Irradiation); [Serrano-Cases et al. TNS 2023, Table I](https://rua.ua.es/server/api/core/bitstreams/72caed0a-eb66-48bd-ae0e-8b3d1928146b/content)
- D. M. Hiemstra, C. Jin, Z. Li, R. Chen, S. Shi, L. Chen, "Single Event Effect Evaluation of the Jetson AGX Xavier Module Using Proton Irradiation", IEEE REDW 2020, DOI 10.1109/REDW51883.2020.9325840. 105 MeV protons; TID, SEFI and SEL; TMR on the CPU with an FFT workload. — [IEEE](https://ieeexplore.ieee.org/document/9325840/); [survey table](https://rua.ua.es/server/api/core/bitstreams/72caed0a-eb66-48bd-ae0e-8b3d1928146b/content)

**G. AFRL / COSMIAC / Troxel Aerospace: W. S. Slater et al., "Single Event Effects and Total Ionizing Dose Radiation Testing of NVIDIA Jetson Orin AGX System on Module", IEEE REDW 2023, DOI 10.1109/REDW61050.2023.10265818** (metadata/abstract only) — [ResearchGate](https://www.researchgate.net/publication/374468482_Single_Event_Effects_and_Total_Ionizing_Dose_Radiation_Testing_of_NVIDIA_Jetson_Orin_AGX_System_on_Module)
- Slater's earlier work is TID-only: "Total Ionizing Dose Radiation Testing of NVIDIA Jetson Nano GPUs". — [Semantic Scholar](https://www.semanticscholar.org/paper/Total-Ionizing-Dose-Radiation-Testing-of-NVIDIA-Slater-Tiwari/4547b1ec8984cec77508e264f6106fd505cc0319)
- A related paper finds that "the reboot process has a significantly higher cross-section compared to application-level operations such as matrix multiplication and neural network inference". — [ResearchGate: Radiation reliability of system reboots in COTS SoC](https://www.researchgate.net/publication/388810931_Radiation_reliability_of_system_reboots_in_commercial_off-the-shelf_SoC)

**H. Alicante / CNA Seville: A. Serrano-Cases, S. Alcaide, M. A. Romero, Y. Morilla, S. Cuenca-Asensi, "Analysis of kernel redundancy for soft error mitigation on embedded GPUs", IEEE TNS 2023, DOI 10.1109/TNS.2023.3291418** — [author PDF](https://rua.ua.es/server/api/core/bitstreams/72caed0a-eb66-48bd-ae0e-8b3d1928146b/content)
- Jetson Nano (Tegra X1, 20 nm, Maxwell GPU: 64 KB shared memory/L1 per SM, 256 KB L2). 15.4 MeV protons at the CNA cyclotron; total fluence 7.4e11 p/cm². — [PDF](https://rua.ua.es/server/api/core/bitstreams/72caed0a-eb66-48bd-ae0e-8b3d1928146b/content)
- "Matrices are stored in DDR memory, which remains out of the beam. However, the L1 and L2 caches … are susceptible." The authors attribute matrix-multiply SDCs to the GPU L1/L2: bigger matrices and higher L2 reuse give more SDCs. The CPU is "the source of the majority of the events, which are mainly dominated by functional interrupts." Cross sections are given only per run class (SDC/SEFI/detected) in figures. — [PDF](https://rua.ua.es/server/api/core/bitstreams/72caed0a-eb66-48bd-ae0e-8b3d1928146b/content)
- The paper's Table I surveys earlier GPU SoC radiation tests: TK1 at p+/120 MeV (Wang 2017) and p+/15 MeV (Badía 2022); TX1 at p+/200 MeV (Wyrwas 2016); TX2 at p+/200 MeV (Wyrwas 2019); AGX Xavier at p+/105 MeV (Hiemstra 2020); Xavier NX/Industrial at p+/200 MeV (Rodriguez 2022); Snapdragon 835/845 with Ar/40 MeV (Guertin 2019). It summarizes that "mature node" CMOS was about 10× more susceptible to SEE than FinFET, that no SEL was seen, and that SEFIs dominate over SEUs. — [PDF](https://rua.ua.es/server/api/core/bitstreams/72caed0a-eb66-48bd-ae0e-8b3d1928146b/content)
- J. M. Badía et al., "Reliability Evaluation of LU Decomposition on GPU-Accelerated SoC Under Proton Irradiation", IEEE TNS 69:1467–1474 (2022). TK1 (28 nm, Kepler) at 15 MeV protons. More intensive GPU use raises the cross section (+15% for the block algorithm); most errors hang the OS. — [Semantic Scholar](https://www.semanticscholar.org/paper/Reliability-Evaluation-of-LU-Decomposition-on-Under-Bad%C3%ADa-Le%C3%B3n/ccb1497d86a43d59b5f1bcd7f3d8a51eb5f7efef)

**I. Stand-alone DRAM heavy-ion data, for use as a DRAM reference: Dufour et al. (Alter/CNES), "Compendium of SEE and TID Test Results for DDR4 SDRAM memories", RADECS 2022, DOI 10.1109/RADECS55911.2022.10412562** — [PDF](https://www.altertechnology.fr/wp-content/uploads/2024/11/Compendium-of-SEE-and-TID-Test-Results-for-DDR4-SDRAM-memories_Paper.pdf)
- Heavy-ion SEU Weibull saturation per bit for five COTS DDR4 parts (Micron MT40A256M16LY and MT40A512M16JY, Samsung K4A4G165WF, Hynix H5AN8G6NCJR, etc.):
  - about 2.6e-11 cm²/bit (static and dynamic, LETth 0.5)
  - 8.5e-11 / 2.3e-11 cm²/bit (static/dynamic, LETth 0.5)
  - 4.4e-12 cm²/bit (LETth 1.45)
  - 1.14e-10 / 4.49e-11 cm²/bit (static/dynamic, LETth 0.5)
  
  Stuck bits and MBUs were observed. — [PDF](https://www.altertechnology.fr/wp-content/uploads/2024/11/Compendium-of-SEE-and-TID-Test-Results-for-DDR4-SDRAM-memories_Paper.pdf)
- Proton DRAM reference (abstract only): Park, Jeon et al., "Soft error study on DDR4 SDRAMs using a 480 MeV proton beam", IRPS 2017. DDR4 showed about 45% higher single-bit-upset cross section than DDR3 from the same vendor. The per-bit value was not accessible. — [IEEE](https://ieeexplore.ieee.org/document/7936404/)

**J. Other Jetson tests (no SRAM/DRAM split)**
- Jetson Xavier NX for the SONATE-2 nanosatellite: protons at 30, 50 and 68 MeV, 10 krad Co-60. Two SEEs occurred: one reboot and one eMMC memory error. Taken from a search summary of the bibliographic entry; the primary text was not read. — [BibSonomy entry](https://www.bibsonomy.org/bibtex/286d756c37afc19180d1181f97437e9f8/tobiasherbst)
- Memon et al., "Where Linux Breaks Under Radiation" (arXiv 2503.03722, 2026). 20–58 MeV protons on RPi Zero 2W (40 nm, LPDDR2), i.MX8M Plus (14 nm FinFET; its LPDDR4 was outside the beam) and an ECP5 RISC-V board. Linux-SEFI cross sections range from 1.98e-10 to 7.63e-9 cm². No per-bit SRAM vs DRAM decomposition. These are not GPUs. — [arXiv](https://arxiv.org/html/2503.03722v4)

### Inferences
- **Per-bit SRAM vs DRAM in one GPU platform is a literature gap.** The closest pairing uses different devices. Snapdragon 835 L2 SRAM is about 1e-11 cm²/bit at LET ≤ 6.9 ([Guertin](https://nepp.nasa.gov/docs/etw/2020/17-JUN-WED/1130-Guertin-NEPP-ETW-CL20-2459-ARM-v5.pdf)). Commercial DDR4 saturates at about 4e-12 to 1e-10 cm²/bit ([Alter/CNES](https://www.altertechnology.fr/wp-content/uploads/2024/11/Compendium-of-SEE-and-TID-Test-Results-for-DDR4-SDRAM-memories_Paper.pdf)). So under **heavy ions**, per-bit SRAM and DRAM are of the same order: DRAM is not intrinsically lower per bit. Caveats: different LETs (DRAM values are saturation fits), different nodes, and the SRAM value is from a live, ECC-reporting cache.
- Orin NX LPDDR5, rough upper-bound estimate. Assume the 16 GB (about 1.37e11 bits) is split over the two irradiated RAM regions, about 6.9e10 bits each. A region SEU cross section of about 2–2.6e-10 cm² would then correspond to about 3–4e-21 cm²/bit at 480 MeV. This is not a valid per-bit DRAM cell cross section: the logged events were interface/communication errors and no data corruption was detected. It only shows that no measurable LPDDR5 data-cell upsets were seen at about 1e10 p/cm². Treat it as an estimate, not a measurement.
- The Xavier result (all events in L2/L3 data and tag arrays, none in the GPU) and the Orin result (GPU SEU about 9× smaller than CPU SEU) suggest that, per device, CPU-complex SRAM dominates the SoC proton response. GPU-internal SRAM is either well protected (ECC on SM/MMU structures, per the Orin dump) or under-exercised.
- For an RTX 4090 (TSMC 4N, GDDR6X, 72 MiB L2) there is no proton or heavy-ion structure-resolved data in this literature. Per-bit values would have to be extrapolated from FinFET SRAM and DRAM device data.

### Gaps
- Per-bit numbers in Wang et al. 2017 (Tegra K1 caches), Hiemstra et al. 2020 (AGX Xavier) and Slater et al. 2023 (Orin AGX) could not be read: IEEE and ResearchGate returned 403/418.
- Xavier L3 per-bit cross section is given only graphically (Fig. 8) in Rodriguez-Ferrandez 2022.
- No proton per-bit value for LPDDR4/LPDDR5 or GDDR5/6/6X was found in these searches. Wyrwas's NEPP TX1/TX2/Xavier/AMD reports are black-box.
- No proton or heavy-ion per-bit data were found for GPU register files, shared memory, or GPU L1/L2 on any COTS GPU, discrete or embedded. Desktop GTX/RTX/Titan proton tests with structure attribution were not found.
- Not covered here: neutron per-structure GPU data (Rech/UFRGS: K40/Volta register file, L1, L2, shared memory, DDR) and HPC field data. A pointer found: "Analyzing the Influence of Memory and Workload on the Reliability of GPUs Under Neutron Radiation". — [ResearchGate](https://www.researchgate.net/publication/379775253_Analysing_the_influence_of_memory_and_workload_on_the_reliability_of_GPUs_under_neutron_radiation)
- Not reviewed: Bruhn/Unibap (AMD G-series), Lentaris/Furano (Myriad VPUs, which are not GPUs), Kosmidis heavy-ion GANIL campaign on Orin NX/Xavier NX (mentioned only in a search summary, no primary source found), and Lüdke.

## Q2. What SRAM:DRAM per-bit cross-section ratio emerges, and under what particle conditions?

### Takeaway
No source states a measured SRAM:DRAM per-bit ratio for a GPU platform. The available numbers come from mismatched devices and conditions. For heavy ions the ratio is about 0.1 to 3 (SRAM L2 about 1e-11 cm²/bit vs DDR4 about 4e-12 to 1e-10 cm²/bit). For protons no defensible ratio can be formed from GPU-platform data. In the SoC tests, cache SRAM dominated the observed events: DRAM data-cell upsets were never isolated, and DRAM effects showed up as interface or SEFI events.

### Cited Findings
- Snapdragon 835 L2: about 1e-11 cm²/bit, flat from LET ≈ 2 to 6.9 MeV·cm²/mg (heavy ion, JPL). — [Guertin 2020](https://nepp.nasa.gov/docs/etw/2020/17-JUN-WED/1130-Guertin-NEPP-ETW-CL20-2459-ARM-v5.pdf)
- COTS DDR4 heavy-ion saturated SEU: 4.4e-12 to 1.14e-10 cm²/bit, LETth about 0.5–1.45 MeV·cm²/mg. — [Dufour et al., RADECS 2022](https://www.altertechnology.fr/wp-content/uploads/2024/11/Compendium-of-SEE-and-TID-Test-Results-for-DDR4-SDRAM-memories_Paper.pdf)
- Jetson Xavier (200 MeV p): the SoC SEE cross section is dominated by L2/L3 SEUs, with SEFIs from cache-tag parity. DRAM is not resolved. — [Rodriguez-Ferrandez 2022](https://upcommons.upc.edu/server/api/core/bitstreams/80d67a8c-34a3-408b-aaf1-5bc5bc48b1c1/content)
- Jetson Orin NX (480 MeV p): SoC SEU 2.5–4.4e-9 cm² vs LPDDR5 region SEU 2.0–2.6e-10 cm² per region, with no DRAM data corruption detected. — [Rodriguez-Ferrandez 2024](https://upcommons.upc.edu/server/api/core/bitstreams/ef814395-1619-43ee-ba00-6a89f0c16d83/content)
- Snapdragon PoP: SEFI cross sections are "much higher than the DDR2 SEFI cross section". — [Guertin 2020](https://nepp.nasa.gov/docs/etw/2020/17-JUN-WED/1130-Guertin-NEPP-ETW-CL20-2459-ARM-v5.pdf)
- Analytical proton reference (generic SRAM, not GPU): about 2e-15 cm²/bit saturated upset cross section at 200 MeV, as reported in a search summary of this modeling paper. Not verified in its full text. — [arXiv 2402.12983](https://arxiv.org/pdf/2402.12983)

### Inferences
- Per device under protons, the Orin data imply that the irradiated SoC region shows about 10–20× more corrected events than each LPDDR5 region. Per bit the gap is far larger, because the DRAM has orders of magnitude more bits (estimate, see Q1).
- The REMU assumption of a uniform BER across a unified LPDDR4 is not backed by GPU-platform beam data showing DRAM cell upsets. The beam data instead point to on-chip SRAM (caches, tags) and control logic as the observed SEE sources. A 4090 extension that adds L2-resident faults is consistent with this evidence. The SRAM:DRAM per-bit weighting would still have to come from generic SRAM/DRAM device data, so it is an estimate.

### Gaps
- No same-campaign, same-spectrum per-bit SRAM vs DRAM measurement on any GPU.
- Proton per-bit DRAM numbers for modern DDR4/LPDDR4/LPDDR5/GDDR6 were not retrieved. The Park et al. IRPS 2017 abstract gives only relative comparisons.

## Q3. Did REMU (DAC 2025) or related Tsinghua work cite DRAM vs SRAM/cache rates?

### Takeaway
No. REMU and its predecessor cite generic, technology-agnostic "COTS memory" rates of 1e-7 to 1e-6 /bit/day, drawn from CNES CARMEN-2 flight data, a neutron SRAM-FPGA test and a HotNets paper. They then apply these rates to the Jetson Xavier NX's unified LPDDR4 DRAM. Neither paper distinguishes SRAM/cache from DRAM rates. Neither reports measured in-orbit error rates from their Chaohu-1 Jetson payload.

### Cited Findings
- REMU: L. Xu, M. Wang, H. Qiu, J. Liu, Y. Li, H. Li, "REMU: Memory-aware Radiation Emulation via Dual Addressing for In-orbit Deep Learning System", DAC 2025, DOI 10.1109/DAC63849.2025.11132935. Platform: Jetson Xavier NX with 16 GB 128-bit LPDDR4 (unified CPU/GPU memory). Dual addressing spans virtual, physical and DRAM address spaces. — [ACM DL](https://dl.acm.org/doi/10.1109/DAC63849.2025.11132935); [IEEE PDF](https://ieeexplore.ieee.org/iel8/11132383/11132091/11132935.pdf) (read from the local copy in the repo)
- REMU text: SEUs and MCUs occur "at a non-negligible rate (e.g., at least 1 × 10−7 /bit/day [12])". [12] is Bezerra, Ecoffet, Lorfèvre, Samaras, Deneau, "CARMEN2/MEX: An in-flight laboratory for the observation of radiation effects on electronic devices", RADECS 2011. — [IEEE PDF](https://ieeexplore.ieee.org/iel8/11132383/11132091/11132935.pdf)
- REMU text: "The probability of SEUs can reach 10−7 to 10−6 /bit/day [9]". [9] is Samaras et al., "CARMEN-2: In flight observation of non destructive single event phenomena on memories", RADECS 2011. — [IEEE PDF](https://ieeexplore.ieee.org/iel8/11132383/11132091/11132935.pdf)
- REMU sets BER to 1e-7 and 1e-6 "based on the statistics of existing real-world radiation tests [11]". [11] is Fabero et al., "Single event upsets under 14-MeV neutrons in a 28-nm SRAM-based FPGA in static mode", IEEE TNS 2020. That is a **terrestrial-neutron SRAM** test, applied in REMU to LPDDR4 DRAM. — [IEEE PDF](https://ieeexplore.ieee.org/iel8/11132383/11132091/11132935.pdf)
- Predecessor: M. Wang, H. Qiu, L. Xu, D. Wang, Y. Li, T. Zhang, J. Liu, H. Li, "A Case for Application-Aware Space Radiation Tolerance in Orbital Computing" (RedNet), arXiv 2407.11853 (2024).
  - It cites a 3-year in-orbit record on JASON-2 (CARMEN-2, Samaras et al.) of "about 4.76 × 10−7 /bit/day", giving "150+ bit errors per day" for a 40 MB footprint.
  - It also states "single-bit errors can reach 10−7 to 10−6 per bit per day", citing Wang et al., "Mars Attacks! Software Protection Against Space Radiation", HotNets 2023.
  - It cites a "private correspondence with leo smallsat operator, 2024".
  
  — [arXiv 2407.11853](https://arxiv.org/pdf/2407.11853)
- RedNet used a Jetson Xavier NX payload "deployed as a payload on Chaohu-1 SAR satellite launched in 2022" (by Spacety), and models the LPDDR4 DRAM hierarchy. Its Chaohu-1 experiments are ground tests with a payload identical to the flight unit. No in-orbit SEU counts are reported. — [arXiv 2407.11853](https://arxiv.org/pdf/2407.11853)
- Comparison point, estimated (OMERE, proton-only AE8+ESP, 800 km/98° LEO): **Xavier L2 ≈ 9e-9 /bit/day and L3 ≈ 1.8e-9 (Industrial) to 3.5e-9 (NX) /bit/day.** GEO: L2 ≈ 3e-9 /bit/day. The table labels these rows "SEFI/bit", which is ambiguous; they are derived from the 200 MeV proton per-bit data. — [Rodriguez-Ferrandez 2022, Table V](https://upcommons.upc.edu/server/api/core/bitstreams/80d67a8c-34a3-408b-aaf1-5bc5bc48b1c1/content)

### Inferences
- REMU's 1e-7 /bit/day is about 10–50× higher than the BSC/ESA LEO estimate for ECC-protected Xavier L2/L3 SRAM (about 2–9e-9 /bit/day). That estimate is proton-only and excludes GCR heavy ions and SAA-dependent orbit choice, so it is a lower bound. REMU's rate is a conservative, memory-agnostic figure from older-technology flight memories (CARMEN-2 on JASON-2 at 1336 km, which crosses the inner belt more heavily). REMU applies it to DRAM without justifying the technology.

### Gaps
- Which CARMEN-2 memory types (SRAM vs SDRAM) produced the 4.76e-7 figure was not verified; the RADECS 2011 primary text was not accessed.
- No other Tsinghua-group paper (Y. Li / H. Li / H. Qiu) with SRAM vs DRAM rates was found.

## Q4. Are there in-orbit measured rates for GPU caches vs DRAM?

### Takeaway
None found. Jetson-class GPUs have flown on several missions: Chaohu-1 (Xavier NX), SpIRIT (Jetson Nano), Aitech S-A1760 (TX2i) and a PolarFire SoC + Orin NX in-orbit validation. None of the public sources found reports per-structure in-orbit SEU rates for GPU caches vs DRAM. Available in-orbit per-bit rates come from dedicated memory experiments (CARMEN-2, Alsat-1), not GPUs.

### Cited Findings
- First in-orbit validation of a heterogeneous Microchip PolarFire SoC + NVIDIA Jetson Orin NX architecture for onboard AI (J. Korean Soc. Aeronaut. Space Sci./Springer, 2026). Per a search summary, the campaign "did not measure SEU rates, inference correctness under bit flips, or ECC event counters". — [Springer](https://link.springer.com/article/10.1007/s42405-026-01130-w)
- SpIRIT (University of Melbourne, launched 2023): Loris AI payload on a Jetson Nano. The paper covers design and mitigation; it reports no in-orbit SEU/ECC statistics, only that the Jetson Nano had been operating in orbit. — [arXiv 2404.08399](https://arxiv.org/pdf/2404.08399)
- The Jetson TX2i is used in Aitech's S-A1760 with LEO flight heritage. This comes from a search summary; no error data. — [arXiv 2302.08952 (Pfandzelter & Bermbach)](https://arxiv.org/pdf/2302.08952)
- The Chaohu-1 Xavier NX payload has flown since 2022, but no in-orbit error rates have been published (see Q3). — [arXiv 2407.11853](https://arxiv.org/pdf/2407.11853)
- HPE Spaceborne Computer (ISS): the searches returned no public per-structure memory-error rates.
- In-orbit memory rates (non-GPU): JASON-2/CARMEN-2 about 4.76e-7 /bit/day, as cited by Tsinghua. — [arXiv 2407.11853](https://arxiv.org/pdf/2407.11853)
- Alsat-1 (LEO) RAM SEU/MBU observations, with a search-summary claim of "one error bit in one million bits per day, 80% single-bit". That figure was not verified in the primary text. — [ResearchGate: Observations of SEUs and MBUs in RAMs on-board the Algerian satellite](https://www.researchgate.net/publication/224380922_Observations_of_single-event_upsets_and_multiple-bit_upsets_in_random_access_memories_on-board_the_Algerian_satellite)

### Inferences
- Any SRAM-vs-DRAM in-orbit rate for a GPU must currently be **estimated**. The approach is to fold ground per-bit data, such as the Xavier L2/L3 per-bit values and DDR4 heavy-ion Weibulls, with an orbit environment model (OMERE/CREME96). No flight measurement exists to anchor it.

### Gaps
- HPE Spaceborne Computer-1/-2 error logs: not found publicly.
- The Kosmidis/BSC GANIL heavy-ion test of Orin NX/Xavier NX and Orin tests "behind shielding with 63 MeV protons and 180 MeV/u Xe" appeared only in search summaries. No primary source was retrieved, so no numbers are given.
