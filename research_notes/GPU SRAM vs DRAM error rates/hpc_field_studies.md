# HPC Field Studies of GPU Memory Errors: On-chip SRAM (L2 / L1-shared / register file) vs Off-chip GPU DRAM (GDDR / HBM)

Scope note: only field (production-fleet) data is covered here. I mention neutron-beam results only where a field paper reports them next to its field data (Tiwari et al. HPCA 2015). All derived per-bit numbers are my own arithmetic. They are labelled as inferences and their normalization assumptions are stated.

Short answer: **only one public field study breaks GPU SBEs down by on-chip structure: Tiwari et al., HPCA 2015, on Titan K20X.** One more study (Tiwari et al., SC15, on Titan) breaks DBEs down by structure. I could only check that DBE split through a secondary citation. Every newer field study I found (Summit V100, Delta A100/H100, Polaris/Perlmutter A100) reports only device-memory errors, or does not attribute errors to a structure at all. I found no field study for Frontier MI250X that separates SRAM from HBM errors.

---

## Q1. What structure breakdowns do the field studies report, and can per-bit rates be derived?

### Takeaway
Only Titan (K20X, 18,688 GPUs) has a per-structure split. Over Feb 2013–Aug 2014, 98% of all corrected single-bit errors (SBEs) were in the 1.5 MB L2, and nearly all of them came from 10 cards. With those 10 cards removed, device memory (6 GB GDDR5) held about 94–96% of SBEs. Of the double-bit errors (DBEs), about 86% were in GDDR5 and 14% in the register file. Per-bit rates can be derived from these fractions plus the published K20X structure sizes (see Q2). The later V100/A100/H100 studies give per-GPU and per-GB device-memory rates only.

### Cited Findings

#### Study 1: Tiwari et al., HPCA 2015 (Titan + LANL Moonlight + neutron beam)
- **Citation:** D. Tiwari, S. Gupta, J. Rogers, D. Maxwell, P. Rech, S. Vazhkudai, D. Oliveira, D. Londo, N. DeBardeleben, P. Navaux, L. Carro, A. Bland, "Understanding GPU Errors on Large-scale HPC Systems and the Implications for System Design and Operation," IEEE HPCA 2015. DOI 10.1109/HPCA.2015.7056044 — [OSTI full text](https://www.osti.gov/servlets/purl/1185857)
- **System:** Titan at OLCF, 18,688 NVIDIA K20X (Kepler GK110), 28 nm. There are 14 SMs per GPU. Each SM has 64K registers, 64 KB of combined shared memory/L1 and a 48 KB read-only data cache. The SMs share 1,536 KB of L2 and 6 GB of GDDR5 — [OSTI](https://www.osti.gov/servlets/purl/1185857)
- **ECC state:** SECDED is on for the register files, shared memory, L1 and L2 caches and device memory. The read-only/texture cache has parity only — [OSTI](https://www.osti.gov/servlets/purl/1185857)
- **Data collection:** DBEs (XID 48) and page-retirement events (XID 63/64) come from console logs. SBEs come from nvidia-smi snapshots, so SBEs have per-structure counters but no continuous timestamps — [OSTI](https://www.osti.gov/servlets/purl/1185857)
- **Period:** about 18 months. SBE analysis covers Feb 2013 – Aug 2014 — [OSTI](https://www.osti.gov/servlets/purl/1185857)
- **Fleet-level SBE result:**
  - "less than 5% of the total GPU cards (899 cards) experienced one or more SBEs during Feb'13 to Aug'14."
  - "close to 98% of all SBEs occurred on only ten cards (out of 899 offender cards)."
  - The paper found no statistically sound correlation between SBEs and node or cage location.
  - [OSTI](https://www.osti.gov/servlets/purl/1185857)
- **SBE breakdown by structure (Fig. 12 and text):**
  - "overall the L2 Cache region is the major contributer in SBE events, showing that 98% of the SBE events happen in that structure alone."
  - "the top 10 offenders also show the same behavior (99% of the SBE events occurred in the L2 Cache)."
  - "once we eliminate the top 10 offenders, the device memory is the structure where most of the SBEs occur (96% of all SBEs)."
  - "the fraction of SBEs occurring in different structures is not proportional to the respective structure sizes."
  - The Fig. 12 bar labels in the extracted PDF read 98%/2% (all cards), 99%/1% (top 10) and 94%/6% (all but top 10). The text says 96% for device memory after excluding the top 10.
  - The legend lists L2 Cache, Device Memory, L1 Cache, Register File and Texture Memory. Only two segments carry labels, so L1, register file and texture are at most a few percent.
  - [OSTI](https://www.osti.gov/servlets/purl/1185857)
- **Internal inconsistency:** Observation 6 says the opposite of the text above it: "GPU cards which experience most of the SBEs are likely to have all the SBEs occur in the device memory instead of the L2 cache." The body text and the figure say the top-10 cards' SBEs are 99% L2. Treat the attribution of the dominant offender cards with care — [OSTI](https://www.osti.gov/servlets/purl/1185857)
- **Authors' recommendation:** device memory and L2 need better protection. L1, register file and texture memory "may not need additional costly protection schemes" — [OSTI](https://www.osti.gov/servlets/purl/1185857)
- **DBEs and page retirement:**
  - Six GPU cards accounted for 25% of all DBEs.
  - One card caused more than 10% of ECC page-retirement errors.
  - Operations staff stress-test and pull repeat-offender cards, which suppresses repeat DBEs.
  - The paper notes DBEs "may be sensitive to temperature."
  - [OSTI](https://www.osti.gov/servlets/purl/1185857)
- **Logging caveat:** in a LANL M2090 HPL test (279 node-hours), syslog recorded 27 DBEs but nvidia-smi captured only 1. The same kind of inconsistency was seen on a Titan card. The authors suspect the DBE takes the node down before the driver can persist the count — [OSTI](https://www.osti.gov/servlets/purl/1185857)
- **Neutron beam (not field data) in the same paper:**
  - Per-bit L2 and register-file cross sections were measured for K20 and C2050.
  - Kepler is lower than Fermi per bit.
  - L2 bits holding 0 are 40% more sensitive than bits holding 1. This asymmetry does not hold in the register file.
  - "only 4-6% errors are double bit errors in L2 cache and register file area."
  - [OSTI](https://www.osti.gov/servlets/purl/1185857)

#### Study 2: Tiwari et al., SC15 (Titan reliability lessons) — DBE split by structure
- **Citation:** D. Tiwari, S. Gupta, G. Gallarno, J. Rogers, D. Maxwell, "Reliability lessons learned from GPU experience with the Titan supercomputer at Oak Ridge Leadership Computing Facility," SC15. DOI 10.1145/2807591.2807666 — [ACM DL](https://dl.acm.org/doi/10.1145/2807591.2807666)
- **What I could verify:** the full text was not reachable in this session. The SC20 Ostrouchov paper summarizes it as follows: "This work observes an ECC double-bit mean-time between errors (MTBE) of about 160 hours, or about one per week, with 86% occurring in GPU memory and the rest in the GPU register file." The SC20 paper also says the SC15 "off the bus" events were a system-integration issue — [Ostrouchov et al. SC20, OSTI](https://www.osti.gov/servlets/purl/1771896)
- **Data window:** "Using Titan data from June 2013 to February 2015, Tiwari et al. and Nie et al. do not address the atypical failure mode [resistor failures]," which appeared in mid-2016 — [Ostrouchov et al. SC20](https://www.osti.gov/servlets/purl/1771896)
- **Related Titan MTTFs:** an NVIDIA/UT Austin SELSE 2019 model takes Titan GPU MTTFs of 144 h (DBE), 178 h (ECC page retirement) and 98 h (off-the-bus) "based on field measurement study in Titan." It cites Tiwari's CUG 2015 paper — [Lee et al., SELSE 2019](https://d1qx31qr3h6wln.cloudfront.net/publications/SELSE2019_GPUDenseChkptAnalysis.pdf)

#### Study 3: Nie et al., HPCA 2016 and follow-ups (Titan SBEs vs temperature, power and space)
- **Citation:** B. Nie, D. Tiwari, S. Gupta, E. Smirni, J. H. Rogers, "A large-scale study of soft-errors on GPUs in the field," HPCA 2016, pp. 519–530. DOI 10.1109/HPCA.2016.7446091 — [ORNL record](https://impact.ornl.gov/en/publications/a-large-scale-study-of-soft-errors-on-gpus-in-the-field/)
  - The abstract says only that it "characterize[s] and quantif[ies] different kinds of soft-errors on the Titan supercomputer's GPU nodes."
  - **I could not retrieve the full text** (IEEE paywall; ResearchGate returned 403), so its per-structure numbers are not reported here.
  - A later paper says "Periodicity of memory errors is observed in the Titan supercomputer [Nie HPCA16]" and that "The temperature effect is found to be related to GPU memory errors [Nie HPCA16, Nie MASCOTS17]" — [Zhu et al. 2025](https://arxiv.org/abs/2508.03513)
- **Follow-up:** B. Nie, J. Xue, S. Gupta, T. Patel, C. Engelmann, E. Smirni, D. Tiwari, "Machine Learning Models for GPU Error Prediction in a Large Scale HPC System," DSN 2018 — [PDF](https://www.christian-engelmann.info/publications/nie18machine.pdf)
  - Titan SBE data from Jan to June 2015, "more than 60 million node hours."
  - SBEs are recorded before and after each batch job, not timestamped individually.
  - Device memory, L2, instruction cache, register files, shared memory and L1 are SECDED. The read-only data cache is parity.
  - "80% of error offender nodes experience a soft error on less than 20% of the total days."
  - Fewer than 20% of applications see 90% or more of the SBEs.
  - Spearman correlation between the cabinet temperature map and the SBE-offender-node map is "as low as 0.07."
  - No per-structure SBE split appears in the parts I extracted.
- **Second follow-up:** B. Nie, J. Xue, S. Gupta, C. Engelmann, E. Smirni, D. Tiwari, "Characterizing temperature, power, and soft-error behaviors in data center systems," MASCOTS 2017, pp. 22–31 (citation as given in [Zhu et al. 2025](https://arxiv.org/abs/2508.03513)). The NSF PAR PDF (https://par.nsf.gov/servlets/purl/10065577) could not be downloaded.

#### Study 4: Ostrouchov et al., SC20 (Titan GPU lifetimes, 2012–2019)
- **Citation:** G. Ostrouchov, D. Maxwell, R. A. Ashraf, C. Engelmann, M. Shankar, J. H. Rogers, "GPU Lifetimes on Titan Supercomputer: Survival Analysis and Reliability," SC20 — [OSTI](https://www.osti.gov/servlets/purl/1771896); data and code at [github.com/olcf/TitanGPULife](https://github.com/olcf/TitanGPULife)
- **What it tracks:** only two event types, DBE ("an error correcting code (ECC) detection in GPU memory") and Off-The-Bus (OTB). From 2016, OTB and DBE "were found to be the 'signature' event of the GPU board failing resistor and a trigger for GPU replacement." The resistor failures were traced to silver-sulfide corrosion — [OSTI](https://www.osti.gov/servlets/purl/1771896)
- **Batch counts:** the old batch had 5,320 DBE+OTB events. The new batch (inserted 2016 or later) had 127 — [OSTI](https://www.osti.gov/servlets/purl/1771896)
- **System MTBF:** above one day (33 h) before 2015-Q4, when the overall MTBF was "determined solely by DBE events" in some quarters. It fell to 7.7 h in 2015-Q4 — [OSTI](https://www.osti.gov/servlets/purl/1771896)
- **Relevance:** Titan DBEs after 2015 are mostly a hardware-degradation signature (failing resistor), not radiation. There is no SRAM/DRAM split.

#### Study 5: Di Martino et al., DSN 2014 (Blue Waters, K20X XK7 partition)
- **Citation:** C. Di Martino, Z. Kalbarczyk, R. K. Iyer, F. Baccanico, J. Fullop, W. Kramer, "Lessons learned from the analysis of system failures at petascale: The case of Blue Waters," DSN 2014, pp. 610–621 — [ResearchGate](https://www.researchgate.net/publication/283380828_Lessons_Learned_from_the_Analysis_of_System_Failures_at_Petascale_The_Case_of_Blue_Waters)
- **Data:** 261 days of failure data, per [Lee et al. SELSE 2019](https://d1qx31qr3h6wln.cloudfront.net/publications/SELSE2019_GPUDenseChkptAnalysis.pdf). The XK7 nodes use the same architecture as Titan but "did not experience the same GPU failure mode" (the resistor problem) — [Ostrouchov SC20](https://www.osti.gov/servlets/purl/1771896)
- **GPU vs CPU memory (secondary, unverified):** a search-engine summary of Sullivan et al., MICRO 2021 says that "a field study of the Blue Waters system found that the DUE rate per GB of Kepler-era GDDR5 was roughly 5 times that of the chipkill-protected CPU memory" — [Sullivan et al., MICRO 2021, ACM DL](https://dl.acm.org/doi/10.1145/3466752.3480111). I could not open the full text (403), so treat this as secondary.
- **Gap:** I found no per-structure GPU SRAM split for Blue Waters.

#### Study 6: Oles et al., ICS 2024 (Summit V100, HBM2) — DBEs only
- **Citation:** V. Oles, A. Schmedding, G. Ostrouchov, W. Shin, E. Smirni, C. Engelmann, "Understanding GPU Memory Corruption at Extreme Scale: The Summit Case Study," ACM ICS 2024. DOI 10.1145/3650200.3656615 — [OSTI full text](https://www.osti.gov/servlets/purl/2378092)
- **System and ECC:** Summit has 4,626 nodes with 6 V100 each, 27,756 GPUs in total. HBM2 is protected by SECDED. A page is retired after one DBE or two SBEs at the same address, with a limit of 64 pages — [OSTI](https://www.osti.gov/servlets/purl/2378092)
- **Counts (Jan 2020 – May 2022):**
  - 295 DBEs (XID 48), 1,430 page-retirement events and 35,791 page-retirement failures (PRFs).
  - These affected 112, 1,011 and 138 GPUs respectively.
  - The top-1 GPU accounts for 10.5% of DBEs.
  - 33 GPUs had multiple DBEs, and those DBEs come in "streaks": the median time between them is 20 h, the mean almost 20 days.
  - After removing DBEs within 120 h of a previous one, 166 "independent" DBEs remain.
  - 97% of PRFs occurred on GPUs with no DBEs and are traced to application behavior.
  - [OSTI](https://www.osti.gov/servlets/purl/2378092)
- **Correlates:** DBEs correlate with short- and long-term high power consumption and with GPU placement in the node and cooling path — [OSTI](https://www.osti.gov/servlets/purl/2378092)
- **What is missing:** SBE counts, and any split of the DBEs into SRAM versus HBM2. The public Summit dataset "does not contain SBE information" — [Zhu et al. 2025](https://arxiv.org/abs/2508.03513)
- **Conflicting Summit table in Zhu et al. 2025 (Table 6):**
  - It lists 28,471 GPUs, 1,026 logged days, 1,088 DBEs, 1,008 DBE events and a "DBE rate 0.00022 per GPU per day."
  - It also lists 124 DBE-occurring GPUs as "(2.6%)."
  - These numbers are internally inconsistent: 1,088/(28,471×1,026) = 3.7e-5 per GPU-day, and 124/28,471 = 0.44%. They also differ from Oles' 295 DBEs.
  - [Zhu et al. 2025](https://arxiv.org/abs/2508.03513)

#### Study 7: Cui et al., SC25 (NCSA Delta A100 + H100) — device memory only
- **Citation:** S. Cui, A. Patke, H. Nguyen, A. Ranjan, Z. Chen, P. Cao, G. Bauer, B. Bode, C. Di Martino, et al., "Story of Two GPUs: Characterizing the Resilience of Hopper H100 and Ampere A100 GPUs," SC '25. DOI 10.1145/3712285.3759821 — [arXiv 2503.11901](https://arxiv.org/abs/2503.11901)
- **Fleet:**
  - 448 A100 40 GB HBM2e, observed for 895 days (Oct 2022–Mar 2025).
  - 608 H100 96 GB HBM3 (GH200), observed for 146 days (Oct 2024–Mar 2025).
  - 11.7 M GPU-hours in total: 9.6 M on A100 and 2.1 M on H100.
  - [arXiv](https://arxiv.org/pdf/2503.11901)
- **Counts (Table 1, A100 / H100):**
  - DBE (XID 48): 1 / 17.
  - Uncorrectable ECC memory errors: 34 / 24. These are inferred from row-remap events (RRE) plus failures (RRF), and include "consecutive SBEs" (33 / 7).
  - XID 63 RRE: 34 / 16.
  - XID 64 RRF: 0 / 8.
  - XID 94 contained: 13 / 14.
  - XID 95 uncontained: 11 / 19.
  - [arXiv](https://arxiv.org/pdf/2503.11901)
- **Rates:**
  - Per-GPU MTBE for uncorrectable memory errors: 283,271 h on A100 and 88,768 h on H100, a 3.2× difference.
  - Per-GB MTBE: 11,330,826 h on A100 HBM2e and 8,521,728 h on H100 HBM3, 24% lower on H100.
  - Row remapping plus containment "alleviated" 92% of uncorrectable memory errors on H100.
  - [arXiv](https://arxiv.org/pdf/2503.11901)
- **What is missing:** correctable SBE counts and any on-chip SRAM (L2/RF) breakdown. The word "SRAM" does not appear in the extracted text.

#### Study 8: Zhu et al., 2025 (Delta, Polaris, Perlmutter — Ampere memory errors)
- **Citation:** Zhu Zhu et al. (GMU, UIUC, PNNL, LBNL, ALCF, NCSA, ANL, et al.), "Understanding the Landscape of Ampere GPU Memory Errors," arXiv 2508.03513 (2025; I did not confirm a venue) — [arXiv](https://arxiv.org/abs/2508.03513)
- **Fleets:**

  | Cluster | GPUs | Logged days | GPU-hours |
  |---|---|---|---|
  | Delta | 849 (400 A40 48 GB GDDR6 + 449 A100) | 388 | 7.32 M |
  | Polaris | 2,240 A100 | 259 | 14.66 M |
  | Perlmutter | 7,604 A100 | 326 | 45.79 M |

  A100 has 40 GB HBM2 (5 stacks), 40 MB L2 and 108 SMs. SECDED protects HBM2 "including [c]aches and register files" — [arXiv](https://arxiv.org/pdf/2508.03513)
- **Data source:** DCGM aggregated SBE/DBE counters. The paper describes its focus as "memory errors in the DRAM identified and reported by ECC." Polaris logged only DBEs — [arXiv](https://arxiv.org/pdf/2508.03513)
- **Counts (Table 2):**

  | Cluster | SBEs | SBE events | SBE-affected GPUs | SBEs per GPU-day | DBEs | DBE events | DBE-affected GPUs | DBEs per GPU-day |
  |---|---|---|---|---|---|---|---|---|
  | Delta | 173,936 | 3,324 | 43 (5.06%) | 0.53 | 9 | 2 | 2 | 0.000027 |
  | Polaris | not recorded | — | — | — | 39,837 | 44 | 68 | 0.069 |
  | Perlmutter | 7,010,888 | 2,016 | 344 (4.52%) | 2.83 | 17,926 | 77 | 35 | 0.0082 |

  [arXiv](https://arxiv.org/pdf/2508.03513)
- **Behavior:**
  - Errors are highly bursty. Delta saw 112,656 SBEs in one day.
  - SBEs and DBEs are correlated (29 Perlmutter GPUs had both).
  - Correlation with temperature, power or utilization is no stronger than 0.32.
  - [arXiv](https://arxiv.org/html/2508.03513v1)
- **What is missing:** no breakdown across L2, register files or other structures.

#### Other field data checked (no SRAM/DRAM split)
- **Tsubame-2 (K20X) and Tsubame-3 (P100), Taherin et al.:** based on failure and repair logs (897 and 338 failures). The paper reports "no GPU memory errors reported by ECC" (as characterized by [Zhu et al.](https://arxiv.org/abs/2508.03513)) — [OSTI 2204463](https://www.osti.gov/servlets/purl/2204463)
- **Haque & Pande, CCGrid 2010 (Folding@home):**
  - Non-ECC consumer GPUs, about 840 TB-hours of memory testing on more than 50,000 GPUs.
  - "two-thirds of GPUs exhibit sensitivity to memory faults in a pattern-dependent manner."
  - GT200 boards are about 10× lower than G80/G92.
  - DRAM only, with no SRAM data.
  - [PDF](https://cs.stanford.edu/people/ihaque/papers/gpuser.pdf)
- **Lim et al. (Titan "interplay" paper):** its error table (L2 FillECC, VbData ECC, etc.) is for the AMD Opteron CPU machine-check banks, not the GPU — [OSTI 1649409](https://www.osti.gov/servlets/purl/1649409)

### Inferences

#### Structure capacities used for normalization (data bits, excluding ECC check bits)
- **K20X (from the Tiwari spec above):**
  - Register file: 14 SMs × 64K × 32 b = 28 Mibit.
  - L1/shared: 14 × 64 KB = 7 Mibit.
  - Read-only cache: 14 × 48 KB = 5.25 Mibit, parity only.
  - L2: 1,536 KB = 12 Mibit.
  - GDDR5: 6 GB = 48 Gibit.
  - Ratios: DRAM/L2 = 4,096; DRAM/RF ≈ 1,755; DRAM/(RF+L2+L1) ≈ 1,046.
  - Cross-check: NVIDIA's K40 (GK110b, 15 SMs) beam paper uses "30 Mbit total RF, 7.86 Mbit total L1/Shared, 12 Mbit L2" — [Hari et al., arXiv 2005.01445](https://arxiv.org/pdf/2005.01445). That is consistent.
  - With ECC on, Kepler reserves part of GDDR5 for check bits, so usable bits are slightly below 48 Gibit. This does not change the order of magnitude.
- **A100 (Zhu et al.):** 40 MB L2 = 320 Mibit and 40 GB HBM2 = 320 Gibit, so DRAM/L2 = 1,024.
- **V100 (Summit):** 16 GB HBM2 per GPU. This is a general spec I did not verify in the papers fetched here.

#### Derived per-bit DRAM rates from newer fleets (device memory only, no SRAM counterpart)
- **Delta, uncorrectable errors (from per-GB MTBE):**
  - A100: 1/11.33 M h per GB ≈ 88 FIT/GB ≈ 11 FIT/Gbit.
  - H100: 1/8.52 M h per GB ≈ 117 FIT/GB ≈ 14.7 FIT/Gbit.
- **Summit, DBEs (Oles counts):** 295 DBEs / (27,756 GPUs × ~867 days × 24 h) ≈ 5.1e-7 per GPU-h ≈ 510 FIT/GPU ≈ 4 FIT/Gibit at 16 GB. Using the 166 independent DBEs gives ≈ 2.3 FIT/Gibit.
- **Perlmutter SBEs:**
  - Raw counts: 2.83 SBE/GPU/day ≈ 3.7e5 FIT/Gbit.
  - Event-based: 2,016 events / 59.5 M GPU-h ≈ 34,000 FIT/GPU ≈ 106 FIT/Gbit.
  - Both are dominated by about 4.5% of GPUs and bursts, so they are not representative of a radiation SER.

### Gaps
- **Nie et al. HPCA 2016 full text** (Titan SBEs by structure, temperature/power correlation) could not be retrieved. Its per-structure numbers, which may update Tiwari's Fig. 12 with 2015 data, are therefore missing.
- **Tiwari SC15 full text** was not retrieved. The 86% GDDR5 / 14% register-file DBE split and the ~160 h MTBE are verified only through Ostrouchov SC20's summary. Absolute DBE counts by structure are not available.
- **Absolute SBE counts per structure on Titan** are not published in the text I retrieved (only percentages), so absolute per-bit SBE FIT rates for L2 vs GDDR5 cannot be computed. Only the ratio can be derived.
- **Blue Waters DSN 2014 full text** was not retrieved. GPU structure-level data, if it exists there, is unknown.
- **No field study found for Frontier MI250X, Sierra/Lassen V100 or Polaris SRAM** (and no separate one for Perlmutter) that splits HBM errors from SRAM or cache errors. The DCGM and XID data used in the A100/H100 studies do not attribute errors to L2 or RF in the published analyses.
- **Delta SC25:** it is not stated whether XID 94/95 (contained/uncontained ECC errors) can originate in on-chip SRAM. The paper treats them as "memory."
- **Zhu et al.:** it is not stated whether the DCGM SBE/DBE fields used are DRAM-only or volatile totals across all structures.

---

## Q2. Do field studies show SRAM structures having higher per-bit rates than DRAM, and by what factor?

### Takeaway
Yes, on the one fleet with data (Titan K20X), but the factor depends heavily on whether the few faulty cards are included.
- **Whole fleet:** the L2 per-bit SBE rate comes out about 2×10^5 times the GDDR5 rate. This is driven by 10 cards and is almost certainly hard or intermittent faults.
- **Excluding those 10 cards:** the SRAM per-bit SBE rate is about 40–260× the DRAM rate. The figure is about 170–260× if the residual non-DRAM SBEs are attributed to L2 alone, and about 45–65× if they are spread over all SECDED SRAM (RF+L2+L1).
- **DBEs:** the register file's per-bit DBE rate is about 285× GDDR5's (14% vs 86% of DBEs; 28 Mibit vs 48 Gibit).

No newer field study allows this calculation.

### Cited Findings
- Titan SBE fractions:
  - All cards: L2 98%, device memory about 2%.
  - Top-10 cards: L2 99%.
  - Excluding the top 10: device memory 96% in the text, about 94% per the Fig. 12 label.
  - "the fraction of SBEs occurring in different structures is not proportional to the respective structure sizes"
  - [Tiwari et al. HPCA 2015](https://www.osti.gov/servlets/purl/1185857)
- K20X sizes: 64K registers per SM × 14 SMs, 64 KB L1/shared per SM, 1,536 KB L2, 6 GB GDDR5 — [Tiwari et al. HPCA 2015](https://www.osti.gov/servlets/purl/1185857)
- Titan DBEs: ~160 h MTBE, "86% occurring in GPU memory and the rest in the GPU register file" — [Ostrouchov et al. SC20 summarizing Tiwari SC15](https://www.osti.gov/servlets/purl/1771896)
- Beam context (not field): in neutron tests the per-bit cross sections of L2 and the register file were within a small factor of each other on K20 (Fig. 14a, normalized). Multi-bit (DBE) outcomes were only 4–6% of L2/RF beam errors — [Tiwari et al. HPCA 2015](https://www.osti.gov/servlets/purl/1185857)

### Inferences
Arithmetic and assumptions: per-bit ratio = (fraction_SRAM / bits_SRAM) / (fraction_DRAM / bits_DRAM), using K20X data-bit capacities.

| Case | Calculation | SRAM : DRAM per-bit ratio |
|---|---|---|
| (a) SBE, all Titan cards, L2 vs GDDR5 | (0.98/12 Mi) / (0.02/48 Gi) = 49 × 4,096 | ≈ 2.0×10^5 |
| (b) SBE, excluding top 10 cards, L2 vs GDDR5 (all residual SBEs assumed in L2) | (0.04/0.96) × 4,096, or (0.06/0.94) × 4,096 with the figure's 94% | ≈ 170 (text), ≈ 260 (figure) |
| (c) SBE, excluding top 10 cards, all SECDED SRAM pooled (RF+L2+L1 = 47 Mibit) | (0.04/0.96) × 1,046, or (0.06/0.94) × 1,046 | ≈ 44 – 67 |
| (d) DBE, register file vs GDDR5 | (0.14/0.86) × 1,755 | ≈ 285 |

Notes on each case:
- **(a)** This reflects about 10 defective cards whose L2 produced most SBEs. It is a hard-fault artifact, not a soft-error ratio.
- **(b)** This is an upper bound for L2 alone. Part of the residual may be RF, L1 or texture.
- **(c)** The most conservative field estimate of the SRAM:DRAM per-bit SBE ratio for "healthy" cards is therefore on the order of 10^1.5–10^2.5.
- **(d)** Absolute values: GDDR5 ≈ 6 FIT/Gibit (≈0.006 FIT/Mibit); RF ≈ 1.7 FIT/Mibit. These follow from 0.86/160 h and 0.14/160 h spread over 18,688 GPUs.

Interpretation:
- A register-file DBE under SECDED needs two flips in one word, so this is not a raw-upset ratio.
- Titan's DBEs are known to be concentrated on a few cards: six cards produced 25% of DBEs.
- The ~285× figure therefore mixes in hard faults as well.

Bottom line for the coordinator: the field data support "per-bit SRAM error rate ≫ per-bit GPU DRAM rate" by roughly 10^2 (range ≈ 40–300×) after removing obvious bad cards. The source is a single 2013–2014 Kepler/GDDR5 fleet, and the attribution is ambiguous. Note that this is the opposite of the conventional per-bit soft-error expectation for modern DRAM vs SRAM (see Q3).

### Gaps
- No field data for HBM-era GPUs (V100/A100/H100/MI250X) separate SRAM from DRAM, so the ~10^2 ratio cannot be checked for newer generations or for HBM.
- Without absolute per-structure SBE counts for Titan, absolute per-bit FIT rates for L2 cannot be computed. Only ratios are possible.
- Whether L2 SBE counters on K20X count repeated reads of the same unrepaired bit as separate SBEs (which would inflate L2 counts relative to DRAM) is not documented in the sources I found.

---

## Q3. Caveats: hard vs soft faults, retirement, temperature, concentration on a few cards

### Takeaway
Field GPU memory error counts are dominated by a small number of faulty devices and by bursts. Where structure is attributed, the per-bit ratios above therefore mostly measure hard or intermittent faults, not radiation SER. No field study separates radiation-induced from hard errors directly. They rely on proxies: concentration, repetition, page retirement and correlation with temperature or power.

### Cited Findings
- **Concentration and repetition:**
  - Titan: 98% of SBEs on 10 of 18,688 cards. Six cards produced 25% of DBEs. Repeat offenders are stress-tested and removed — [Tiwari HPCA 2015](https://www.osti.gov/servlets/purl/1185857)
  - Summit: 112 of ~27,756 GPUs had DBEs. The top-1 GPU had 10.5%. DBEs come in streaks (median 20 h apart on multi-DBE GPUs). The authors cite "manufacturing variability as a factor" and link DBEs to high power — [Oles ICS 2024](https://www.osti.gov/servlets/purl/2378092)
  - Ampere: about 5% of GPUs have any SBE. 112,656 SBEs in a single day on Delta. "Erroneous GPUs vary over time." Most error GPUs are active for only 1–3 weeks — [Zhu et al. 2025](https://arxiv.org/html/2508.03513v1)
- **Hardware degradation:** after 2016, Titan DBEs and Off-The-Bus events were the signature of a corroding board resistor (hard fault), and they triggered GPU replacement — [Ostrouchov SC20](https://www.osti.gov/servlets/purl/1771896)
- **Retirement and remapping change the counts:**
  - Pre-Ampere GPUs retire pages after 1 DBE or 2 SBEs at one address, with at most 64 pages — [Oles ICS 2024](https://www.osti.gov/servlets/purl/2378092)
  - A100/H100 support up to 512 row remappings plus containment — [Cui SC25](https://arxiv.org/pdf/2503.11901)
  - So SBEs from stuck DRAM cells are suppressed after retirement. On Summit, PRFs (35,791) were driven by 10 project-user combinations, that is, by applications — [Oles ICS 2024](https://www.osti.gov/servlets/purl/2378092)
- **Temperature:**
  - Titan DBEs "may be sensitive to temperature" — [Tiwari HPCA 2015](https://www.osti.gov/servlets/purl/1185857)
  - SBE-offender node locations show only 0.07 Spearman correlation with the temperature map — [Nie DSN 2018](https://www.christian-engelmann.info/publications/nie18machine.pdf)
  - On A100, correlations with temperature, power or utilization are ≤ 0.32 — [Zhu et al. 2025](https://arxiv.org/html/2508.03513v1)
  - Summit DBEs correlate with power intake — [Oles ICS 2024](https://www.osti.gov/servlets/purl/2378092)
- **Logging fidelity:** nvidia-smi snapshot counters miss DBEs (27 in syslog vs 1 in nvidia-smi on an M2090 test) — [Tiwari HPCA 2015](https://www.osti.gov/servlets/purl/1185857). Counting raw counter increments vs "events" changes rates by orders of magnitude (Delta: 173,936 SBEs vs 3,324 events) — [Zhu et al. 2025](https://arxiv.org/pdf/2508.03513)
- **Raw vs grouped DBE counts:**
  - Oles: 295 DBEs; 166 "independent."
  - Zhu Table 6 for Summit: 1,088 DBEs / 1,008 events.
  - The same fleet gives rates that differ by a large factor depending on grouping and window.
  - [Oles](https://www.osti.gov/servlets/purl/2378092); [Zhu](https://arxiv.org/pdf/2508.03513)

### Inferences
- **Hard faults dominate the field SRAM signal.**
  - The Titan whole-fleet ratio (~2×10^5) is a weak-cell or defect signature in a few L2s.
  - Even the "excluding top-10" ratio (~40–260×) almost certainly still contains intermittent or weak-cell contributions.
  - For a radiation-only per-bit SRAM:DRAM ratio, beam data (covered by other researchers) are a better basis. The field ratio is best treated as an upper bound on the "effective" ratio a system sees.
- **Observation bias pushes the measured ratio up.**
  - ECC errors are detected only when data is read.
  - L2 and RF are read continuously, while most of 6–96 GB of DRAM may sit idle or unread between overwrites, and there is no evidence of DRAM scrubbing on these GPUs.
  - Field counts therefore under-sample DRAM upsets relative to SRAM.
  - This is an inference, not stated in the sources.
- **Repeated counting may push it up further.** If corrected data is not written back, one persistent flipped bit can be counted many times. This may contribute to the extreme L2 counts on offender cards. This is an inference: the documentation was not found.
- **Unit conventions:** 1 FIT/Mbit = 1e-15 upsets per bit-hour. The DBE-derived values above (GDDR5 ~6 FIT/Gibit, Delta HBM ~11–15 FIT/Gbit, Summit HBM2 ~2–4 FIT/Gibit) are uncorrectable-error rates and are much lower than raw SBE/upset rates.

### Gaps
- No field study reports a soft vs hard classification per structure (for example, unique-address analysis for L2 vs DRAM SBEs) for GPUs.
- No field study reports DRAM scrubbing or patrol-read behavior for the GPUs studied, which is needed to correct for observation bias.
- Altitude and neutron-flux normalization is not reported in the field studies. Titan, Summit and Delta are at low altitude; Polaris and Perlmutter are also low; none give flux.
