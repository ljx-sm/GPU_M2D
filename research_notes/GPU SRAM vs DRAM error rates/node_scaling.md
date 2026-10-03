# Per-bit soft-error rate (SER) scaling: FinFET SRAM caches vs DRAM/graphics DRAM

Conventions used in these notes:
- FIT = failures per 10^9 device-hours. FIT/Mb = FIT per 10^6 bits.
- Conversion of a high-energy neutron cross section to a sea-level rate uses the JESD89A NYC reference flux of about 13 n/cm^2/h (E > 10 MeV): FIT/Mb ≈ sigma[cm^2/bit] x 13 x 1e9 x 1e6 = sigma x 1.3e16. Sources quote this flux as "approximately 13 neutrons per cm2 per hour" ([Agiakatsikas et al., arXiv 2303.08098](https://arxiv.org/pdf/2303.08098)) and 14 n/cm^2/h ([Sullivan et al., MICRO 2021](https://www.mbsullivan.info/attachments/papers/sullivan2021characterizing.pdf)). Wherever these notes convert a cross section with this formula, the conversion is marked "(converted)" and is my arithmetic, not the source's.
- Every value is tagged MEASURED (beam, field or real-time test), MODELED, or ASSUMED.

---

## Q1. How does FinFET SRAM per-bit SER compare with planar SRAM, and what values exist for 7 nm and 5 nm?

### Takeaway
Measured data from several sources agree on the trend. Moving from planar (28/20/32 nm) to FinFET (22/16/14 nm) cut SRAM per-bit SER by about 3x to more than 10x, depending on the particle and on the cell. The reduction continued down to 7 nm. At 5 nm, per-bit SER rose again, by about 2x for low-LET particles compared with 7 nm. The best public absolute numbers for a TSMC-16FF-class logic-process SRAM put high-energy-neutron per-bit cross sections at about 3e-16 to 1.5e-15 cm^2/bit, or roughly 3 to 20 FIT/Mb at NYC sea level. Alpha contributions are about 0.1 to 0.2 FIT/Mb when low-alpha packaging is used. I found no public absolute FIT/Mb values for 7 nm or 5 nm SRAM; those papers report only normalized values.

### Cited Findings
**Absolute per-bit data, planar to FinFET. Xilinx/AMD FPGA SRAM made on TSMC processes (MEASURED: LANSCE beam per JESD89A/89-3A; real-time NYC-corrected per JESD89A/89-1A)** — [Xilinx UG116 v10.9 Device Reliability Report, Sept 2018 (mirror)](https://0x04.net/~mwk/xidocs/ug/drr.pdf)
- The table below lists BRAM, which is a 6T-style dense SRAM block and the closer analogue to a cache. Units: LANSCE neutron cross section in cm^2/bit; thermal-neutron, alpha and real-time rates in FIT/Mb.

| Node | Family | Neutron cross section | Thermal | Alpha | Real-time |
|---|---|---|---|---|---|
| 130 nm | Virtex-II Pro | 3.91e-14 | – | – | 770 |
| 90 nm | Virtex-4 | 2.74e-14 | – | – | 484 |
| 65 nm | Virtex-5 | 3.96e-14 | – | – | 692 |
| 40 nm | Virtex-6 | 1.14e-14 | 1.4 | 120 | 213 |
| 28 nm planar | Kintex-7/Virtex-7 | 5.57e-15 | 1.8 | 39 | 66 |
| 20 nm planar | UltraScale | 4.43e-15 | 1.1 | 16 | 44 |
| 16 nm FinFET | UltraScale+ | 9.82e-16 | 4.7 | 0.2 | 20 |

- Configuration RAM (CRAM) in the same report: 28 nm Kintex-7 is 5.69e-15 cm^2/bit and 67 FIT/Mb real-time. 20 nm is 2.55e-15 and 30 FIT/Mb. 16 nm UltraScale+ is 2.67e-16 cm^2/bit, with 0.35 FIT/Mb thermal, 0.1 FIT/Mb alpha and 6 FIT/Mb real-time. Notes in the report: the UltraScale+ alpha data "is based on alpha foil testing and package alpha emissivity of 0.001 counts/cm2/hr". The real-time data comes from the Rosetta experiment, includes neutrons, protons and thermal-neutron secondaries, and excludes alpha particles. Experiments ran "at ambient temperature with typical power supply voltages". — [UG116 v10.9](https://0x04.net/~mwk/xidocs/ug/drr.pdf)
- Ratios computed from the table: going from 28 nm planar to 16 nm FinFET reduces BRAM per bit by about 5.7x (cross section) and 3.3x (real-time FIT/Mb). CRAM drops about 21x (cross section) and 11x (real-time). The alpha rate drops about 200x (39 to 0.2 FIT/Mb), but part of that comes from the stated low-emissivity package assumption.
- A separate summary of the 16 nm XCZU9EG MPSoC (TSMC 16FF+) reports that CRAM and BRAM static cross section per bit "was reduced by 20X and 16X, respectively, compared to the AMD Kintex-7 FPGA that uses 28nm TSMC's HKMG process". The same paper tabulates neutron (≥10 MeV) cross sections of CRAM 1.1e-16 to 3.4e-16 and BRAM 4.1e-16 to 3.02e-15 cm^2/bit across different test campaigns. For the processor-subsystem SRAMs (ARM caches and on-chip memory, a true logic-process cache), it reports "OCM: 1.47E-16, Caches: 1.5E-15" cm^2/bit (MEASURED; the ChipIR campaign is ref. [13] in that paper). Protons at 64 MeV gave CRAM 3.3e-16 and BRAM 1.1e-15 cm^2/bit. — [Agiakatsikas et al., "Single Event Effects Assessment of UltraScale+ MPSoC Systems under Atmospheric Radiation", arXiv 2303.08098 (under review, IEEE Trans. Reliability, 2023)](https://arxiv.org/pdf/2303.08098)
- (converted) A 16 nm cache at 1.5e-15 cm^2/bit works out to about 20 FIT/Mb (neutrons only, NYC). The 16 nm OCM at 1.47e-16 cm^2/bit works out to about 2 FIT/Mb.

**Intel tri-gate (MEASURED; accelerated beams for thermal and high-energy neutrons, protons and alpha)**
- 22 nm tri-gate: the paper reports measured SER of memory and logic in 22 nm HK/MG bulk tri-gate. SEU SER for cosmic radiation was reduced relative to 32 nm planar, and the alpha-particle benefit was larger still. MCU rates in memory arrays improved similarly. — [Seifert et al., "Soft Error Susceptibilities of 22 nm Tri-Gate Devices", IEEE TNS 59(6):2666–2673, Dec 2012](https://www.researchgate.net/publication/258657647_Soft_Error_Susceptibilities_of_22_nm_Tri-Gate_Devices). The exact reduction factors were not retrievable; see Gaps.
- 14 nm (second-generation tri-gate): "SER improvements up to approximately 23× with respect to devices manufactured in a 32-nm planar technology". The improvement is largest in logic, where fin depopulation gave about 8x relative to first-generation tri-gate. High-energy neutrons "continue to dominate the total SER". — [Seifert et al., "Soft Error Rate Improvements in 14-nm Technology Featuring Second-Generation 3D Tri-Gate Transistors", IEEE TNS 62(6):2570–2577, Dec 2015](https://www.semanticscholar.org/paper/Soft-Error-Rate-Improvements-in-14-nm-Technology-3D-Seifert-Jahinuzzaman/721b9a76b8a82784d9f89abd2c86aaf4401940d4)

**Samsung 14 nm FinFET SRAM (MEASURED; alpha, thermal neutrons, high-energy neutrons; HP and HD cells)**
- FinFET "drastically reduc[ed] SER FIT rate by 5-10X" against prior planar nodes, credited to fin isolation from the substrate. The thermal-neutron reduction was a smaller 1.6x. — ["Radiation-induced soft error rate analyses for 14 nm FinFET SRAM devices" (IRPS 2015; text from search abstract)](https://www.researchgate.net/publication/283809902_Radiation-induced_soft_error_rate_analyses_for_14_nm_FinFET_SRAM_devices)

**TSMC 16 nm and 7 nm (MEASURED)**
- Fang & Oates (TSMC): with technology scaling, "voltage-dependent soft error rate (SER) due to intrinsic alpha particles as well as high-energy neutrons is significantly reduced in 7nm SRAM". The InFO package's redistribution layers block extrinsic alpha emitters such as solder and bumps, which leaves only intrinsic alpha sources. — [Y. Fang & A. S. Oates, "Soft errors in 7nm FinFET SRAMs with integrated fan-out packaging", IRPS 2018](https://ieeexplore.ieee.org/document/8353584/)
- Narasimham et al. (Broadcom) studied 16 nm vs 7 nm FinFET SRAM SER and its bias dependence. — [Narasimham, Gupta, Reed, Wang, Hendrickson, Taufique, "Scaling trends and bias dependence of the soft error rate of 16 nm and 7 nm FinFET SRAMs", IRPS 2018, DOI 10.1109/IRPS.2018.8353583](https://www.semanticscholar.org/paper/Scaling-trends-and-bias-dependence-of-the-soft-rate-Narasimham-Gupta/22072b12cf18dd1565048165be46868bdece06ab). The abstract and numbers were paywalled; the publisher elided the abstract.
- Pieper et al., summarizing that data, say scaling "has led to overall decreasing SER rates per SRAM cell in recent technologies … as seen for 16-nm and 7-nm nodes". — [Pieper et al., TNS 2023 (SAND2022-9599C)](https://www.osti.gov/servlets/purl/2004056)

**5 nm (MEASURED)**
- "SRAM SER measurements indicate that while scaling from planar processes down to the 7-nm FinFET process provided a reduction in per-bit SER at every node, subsequent scaling to the 5-nm FinFET process results in an increase in the per-bit SER relative to the 7-nm FinFET process." Alpha and neutron SER both show "an increase in the per-bit SER and percent multi-cell upsets at the 5-nm FinFET process compared to the 7-nm process". Neutron SER "across process corners show[s] that the faster process corner SER is up to 2× higher than the slower process corner SER in 7-nm and 5-nm". — [Narasimham et al., "Scaling Trends and the Effect of Process Variations on the Soft Error Rate of advanced FinFET SRAMs", IRPS 2023](https://ieeexplore.ieee.org/document/10118025/) (abstract text via search); the companion paper is ["Scaling Trends in the Soft Error Rate of SRAMs from Planar to 5-nm FinFET"](https://www.semanticscholar.org/paper/Scaling-Trends-in-the-Soft-Error-Rate-of-SRAMs-from-Narasimham-Chaudhary/4d493609e82a346be21f64db713169d8c276a25c)
- "at the 5-nm node, the decreases in critical charge and collected charge have resulted in a ~2x increase in SER for low-LET particles" relative to 7 nm. Cell area went from 0.074 µm^2 (16 nm) to 0.027 µm^2 (7 nm) to 0.021 µm^2 (5 nm). — [Pieper, Xiong, Feeley, Pasternak, Dodds, Ball, Bhuva, "SRAM Multi-Cell Upset Vulnerability at the 5-nm FinFET Node", IEEE TNS (2023), SAND2022-9599C](https://www.osti.gov/servlets/purl/2004056)
- In that 5 nm test chip, nominal core voltage was 750 mV and I/O was 1.2 V. Arrays were a 256 Kb single-port (8K x 32) and a 76 Kb two-port (1K x 72), built in "5-nm bulk FinFET technology at a commercial foundry"; the foundry is not named. — [Pieper et al.](https://www.osti.gov/servlets/purl/2004056)

**8 nm FinFET (MEASURED, CSNS atmospheric-like neutron spectrum)**
- The SRAM in an 8 nm FinFET AI chip had a neutron SER "at the level of 10^-16 cm^2·bit^-1". From 14 nm to 8 nm, "the SEU cross section per bit decreases while the non-single bit upset proportion goes up". The largest MCU cluster was 7 bits. — ["Atmospheric neutron inducing single event effects on AI chips manufacturing with 8 nm FinFET", Nuclear Engineering and Technology (2025)](https://www.sciencedirect.com/science/article/pii/S1738573325003535). Text is from the search abstract; the full text returned 403.

**Space-relevant (heavy ions and protons), 16 nm FinFET (MEASURED)**
- Kintex UltraScale+ at high LET: BRAM about 2e-9 cm^2/bit and flip-flops about 1e-8 cm^2 per FF. At 64 MeV protons: CRAM 3.3e-16 and BRAM 1e-15 cm^2/bit. — [Alexandrescu et al. (IROC/ESA), "Applicability of FinFET Technologies for Space Applications"](https://indico.esa.int/event/300/contributions/6085/attachments/4167/6213/Applicability_of_FinFET_Technologies_for_Space_Applications.pdf)
- A 16 nm UltraScale+ heavy-ion CRAM Weibull fit gives a CREME96 GEO (solar-min, 100 mil Al) upset rate of 9.18e-12 events/day/bit. — [Lee et al., "Single-Event Characterization of 16 nm FinFET Xilinx UltraScale+ Devices with Heavy Ion and Neutron Irradiation", SAND2018-7613C](https://www.osti.gov/servlets/purl/1570816)

### Inferences
- Taking the measured 16 nm FinFET SRAM data together (Xilinx BRAM 20 FIT/Mb real-time, CRAM 6 FIT/Mb, MPSoC caches about 20 FIT/Mb converted), a 16 nm-class FinFET cache SRAM sits at about 5 to 20 FIT/Mb at NYC sea level. Most of that comes from high-energy neutrons. Alpha adds about 0.1 to 0.2 FIT/Mb with low-alpha packaging, and thermal neutrons add about 0.35 to 4.7 FIT/Mb.
- 7 nm is probably below 16 nm per bit, given "significantly reduced" in Fang & Oates and "decreasing" in Narasimham and Pieper. 5 nm is about 2x above 7 nm, and a fast process corner can add up to 2x more. A plausible 5 nm/N4-class cache estimate is therefore of the same order as 16 nm, roughly 2 to 20 FIT/Mb. This is an INFERENCE: no absolute 7 or 5 nm value is published in the sources I could read.
- The planar-to-FinFET drop (3 to 20x depending on metric) is far smaller than the drop DRAM achieved over the same decades (see Q2). That difference is why the SRAM:DRAM per-bit ratio grew over time.

### Gaps
- I found no absolute (non-normalized) FIT/Mb or cm^2/bit for TSMC N7, N5 or N4 SRAM. Narasimham 2018/2021/2023 and Fang & Oates 2018 are paywalled and report normalized SER in their abstracts. The newest public UG116 Versal (7 nm) table could not be retrieved. A search-engine summary claimed Versal 7 nm CRAM is about 3e-17 and BRAM about 1e-15 cm^2/bit, but I could not verify this against a primary document, so it is excluded.
- The exact Intel 22 nm reduction factors versus 32 nm planar (often quoted as roughly 1.5 to 4x for neutrons and more for alpha) could not be verified from accessible text.
- A search result stated that "10 nm FinFET: alpha SER down 13.4X, neutron SER down 3.1X vs 14 nm FinFET", but I could not identify or verify the primary source, so it is excluded from the findings.
- I found no Samsung 7/5 nm or GlobalFoundries 12/14 nm SRAM SER absolute values in open sources.

---

## Q2. What is the per-bit SER of modern DRAM (DDR4/DDR5/LPDDR/GDDR/HBM) under neutrons and alpha, and how much lower is it than SRAM?

### Takeaway
DRAM bit-cell SER fell about 4 to 5x per generation and more than 1000x over seven generations up to about 2005. Later reviews put DRAM bit error rates at about 1e-9 to 1e-8 FIT/bit, which is about 0.001 to 0.01 FIT/Mb. Modern 3D/HBM DRAM shows a low raw rate but a large multi-bit fraction. NVIDIA's MICRO 2021 HBM2 beam test publishes only a normalized rate. Its system analysis assumes 12.51 FIT/Gb (about 0.0125 FIT/Mb), a figure "inspired by" Titan GDDR5 field rates. Compared with a 16 nm FinFET cache SRAM at about 5 to 20 FIT/Mb, modern DRAM is lower per bit by roughly 3 orders of magnitude, with a range of about 10^2.7 to 10^4.

### Cited Findings
- "The DRAM SER of a single bit is shrinking about 4× to 5× per generation. Although the DRAM bit SER has decreased by more than 1,000× over seven generations, the DRAM system SER has remained essentially unchanged." Baumann also calls DRAM "one of the more robust devices in terms of soft-error immunity". By contrast, the SRAM bit SER, after rising in BPSG-era nodes, "has reached saturation and might even be decreasing" below 250 nm. His mitigation example assumed an uncorrected SRAM SER of "1.6-kFIT-per-megabit" (circa 130/90 nm). — [R. Baumann, "Soft Errors in Advanced Computer Systems", IEEE Design & Test of Computers, May–June 2005](http://www.cs.columbia.edu/~cs4823/handouts/baumann-soft-errors-DT-05.pdf) (MEASURED trend, normalized figures)
- DRAM cell SER across generations spans "10^-10 to 10^-5 FIT/bit", and "DRAM error rates are dropping below 10^-9 to 10^-8 FIT/bit". These figures come from search-result summaries of the paper. — [C. Slayman, "Soft Error Trends and Mitigation Techniques in Memory Devices", RAMS 2011](https://www.researchgate.net/publication/224231257_Soft_error_trends_and_mitigation_techniques_in_memory_devices). The PDF was not accessible, so the exact wording is unverified.
- Older, pre-2005 DRAM: "200-5000 FIT per Mbit … 1000 to 5000 FIT per Mbit seems to be a reasonable SER". — [Tezzaron Semiconductor, "Soft Errors in Electronic Memory" white paper (2004)](https://tezzaron.com/media/soft_errors_1_1_secure.pdf). This is an aggregator of SDRAM-era data and is NOT representative of DDR4/DDR5/HBM.
- **HBM2 on GPU (MEASURED, ChipIR neutron beam).** Setup: NVIDIA Quadro V100 with 32 GB HBM2, running at full voltage and speed with DRAM ECC disabled and SRAM ECC enabled. Average flux was 9.8e5 n/cm^2/s, an acceleration factor of 2.52e8 over 14 n/cm^2/h at NYC sea level. Fig. 1 plots the measured HBM2 per-Gb error rate as normalized, alongside Slayman (2010) historical data. The authors call the low HBM2 rate "within expectations given the historical trends". They also note that falling per-bit capacitance in recent DRAM generations means "this historical decrease no longer holds". — [Sullivan, Saxena, O'Connor, Lee, et al., "Characterizing and Mitigating Soft Errors in GPU DRAM", MICRO 2021, DOI 10.1145/3466752.3480111](https://www.mbsullivan.info/attachments/papers/sullivan2021characterizing.pdf)
- **Origin of the 12.51 FIT/Gb figure.** The value is ASSUMED, not a published beam measurement. Sullivan's system analysis is "drawn assuming a failure rate of 12.51 FIT/Gb, inspired by the GDDR5 memory failure rates in the Titan supercomputer". With SEC-DED, an A100 then "suffers from 216 FIT of HBM2 SDC". — [Sullivan et al. 2021](https://www.mbsullivan.info/attachments/papers/sullivan2021characterizing.pdf). Several search engines misreport this value as the measured HBM2 rate.
- Displacement damage in the beam caused intermittent retention errors in HBM2, "up to several thousand per GPU". The authors filtered these out; they do not occur in the field. — [Sullivan et al. 2021](https://www.mbsullivan.info/attachments/papers/sullivan2021characterizing.pdf)
- **DDR4 vs DDR3 (MEASURED, 480 MeV protons at TRIUMF).** DDR4 from the same manufacturer showed "about 45% SBU cross-section increase, and 17% logic upset decrease" compared with DDR3. Logic cross section differed by 1.9x between vendors at the same density and speed. — [Park, Jeon, Yu, et al., "Soft error study on DDR4 SDRAMs using a 480 MeV proton beam", IRPS 2017](https://ieeexplore.ieee.org/document/7936404/)
- **DDR4 component and system level (MEASURED, 2025).** The study found 16 distinct fault modes. "Over 45% of the faults that occurred affected multiple DRAM bits". Block errors reached "up to two thousand faulty words per event", and fault rates differed by more than 1.34x between vendors. — [Jiao et al., "Soft Error Study on Advanced Process Node DRAMs at Component and System Level", IEEE DFT 2025](https://www.researchgate.net/publication/397981983_Soft_Error_Study_on_Advanced_Process_Node_DRAMs_at_Component_and_System_Level) (abstract via search)
- **Thermal neutrons.** Rech's group tested two DDR memories, GPUs, an APU, a Xeon Phi and an FPGA under ChipIR (high-energy) and ROTAX (thermal) beams. Thermal-neutron FIT "could be comparable to the high energy neutron FIT rate" for some devices because of boron-10. At Leadville, 29% of the K20 GPU's SDC FIT rate came from thermal neutrons. — [Oliveira, Blanchard, DeBardeleben, …, Baumann, Rech, "Thermal Neutrons: a Possible Threat for Supercomputers and Safety Critical Applications", ETS 2020 (LA-UR-21-20209)](https://www.osti.gov/servlets/purl/1756791); journal version: ["Thermal neutrons: a possible threat for supercomputer reliability", J. Supercomputing 2021](https://link.springer.com/article/10.1007/s11227-020-03324-9)

### Inferences
- **Per-bit ratio estimate (INFERENCE from the numbers above).**
  - FinFET 16 nm cache SRAM is about 5 to 20 FIT/Mb (MEASURED).
  - Modern DRAM is about 0.001 to 0.0125 FIT/Mb, from Slayman's 1e-9 to 1e-8 FIT/bit (MEASURED trend) and Sullivan's ASSUMED 12.51 FIT/Gb.
  - The resulting SRAM:DRAM per-bit ratio is about 400x to 20,000x, with a central value of about 1,000 to 2,000x (for example, 20 FIT/Mb ÷ 0.0125 FIT/Mb ≈ 1,600x).
  - For 5 nm SRAM, which runs about 2x above 7 nm, the ratio is probably of the same order.
- **Caveat 1:** DRAM rates are often counted as events, and about 31% of HBM2 events (and over 45% of DDR4 faults) hit multiple bits. Per raw bit flipped, the DRAM rate could be up to a few times higher than the per-event rate.
- **Caveat 2:** SRAM rates assume ECC is off and nominal voltage. Lower voltage or a fast process corner raises SRAM SER by up to 2x or more.
- **History of the ratio.** In the 1980s and 1990s, DRAM per-bit SER was comparable to or higher than SRAM's. Over seven or more DRAM generations, DRAM fell more than 1000x while SRAM stayed flat (Baumann) and then fell another 3 to 20x at FinFET. That turned a ratio near 1 into one of about 10^3 today.

### Gaps
- I found no publicly available absolute neutron per-bit SER for DDR4, DDR5, LPDDR4/5, HBM2/2e/3 or GDDR6. Vendors (Micron, Samsung, SK hynix) do not publish these numbers openly. JEDEC JEP151 and similar documents were not accessible.
- The Micron TN "DRAM Soft Error Rate Calculations" was referenced by search results but not retrieved.
- Search snippets carried a conflicting claim that "DDR4 memory cross section is approximately one order of magnitude lower than the DDR3 one". It contradicts Park et al. (+45% SBU cross section), and I could not confirm its primary source.
- I found no measured DRAM alpha SER at modern nodes. The literature says neutrons dominate modern DRAM, but I found no number.

---

## Q3. Are there published SRAM:DRAM per-bit ratios or textbook/review numbers?

### Takeaway
I found no single primary source that quotes an explicit "SRAM:DRAM per-bit SER ratio" for modern nodes. Reviews give the components instead. Baumann (2005) covers DRAM bit SER falling more than 1000x over seven generations, SRAM bit SER saturating, and an example SRAM figure of 1.6 kFIT/Mb. Slayman (2011) puts DRAM at 1e-10 to 1e-5 FIT/bit, falling below 1e-9 to 1e-8. The ratio has to be derived. Using measured FinFET SRAM data and DRAM trend data, it is about 10^3 (range roughly 10^2.6 to 10^4.3).

### Cited Findings
- Baumann 2005: DRAM bit SER down 4 to 5x per generation and more than 1000x over seven generations; SRAM bit SER "reached saturation" below 250 nm; DRAM system SER roughly constant while SRAM system SER rises with integrated SRAM. — [Baumann, IEEE D&T 2005](http://www.cs.columbia.edu/~cs4823/handouts/baumann-soft-errors-DT-05.pdf)
- An aggregator quoting Baumann: "DRAM sensitivity to cosmic rays has dropped about six orders of magnitude over 15 years, on a per bit basis", while "the total fail rate per bit over 16 years has dropped about 10x for SRAM". — [J. Ziegler / SRIM, "Review of Accelerated Testing of SRAMs" (SER trends page)](http://www.srim.org/SER/SERTrends.htm). This text comes from a search-result snippet; the page's TLS certificate failed, so I could not verify the context. These figures imply a ratio change of about 10^5 over that period.
- Slayman 2011 (RAMS) gives the DRAM per-bit range of 1e-10 to 1e-5 FIT/bit with the latest DRAM below 1e-9 to 1e-8 FIT/bit. — [Slayman 2011](https://www.researchgate.net/publication/224231257_Soft_error_trends_and_mitigation_techniques_in_memory_devices)
- Sullivan et al. reproduce Slayman's historical DRAM beam data (Fig. 1) and the Borucki 2008 finding that DRAM non-bitcell (logic) upset rates "stay within a two-order-of-magnitude range" with no strong scaling trend. — [Sullivan et al. 2021](https://www.mbsullivan.info/attachments/papers/sullivan2021characterizing.pdf); original: [Borucki, Schindlbeck, Slayman, "Comparison of accelerated DRAM soft error rates measured at component and system level", IRPS 2008](https://ieeexplore.ieee.org/document/4558933)

### Inferences
- Combining the Xilinx 16 nm data (6 to 20 FIT/Mb) and the ArM cache cross section (about 20 FIT/Mb converted) with DRAM at 1e-9 to 1e-8 FIT/bit gives an SRAM:DRAM per-bit ratio of about 600x to 20,000x. Using the Sullivan 12.51 FIT/Gb assumption gives about 500x to 1,600x.
- A defensible engineering statement: **"a modern FinFET SRAM cell is roughly 10^3x (order-of-magnitude range 10^2.5 to 10^4) more likely per bit to be upset by terrestrial radiation than a modern DRAM/HBM cell."** This ratio is derived, not taken from a single paper.

### Gaps
- I could not access the Autran & Munteanu "Soft Errors: From Particles to Circuits" (CRC 2015) or Ibe "Terrestrial Radiation Effects in ULSI Devices and Electronic Systems" (Wiley 2015) texts to quote their tabulated values.
- JEDEC JEP151 and JESD89B were not accessible.
- The srim.org statement is not verified in context.

---

## Q4. Is there radiation data on GDDR6/GDDR6X or Micron graphics memory, and on TSMC N5/N4 SRAM?

### Takeaway
I found no public neutron, alpha or heavy-ion per-bit SER data for GDDR6, GDDR6X or any Micron graphics DRAM. The closest graphics-memory evidence is GDDR5 on NVIDIA K20/K40-class GPUs, from Titan field data and LANSCE/ISIS beam tests reported as normalized or system-level results, plus HBM2 on V100 (Sullivan 2021). For 5 nm SRAM, the Pieper/Bhuva (Vanderbilt/Sandia) and Narasimham (Broadcom) studies used an unnamed "commercial foundry" 5 nm bulk FinFET, which is very likely TSMC N5 but not stated. I found no N4 SRAM SER data.

### Cited Findings
- Titan (18,688 K20X GPUs, 6 GB GDDR5): only 899 cards had any SBE between Feb 2013 and Aug 2014. 98% of SBEs were in L2, dominated by the 10 worst cards. Excluding those 10, "the device memory is the structure where most of the SBEs occur (96% of all SBEs)". — [Tiwari, Gupta, Rogers, Maxwell, Rech, et al., "Understanding GPU Errors on Large-Scale HPC Systems and the Implications for System Design and Operation", HPCA 2015](https://www.osti.gov/servlets/purl/1185857). These are field data that mix soft and hard faults.
- In the same paper's beam tests (LANSCE and ISIS, 2013–2014), K20 (Kepler) L2 and register-file per-bit cross sections were lower than C2050 (Fermi); the paper gives only normalized values. "Only 4-6% errors are double bit errors in L2 cache and register file area for the K20 card". In L2, bits storing zero were 40% more prone to upset. — [Tiwari et al. 2015](https://www.osti.gov/servlets/purl/1185857)
- The 12.51 FIT/Gb HBM2 analysis assumption was "inspired by the GDDR5 memory failure rates in the Titan supercomputer". — [Sullivan et al. 2021](https://www.mbsullivan.info/attachments/papers/sullivan2021characterizing.pdf)
- 5 nm SRAM details: commercial-foundry 5 nm bulk FinFET; 0.021 µm^2 cell; 750 mV nominal; tested with alpha (Am-241), 14 MeV neutrons (Sandia), LANSCE terrestrial neutrons (fluence about 1e11 n/cm^2), thermal neutrons (MURR) and heavy ions (LBNL, LET 2 to 86 MeV·cm^2/mg). — [Pieper et al.](https://www.osti.gov/servlets/purl/2004056)

### Inferences
- GDDR6/6X are 1x/1y/1z-nm-class DRAM cells with large cell capacitance, like DDR4 and HBM2. By physics and trend, their per-bit cell SER should be of the same order as DDR4 and HBM2, about 1e-9 to 1e-8 FIT/bit. This is an INFERENCE; the high-speed interface and logic of GDDR6X (PAM4) may add non-cell, multi-bit error modes, as HBM2 showed.

### Gaps
- There is no public GDDR6, GDDR6X, GDDR7 or HBM3 beam data with absolute per-bit numbers.
- I found no Micron graphics-memory SER white paper.
- I found no TSMC N4/N3 SRAM SER publications in open sources.

---

## Q5. How have MCU/MBU fractions and multi-bit patterns changed for FinFET SRAM and for DRAM?

### Takeaway
In FinFET SRAM, the MCU fraction rises with scaling and especially at low voltage. At 5 nm, MCUs are a majority of events for neutrons at reduced voltage. At nominal voltage, terrestrial-neutron MCUs stay small, with a bit-line range of 2 to 3 cells. DRAM bit-cell upsets are mostly single-bit, but modern DRAM has a large, severe multi-bit component from logic and periphery: about 31.5% of HBM2 SEUs are multi-bit, and over 45% of DDR4 faults affect multiple bits.

### Cited Findings
**SRAM**
- **5 nm SRAM, alpha:** MBUs "contribute to >15% of overall errors for both SRAM designs when supply voltages were decreased below 550 mV". MCUs were observed even at nominal and elevated voltage. — [Pieper et al.](https://www.osti.gov/servlets/purl/2004056)
- **5 nm SRAM, 14 MeV neutrons:** "At 550 mV supply voltage, SBUs only account for ~50% of overall events". Bit-line upset ranges were larger than for alpha. — [Pieper et al.](https://www.osti.gov/servlets/purl/2004056)
- **5 nm SRAM, terrestrial neutrons (LANSCE, nominal 750 mV):** the maximum bit-line upset range was 3 cells (single-port) and 2 cells (two-port). — [Pieper et al.](https://www.osti.gov/servlets/purl/2004056)
- **5 nm SRAM, thermal neutrons:** the largest MCU at nominal voltage was 6 bits (single-port) and 3 bits (two-port). — [Pieper et al.](https://www.osti.gov/servlets/purl/2004056)
- **5 nm SRAM, heavy ions** (10 MeV·cm^2/mg Ar): the maximum MCU grew from 7 bits at 750 mV to 29 bits at 550 mV in single-port SRAM. High-LET ions caused "multiple 200+ bit upsets" at 550 mV. At nominal voltage the largest cluster was 76 bits (two-port) and 12 bits (single-port). — [Pieper et al.](https://www.osti.gov/servlets/purl/2004056)
- **Low-voltage FinFET SRAM:** lowering supply from 0.7 V to 0.35 V cut the 1-bit event share from 94% to 78.5% and raised 2-bit events from 5% to 20%. — [search-result abstract from the FinFET SRAM SER literature](https://www.researchgate.net/publication/325031015_Scaling_trends_and_bias_dependence_of_the_soft_error_rate_of_16_nm_and_7_nm_FinFET_SRAMs). The attribution to this exact paper is uncertain; treat it as indicative.
- **5 nm vs 7 nm:** "percent multi-cell upsets" increase at 5 nm compared with 7 nm. — [Narasimham et al., IRPS 2023](https://ieeexplore.ieee.org/document/10118025/)
- **8 nm vs 14 nm:** going to 8 nm, "the non-single bit upset proportion goes up", with a maximum cluster of 7. — [NET 2025, 8 nm FinFET AI chip](https://www.sciencedirect.com/science/article/pii/S1738573325003535)
- **16 nm MPSoC:** "99.99% of the events were correctable due to the interleaving layout". — [Agiakatsikas et al.](https://arxiv.org/pdf/2303.08098)
- **K20 GPU SRAM (28 nm planar):** 4 to 6% of neutron-induced errors were double-bit. — [Tiwari et al. 2015](https://www.osti.gov/servlets/purl/1185857)
- **Intel 22 nm tri-gate:** MCU rates improved along with SBU rates. — [Seifert et al. 2012](https://www.researchgate.net/publication/258657647_Soft_Error_Susceptibilities_of_22_nm_Tri-Gate_Devices)

**DRAM**
- **HBM2 (V100, ChipIR):**
  - "∼31.5% of single event upsets (SEUs) in device memory affect multiple bits in at least one word". Isolated single-bit events are 65% ± 2.3%.
  - Multi-bit multi-entry events are 28% ± 2.1%, with the broadest affecting 5,359 32-byte entries.
  - Of multi-bit errors, about 75% (73.8%) are byte-aligned, attributed to 8-bit mats.
  - About 15% of errors flip all bits in a byte or word (data-dependent inversions).
  - These severe errors "likely originate in DRAM logic structures".
  — [Sullivan et al. 2021](https://www.mbsullivan.info/attachments/papers/sullivan2021characterizing.pdf)
- **DDR4:** ">45% of faults affected multiple DRAM bits", with block errors of up to about 2,000 words per event. — [Jiao et al., DFT 2025](https://www.researchgate.net/publication/397981983_Soft_Error_Study_on_Advanced_Process_Node_DRAMs_at_Component_and_System_Level)
- **DRAM logic vs cell scaling:** logic and periphery upset rates do not scale down like cell rates; Borucki 2008 found them within a two-order-of-magnitude band. Their relative share therefore rises. — [Sullivan et al. 2021](https://www.mbsullivan.info/attachments/papers/sullivan2021characterizing.pdf); [Park et al. IRPS 2017](https://ieeexplore.ieee.org/document/7936404/) (DDR4 logic upset 17% lower than DDR3)

### Inferences
- For a GPU-cache vs GDDR comparison: SRAM upsets at nominal voltage are mostly single-bit and physically clustered small MCUs, which column interleaving plus SEC-DED or parity largely handles. DRAM upsets are rarer per bit, but about a third are multi-bit, structured byte, entry or burst errors from periphery. So the per-bit cell ratio (about 10^3) overstates the ratio of severe multi-bit events, which is probably lower, perhaps 10^2 to 10^3.
- This is an INFERENCE: the absolute DRAM MBU rate is not published in absolute units.

### Gaps
- I found no published MCU statistics for GDDR6/6X or DDR5. DDR5 on-die ECC hides cell SBUs from the system, and no open data separates the two.
- I found no absolute 7 nm SRAM MCU fractions in open text.
