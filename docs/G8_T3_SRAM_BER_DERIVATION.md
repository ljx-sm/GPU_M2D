# G8-T3: deriving the L2 (SRAM) BER from the GDDR6X (DRAM) BER

Written 2026-10-12. This document re-examines the G8-T3 decision
(`ρ = 1`, [G8_CACHE_FAULT_PLAN.md §3.3, §14](G8_CACHE_FAULT_PLAN.md)) from
first principles. It derives which radiation environment applies, how
errors arise in each memory, and how a DRAM BER level maps to an SRAM BER.
Every number in §4–§7 is printed by
[`tools/g8_cache/t3_rho_derivation.py`](../tools/g8_cache/t3_rho_derivation.py)
(standard library only), so the arithmetic can be re-checked. The script
holds every input with its source.

Verification tags used for each source value:
- **[V]**: checked against the primary text.
- **[F]**: read off a figure in the primary text, about ±30 %.
- **[S]**: secondary citation only.
- **[A]**: abstract only.
- **[ASSUMED]**: a modelling choice, not a literature value.

## 0. Bottom line

1. **Environment.** The reference mission is a COTS GPU on a low-Earth,
   sun-synchronous orbit (500–750 km, about 98°) behind about 2.5 mm Al.
   - Trapped protons in the South Atlantic Anomaly (SAA) cause most upsets
     in modern SRAM: about 70–95 %.
   - Galactic-cosmic-ray (GCR) heavy ions, seen mostly on the high-latitude
     parts of the orbit, cause the rest.
   - Solar particle events are episodic worst-case days.
   - Trapped electrons and secondary neutrons are negligible.
2. **ρ = λ_SRAM / λ_DRAM ≈ 2.5 × 10³ (best estimate).**
   - Plausible range 10²–10⁴; the extreme bounds are 5 and 10⁵.
   - A same-orbit flight measurement agrees: on CARMEN-2/JASON-2, the SRAMs
     upset 600× more often per bit than the SDRAMs (geometric means).
3. **The earlier "ρ ≈ 1 under heavy ions" used the wrong quantity.**
   - It compared saturation cross sections, the plateau reached at LET
     ≈ 60 MeV·cm²/mg.
   - The space rate is dominated by low-LET particles, where FinFET SRAM is
     already sensitive and DRAM is not.
   - Integrated over the real GCR spectrum, heavy ions alone give
     ρ_HI ≈ 1.9 × 10³.
4. **REMU's 1e-7 /bit/day is an SRAM number.** It comes from the most
   sensitive SSRAM on CARMEN-2. The SDRAMs on the same experiment measured
   2e-12 to 1.2e-10 /bit/day.
5. **What the ladder means.** In the reference orbit, our BER ladder
   1e-7 … 1e-5 corresponds to:
   - **34 days – 9 years** of upset accumulation in the L2 (λ_S ≈ 3e-9
     /bit/day);
   - **230 – 23,000 years** in modern DRAM (λ_D ≈ 1.2e-12 /bit/day).

   So the ladder is a physically meaningful **SRAM** exposure range. As a
   **DRAM** exposure it is only a stress axis.
6. **Derivation result.**
   - Formula: `BER_SRAM = ρ · min(1, T_res / T_DRAM) · BER_DRAM`, for the
     read-only lines that stay resident in L2.
   - At equal exposure the physically paired point of a DRAM level is
     **BER_SRAM ≈ 2.5 × 10³ × BER_DRAM**.
   - Equivalently, the DRAM BER that goes with one of our SRAM levels is
     BER_SRAM / 2.5 × 10³ = **4e-11 … 4e-9**. That is at most 1–3 DRAM bits
     per trial for every model.
   - The T4 campaign (ρ = 1, same BER in both memories) is therefore an
     **iso-BER comparison** of per-flip vulnerability, not an equal-exposure
     pairing.
   - The **SRAM-only curves are the physically representative ones**: a
     physically paired DRAM+SRAM point adds only 1–3 DRAM bits to them.
   - §8 gives the exact reading of each T4 curve.
7. **Largest open unknowns:**
   - whether the GeForce L2 has always-on parity or ECC (not public);
   - how long weight lines stay resident in L2 in a deployed system;
   - the absence of any GDDR6X or 4N radiation data.

   §7 rates each one.

## 1. Reference mission and why

| Item | Choice | Why |
| --- | --- | --- |
| Orbit | LEO, sun-synchronous, 500–750 km, about 98° | On-board DNN inference with COTS hardware is a LEO use case. REMU's Jetson Xavier NX platform flew on Chaohu-1, a LEO SAR satellite [S: RedNet arXiv 2407.11853]. Remote-sensing satellites use SSO. A non-hardened desktop GPU is not a GEO or deep-space candidate. |
| Shielding | 100 mil (2.54 mm) Al equivalent | The standard reference thickness for the published flux tables used here [Barak 1999 Table 1; CREME96 outputs in Poivey 2002]. |
| Solar cycle | Solar minimum | GCR flux is highest at solar minimum, and AP-8 MIN is the standard trapped-proton case. Solar flares are treated separately as worst-case days (§2). |

**Sensitivity to altitude.** The trapped-proton flux behind 100 mil grows
5.2× from 500 km (1.86e6 /cm²/day) to 750 km (9.65e6 /cm²/day) [V: Barak
1999 Table 1]. AP-8 under-predicts the flux above 10 MeV by at least 2×
[V: Lauenstein & Barth 2005]. The derivation therefore carries a low case
(500 km), a central case (600 km, log-interpolated, 3.59e6 /cm²/day) and a
high case (2 × the 750 km flux).

## 2. Which particles, and why

| Particle | Role in LEO SSO | Evidence |
| --- | --- | --- |
| **Trapped protons (SAA)** | **Dominant for SRAM.** | ECSS-E-ST-10-12C Table 6-1 lists SEE sources in LEO as "trapped protons / solar energetic particles / cosmic rays" [V]. ECSS-E-ST-10-04C I.1.2: LEO "is dominated by the South Atlantic anomaly … high-energy radiation-belt trapped protons" [V]. NASA NEPP: "LEO mission radiation environments tend to be dominated by trapped protons, especially in the SAA" [V: Ladbury et al.]. Barak et al.: below 50° and 10,000 km, "most SEU events, for devices with Lc ≤ 10 MeVcm²/mg, are expected to come from protons" [V]. Flight data: PROBA-2 (720 km, 98°) saw 86–90 % of its SRAM upsets in the SAA, and OMERE attributes 93–98 % of the predicted rate to protons [V: Poivey 2015]. SeaStar (705 km, 98°) saw more than 80 % of its upsets in the SAA, where it spends about 5 % of its time [V: Ladbury]. |
| **GCR heavy ions** | Second; mainly on the high-latitude parts of the orbit. Comparable to protons for DRAM. | ECSS-E-ST-10-04C I.1.2: on polar orbits "on the high-latitude parts … almost unattenuated fluxes of cosmic rays" [V]. In Dodds 2015 Table V, heavy ions dominate only where GCR dominates (GEO), and the heavy-ion rate at ISS is 1/7–1/16 of the GEO value [V]. |
| Solar energetic particles | Episodic: days of 10²–10³× higher rates | Example: DDR3 at 1–4e-9 /bit/day during a flare vs about 1e-12 quiet [V: ESA RHM Table 7-1]. Treated as worst-case days, not part of the average λ. |
| Trapped electrons | Negligible for SEU | Electrons matter only for critical charges ≲ 0.2 fC [A: Trippe 2015]. A 4N SRAM could approach this; flagged in §7. |
| Secondary neutrons | Negligible | "not typically a concern for commercial missions" [V: ECSS-E-ST-10-12C Table 6-1]. |
| Low-energy protons (< 2 MeV, direct ionization) | Raise the SRAM rate by less than 2× | Cross sections up to 10³–10⁴× the high-energy value, but low-energy protons are rare behind shielding. Dodds 2015: the total rate rises by "less than double" except on the GEO worst day (4.3×) [V]. Carried as a factor of 1–2 for SRAM only. |

**Why protons dominate SRAM.**
- The particle counts differ by four orders of magnitude. Orbit-averaged
  trapped protons behind 100 mil number 1.9e6–9.7e6 per cm² per day. GCR
  ions above LET 1 MeV·cm²/mg number about 155 per cm² per day, even
  without geomagnetic shielding (§4.1).
- A modern SRAM cell flips on very little charge. A proton's nuclear
  reaction in silicon (a recoil of LET up to about 10–15) is enough, so its
  per-bit proton cross section is large: about 1e-15 cm².
- A DRAM cell stores far more charge per bit, so its proton cross section
  is about 10⁴× smaller (about 2e-19 cm²). For DRAM the few heavy ions
  therefore matter as much as the many protons.

## 3. How errors arise in each memory

| Aspect | GDDR6X main memory (DRAM) | L2 cache (SRAM, TSMC 4N) |
| --- | --- | --- |
| Cell | 1T1C. Charge stored on a capacitor; refreshed every few ms. | 6T cross-coupled inverters. Static; no refresh. |
| Upset mechanism | An ion or recoil discharges a **charged** cell, so flips go to the discharged value. With true and anti cells, the logical direction is 1→0 or 0→1: Hynix and Samsung DDR3 show more than 99 % 1→0, Micron about 50/50 [V: ESA/IDA TN-IDA-RAD-14/8; Ladbury NEPP]. | Collected charge flips the stored state. The direction is near-symmetric at 5 and 3 nm ("similar", "negligible differences" [V: Pieper]). At 16 nm, CRAM upsets stored 0s about 1.9× more often than 1s, and BRAM shows no bias [V: Lee 2018]. |
| Persistence | **Persistent until overwritten.** Refresh rewrites the wrong value [V: Kamibayashi 2023]. Weights loaded once accumulate flips for as long as they stay in DRAM. | **Persistent while resident.** A read-only (clean) line keeps the flip until eviction; the next fill from DRAM is correct. A dirty (engine-written) line's flip lasts until the engine overwrites it or it is written back to DRAM. |
| Multi-bit | GDDR5 under neutrons: 59 % single-bit upsets, 38 % bursts (often one byte per word), 3 % SEFI [V: Kamibayashi]. HBM2: 31.5 % multi-bit, about 75 % within a byte [V: Sullivan]. Our DRAM model uses 60 % single-bit and 40 % 2–3-bit events. | Mostly single-bit at nominal voltage: 16 nm CRAM about 4 % MCU under neutrons [V: Lee]. The MCU share rises with LET (7 nm CRAM: 0 % at LET 2.8, 14 % at 8.8 [V: Mayo]). Our cache model uses single-bit flips. |
| Other modes | Row, column and device SEFIs, each corrupting hundreds of bits. Their count is comparable to cell SEUs under heavy ions [V: ESA RHM §6.1.2]. Stuck or weak bits from displacement damage. **Not part of our BER model** (§7). | Peripheral or control upsets (crashes, hangs) are outside the bit-flip model. |
| Protection | GDDR6X with ECC off: no storage protection (EDC covers only the link). | **Unknown.** NVIDIA documents SECDED on L2 for A100 [V], but publishes nothing for GeForce Ada [V: absent from the Ada whitepaper]. |

## 4. The derivation

### 4.1 Invariants and variables

The derivation rests on four invariants: quantities that are the same for
the L2 and the GDDR6X, or are defined the same way.

| # | Invariant | Why it holds |
| --- | --- | --- |
| I1 | **The particle environment** φ_k(E): the flux spectrum of each species k per unit time. | The AD102 die and the GDDR6X packages sit centimetres apart on one board behind the same spacecraft shielding. Package lids are negligible next to 2.5 mm Al. |
| I2 | **The mission clock** t. | Both memories are exposed during the same mission time. |
| I3 | **Independent, memoryless upsets** (Poisson, constant rate). The expected flipped fraction after exposure T is 1 − e^(−λT) ≈ λT. | Standard SEE rate theory (CREME96, OMERE). λT ≤ 1e-2 everywhere here, so the linear form is exact to < 1 %. |
| I4 | **The definition of BER**: flipped bits ÷ exposed bits, the same in both memories. | Our DRAM count B = round(BER_D × R × 8) and cache count n_cache = round(BER_S × R_eff_bits) use the same definition (§5). |

What differs between the memories (the variables):
- the per-bit cross section σ_X,k(E) of each memory X for each particle k;
- the exposure window T_X: how long a bit holds the data without being
  restored from a clean copy;
- event shape (multi-bit) and flip direction.

### 4.2 From invariants to the SRAM BER

```text
per-bit rate          λ_X = Σ_k ∫ σ_X,k(E) φ_k(E) dE              (I1, I3)
BER after exposure    BER_X = λ_X · T_X                            (I3, I4)
ratio                 BER_S / BER_D = (λ_S / λ_D) · (T_S / T_D) = ρ · τ

  ρ = λ_S / λ_D   technology ratio in the same environment   (§4.3–4.5)
  τ = T_S / T_D   exposure-window ratio                        (§4.6)
```

A DRAM BER level therefore means an exposure time
`T_D = BER_D / λ_D`. Over that time the cache accumulates

```text
BER_S = λ_S · min(T_D, T_res) = ρ · min(1, T_res / T_D) · BER_D       (*)
```

Here T_res is how long a cache line stays continuously resident; its flips
are lost when it is evicted and refilled. Equation (*) is the complete
answer to "given the DRAM BER, what is the SRAM BER". It needs exactly two
numbers, ρ and T_res/T_D, and the rest of this section derives them.

### 4.3 Heavy-ion term

The GCR rate is `λ_HI = ∫ σ(L) f(L) dL`, using the GCR differential LET
spectrum f(L) = 353 · L^−3.27 ions/(cm²·day·MeV·cm²/mg). This form is
valid for LET 1–28; above 28 the flux "falls … by more than two orders of
magnitude" [V: Barak 1999, Fig. 1 and p. 1506]. It integrates to
155 ions/cm²/day above LET 1, which matches about 2e-3 /cm²/s read from
the CREME96 solar-minimum curve [F: Poivey 2002 Fig. 12].

**SRAM σ(L).** FinFET SRAM does not saturate, so a power law is fitted to
the 16 nm SRAM-array data:
- TSMC 16 nm single-port SRAM [F: Tam 2015, via Pieper 2023 Fig. 27];
- TSMC N16FFC [F: Glorieux 2024].

The fit is **σ_S = 9.3e-12 · L^1.07 cm²/bit**, which gives 2e-11 at LET 2
and 2.3e-10 at LET 20.

| Case | Source of σ(L) | λ_HI, interplanetary (/bit/day) |
| --- | --- | --- |
| low | 7 nm Versal CRAM fit, 3.4e-13 · L^1.63 [V: Mayo 2025] | 1.7e-10 |
| central | 16 nm SRAM fit | **2.7e-9** |
| high | 16 nm UltraScale+ BRAM Weibull (σ_sat 3e-9, L_th 1.5, W 10, s 1.5) [V: Lee 2018] | 9.2e-9 |

**DRAM σ(L).** Weibull fits of six 2–4 Gb DDR3 parts give σ_sat
1.7e-11 to 3.9e-10 cm²/bit and L_th 0.48–1.09. Their **L_0.25 is 26–48
MeV·cm²/mg** (LET at 25 % of saturation, computed from the fits) [V:
ESA/IDA TN-IDA-RAD-14/14 Table 1, TN-IDA-RAD-13/1A Table 5]. DDR3 is
nearly insensitive below LET of about 10. Samsung DDR3 showed "no errors
of any type … for Ne ions at normal incidence (LET = 2.1)" [V: Ladbury
2013].
- **Central rate:** the geometric mean of the seven DDR3 fits is
  **1.4e-12** /bit/day (range 2.3e-13 to 1.4e-11).
- **Cross-check:** this agrees with ESA's OMERE prediction for 4 Gb DDR3
  in interplanetary cruise, 0.9–1.6e-12 [V: RHM Table 7-1]. It also agrees
  with the LRO SDRAM flight value of 6.8e-12 in lunar orbit [V: Ladbury
  2010].
- **High bound (worst case for DRAM):** the DDR4 parts in Dufour 2022 publish
  only L_th and σ_sat. Assuming a narrow Weibull (W = 10, s = 2) [ASSUMED]
  gives up to **4.7e-10**.

**Heavy-ion-only ρ_HI = 2.7e-9 / 1.4e-12 ≈ 1.9 × 10³** (range 0.35 to
4 × 10⁴; the 0.35 needs the assumed-shape DDR4 bound).

**Why the earlier ρ ≈ 1 was wrong.** The 2026-10-03 survey divided the
Snapdragon 835 L2 value (about 1e-11 cm²/bit, flat up to LET 6.9) by the
DDR4 **saturation** cross sections (8.9e-13 to 1.14e-10). That gives
0.09–11; the survey reported 0.1–2 because it missed one Samsung row. The
comparison fails for two reasons:
- **Saturation is the wrong quantity.** Saturation is reached only at
  LET ≈ 30–60. f(L) there is below 10⁻⁵ of its value at LET 1–3. The rate
  is set where DRAM is still near its threshold and SRAM is already
  sensitive.
- **The Snapdragon value is a lower bound, not a saturation value.** Every
  FinFET array rises 3–18× between LET 2 and 7, and the Snapdragon value
  stays flat. The counts were OS-visible correctable errors on accessed
  lines, over an unknown number of bits [V: Guertin 2020: "We lack
  understanding of the cache bits used in the testing"].

### 4.4 Proton term

High-energy protons and neutrons upset memory through the same
nuclear-reaction mechanism. Above about 50 MeV their cross sections agree
within about 2× [S: JESD89A via CERN; V: ECSS-E-ST-10-12C §9.5.3 uses one
σ_nucleon(E)]. Neutron data can therefore stand in where proton data are
missing. The proton rate is `λ_p = σ_p · Φ_p`, with Φ_p the
orbit-averaged trapped-proton flux behind 100 mil (§1).

| Quantity | low | central | high | Basis |
| --- | --- | --- | --- | --- |
| σ_p, SRAM (cm²/bit) | 2.5e-16 | **6e-16** | 1.5e-15 | 16 nm BRAM 1.1e-15 at 64 MeV [V: Agiakatsikas Tab. I]; 7 nm Versal BRAM 1.2e-15 under neutrons [V: AMD UG116]; 16 nm A53 L2 data 6.3e-16 under neutrons [V: Agiakatsikas Tab. IX]; N16FFC 4.5e-16 at 50–230 MeV [F: Glorieux]. Node trend: 7 nm about 3.4× below 16 nm, 5 nm about 1.4× above 7 nm [F: Narasimham 2021 via Pieper]. 4N is a 5 nm-class node, so the central value sits near the A53 L2 and below the BRAM values. |
| Low-energy-proton factor, SRAM | 1 | 1 | 2 | Dodds 2015 [V] |
| σ_p, DRAM (cm²/bit) | 2e-20 | **2.15e-19** | 5e-19 | Central: **Micron GDDR5, 8 Gb, 1X nm** (on a Quadro P2000), 2.99 FIT/Gb under neutrons [V: Kamibayashi 2023], converted as FIT ÷ (2³⁰ × 12.946 n/cm²/h × 10⁹ h). Range: 4 Gb DDR3 under 21–230 MeV protons, 2e-20 to 5e-19 [V: ESA/IDA TN-IDA-RAD-14/8]. |
| Φ_p (/cm²/day) | 1.86e6 | **3.59e6** | 1.93e7 | §1 |

**Proton-only ρ_p = 6e-16 / 2.15e-19 ≈ 2.8 × 10³** (range 500 to
7.5 × 10⁴).

### 4.5 Total ρ for the reference mission

GCR reaching a 98° orbit is reduced by the geomagnetic field. No verified
SSO factor was found. The derivation uses **k_geo = 0.3** (0.2–0.5)
[ASSUMED]. This sits between "almost unattenuated" on polar passes
(ECSS-E-ST-10-04C) and 1/7–1/16 at ISS inclination (Dodds 2015). It
affects only the heavy-ion share.

```text
λ = σ_p · Φ_p · (LEP factor) + k_geo · λ_HI
```

| Case | λ_SRAM (/bit/day) | Proton share, SRAM | λ_DRAM (/bit/day) | Proton share, DRAM | **ρ** |
| --- | --- | --- | --- | --- | --- |
| low (all choices against a large ρ) | 5.0e-10 | 93 % | 9.4e-11 | 1 % | 5 |
| **central** | **3.0e-9** | **73 %** | **1.2e-12** | **64 %** | **2.5 × 10³** |
| high (all choices toward a large ρ) | 6.3e-8 | 93 % | 5.0e-13 | 77 % | 1.3 × 10⁵ |

The low case needs every pessimistic choice at once, including the
**assumed** narrow DDR4 Weibull. The plausible range is about 10²–10⁴.
**Independent cross-checks of the central rates:**

- **SRAM 3.0e-9 /bit/day.**
  - BSC/ESA estimated **2–9e-9** for the ECC-protected 12 nm Xavier L2/L3
    in LEO from protons alone [V: Rodriguez-Ferrandez 2022 Table V].
  - Older SRAMs on PROBA-2 (720 km, 98°) measured 1.1–1.9e-7 [V counts:
    Poivey 2015]. That is 40–60× higher, consistent with the measured
    per-bit drop from planar to FinFET (planar heavy-ion σ_sat 5e-9 to
    1e-8 vs FinFET ≲ 5e-10 [V: Glorieux 2024]).
- **DRAM 1.2e-12 /bit/day.** This is within the flight and prediction band
  for SDRAM and DDR: LRO 6.8e-12, CARMEN-2 SDRAM 2e-12 to 1.2e-10, ESA DDR3
  quiet 0.9–1.6e-12, GEO 90 nm DDR 1e-11 to 1e-10 [V: Ladbury 2006].
- **Same orbit, same shielding, both technologies: CARMEN-2 on JASON-2**
  (1336 km, 66°, 954 days) [V counts: Samaras 2011].
  - SRAMs and SSRAMs: 5e-10 to 2.5e-7 /bit/day.
  - SDRAMs: 2e-12 to 1.2e-10 /bit/day.
  - **Ratio of geometric means: 600** (extremes 4 to 10⁵).

  These are 2000s-era parts and a harsher proton orbit, but it is the one
  measurement that holds I1 and I2 fixed by construction. It lands within
  a factor of 4 of the central ρ.

**Best estimate: ρ ≈ 2.5 × 10³, quoted as "order 10³ (10²–10⁴)".**

### 4.6 The exposure-window ratio τ = T_S / T_D

- **DRAM: T_D = time since the weights were last written to GDDR6X**
  (model load or reload, or a scrub). Refresh does not restore data
  (§3), so T_D is as long as the operator lets it be.
- **L2, read-only lines (weights, constants): T_S = T_res,** the
  continuous residency time.
  - **Measured:** in every T4 run of the five models whose surface fits
    in the 72 MiB L2, R_eff equals the whole surface with stride 1. All
    resident lines stayed in L2 for the whole trial (10,000 images,
    4–10 s per trial, §5).
  - **Not measured:** whether lines stay resident across hours or days
    of deployment. That depends on the driver (flushes on context
    switches or power-state changes) and on co-running work.
  - **Bounds:**
    - **τ ≈ 1** when residency lasts as long as the DRAM accumulation (a
      dedicated inference accelerator whose model fits in L2).
    - **τ ≈ T_trial / T_D**, a few seconds out of months, when lines are
      refreshed every trial. Cache flips per trial are then about 0.
- **L2, engine-written lines (activations, scratch, bindings):** a flip
  lives for at most one image, so T_S ≤ T_image (about 1 ms). These lines
  contribute nothing in an accumulation reading.

## 5. Quantities used in the derivation

| Symbol | Meaning | Value | Why this value / source | Reasonable? |
| --- | --- | --- | --- | --- |
| φ_k(E), Φ_p | Trapped-proton flux, orbit-averaged, behind 100 mil Al | 1.86e6 / **3.59e6** / 1.93e7 per cm² per day | AP-8 MIN tables [V: Barak 1999 Table 1]; ×2 high case for AP-8 under-prediction [V: Lauenstein & Barth 2005] | Yes. A standard model. The altitude choice moves it 5×, which is carried as a range. |
| f(L) | GCR differential LET spectrum | 353 · L^−3.27, LET 1–28; ×0.01 above 28 | [V: Barak 1999]; integral checked against CREME96 [F: Poivey 2002] | Yes, for 1–28. Below LET 1 the integration is cut off, which under-counts SRAM and makes ρ conservative. |
| k_geo | GCR fraction reaching 98° LEO | 0.2 / **0.3** / 0.5 | [ASSUMED] between ECSS's "almost unattenuated" on polar passes and ISS's 1/7–1/16 | Weakest environment input. It changes ρ by less than 2× because protons carry 64–73 % of both rates. |
| σ_S(L) | SRAM heavy-ion cross section per bit | 9.3e-12 · L^1.07 | 16 nm SRAM arrays [F: Tam 2015; Glorieux 2024]; range from 7 nm CRAM and 16 nm BRAM [V] | Good for 16 nm. 4N has no public data; the node trend suggests within ±3×. |
| σ_D(L) | DRAM heavy-ion Weibull per bit | 6 DDR3 fits, L_0.25 26–48 | [V: ESA/IDA TN-IDA-RAD-14/14, 13/1A] | DDR3, not GDDR6X. Validated against OMERE and LRO rates (§4.3). DRAM sensitivity falls 4–5× per generation [V: Baumann 2005], so GDDR6X is probably lower, which raises ρ. |
| σ_p,S | SRAM proton/neutron cross section per bit | 2.5e-16 / **6e-16** / 1.5e-15 cm² | 16 nm and 7 nm FinFET data (§4.4) | Good. Several independent FinFET arrays agree within 3×. |
| σ_p,D | DRAM proton/neutron cross section per bit | 2e-20 / **2.15e-19** / 5e-19 cm² | Micron GDDR5 1X nm [V: Kamibayashi 2023]; DDR3 protons [V] | GDDR5 is the closest relative of GDDR6X with any data. No GDDR6/6X radiation data exist [V: searched, none]. |
| LEP | Low-energy-proton factor (SRAM) | 1 / 1 / 2 | [V: Dodds 2015] | Yes. |
| λ_S, λ_D | Per-bit upset rates in the reference mission | **3.0e-9**, **1.2e-12** /bit/day | §4.5 | Cross-checked against flight data (§4.5). |
| ρ | λ_S / λ_D | **2.5 × 10³** (10²–10⁴; bounds 5 to 10⁵) | §4.5 | Matches the same-orbit CARMEN-2 ratio of 600 within 4×. |
| T_res, τ | L2 residency of read-only lines, and T_S / T_D | ≥ one trial (measured); long-term unknown | T4 residency maps | The main modelling unknown (§4.6, §7). |
| R | DRAM injection surface | 9.1–93.9 MB per model | Frozen G7-v2 surfaces ([G7V2_RESULTS.md §4](G7V2_RESULTS.md)) | Measured. |
| R_eff | L2-resident surface (time-averaged) | equals R for five models; ViT-B 74.2 of 93.9 MB (79 %) | T4 residency maps, idle GPU 0 | Measured. |
| T_trial | One trial (10,000 images) | 4–10 s (median per model: ResNet-50 4.2, DeiT-S 5.9, MobileNetV3 6.8, EfficientNet-B0 8.9, Swin-T 9.1, ViT-B 10.0) | T4 run windows | Measured. |
| B, n_cache | DRAM and cache flips per trial | round(BER × R × 8), round(BER × R_eff_bits) | G5/G8 fault model | Exact under I4. |

## 6. What each BER level means in the reference mission

These use the central rates. "Same exposure" means the same mission time T
for both memories, with τ = 1.

| Level BER | T for DRAM to reach it | T for L2 to reach it | SRAM BER at the DRAM level's T (ρ × BER) | DRAM BER at the L2 level's T (BER / ρ) |
| --- | --- | --- | --- | --- |
| 1e-7 | 228 years | 34 days | 2.5e-4 | 4.0e-11 |
| 5e-7 | 1,140 years | 169 days | 1.2e-3 | 2.0e-10 |
| 1e-6 | 2,280 years | 337 days | 2.5e-3 | 4.0e-10 |
| 3e-6 | 6,840 years | 2.8 years | 7.4e-3 | 1.2e-9 |
| 5e-6 | 11,400 years | 4.6 years | 1.2e-2 | 2.0e-9 |
| 7e-6 | 16,000 years | 6.5 years | 1.7e-2 | 2.8e-9 |
| 1e-5 | 22,800 years | 9.2 years | 2.5e-2 | 4.0e-9 |

What the table means:

- **The ladder is realistic for SRAM.** Reaching 1e-7 … 1e-5 in L2 needs a
  model that stays resident for weeks to years, which is a deployment
  timescale.
- **For modern DRAM the ladder is a stress axis.** No mission accumulates
  1e-7 through cell upsets. Real DRAM data loss in orbit is dominated by
  SEFIs and bursts (§7). LRO's SDRAM stayed below 1e-11 errors/bit/day on
  97.6 % of days, but reached up to 1e-6 on bad days, when a few events
  corrupted many words [V: Ladbury 2010]. That is a different fault model
  from independent bit flips.
- **REMU's 1e-7 … 1e-6 /bit/day** [V: REMU §I, refs 9/11/12] comes from:
  - CARMEN-2's most sensitive device, the IDT 71V3558S SSRAM (1970 upsets
    in 954 days, about 2.5–4.4e-7 /bit/day depending on the slide's word
    organization) [V: Samaras 2011];
  - a 28 nm SRAM-FPGA neutron test [V: REMU ref. 11].

  Applied to DRAM, as in REMU, it overstates the DRAM rate by about
  8 × 10⁴ relative to our central λ_D. It is 34× our 4N L2 estimate, which
  is plausible for 2000s-era SRAM.

## 7. Uncertainties, ranked

| Rank | Unknown | Direction / size | How to remove it |
| --- | --- | --- | --- |
| 1 | **GeForce L2 protection.** Always-on parity or ECC independent of the user ECC toggle is not documented. A100 has SECDED on L2 [V]; Ada says nothing [V]. | With SECDED, single-bit L2 upsets never become silent corruption, and the effective ρ for SDC drops by orders of magnitude. With parity, they become detected errors (DUE or crash). | Not decidable from public documents. A beam test, or a fault test that writes L2 SRAM directly, would settle it. |
| 2 | **Long-term L2 residency T_res** | τ ranges from about 0 (lines refreshed every trial) to 1 (dedicated accelerator). This decides whether cache faults matter at all in deployment. | Measurable on our setup: re-probe residency after idle gaps, context switches and host copies. **Not started.** |
| 3 | **No 4N SRAM or GDDR6X radiation data** | ρ within about ×3 per side. The DRAM trend (4–5× per generation lower) points to a higher ρ. | Only a beam test on the RTX 4090 (survey report, conclusion). |
| 4 | **DRAM SEFI and burst errors** are excluded from λ_D. | Including them raises the effective DRAM data loss by up to about 10× and lowers ρ for "corrupted bits". They are clustered, not independent flips, so they need their own fault model. | Separate SEFI and burst model (future work). |
| 5 | k_geo for SSO, AP-8 accuracy, low-energy protons, electrons at 4N | Each changes ρ by less than 2×. | CREME96 or OMERE run for the exact orbit. |

## 8. Consequences for the G8-T4 results

Nothing has been re-run. This section only reinterprets the results already
recorded in [G8_T4_RESULTS.md](G8_T4_RESULTS.md).

1. **ρ = 1 is not supported for LEO.** At equal exposure, physics gives
   ρ ≈ 10³. ρ = 1 is equivalent to `ρ · τ = 1`, i.e. τ ≈ 1/2,500 (an L2
   line held for 1/2,500 of the DRAM accumulation time).
   - At our levels that means T_res = 34 days … 9 years against DRAM
     accumulation times of 230 … 23,000 years.
   - This is a consistent physical scenario only as a description of the
     **SRAM** side.
2. **SRAM-only curves.** These are physically meaningful. The x-axis value
   BER_S corresponds to an L2 exposure of BER_S / λ_S (34 days – 9 years,
   central rates).
   - The physically paired DRAM BER at the same exposure is BER_S / ρ =
     4e-11 … 4e-9. That is **at most about 1 bit (ResNet-50) to 3 bits
     (ViT-B) per trial**, against 727–5,937 cache flips at 1e-5.
   - So the SRAM-only curve is also, within statistics, the physically
     paired DRAM+SRAM curve. That holds if read-only lines stay resident
     (τ = 1, rank 2 in §7) and the L2 is unprotected (rank 1).
3. **DRAM-only curves** are valid as **per-bit vulnerability (iso-BER)**
   curves. That matches how REMU and the GPU fault-injection literature use
   BER sweeps. Their BER axis is not a realistic DRAM exposure in LEO.
4. **DRAM+SRAM at equal BER** is a controlled **iso-BER combination**: both
   memories receive the same flipped-bit fraction. It answers "does a flip
   in L2 hurt as much as a flip in DRAM, and do they add?", which
   [G8_T4_DISCUSSION.md](G8_T4_DISCUSSION.md) shows they do. It is not an
   equal-exposure point.
5. **Timing of cache flips.** These are two implementation choices relative
   to the accumulation reading of (*):
   - **Read-only lines:** our injector starts each cache flip at a uniform
     random image (plan §3.4). Under accumulation, a flip on a read-only
     line would be present from image 0. On average the T4 runs therefore
     expose about half as many images per cache flip, an under-weight of
     up to 2×.
   - **Engine-written lines:** under accumulation these get only
     λ_S × T_image, effectively 0. The T4 runs placed 5 % (ViT-B) to 61 %
     (EfficientNet-B0) of their cache flips there (`overwritten` +
     `restored` removal counts), which is an over-weight.

   Both effects are within the ρ uncertainty. They are listed so the
   reading of the curves is exact.

Options, for the user to decide; none started:
- **(a) Keep T4 as is and label the axes.**
  - Label the T4 figures and tables as iso-BER.
  - Add this document's mapping: SRAM-only BER ↔ L2 exposure time, with the
    physically paired DRAM contribution negligible.
- **(b) Run a physically paired campaign.** BER_S on the ladder, BER_D =
  BER_S / 2.5 × 10³ (0–3 DRAM bits), and cache flips present from image 0
  on read-only lines only.
- **(c) Measure long-term L2 residency (τ).** This is the cheapest
  experiment that changes the physical conclusion.

## 9. References

Local copies of the sources are in the session scratchpad
(`lit/`, `lit_env/dl/`, `lit_sram/`, `lit_dram/`); the URLs below are the
originals.

**Environment and rate methods**
- ECSS-E-ST-10-12C, *Methods for the calculation of radiation received and its effects*, 2008 (Corr.1 2017), Table 6-1, §9.5.3 [V]. https://ecss.nl/wp-content/uploads/2008/11/ECSS-E-ST-10-12C(15November2008)Corr.1(22February2017).pdf
- ECSS-E-ST-10-04C, *Space environment*, 2008, Annex I.1.2, I.1.4.1 [V]. https://ecss.nl/wp-content/uploads/standards/ecss-e/ECSS-E-ST-10-04C15November2008.pdf
- J. L. Barth, C. S. Dyer, E. G. Stassinopoulos, "Space, atmospheric, and terrestrial radiation environments", IEEE TNS 50(3), 2003 [V]. https://ntrs.nasa.gov/api/citations/20030054354/downloads/20030054354.pdf
- J. Barak, R. A. Reed, K. A. LaBel, "On the figure of merit model for SEU rate calculations", IEEE TNS 46(6), 1999, DOI 10.1109/23.819114. Table 1, Fig. 1, eq. 10 [V].
- E. L. Petersen, "The SEU figure of merit and proton upset rate calculations", IEEE TNS 45(6), 1998, DOI 10.1109/23.736497 [S via Barak 1999].
- J.-M. Lauenstein & J. L. Barth, "Radiation belt modeling for spacecraft design: model comparisons for common orbits", NSREC 2005 Data Workshop W-16 [V]. https://ntrs.nasa.gov/api/citations/20050209938/downloads/20050209938.pdf
- C. Poivey, NSREC 2002 Short Course, Fig. 12, CREME96 GCR LET spectrum [F]. https://nepp.nasa.gov/docuploads/A6B8B953-E2DD-4D92-AB8A873A04F0B10A/NSREC02_SC_Poivey.pdf
- N. A. Dodds et al., "The contribution of low-energy protons to the total on-orbit SEU rate", IEEE TNS 62(6), 2015 (SAND2015-7240C), Tables I, V [V]. https://www.osti.gov/servlets/purl/1326915
- B. D. Sierawski et al., IEEE TNS 56(6), 2009; D. F. Heidel et al., IEEE TNS 55(6), 2008 (low-energy protons) [A].
- J. M. Trippe et al., electron-induced upsets in bulk SRAMs, IEEE TNS 62(6), 2015, DOI 10.1109/TNS.2015.2496967 [A].
- R. Ladbury et al., NEPP solid-state recorder lessons learned (SeaStar SAA statistics, DRAM upset physics) [V]. https://nepp.nasa.gov/files/24662/Ladbury_2013_n224_SSR%20Update.pdf

**Flight data**
- A. Samaras et al., CARMEN-2/JASON-2 memory results, CNES slides 28 Mar 2011 (paper: RADECS 2011) [V counts]. https://escies.org/download/webDocumentFile?id=49242
- C. Poivey et al., PROBA-2 TDM final presentation, CNES-ESA 2015 [V]. https://escies.org/download/webDocumentFile?id=63087
- R. Ladbury, LRO SDRAM flight data, SEE Symposium 2010 [V]. https://nepp.nasa.gov/files/25473/SEESYM2010_Ladbury_SDRAM.pdf
- Y. Bentoutou, Alsat-1 SRAM SEU statistics, WASET 5(5), 2011 [V]. Zenodo 1072680.
- J. Rodriguez-Ferrandez et al., Jetson Xavier proton test and LEO rate estimate, IOLTS 2022, Table V [V]. https://upcommons.upc.edu/server/api/core/bitstreams/80d67a8c-34a3-408b-aaf1-5bc5bc48b1c1/content

**SRAM (FinFET)**
- D. S. Lee et al., heavy-ion and neutron characterization of 16 nm UltraScale+ (KU9P), SAND2018-7612C [V]. https://www.osti.gov/servlets/purl/1570815
- M. Glorieux et al., TSMC N16FFC SRAM heavy-ion and proton (incl. low-energy) results, IEEE TNS 2024 [F/V]. https://jyx.jyu.fi/bitstreams/0e982989-6555-43bb-a552-13b04b3fdb44/download
- J. Mayo et al., Versal VC1902 (7 nm) heavy-ion CRAM, IEEE TNS 2025 [V]. https://jyx.jyu.fi/bitstreams/d27614ac-5e21-4ef3-abb5-95c3224c7fce/download
- A. Pieper, Vanderbilt dissertation, 2023, Fig. 11 (TSMC node trend after Narasimham IRPS 2021) and Fig. 27 (Tam IRPS 2015 16 nm SRAM) [F]. https://irbe.library.vanderbilt.edu/server/api/core/bitstreams/352f4daf-7bbc-4def-9743-46a7c32de0ff/content
- A. Agiakatsikas et al., Zynq UltraScale+ cache and BRAM neutron/proton cross sections, arXiv 2303.08098, Tables I, IX [V]. https://arxiv.org/pdf/2303.08098
- AMD UG116, *Device Reliability Report* (UltraScale+ and Versal BRAM/URAM/CRAM neutron cross sections) [V].
- S. M. Guertin, "ARM Radiation Testing & Collaborations", NEPP ETW 2020, slide 20 [V]. https://nepp.nasa.gov/docs/etw/2020/17-JUN-WED/1130-Guertin-NEPP-ETW-CL20-2459-ARM-v5.pdf
- NVIDIA A100 Tensor Core GPU Architecture whitepaper (L2 SECDED) [V]; NVIDIA Ada GPU Architecture whitepaper (no L2 ECC statement) [V].

**DRAM**
- IDA TU Braunschweig / Airbus for ESA, TN-IDA-RAD-14/14 (DDR3 heavy-ion Weibull fits) [V]. https://escies.org/download/webDocumentFile?id=62854; TN-IDA-RAD-13/1A [V] https://escies.org/download/webDocumentFile?id=62805; TN-IDA-RAD-14/8 (DDR3 protons, 21–230 MeV) [V] https://escies.org/download/webDocumentFile?id=62807; RHM-ASG-RP-0002 (parts; rate predictions Tables 7-1, 7-2) [V] https://escies.org/download/webDocumentFile?id=62803
- R. Ladbury et al., DDR2/DDR3 heavy-ion SEE, IEEE TNS 2013 [V]. https://ntrs.nasa.gov/api/citations/20160000788/downloads/20160000788.pdf
- R. Ladbury et al., 1 Gb DDR (90 nm) SEE and GEO rate, 2006 [V]. https://ntrs.nasa.gov/api/citations/20080039559/downloads/20080039559.pdf
- A. Dufour et al., "Compendium of SEE and TID Test Results for DDR4 SDRAM memories", RADECS 2022, DOI 10.1109/RADECS55911.2022.10412562 [V]. https://www.altertechnology.fr/wp-content/uploads/2024/11/Compendium-of-SEE-and-TID-Test-Results-for-DDR4-SDRAM-memories_Paper.pdf
- Kamibayashi, Kobayashi, Hashimoto, neutron SER of GDDR5 and LPDDR4, IEICE Tech. Rep. VLD2022-65, 2023 [V]. https://www-vlsi.es.kit.ac.jp/thesis/papers/pdfs/VLD_2023_01_kamibayashi.pdf
- M. B. Sullivan et al., "Characterizing and mitigating soft errors in GPU DRAM", MICRO 2021 [V]. https://www.mbsullivan.info/attachments/papers/sullivan2021characterizing.pdf
- R. C. Baumann, "Radiation-induced soft errors in advanced semiconductor technologies", IEEE D&T 2005 [V]. http://www.cs.columbia.edu/~cs4823/handouts/baumann-soft-errors-DT-05.pdf

**Fault-injection context**
- L. Xu et al., "REMU: Memory-aware radiation emulation via dual addressing for in-orbit deep learning system", DAC 2025, DOI 10.1109/DAC63849.2025.11132935 (refs 9, 11, 12) [V, local PDF in repo root].
- The 2026-10-03 survey: [reports/GPU SRAM vs DRAM error rates.md](../reports/GPU%20SRAM%20vs%20DRAM%20error%20rates.md). Its heavy-ion ρ ≈ 0.1–2 is superseded by §4.3.
