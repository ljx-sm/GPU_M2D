# G8-T4 discussion: questions raised by the three-way results

Analysis of the G8-T4 results (six models × seven BER levels × three fault
modes; tables in [G8_T4_RESULTS.md](G8_T4_RESULTS.md), figures in
`artifacts/g8/t4/fig/`), written 2026-10-08. It answers four questions
raised by the user after reviewing the figures.

Every number below is recomputed from the VERIFIED run directories by
`tools/g8_cache/t4_discussion_numbers.py`. Its complete output is
committed as [G8_T4_DISCUSSION_numbers.txt](G8_T4_DISCUSSION_numbers.txt).
No new experiments were run for this analysis, except the 3-trial
zero-fault check in §3, which ran on 2026-10-05.

Terms used:

- **Collapsed trial:** a completed trial whose top-1 accuracy is below
  40 %. From 3e-6 on, most trials stay near clean accuracy and a minority
  collapse.
- **Hit:** one independent fault location in a given allocation. For DRAM
  this is one fault event (1–3 bits); for SRAM it is one applied cache
  flip.
- **Per-hit fatality q:** the maximum-likelihood fit of
  P(no crash | k hits) = (1 − q)^k over all trials of a model and mode.

## 1. Why does accuracy sometimes rise as BER increases?

**It is sampling noise, not a real effect.**

- **Not significant.** Across 6 models × 3 modes there are 108 steps
  between adjacent levels. Accuracy rises in 12 of them, and none is
  significant at 95 %: the largest is z = 0.82, where 1.96 would be
  needed.
- **Driven by collapsed trials.** The level mean depends heavily on how
  many trials collapsed, and in almost every rise the higher level
  simply had fewer:

  | Rise | Mean | z | Collapsed trials | Median trial |
  | --- | --- | --- | --- | --- |
  | ViT-B DRAM-only L1→L2 | 76.80 → 77.72 | 0.82 | 2 → 1 | 78.36 → 78.41 |
  | ViT-B DRAM-only L6→L7 | 70.86 → 71.73 | 0.32 | 9 → 5 | 77.97 → 77.77 |
  | Swin-T SRAM-only L5→L6 | 74.29 → 75.31 | 0.44 | 6 → 3 | 81.05 → 80.97 |
  | Swin-T DRAM-only L4→L5 | 74.20 → 74.82 | 0.21 | 8 → 7 | 81.22 → 81.06 |
  | MobileNetV3-L DRAM + SRAM L2→L3 | 72.71 → 73.21 | 0.50 | 3 → 0 | 75.00 → 74.97 |
  | ResNet-50 DRAM-only L7→L8 | 66.62 → 67.14 | 0.16 | 13 → 13 | 78.18 → 77.53 |

  The other six rises are below 0.4 pp.
- **The median does not rise.** The median trial accuracy rises by more
  than 0.05 pp in only 1 of the 108 steps.
- **Levels are independent draws.** Each level samples its own fault
  sites; it is not the previous level plus extra flips. Nothing makes the
  mean curve smooth: it is a noisy estimate of a mixture of "nearly
  intact" and "collapsed" trials.

## 2. Why does SRAM-only crash more than DRAM-only (Swin-T, EfficientNet-B0)?

The expectation that SRAM faults matter less holds for accuracy: a cache
flip does 0.32–0.60× the damage of a DRAM flip (G8 plan §15.11). A crash
is different. The process dies the first time one critical value is
read, so it depends on *where* faults land and how many independent ones
there are, not on how long they last.

- **One small allocation causes every crash.**
  - In Swin-T and EfficientNet-B0, every crash in every mode involves a
    fault in `trt-internal-2`. That is an 18,944-byte allocation TensorRT
    creates with the execution context, holding kernel control state:
    0.043 % of Swin-T's memory R and 0.110 % of EfficientNet-B0's.
  - For cache-caused crashes the culprit is identifiable, because the
    process dies at the image where the flip is applied. A
    `trt-internal-2` flip starts at the crash image in 122 of 128
    (Swin-T) and 137 of 140 (EfficientNet-B0) mid-trial crashes. In the
    remaining 9, the trial already had an earlier `trt-internal-2` hit.
  - Across all modes, not one crash happened in a trial without such a
    hit.
- **A hit is about equally fatal in every mode; SRAM simply gets more
  hits.**

  | Model | Mode | Crashes / 700 trials | Hits per trial | Fatality per hit q |
  | --- | --- | --- | --- | --- |
  | Swin-T | DRAM-only | 33 | 0.26 | 0.191 |
  | Swin-T | SRAM-only | 63 | 0.49 | 0.189 |
  | Swin-T | DRAM + SRAM | 87 | 0.73 | 0.180 |
  | EfficientNet-B0 | DRAM-only | 24 | 0.26 | 0.137 |
  | EfficientNet-B0 | SRAM-only | 70 | 0.52 | 0.197 |
  | EfficientNet-B0 | DRAM + SRAM | 89 | 0.74 | 0.183 |

  EfficientNet-B0's lower DRAM-only q rests on only 24 crashes, about
  1.5 standard errors from the cache modes. DRAM + SRAM hits are the sum
  of the two single modes.
- **Why SRAM gets about 2× the hits at the same bit count.**
  - DRAM bits come in 1–3-bit events, 1.60 bits per event on average,
    while cache flips are all single bits. The same number of bits
    therefore means 1.6× more independent locations: the fault model
    predicts 0.58 vs 0.36 hits per trial.
  - The DRAM sampler also lands on this small allocation less than its
    size share predicts: 0.26 observed vs 0.36 expected. See §4, item 7.
- **Shorter lifetime does not help.** `trt-internal-2` is read at every
  inference. A cache flip that starts at image 5,000 still crashes the
  process within one image, so "present for half a trial on average"
  reduces gradual accuracy damage but not crash probability.
- **Other crashes.** All crashes of the other models in cache modes are
  DRAM faults at the first inference: DeiT-S 3 and ViT-B 4, in DRAM +
  SRAM only. ResNet-50 and MobileNetV3-L have none.

## 3. ViT-B: accuracy above clean, and DRAM + SRAM above DRAM-only

**Above clean (SRAM-only L1–L3 78.36–78.38 vs clean 78.21).** This is a
real fault effect, not a measurement offset.

- A zero-fault campaign (cache BER 0, no DRAM faults, 3 trials) matches
  the clean pass on 10,000 of 10,000 images in every trial, class and
  probability alike. The trial pipeline is deterministic and unbiased.
- The gain comes from borderline images: in SRAM-only at 1e-7, 156
  images change top-1 in at least 90 % of trials.
  - Their median clean top-1 probability is 0.185, against 0.988 for
    images that never change.
  - 134 of the 156 are wrong in the clean pass, and about 35 % of their
    changes land on the true label. A few random flips therefore give a
    small net gain of about +0.17 pp.

**DRAM + SRAM above DRAM-only (L1–L3, L5).** This is not SRAM
correcting DRAM errors.

- **Different DRAM faults.** ViT-B landed on different physical pages in
  the two runs (0 of 51 equal), so DRAM-only (G7-v2) and DRAM + SRAM
  are independent DRAM draws, not the same faults with cache flips
  added.
- **The gap is collapsed trials.** Without them the modes agree:

  | Level | Mode | Mean | Median | Collapsed | Mean of non-collapsed |
  | --- | --- | --- | --- | --- | --- |
  | L1 | DRAM-only | 76.80 | 78.36 | 2 | 78.20 |
  | L1 | DRAM + SRAM | 78.20 | 78.40 | 0 | 78.20 |
  | L4 | DRAM-only | 77.08 | 78.30 | 1 | 77.86 |
  | L4 | DRAM + SRAM | 74.68 | 78.27 | 4 | 77.62 |
  | L7 | DRAM-only | 71.73 | 77.77 | 5 | 75.60 |
  | L7 | DRAM + SRAM | 66.73 | 76.85 | 11 | 74.10 |

  Where DRAM + SRAM happens to draw more collapses (L4, L7), it is
  clearly worse.
- **Correction is physically negligible.** A cache flip can undo a DRAM
  fault only by landing on the same bit. Of 1,545,310 applied cache
  flips, 10 did (6 per million); 94 landed in a DRAM-flipped byte.

## 4. Points worth further analysis and discussion

1. **How crashes and DUEs enter the headline accuracy.** Accuracy over
   completed trials excludes DUE and crash trials, which flatters the
   crash-heavy models. Counting them as 0 (accuracy per attempted
   trial) at 1e-5:

   | Model | DRAM-only | SRAM-only | DRAM + SRAM |
   | --- | --- | --- | --- |
   | ResNet-50 | 61.23 → 60.01 | 70.36 → 69.65 | 53.25 → 52.19 |
   | MobileNetV3-L | 47.27 → 44.44 | 58.33 → 53.66 | 30.02 → 27.02 |
   | EfficientNet-B0 | 61.34 → 57.05 | 69.79 → 51.64 | 53.84 → 37.69 |
   | DeiT-S | 61.11 → 61.11 | 70.58 → 69.16 | 56.17 → 55.05 |
   | Swin-T | 58.84 → 48.25 | 68.02 → 53.06 | 51.30 → 34.89 |
   | ViT-B | 71.73 → 69.58 | 76.13 → 76.13 | 66.73 → 64.06 |

   For EfficientNet-B0, SRAM-only even drops below DRAM-only. Reporting
   both conventions, or crash and DUE rates next to accuracy, avoids
   hiding this.
2. **Distributions rather than means.** Trials are bimodal from 3e-6 on.
   The collapse rate plus the median (or confidence intervals on the
   figures) would make the curves monotonic and honest. Most of the
   visible wiggles are a handful of collapsed trials (§1).
3. **The crash-critical block.** One 19 KB TensorRT control allocation
   accounts for every process crash in all three modes, and about 14–20 %
   of the hits on it are fatal. That makes it a concrete, cheap
   protection target, for example a checksum or duplication. Mapping
   which offsets and bits in it are fatal would sharpen the result.
4. **Finding the flip behind each collapse.** Cache flips start at known
   images, so the per-image accuracy within a trial shows which single
   flip collapsed it. DRAM injection cannot give this, because all its
   faults are present from the first image. The same method applies to
   MobileNetV3-L's recurring DUE trials (85 and 89 from L4; 36, 63 and 70
   from L5). It would identify the catastrophic bits, likely in
   quantisation scales, biases or normalisation parameters. This is the
   suggested next experiment and has not been started.
5. **The SRAM / DRAM damage ratio is largely set by the model.** Models
   that fit in L2 stay resident on an idle GPU, so a cache flip persists
   from a uniformly random start image to the end of the 10K-image
   trial: about half a trial. That is the origin of ratios around 0.5
   (0.32–0.60). With longer trials, SRAM damage would approach DRAM
   damage.
   - The trial length therefore sets the ratio. This connects to the
     time-window factor deliberately left out in T3 (G8 plan §3.3), and
     the paper should state it.
   - The engine-written share of cache flips (5.4 % ViT-B to 60.6 %
     EfficientNet-B0) is not an explanation: DRAM faults in
     engine-written memory are also overwritten at the first inference.
6. **Additivity is exact only for ResNet-50.** It is the only model whose
   DRAM faults are paired between DRAM-only and DRAM + SRAM (20 of 20
   physical pages equal). The other five got different physical pages
   (0 of 14, 15, 19, 27 and 51 equal), so their additivity check
   compares level means with wide intervals. That is consistent with
   additivity, but weaker evidence.
7. **DRAM sampling of small allocations.**
   - EfficientNet-B0's DRAM-only crash count falls at the top levels:
     7 → 6 → 3 at L5–L7. Its DRAM faults hit `trt-internal-2` in 30,
     44 and 34 trials, so at L7 fewer trials were exposed than at L6.
     In general the DRAM sampler hits this block at about 0.26 events
     per trial, against 0.36 expected from its size.
   - It is worth checking whether the G5 DRAM sampler covers small
     allocations on shared physical pages in proportion. If not,
     DRAM-only crash rates are slightly underestimated.
8. **Borderline images inflate the SDC metric.** Images changed in at
   least 90 % of SRAM-only trials at the lowest BER: ResNet-50 5,
   MobileNetV3-L 0, EfficientNet-B0 0, DeiT-S 118, Swin-T 11, ViT-B 156.
   For DeiT-S and ViT-B they dominate the "top-1 changed" rate at low
   BER. A metric restricted to confident images would separate real
   fault sensitivity from tie-breaking.
