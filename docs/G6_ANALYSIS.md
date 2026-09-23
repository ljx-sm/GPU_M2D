# G6 Analysis — First Attribution Pass over the Nine-Level Campaign

Status: **G6-T0 complete 2026-09-22** (analysis only, zero new
collection). Inputs: the 19 VERIFIED G5 campaigns (L1–L5 × 3 cards,
L6–L9 single card; 1900 trials). Every number in this document is
reproduced by `tools/g5_faultinj/analyze_g6_attribution.py` (pure
stdlib; run from the repo root). The four questions are the ones posed
in the 2026-09-22 discussion; each is answered from the campaign CSVs
alone.

## 1. Is the plateau an artifact of missing flips? — No (three independent proofs)

1. **Counting + per-site verification.** Site rows equal frozen
   B × trials exactly at every level (600 / 3,300 / 6,300 / 31,800 /
   63,300 / 105,700 / 211,400 / 1,057,100 / 2,114,300), with **zero**
   `guard_bytes_unchanged` / `reverse_map_ok` violations. Every site was
   verified online as `after == before ^ mask` by the runner and
   re-verified from the logged before/after values by the orchestrator
   (a precondition of `G5_CAMPAIGN_VERIFIED`), and every trial's
   restore returned the weights byte-exact.
2. **Behavioral evidence.** The flips demonstrably reach the output at
   every level: 63.1% of (trial, image) pairs already show a numerical
   output deviation (|ΔP| > 1e-6) at L1 (B = 2), rising to ~100% from
   L3. The corruption propagates through all ~50 layers into the
   logits; it is simply too small to cross decision boundaries.
3. **Amplitude quantification.** Clean top-1 confidence percentiles are
   p10/p50/p90 = 0.554 / 0.722 / 0.809 (decision margins of order
   0.1–0.7). At L5 the per-image |ΔP| median is 0.0078 — two orders of
   magnitude below the typical margin; only tail images flip (top-1
   change 1.03%), and half of those flip "back to correct"
   (r2w:w2r ≈ 2:1), netting −0.30 pp.

**Conclusion: the plateau is genuine absorption by the model's decision
margins, not missed injection.**

## 2. Where do the flipped bits land? — 91.8% in the INT8 weight blob

Pooled over all 19 runs (allocation ids normalized across cards;
residency share vs site share shows the sampler is unbiased):

| allocation | content | residency | sites | restore_check (all runs) |
|---|---|---|---|---|
| trt-internal-0 | INT8 weight blob, 23,995,396 B (≈ ResNet-50's 25.6 M params in INT8) | 90.79% | 91.79% | exact = 3,298,687 |
| trt-internal-3 | activation / engine scratch, 1.8 MiB | 6.83% | 6.84% | mismatch = 245,987 |
| trt-binding-data | input binding, 602 KiB | 2.28% | 1.29% | exact |
| trt-internal-1/2 | small TRT objects | 0.10% | 0.08% | exact |
| trt-binding-prob/index | output bindings, 4 B each | 3e-5% | 0.00% | never hit |

Mechanistic confirmations:

- **Weight corrosion is persistent.** Every one of the 3.3 M weight
  site rows restores `exact` — the engine never rewrites those bytes
  during a pass, so all 1000 images of a trial are inferred with the
  same corrupted weights. The accuracy effect is systematic, not
  per-image luck. This is the carrier of the level curve.
- **Scratch corrosion is transient.** `trt-internal-3` is rewritten by
  the engine during the pass (soft-upset semantics, the recorded
  `mismatch` restores): flips there are mostly overwritten before they
  matter.
- **Input flips are doubly absorbed:** under-sampled by the V/L
  anchor-mate residency effect (1.29% of sites vs 2.28% residency,
  docs/G5_FAULT_MODEL.md §4) and absorbed by INT8 quantization
  (measured zero output effect at L1 and L5).
- Honest boundary (R4): attribution is allocation-level. Inside
  trt-internal-0 we cannot split "pure weights" from engine metadata;
  the size match to the INT8 parameter count and the never-rewritten
  byte behavior are the evidence, and they are consistent.

## 3. Why does accuracy collapse from 5e-6? — perturbation crosses the margin distribution

| level | BER | weight bytes / trial | % of 24 MB | \|ΔP\| median | top-1 change | accuracy drop | r2w:w2r |
|---|---|---|---|---|---|---|---|
| L5 | 1e-6 | 193 | 0.0008% | 0.008 | 1.0% | −0.30 pp | 2:1 |
| L6 | 5e-6 | 974 | 0.0041% | 0.018 | 5.1% | −3.42 pp | 6.6:1 |
| L7 | 1e-5 | 1,948 | 0.0081% | 0.028 | 6.7% | −4.62 pp | 7.1:1 |
| L8 | 5e-5 | 9,731 | 0.0406% | 0.132 | 31.2% | −27.5 pp | 31.9:1 |
| L9 | 1e-4 | 19,362 | 0.0807% | 0.260 | 63.6% | −59.2 pp | 28:1 |

The knee is the crossing of two distributions: the perturbation
amplitude (per-image |ΔP|, growing with the number of corrupted weight
bytes) and the fixed decision-margin distribution (7.4% of images have
clean confidence < 0.5 and fall first). At L5 the median perturbation
is ~2 orders below the margins; at L6–L7 it starts consuming the
low-margin population and the damage turns asymmetric (r2w:w2r rises
from 2:1 to ~7:1); at L9 the median perturbation (0.26) is the same
order as the margins themselves, so most images flip regardless of
their original correctness. A second, milder superlinearity: damage
per corrupted weight byte grows from 0.0016 pp (L5) to ~0.003 pp (L9) —
multiple corrupted weights on one forward path compound through depth.

The correct phrasing for the paper: **not a hardware threshold but the
point (BER ≈ 2e-6) where accumulated logit perturbation enters the bulk
of the decision-margin distribution.**

## 4. Why zero DUE at every level? — structural, not luck

- The only allocations whose corruption can produce an invalid output
  are the two output bindings: 8 B of 26.4 MiB (3e-7 of residency).
  Expected hits over the L9 campaign λ ≈ 0.64 → P(0) = e^-0.64 ≈ 53%;
  observed 0, consistent (the coin landed on the no-DUE side). L1–L5
  expectations are ≪ 1.
- INT8 TensorRT execution is saturating fixed-point arithmetic: a
  corrupted weight produces bounded wrong logits, never NaN/Inf
  propagation — the worst case is a wrong top-1 (SDC), not an invalid
  output. Quantization acts as a de-facto DUE firewall.
- The fault channel itself (XOR on a healthy process's device memory)
  has no crash path: no ECC trip, no page fault. Honest boundary:
  hardware-level DUE (e.g. ECC uncorrectable) is outside this fault
  model, as is the cache hierarchy (plan §8).

## 5. Overall conclusion — two-regime robustness

On RESISC45, the INT8 ResNet-50 shows a **high tolerance floor with
cliff-type failure**: full absorption up to BER 1e-6 (≤ 0.0008% of
weight bytes corrupted per pass; ≤ 0.3 pp, half of the damage
self-correcting), then superlinear collapse past the knee (≈ 2e-6),
reaching 36% accuracy at 1e-4 (0.08% of weights). There is no gradual
linear-degradation middle band — a reliability warning in itself:
**low-BER behavior cannot be extrapolated to high BER** (the campaign's
own pre-run power-law extrapolation under-predicted the L9 loss by
57 pp).

## 6. Reproduction

```bash
python3 tools/g5_faultinj/analyze_g6_attribution.py            # all tables
python3 tools/g5_faultinj/analyze_campaign.py --min-trials 100 # level curve
/data1/luojx/miniforge3/envs/vit_fault/bin/python \
    tools/g5_faultinj/plot_accuracy_curve.py                   # the figure
```

Data: `artifacts/g5/campaign/` (untracked by design; the four result
CSVs per run directory are the complete source for every number above).
