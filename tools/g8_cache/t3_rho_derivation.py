#!/usr/bin/env python3
"""G8-T3 derivation of rho = lambda_SRAM / lambda_DRAM (per-bit upset rate
ratio of the RTX 4090 L2 vs its GDDR6X) for the reference mission of
docs/G8_T3_SRAM_BER_DERIVATION.md, and of the SRAM BER that pairs with
each DRAM BER level.

Every input below is a literature value; the comment on each names the
source (full references in the document, §9). Every output printed here
is quoted in the document, so the document can be re-checked by running

  python3 tools/g8_cache/t3_rho_derivation.py

Standard library only. Units: cross sections cm^2/bit, LET MeV*cm^2/mg,
fluxes per cm^2 per day, rates upsets per bit per day.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------
# 1. Environment (reference mission: LEO sun-synchronous 500-750 km,
#    ~2.5 mm (100 mil) Al equivalent shielding, solar minimum)
# ---------------------------------------------------------------------

# GCR differential LET spectrum, solar minimum, ~200 mil Al, no
# geomagnetic shielding: f(L) = 353 L^-3.27 ions/(cm2 day LET), valid
# 1..28; above 28 the flux "falls ... by more than two orders of
# magnitude" [Barak 1999, Fig. 1 and p. 1506].
GCR_P, GCR_N = 353.0, 3.27
L_MIN, L_KNEE, L_MAX = 1.0, 28.0, 100.0
ABOVE_KNEE = 0.01

# Fraction of the interplanetary GCR flux reaching a ~98 deg LEO orbit
# (geomagnetic shielding). ASSUMPTION: no verified SSO number was found;
# ECSS-E-ST-10-04C I.1.2 says high-latitude parts see "almost unattenuated"
# GCR, while ISS (51.6 deg) sees 1/7..1/16 of GEO [Dodds 2015 Table V].
K_GEO = {"low": 0.2, "central": 0.3, "high": 0.5}

# Trapped protons, total flux behind 100 mil Al, solar minimum (AP-8),
# orbit-averaged [Barak 1999 Table 1]: 500 km/90 deg 1.86e6, 750 km/90 deg
# 9.65e6 per cm2 per day. 600 km by log interpolation. AP-8 under-predicts
# > 10 MeV by >= 2x [Lauenstein & Barth 2005] -> high case x2.
PHI_P_500, PHI_P_750 = 1.86e6, 9.65e6
PHI_P_600 = PHI_P_500 * (PHI_P_750 / PHI_P_500) ** (100 / 250)
PHI_P = {"low": PHI_P_500, "central": PHI_P_600, "high": 2 * PHI_P_750}

# ---------------------------------------------------------------------
# 2. SRAM (proxy for TSMC 4N L2): per-bit cross sections
# ---------------------------------------------------------------------

# Heavy ions, 16 nm FinFET SRAM arrays (points read off figures, ~+-30%):
# TSMC 16 nm SP SRAM [Tam IRPS 2015 via Pieper 2023 Fig. 27] and TSMC
# N16FFC [Glorieux TNS 2024]. Fitted with a power law sigma = a L^b.
SRAM_HI_POINTS = [(2.4, 1.7e-11), (7.0, 6e-11), (16.5, 1.6e-10),
                  (50.0, 5.5e-10), (1.5, 1.5e-11), (2.3, 3e-11),
                  (7.0, 9e-11), (22.0, 3.5e-10)]
# low case: 7 nm Versal CRAM per-bit points [Mayo TNS 2025]
SRAM_HI_POINTS_LOW = [(2.8, 1.2e-12), (8.8, 2.2e-11), (32.0, 1.1e-10),
                      (62.4, 2.1e-10)]
# high case: 16 nm UltraScale+ BRAM Weibull [Lee SAND2018-7612C]
SRAM_HI_WEIBULL_HIGH = (1.5, 10.0, 1.5, 3e-9)   # L0, W, s, sigma_sat

# High-energy protons / neutrons (same nuclear-reaction mechanism):
# 16 nm BRAM 1.1e-15 (64 MeV p) [Agiakatsikas 2023 Tab. I]; 7 nm Versal
# BRAM 1.2e-15 (neutron) [AMD UG116]; 16 nm A53 L2 data 6.29e-16 (neutron)
# [Agiakatsikas 2023 Tab. IX]; N16FFC 4.5e-16 (50-230 MeV p) [Glorieux
# 2024]; TSMC 7 nm ~3.4x below 16 nm, 5 nm ~1.4x above 7 nm [Narasimham
# IRPS 2021 via Pieper Fig. 11].
SRAM_P = {"low": 2.5e-16, "central": 6e-16, "high": 1.5e-15}
# Low-energy-proton direct ionization multiplies the total rate by
# < 2 except GEO worst day (4.3x) [Dodds 2015].
SRAM_LEP = {"low": 1.0, "central": 1.0, "high": 2.0}

# ---------------------------------------------------------------------
# 3. DRAM (proxy for Micron GDDR6X): per-bit cross sections
# ---------------------------------------------------------------------

# Heavy-ion Weibull fits, 2-4 Gb DDR3 (L0, W, s, sigma_sat)
# [ESA/IDA TN-IDA-RAD-14/14 Table 1; TN-IDA-RAD-13/1A Table 5].
DRAM_HI_WEIBULL = {
    "Samsung 4Gb DDR3 (35 nm)": (1.08901, 42.82953, 2.71533, 2.38e-10),
    "Elpida 4Gb DDR3": (1.08901, 39.29846, 2.80791, 4.57e-11),
    "Samsung 2Gb DDR3 rev D": (1.08901, 59.09633, 3.96459, 3.88e-10),
    "Micron 2Gb DDR3": (1.08901, 39.73573, 4.02061, 8.06e-11),
    "Hynix 4Gb DDR3 (44 nm)": (0.47768, 47.4727, 2.38903, 1.71e-11),
    "Hynix 2Gb DDR3": (1.08901, 65.69309, 3.77954, 6.43e-11),
    "Samsung 4Gb DDR3 (TAMU fit)": (0.47768, 57.56, 3.63, 3.38e-11),
}
# DDR4 [Dufour RADECS 2022]: only L_th and sigma_sat are published. As a
# pessimistic bound (raises the DRAM rate, lowers rho) we assume a narrow
# Weibull W = 10, s = 2. ASSUMED shape, published L_th and sigma_sat.
DDR4_DUFOUR = {
    "Micron 4Gb DDR4 static": (0.5, 2.6e-11),
    "Samsung 4Gb DDR4 static": (2.5, 3.1e-11),
    "Samsung 4Gb DDR4 dynamic": (0.2, 8.9e-13),
    "Hynix 8Gb DDR4 static": (0.5, 8.5e-11),
    "Nanya 4Gb DDR4": (1.45, 4.4e-12),
    "Micron 8Gb DDR4 static": (0.5, 1.14e-10),
}
DDR4_ASSUMED_W, DDR4_ASSUMED_S = 10.0, 2.0

# High-energy protons / neutrons: Micron GDDR5 8 Gb 1X nm 2.99 FIT/Gb
# (neutron) [Kamibayashi 2023] -> central; DDR3 4 Gb protons 21-230 MeV
# 2e-20..5e-19 [ESA/IDA TN-IDA-RAD-14/8] -> low/high.
NYC_FLUX_PER_H = 12.946          # > 10 MeV neutrons, JESD89A sea level NYC


def fit_to_sigma(fit_per_gbit: float) -> float:
    return fit_per_gbit / (2 ** 30) / (NYC_FLUX_PER_H * 1e9)


DRAM_P = {"low": 2e-20, "central": fit_to_sigma(2.99), "high": 5e-19}

# ---------------------------------------------------------------------
# 4. Same-orbit flight check: CARMEN-2 on JASON-2 (1336 km, 66 deg),
#    pattern 5A5A, 2008-07-16 .. 2011-02-25 [Samaras 2011 CNES slides].
#    (name, kind, bits per DUT, upsets per DUT). Rates assume the whole
#    954-day window was powered (lower bounds if not).
# ---------------------------------------------------------------------
CARMEN_DAYS = 954
CARMEN = [
    ("Hitachi HM628512C", "SRAM", 512 * 1024 * 8, [2, 2]),
    ("Samsung KM684000", "SRAM", 512 * 1024 * 8, [13, 28]),
    ("3D+ MMSR32001608", "SRAM", 1024 * 1024 * 32, [349]),
    ("Samsung K7A801800M", "SSRAM", 512 * 1024 * 16, [334]),
    ("IDT 71V3558S", "SSRAM", 512 * 1024 * 16, [1970]),
    ("Micron MT48LC64M8A2", "SDRAM", 64 * 2 ** 20 * 8, [3, 3]),
    ("3D+ MMSD08512408S (Elpida)", "SDRAM", 512 * 2 ** 20 * 8, [8, 12]),
    ("Samsung K4S560432C", "SDRAM", 64 * 2 ** 20 * 4, [23, 30]),
]

# ---------------------------------------------------------------------
# 5. Campaign quantities
# ---------------------------------------------------------------------
LEVELS = [1e-7, 5e-7, 1e-6, 3e-6, 5e-6, 7e-6, 1e-5]
# REMU's rate (applied to DRAM in REMU) = CARMEN-2 most sensitive SSRAM
REMU_RATE = 1e-7


# ---------------------------------------------------------------------
def weibull(L: float, l0: float, w: float, s: float, sat: float) -> float:
    if L <= l0:
        return 0.0
    return sat * (1.0 - math.exp(-((L - l0) / w) ** s))


def power_fit(points):
    xs = [math.log(L) for L, _ in points]
    ys = [math.log(sig) for _, sig in points]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / \
        sum((x - mx) ** 2 for x in xs)
    a = math.exp(my - b * mx)
    return a, b


def gcr_rate(sigma, steps: int = 4000) -> float:
    """Upsets/bit/day = integral sigma(L) f(L) dL (normal-incidence,
    effective-LET corrections ignored), GEO/interplanetary solar min."""
    total = 0.0
    lo, hi = math.log(L_MIN), math.log(L_MAX)
    h = (hi - lo) / steps
    for i in range(steps):
        L0, L1 = math.exp(lo + i * h), math.exp(lo + (i + 1) * h)
        Lm = math.sqrt(L0 * L1)
        f = GCR_P * Lm ** -GCR_N * (1.0 if Lm <= L_KNEE else ABOVE_KNEE)
        total += sigma(Lm) * f * (L1 - L0)
    return total


def main() -> int:
    print("== Environment ==")
    print(f"trapped protons behind 100 mil: 500 km {PHI_P_500:.3g}, 600 km "
          f"{PHI_P_600:.3g}, 750 km {PHI_P_750:.3g} /cm2/day "
          f"(cases {PHI_P['low']:.3g} / {PHI_P['central']:.3g} / "
          f"{PHI_P['high']:.3g})")
    gcr_int = sum(GCR_P / (GCR_N - 1) * L ** (1 - GCR_N) for L in [1.0]) \
        - GCR_P / (GCR_N - 1) * L_KNEE ** (1 - GCR_N)
    print(f"GCR integral flux LET 1..28 (interplanetary): {gcr_int:.3g} "
          "/cm2/day")

    print("\n== SRAM heavy ion ==")
    a, b = power_fit(SRAM_HI_POINTS)
    al, bl = power_fit(SRAM_HI_POINTS_LOW)
    print(f"16 nm SRAM fit sigma = {a:.3g} L^{b:.3f}; at L=2,7,20: "
          + ", ".join(f"{a * L ** b:.2g}" for L in (2, 7, 20)))
    print(f"7 nm CRAM fit  sigma = {al:.3g} L^{bl:.3f}")
    hi_sram = {
        "low": gcr_rate(lambda L: al * L ** bl),
        "central": gcr_rate(lambda L: a * L ** b),
        "high": gcr_rate(lambda L: weibull(L, *SRAM_HI_WEIBULL_HIGH)),
    }
    for k, v in hi_sram.items():
        print(f"  GCR rate (interplanetary) {k:8}: {v:.3g} /bit/day")

    print("\n== DRAM heavy ion ==")
    ddr3 = {}
    for name, p in DRAM_HI_WEIBULL.items():
        ddr3[name] = gcr_rate(lambda L, p=p: weibull(L, *p))
        l0, w, s, sat = p
        l25 = l0 + w * (-math.log(0.75)) ** (1 / s)
        print(f"  {name:30} L0.25 {l25:5.1f}  rate {ddr3[name]:.3g}")
    ddr4 = {}
    for name, (l0, sat) in DDR4_DUFOUR.items():
        ddr4[name] = gcr_rate(lambda L, l0=l0, sat=sat: weibull(
            L, l0, DDR4_ASSUMED_W, DDR4_ASSUMED_S, sat))
        print(f"  {name:30} (W=10,s=2 assumed) rate {ddr4[name]:.3g}")
    v3 = sorted(ddr3.values())
    hi_dram = {"low": v3[0], "central": math.exp(
        sum(math.log(v) for v in v3) / len(v3)), "high": max(ddr4.values())}
    for k, v in hi_dram.items():
        print(f"  GCR rate (interplanetary) {k:8}: {v:.3g} /bit/day")
    print(f"heavy-ion-only rho (interplanetary, same spectrum): central "
          f"{hi_sram['central'] / hi_dram['central']:.3g}, range "
          f"{hi_sram['low'] / hi_dram['high']:.3g} .. "
          f"{hi_sram['high'] / hi_dram['low']:.3g}")
    print(f"for contrast, sigma_sat ratio (the old 'rho ~ 1' metric): "
          f"1e-11 / (8.9e-13 .. 1.14e-10) = {1e-11 / 1.14e-10:.2g} .. "
          f"{1e-11 / 8.9e-13:.2g}")

    print("\n== Protons ==")
    print(f"DRAM central sigma_p from 2.99 FIT/Gb = {DRAM_P['central']:.3g}")
    print(f"proton-only rho: central {SRAM_P['central'] / DRAM_P['central']:.3g}"
          f", range {SRAM_P['low'] / DRAM_P['high']:.3g} .. "
          f"{SRAM_P['high'] / DRAM_P['low']:.3g}")

    print("\n== LEO reference mission rates ==")
    lam = {}
    for case in ("low", "central", "high"):
        s_p = SRAM_P[case] * PHI_P[case] * SRAM_LEP[case]
        s_hi = hi_sram[case] * K_GEO[case]
        # DRAM: the 'high' DRAM rate pairs with the 'low' rho case
        opp = {"low": "high", "central": "central", "high": "low"}[case]
        d_p = DRAM_P[opp] * PHI_P[case]
        d_hi = hi_dram[opp] * K_GEO[case]
        lam[case] = (s_p + s_hi, d_p + d_hi)
        print(f"  {case:8} SRAM {s_p + s_hi:.3g} (protons {s_p:.3g}, "
              f"{100 * s_p / (s_p + s_hi):.0f} %)  DRAM {d_p + d_hi:.3g} "
              f"(protons {d_p:.3g}, {100 * d_p / (d_p + d_hi):.0f} %)  "
              f"rho {(s_p + s_hi) / (d_p + d_hi):.3g}")
    # rho range: low = SRAM low / DRAM high, high = SRAM high / DRAM low
    rho_c = lam["central"][0] / lam["central"][1]
    print(f"rho central {rho_c:.3g}, range {lam['low'][0] / lam['low'][1]:.3g}"
          f" .. {lam['high'][0] / lam['high'][1]:.3g}")

    print("\n== CARMEN-2 same-orbit flight ratio ==")
    by_kind = {"SRAM": [], "SDRAM": []}
    for name, kind, bits, counts in CARMEN:
        rates = [c / (bits * CARMEN_DAYS) for c in counts]
        by_kind["SDRAM" if kind == "SDRAM" else "SRAM"] += rates
        print(f"  {name:28} {kind:5} {bits / 2 ** 20:6.0f} Mb  "
              + ", ".join(f"{r:.2g}" for r in rates))
    gm = {k: math.exp(sum(math.log(r) for r in v) / len(v))
          for k, v in by_kind.items()}
    print(f"  geometric means: SRAM {gm['SRAM']:.3g}, SDRAM {gm['SDRAM']:.3g}"
          f", ratio {gm['SRAM'] / gm['SDRAM']:.3g}; extremes "
          f"{min(by_kind['SRAM']) / max(by_kind['SDRAM']):.2g} .. "
          f"{max(by_kind['SRAM']) / min(by_kind['SDRAM']):.2g}")

    print("\n== What the BER levels mean in the reference mission ==")
    lam_s, lam_d = lam["central"]
    print(f"  central lambda_SRAM {lam_s:.3g}, lambda_DRAM {lam_d:.3g} "
          "/bit/day")
    print("  BER      T for DRAM (years)  T for SRAM (days)  "
          "SRAM BER at the same T as DRAM (rho x BER)  DRAM BER at the "
          "same T as SRAM (BER / rho)")
    for ber in LEVELS:
        print(f"  {ber:7.0e}  {ber / lam_d / 365.25:16.3g}  "
              f"{ber / lam_s:16.3g}  {rho_c * ber:12.3g}  "
              f"{ber / rho_c:12.3g}")
    print(f"  REMU's {REMU_RATE:g} /bit/day / central lambda_DRAM = "
          f"{REMU_RATE / lam_d:.3g}; / central lambda_SRAM = "
          f"{REMU_RATE / lam_s:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
