#!/usr/bin/env python3
"""GPU_M2D G3 S4 solver: fit the PA -> GDDR bank/row model to pair constraints.

Reads the S3 bit-scan ``constraints.csv`` and the S3b pair-scan
``pair_constraints.csv`` (run directories or CSV paths; both are just
(pa_a, pa_b, class) triples to the solver) and fits the two functions the
timing primitive measured:

    conflict(x, y)  <=>  same_bank(x, y) AND row(x) != row(y)

with the model class chosen in the S3/S3b analysis:

    bank(x)  = one GF(2) polynomial of degree <= 2 per bank output bit --
               the linearized form of a row-seeded hash: a quadratic term
               pa_i*pa_k is exactly "seed bit i x flipped bit k";
    row(x)   = the PA bits in the row support R (the row differs iff the
               pair xor touches R).

Conflicts give homogeneous equations: every bank functional must vanish on
their feature difference phi(x, y) = mu(x) ^ mu(y). Lows whose xor touches
R must be separated by at least one functional -- the separating family is
grown by greedy sparse search over valid functionals (valid = vanishing on
all conflicts; the set is XOR-closed, so pairs of columns with equal
conflict signatures, beam growth, and affine solves with pair-matching
sparsification all stay valid). R is re-solved as monotone clauses (hit the
xor of every conflict) plus units (avoid the xor of every low predicted
same-bank) with violation counting. The two stages alternate; mid pairs are
never fitted, only scored.

Residuals (lows no valid functional can separate, conflicts no R can hit)
are the honest measure of what the model class cannot express -- run with
``--degree 1`` to quantify the linear baseline that S3/S3b ruled out.

    python3 solve_mapping.py RUN_DIR_OR_CSV... [--degree {1,2}]
        [--holdout FRAC] [--seed N] [--rounds N] [--model-out PATH]
    python3 solve_mapping.py --predict MODEL.json 0xPA_A 0xPA_B

Writes mapping_model.json (bank functional term lists, row support, stats)
for the S5 prediction gate. ``--self-test`` pins the whole pipeline on a
synthetic seeded truth: the degree-2 fit must reach zero residuals and
violations with >=99% holdout accuracy, and the degree-1 fit must report
the insufficiency. Pure stdlib; never imports bcc.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

MODEL_SCHEMA = "gpu-m2d.g3-mapping-model.v1"
HOLDOUT_GATE = 0.95
USABLE_CLASSES = ("low", "mid", "conflict")


def dot(u: int, v: int) -> int:
    """GF(2) inner product of two bitmask vectors."""
    return (u & v).bit_count() & 1


def iter_bits(mask: int):
    """Yields the set bit indices of a bitmask, ascending."""
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def mask_of(bits) -> int:
    out = 0
    for bit in bits:
        out |= 1 << bit
    return out


class Features:
    """Degree-<=2 GF(2) monomial map over PA bits 0..n_bits-1."""

    def __init__(self, n_bits: int, degree: int):
        if degree not in (1, 2):
            raise ValueError("degree must be 1 or 2")
        self.n_bits = n_bits
        self.degree = degree
        self.terms = [f"pa{i}" for i in range(n_bits)]
        self.quad_index: dict[tuple[int, int], int] = {}
        if degree >= 2:
            for i in range(n_bits):
                for k in range(i + 1, n_bits):
                    self.quad_index[(i, k)] = len(self.terms)
                    self.terms.append(f"pa{i}*pa{k}")
        self.size = len(self.terms)

    def mu(self, pa: int) -> int:
        """Monomial evaluation bitmask of one address."""
        set_bits = [i for i in range(self.n_bits) if (pa >> i) & 1]
        out = 0
        for i in set_bits:
            out |= 1 << i
        if self.degree >= 2:
            for pos, i in enumerate(set_bits):
                for k in set_bits[pos + 1:]:
                    out |= 1 << self.quad_index[(i, k)]
        return out

    def phi(self, pa_a: int, pa_b: int) -> int:
        """Feature difference of a pair (mu(a) ^ mu(b))."""
        return self.mu(pa_a) ^ self.mu(pa_b)


class Pair:
    """One measured pair: both PAs, the timing class, and its provenance."""

    __slots__ = ("a", "b", "d", "cls", "source", "query_id", "phi")

    def __init__(self, a: int, b: int, cls: str, source: str, query_id: str):
        self.a = a
        self.b = b
        self.d = a ^ b
        self.cls = cls
        self.source = source
        self.query_id = query_id
        self.phi = 0

    def attach(self, feats: Features) -> None:
        self.phi = feats.phi(self.a, self.b)


def resolve_csvs(paths: list[str]) -> list[Path]:
    """Expands run directories into their constraint CSVs."""
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found = [path / name for name in
                     ("constraints.csv", "pair_constraints.csv")
                     if (path / name).is_file()]
            if not found:
                raise RuntimeError(f"no constraints CSV in run dir: {path}")
            out.extend(found)
        else:
            out.append(path)
    if not out:
        raise RuntimeError("no constraint CSVs given")
    return out


def read_pairs(csv_path: Path) -> list[Pair]:
    """Reads one analyzer CSV; both schemas carry pa_a/pa_b/class."""
    with csv_path.open(encoding="utf-8", newline="") as source:
        lines = [line for line in source if not line.startswith("#")]
    reader = csv.DictReader(lines)
    fields = reader.fieldnames or []
    if "section" in fields:
        kind = "pair-scan"
    elif "kind" in fields:
        kind = "bit-scan"
    else:
        raise RuntimeError(f"{csv_path}: unrecognized constraint header")
    pairs = []
    for row in reader:
        cls = row.get("class", "")
        if cls not in USABLE_CLASSES:
            continue  # asymmetric rows are not usable constraints
        pairs.append(Pair(int(row["pa_a"], 16), int(row["pa_b"], 16), cls,
                          str(csv_path), row.get("query_id", "")))
    if not pairs:
        raise RuntimeError(f"{csv_path}: no usable constraint rows ({kind})")
    return pairs


def split_holdout(pairs: list[Pair], frac: float, seed: int) \
        -> tuple[list[Pair], list[Pair]]:
    """Deterministic stratified split by class."""
    rng = random.Random(seed)
    by_cls: dict[str, list[Pair]] = {}
    for pair in pairs:
        by_cls.setdefault(pair.cls, []).append(pair)
    train: list[Pair] = []
    holdout: list[Pair] = []
    for cls in sorted(by_cls):
        items = sorted(by_cls[cls], key=lambda p: (p.source, p.query_id))
        rng.shuffle(items)
        cut = int(len(items) * frac)
        holdout.extend(items[:cut])
        train.extend(items[cut:])
    return train, holdout


def solve_affine(eq_rows: list[int], rhs: list[int]) -> int | None:
    """One solution theta of dot(eq, theta) = rhs over GF(2), or None."""
    basis: dict[int, int] = {}  # pivot -> augmented row (equation << 1 | rhs)
    for eq, r in zip(eq_rows, rhs):
        v = (eq << 1) | (r & 1)
        while v >> 1:
            eq_part = v >> 1
            pivot = (eq_part & -eq_part).bit_length() - 1
            if pivot in basis:
                v ^= basis[pivot]
            else:
                basis[pivot] = v
                break
        if not (v >> 1) and (v & 1):
            return None  # 0 = 1: inconsistent
    theta = 0
    for pivot in sorted(basis, reverse=True):
        row = basis[pivot]
        eq = row >> 1
        value = (row & 1) ^ dot(eq & ~(1 << pivot), theta)
        if value:
            theta |= 1 << pivot
    return theta


def solve_bank_bits(positives: list[Pair], negatives: list[Pair]) \
        -> tuple[list[int], list[Pair]]:
    """Greedy sparse family of bank functionals.

    A functional is valid iff it vanishes on every conflict (positive); the
    valid set is XOR-closed. The family must make every negative (a low
    whose xor touches the row support) be fired by at least one member.
    Negatives that no valid functional can fire are returned as residuals.

    Candidate construction is exact over weights 1-3: a functional is valid
    iff the conflict signatures (colmasks) of its terms XOR to zero, so
    valid pairs are equal-signature columns and valid triples are found by
    dict lookup (signature of the third = XOR of the other two) -- the
    linearized seed terms genuinely live at weight 3. Beam growth (XOR with
    the current pick) extends to higher weights; the dense affine solve is
    a last-resort fallback and is reported.
    """
    if not negatives:
        return [], []
    pos_phis = [pair.phi for pair in positives]
    full_remaining = (1 << len(negatives)) - 1
    columns = sorted({j for pair in negatives for j in iter_bits(pair.phi)})

    colmask_cache: dict[int, int] = {}

    def colmask(j: int) -> int:
        """Which positives column j fires on, as a bitmask."""
        cached = colmask_cache.get(j)
        if cached is None:
            cached = 0
            bit = 1 << j
            for idx, phi in enumerate(pos_phis):
                if phi & bit:
                    cached |= 1 << idx
            colmask_cache[j] = cached
        return cached

    def cover_of(theta: int) -> int:
        """Which negatives theta fires on, as a bitmask."""
        out = 0
        for idx, pair in enumerate(negatives):
            if dot(theta, pair.phi):
                out |= 1 << idx
        return out

    single_covers = {j: cover_of(1 << j) for j in columns}
    signature_of = {j: colmask(j) for j in columns}
    by_signature: dict[int, list[int]] = {}
    for j in columns:
        by_signature.setdefault(signature_of[j], []).append(j)

    pool: list[tuple[int, int, int]] = []  # (weight, cover, theta)
    seen_thetas: set[int] = set()

    def add_candidate(weight: int, cover: int, theta: int) -> None:
        if cover and theta not in seen_thetas:
            pool.append((weight, cover, theta))
            seen_thetas.add(theta)

    # Weight 1 and 2: zero-signature columns; equal-signature pairs.
    for j in columns:
        if signature_of[j] == 0:
            add_candidate(1, single_covers[j], 1 << j)
    pair_budget = 64
    for signature in sorted(by_signature):
        if signature == 0:
            continue
        js = sorted(by_signature[signature],
                    key=lambda j: -single_covers[j].bit_count())
        pairs_here = [(js[0], j) for j in js[1:1 + pair_budget]]
        for a, b in pairs_here:
            add_candidate(2, single_covers[a] ^ single_covers[b],
                          (1 << a) | (1 << b))

    # Weight 3: for each column pair, any column whose signature completes
    # the XOR to zero. The triple terms are exactly where seeded bank bits
    # (linear core ^ seed1*flip ^ seed2*flip) show up.
    triples: list[tuple[int, int, int]] = []
    n_columns = len(columns)
    for i1 in range(n_columns):
        j1 = columns[i1]
        c1, s1 = single_covers[j1], signature_of[j1]
        for i2 in range(i1 + 1, n_columns):
            j2 = columns[i2]
            wanted = s1 ^ signature_of[j2]
            if not wanted:
                continue  # equal signatures: weight-2 territory
            for j3 in by_signature.get(wanted, ()):
                if j3 <= j2:
                    continue  # keep triples sorted: dedupe
                cover = c1 ^ single_covers[j2] ^ single_covers[j3]
                if cover:
                    triples.append((3, cover,
                                    (1 << j1) | (1 << j2) | (1 << j3)))
    triples.sort(key=lambda item: (-item[1].bit_count(), item[2]))
    for weight, cover, theta in triples[:512]:
        add_candidate(weight, cover, theta)

    def pool_order(item):
        weight, cover, theta = item
        return (weight, theta)

    pool.sort(key=lambda item: (-item[1].bit_count(),) + pool_order(item))
    pool = pool[:256]

    def pool_sort(item):
        return (-(item[1] & remaining).bit_count(),) + pool_order(item)

    chosen: list[int] = []
    dense_fallbacks = 0
    remaining = full_remaining
    residual_pairs: list[Pair] = []
    guard = len(negatives) + 64
    while remaining and guard:
        guard -= 1
        best = None
        for weight, cover, theta in pool:
            hit = (cover & remaining).bit_count()
            if not hit:
                continue
            key = (-hit, weight, theta)
            if best is None or key < best[0]:
                best = (key, weight, cover, theta)
        if best is not None:
            _, weight, cover, theta = best
            chosen.append(theta)
            remaining &= ~cover
            # Beam growth: XOR of the pick with pool members stays valid.
            fresh = []
            for w, c, t in pool[:32]:
                new_theta = theta ^ t
                if new_theta == 0 or new_theta in seen_thetas:
                    continue
                new_cover = cover ^ c
                if new_cover & remaining:
                    fresh.append((new_theta.bit_count(), new_cover, new_theta))
                    seen_thetas.add(new_theta)
            pool.extend(fresh)
            pool.sort(key=pool_sort)
            pool = pool[:256]
            continue
        # Stall: solve for one uncovered negative directly (affine system:
        # vanish on every positive, fire on the target). Dense by nature.
        target_idx = next(iter_bits(remaining))
        theta_star = solve_affine(pos_phis + [negatives[target_idx].phi],
                                  [0] * len(pos_phis) + [1])
        fired = cover_of(theta_star) if theta_star is not None else 0
        if theta_star is None or not (fired & remaining):
            residual_pairs.append(negatives[target_idx])
            remaining &= ~(1 << target_idx)
            continue
        dense_fallbacks += 1
        add_candidate(theta_star.bit_count(), fired, theta_star)
        pool.sort(key=pool_sort)

    # Redundancy removal: drop members whose coverage is contained in the rest.
    if chosen:
        covers = [cover_of(theta) for theta in chosen]
        drop = set()
        for idx, theta in enumerate(chosen):
            others = 0
            for j, cover in enumerate(covers):
                if j != idx and j not in drop:
                    others |= cover
            if covers[idx] & ~others == 0:
                drop.add(idx)
        chosen = [theta for idx, theta in enumerate(chosen) if idx not in drop]
    # Safety: a chosen functional must vanish on every training conflict.
    chosen = [theta for theta in chosen
              if all(dot(theta, phi) == 0 for phi in pos_phis)]
    chosen.sort(key=lambda theta: (theta.bit_count(), theta))
    if dense_fallbacks:
        print(f"note: {dense_fallbacks} dense affine fallback functional(s)")
    return chosen, residual_pairs


def solve_row_support(pairs: list[Pair], sb_flags: list[bool] | None) \
        -> tuple[set[int], int]:
    """Row support R: hit every conflict xor, avoid every same-bank low xor.

    Greedy clause cover followed by local search on the exact violation
    count (unhittable conflicts + broken same-bank lows). ``sb_flags`` are
    the current same-bank predictions aligned with ``pairs``; None means
    no same-bank knowledge yet (initial cover, units ignored).
    """
    if sb_flags is None:
        sb_flags = [False] * len(pairs)
    conflicts = [pair for pair in pairs if pair.cls == "conflict"]
    universe = sorted({b for pair in pairs for b in iter_bits(pair.d)})
    freq: Counter = Counter(b for pair in conflicts for b in iter_bits(pair.d))
    clauses = [tuple(sorted(iter_bits(pair.d))) for pair in conflicts]

    support = {clause[0] for clause in clauses if len(clause) == 1}
    for clause in sorted(clauses, key=lambda c: (len(c), c)):
        if not set(clause) & support:
            avail = [b for b in clause if b not in support]
            if avail:
                support.add(max(avail, key=lambda b: (freq[b], -b)))

    def violations(candidate: set[int]) -> int:
        rmask = mask_of(candidate)
        count = 0
        for pair, same_bank in zip(pairs, sb_flags):
            row_diff = bool(pair.d & rmask)
            if pair.cls == "conflict":
                if not row_diff:
                    count += 1
            elif same_bank and row_diff:
                count += 1
        return count

    best = violations(support)
    improved = True
    while improved:
        improved = False
        for bit in universe:
            candidate = support ^ {bit}
            count = violations(candidate)
            if count < best:
                support, best, improved = candidate, count, True
    return support, best


def run_fit(train_hard: list[Pair], feats: Features, rounds: int) -> dict:
    """Alternates bank-functional and row-support solves; keeps the best."""
    positives = [pair for pair in train_hard if pair.cls == "conflict"]
    lows = [pair for pair in train_hard if pair.cls == "low"]

    def negatives_for(rmask: int) -> list[Pair]:
        return [pair for pair in lows if pair.d & rmask]

    support, _ = solve_row_support(train_hard, None)
    seen: set = set()
    best: tuple | None = None
    for _ in range(rounds):
        theta, residuals = solve_bank_bits(positives,
                                           negatives_for(mask_of(support)))
        sb_flags = [all(dot(t, pair.phi) == 0 for t in theta)
                    for pair in train_hard]
        support2, violations = solve_row_support(train_hard, sb_flags)
        key = (len(residuals) + violations, len(theta),
               sum(t.bit_count() for t in theta))
        if best is None or key < best[0]:
            best = (key, theta, support2, residuals, violations)
        state = (tuple(theta), frozenset(support2))
        if state in seen:
            break
        seen.add(state)
        support = support2
    _, theta, support, residuals, violations = best
    return {"theta": theta, "row_support": sorted(support),
            "residuals": residuals, "violations": violations}


def classify(theta: list[int], rmask: int, pair: Pair) -> str:
    same_bank = all(dot(t, pair.phi) == 0 for t in theta)
    return "conflict" if same_bank and (pair.d & rmask) else "low"


def evaluate(theta: list[int], rmask: int, pairs: list[Pair]) -> dict:
    correct = 0
    total = 0
    confusion: Counter = Counter()
    mid_votes: Counter = Counter()
    for pair in pairs:
        predicted = classify(theta, rmask, pair)
        if pair.cls == "mid":
            mid_votes[predicted] += 1
            continue
        total += 1
        correct += predicted == pair.cls
        confusion[(pair.cls, predicted)] += 1
    return {"correct": correct, "total": total,
            "accuracy": correct / total if total else 0.0,
            "confusion": confusion, "mid_votes": mid_votes}


def term_list(feats: Features, theta: int) -> list[str]:
    return [feats.terms[j] for j in iter_bits(theta)]


def print_report(feats: Features, inputs: list[dict], pairs: list[Pair],
                 train: list[Pair], holdout: list[Pair], result: dict,
                 train_stats: dict, hold_stats: dict, model_path: Path) -> None:
    for item in inputs:
        print(f"input: {item['path']} rows={item['rows']} "
              f"sha256={item['sha256'][:16]}")
    classes = Counter(pair.cls for pair in pairs)
    print(f"pairs: {len(pairs)} (low/mid/conflict = "
          f"{classes['low']}/{classes['mid']}/{classes['conflict']})")
    quad = feats.size - feats.n_bits
    print(f"features: {feats.n_bits} linear + {quad} quadratic = "
          f"{feats.size} monomials, PA bits 0..{feats.n_bits - 1}")
    train_hard = [p for p in train if p.cls != "mid"]
    hold_hard = [p for p in holdout if p.cls != "mid"]
    print(f"split: train {len(train_hard)} hard + "
          f"{len(train) - len(train_hard)} mid / holdout {len(hold_hard)} hard "
          f"+ {len(holdout) - len(hold_hard)} mid")
    print(f"fit: bank bits {len(result['theta'])} "
          f"(weights {','.join(str(t.bit_count()) for t in result['theta']) or '-'}) "
          f"| row support {result['row_support']}")
    print(f"residual unseparable lows: {len(result['residuals'])} "
          f"{[p.query_id for p in result['residuals']][:8]}")
    print(f"row-support violations: {result['violations']}")
    for idx, theta in enumerate(result["theta"]):
        print(f"  bank{idx} = " + " ^ ".join(term_list(feats, theta)))
    print(f"train accuracy: {train_stats['correct']}/{train_stats['total']} "
          f"= {train_stats['accuracy']:.4f}")
    gate = "PASS" if hold_stats["accuracy"] >= HOLDOUT_GATE else "FAIL"
    print(f"holdout accuracy: {hold_stats['correct']}/{hold_stats['total']} "
          f"= {hold_stats['accuracy']:.4f} "
          f"[gate >= {HOLDOUT_GATE:.2f} {gate}]")
    if hold_stats["mid_votes"]:
        votes = hold_stats["mid_votes"]
        print(f"holdout mid (soft, {sum(votes.values())}): "
              f"predicted low {votes['low']} / conflict {votes['conflict']}")
    print(f"model: {model_path}")


def write_model(path: Path, feats: Features, inputs: list[dict], result: dict,
                train_stats: dict, hold_stats: dict, args: argparse.Namespace,
                n_pairs: int) -> dict:
    model = {
        "schema_version": MODEL_SCHEMA,
        "created_wall_time_ns": time.time_ns(),
        "meaning": "conflict(x,y) <=> all bank functionals vanish on "
                   "mu(x)^mu(y) AND (x^y) touches the row support; "
                   "degree<=2 GF(2) monomials over PA bits",
        "inputs": inputs,
        "pair_count": n_pairs,
        "degree": feats.degree,
        "n_bits": feats.n_bits,
        "monomial_count": feats.size,
        "terms": feats.terms,
        "bank_functionals": [
            {"mask": theta, "terms": term_list(feats, theta)}
            for theta in result["theta"]
        ],
        "row_support": result["row_support"],
        "row_mask": mask_of(result["row_support"]),
        "residual_unseparable_lows": len(result["residuals"]),
        "row_support_violations": result["violations"],
        "holdout_fraction": args.holdout,
        "holdout_seed": args.seed,
        "train_accuracy": train_stats["accuracy"],
        "holdout_accuracy": hold_stats["accuracy"],
        "holdout_gate": HOLDOUT_GATE,
        "holdout_gate_pass": hold_stats["accuracy"] >= HOLDOUT_GATE,
    }
    path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return model


def load_model(path: Path) -> tuple[Features, list[int], int]:
    model = json.loads(path.read_text(encoding="utf-8"))
    if model.get("schema_version") != MODEL_SCHEMA:
        raise RuntimeError(f"{path}: unknown model schema")
    feats = Features(int(model["n_bits"]), int(model["degree"]))
    theta = [int(item["mask"]) for item in model["bank_functionals"]]
    return feats, theta, int(model["row_mask"])


def _synthetic_truth(x: int) -> list[int]:
    """4 bank output bits: linear core + row-seeded quadratic terms."""
    def b(i: int) -> int:
        return (x >> i) & 1
    return [b(8) ^ b(12) ^ (b(16) & b(24)),
            b(12) ^ (b(17) & b(25)),
            b(8) ^ b(13) ^ (b(18) & b(27)),
            b(14) ^ (b(21) & b(28))]


_SYNTH_ROW_SUPPORT = {10, 16, 18, 19, 21, 24, 25, 27, 28}
_SYNTH_ANCHOR = 0xD0100


def _synthetic_pairs(rng: random.Random, count: int) -> list[Pair]:
    """Pairs shaped like the S3/S3b plans against the synthetic truth."""
    pairs: list[Pair] = []
    for idx in range(count):
        x = rng.getrandbits(33)
        style = rng.random()
        if style < 0.35:  # single-bit pairs (in-page and page-level bits)
            d = 1 << rng.randrange(33)
        elif style < 0.55:  # anchored-style masks
            d = _SYNTH_ANCHOR ^ (1 << rng.randrange(21))
        elif style < 0.8:  # two-bit pairs
            d = (1 << rng.randrange(21)) | (1 << rng.randrange(21))
        else:  # small random masks
            d = 0
            for _ in range(rng.randrange(1, 4)):
                d |= 1 << rng.randrange(33)
        y = x ^ d
        same_bank = _synthetic_truth(x) == _synthetic_truth(y)
        row_diff = bool(d & mask_of(_SYNTH_ROW_SUPPORT))
        cls = "conflict" if same_bank and row_diff else "low"
        if rng.random() < 0.04:
            cls = "mid"
        pairs.append(Pair(x, y, cls, "synthetic", str(idx)))
    return pairs


def _fit_and_score(pairs: list[Pair], degree: int, seed: int) -> dict:
    feats = Features(33, degree)
    for pair in pairs:
        pair.attach(feats)
    train, holdout = split_holdout(pairs, 0.2, seed)
    train_hard = [p for p in train if p.cls != "mid"]
    result = run_fit(train_hard, feats, rounds=4)
    rmask = mask_of(result["row_support"])
    train_stats = evaluate(result["theta"], rmask, train_hard)
    hold_stats = evaluate(result["theta"], rmask,
                          [p for p in holdout if p.cls != "mid"])
    return {"result": result, "train": train_stats, "holdout": hold_stats}


def self_test() -> int:
    """Synthetic seeded truth must be recovered; linear model must fail."""
    rng = random.Random(20260916)
    pairs = _synthetic_pairs(rng, 2400)
    classes = Counter(pair.cls for pair in pairs)
    assert classes["conflict"] > 100, classes  # the fixture must be informative

    scored = _fit_and_score(pairs, degree=2, seed=7)
    result = scored["result"]
    assert not result["residuals"], len(result["residuals"])
    assert result["violations"] == 0, result["violations"]
    assert scored["holdout"]["accuracy"] >= 0.99, scored["holdout"]
    assert scored["train"]["accuracy"] >= 0.99, scored["train"]
    # Fresh pairs from the same generator must agree (same-bank predicate
    # and end-to-end class), not just the held-out split.
    probe = _synthetic_pairs(random.Random(99), 3000)
    feats = Features(33, 2)
    for pair in probe:
        pair.attach(feats)
    rmask = mask_of(result["row_support"])
    same_ok = 0
    class_ok = 0
    for pair in probe:
        truth_bank = _synthetic_truth(pair.a) == _synthetic_truth(pair.b)
        pred_bank = all(dot(t, pair.phi) == 0 for t in result["theta"])
        same_ok += truth_bank == pred_bank
        class_ok += classify(result["theta"], rmask, pair) == (
            "conflict" if truth_bank and
            (pair.d & mask_of(_SYNTH_ROW_SUPPORT)) else "low")
    assert same_ok >= 0.99 * len(probe), same_ok
    assert class_ok >= 0.99 * len(probe), class_ok

    linear = _fit_and_score(pairs, degree=1, seed=7)
    insufficient = (linear["result"]["residuals"]
                    or linear["result"]["violations"]
                    or linear["holdout"]["accuracy"] < HOLDOUT_GATE)
    assert insufficient, linear  # the linear baseline must be caught out

    # CSV loading: both analyzer schemas, asymmetric rows dropped.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bit_scan = tmp_path / "constraints.csv"
        bit_scan.write_text(
            "query_id,kind,bit,pa_a,pa_b,xor,cycles_a,cycles_b,class,band_cycles\n"
            "0,in_page,0,0x1ee00000,0x1ee00001,0x1,1021,1021,low,x\n"
            "1,in_page,16,0x1ee00000,0x1ee10000,0x10000,1140,1140,conflict,x\n"
            "2,in_page,8,0x1ee00000,0x1ee00100,0x100,1050,1050,asymmetric,x\n",
            encoding="utf-8")
        pair_scan = tmp_path / "pair_constraints.csv"
        pair_scan.write_text(
            "query_id,section,bit,bit2,base_index,sample_index,role,pa_a,pa_b,"
            "xor,cycles_a,cycles_b,class,interpretation\n"
            "0,calibration,,,0,,floor,0x1ee00000,0x1ee00000,0x0,1028,1028,low,x\n"
            "1,anchor_base,,,0,,,\"0x1ee00000\",\"0x1eed0100\",0xd0100,"
            "1144,1144,conflict,anchor_valid\n",
            encoding="utf-8")
        loaded = [read_pairs(path) for path in resolve_csvs([str(tmp_path)])]
        rows = [pair for batch in loaded for pair in batch]
        assert len(rows) == 4, rows  # the asymmetric row was dropped
        assert {pair.cls for pair in rows} == {"low", "conflict"}
        assert rows[0].d == 1 and rows[1].d == 0x10000
        assert rows[2].d == 0  # pair-scan calibration floor (same address)
        assert rows[3].d == 0xD0100

    print("solve_mapping self-test: PASS")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--predict", metavar="MODEL.json",
                        help="classify one PA pair with a saved model")
    parser.add_argument("pas", nargs="*",
                        help="run dirs / CSVs to fit, or PA_A PA_B with --predict")
    parser.add_argument("--degree", type=int, choices=(1, 2), default=2)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--model-out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if args.predict:
        if len(args.pas) != 2:
            raise RuntimeError("--predict needs exactly PA_A PA_B")
        feats, theta, rmask = load_model(Path(args.predict))
        pair = Pair(int(args.pas[0], 0), int(args.pas[1], 0), "", "", "")
        pair.attach(feats)
        same_bank = all(dot(t, pair.phi) == 0 for t in theta)
        row_diff = bool(pair.d & rmask)
        print(f"same_bank={same_bank} row_differs={row_diff} "
              f"predicted={classify(theta, rmask, pair)}")
        return 0
    if not args.pas:
        raise RuntimeError("give run dirs / constraint CSVs to fit")

    csvs = resolve_csvs(args.pas)
    pairs: list[Pair] = []
    inputs = []
    for path in csvs:
        rows = read_pairs(path)
        pairs.extend(rows)
        inputs.append({"path": str(path),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "rows": len(rows)})
    n_bits = max(max(pair.a, pair.b) for pair in pairs).bit_length()
    feats = Features(n_bits, args.degree)
    for pair in pairs:
        pair.attach(feats)
    train, holdout = split_holdout(pairs, args.holdout, args.seed)
    train_hard = [p for p in train if p.cls != "mid"]
    result = run_fit(train_hard, feats, args.rounds)
    rmask = mask_of(result["row_support"])
    train_stats = evaluate(result["theta"], rmask, train_hard)
    hold_stats = evaluate(result["theta"], rmask,
                          [p for p in holdout if p.cls != "mid"])

    model_path = args.model_out or csvs[0].parent / f"mapping_model_d{args.degree}.json"
    write_model(model_path, feats, inputs, result, train_stats, hold_stats,
                args, len(pairs))
    print_report(feats, inputs, pairs, train, holdout, result,
                 train_stats, hold_stats, model_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:
        print(f"GPU_M2D_SOLVE_MAPPING_ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
