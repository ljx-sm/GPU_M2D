# G2 GPU VA → Local Framebuffer PA

Status: **PASS** on 2026-09-15 (all three GPUs, three validation layers).

## Scope

G2 establishes the missing half of the dual-addressing chain on the real
workload:

```text
(device, allocation ID, byte offset, bit)   <- G1.5 AllocationRegistry
        <-> (device, active GPU VA, bit)     <- G1.5 AllocationRegistry
        <-> (GPU UUID, aperture=VIDEO, local framebuffer PA page, page size,
             raw PTE, mapping epoch)         <- G2 (this stage)
```

The physical claim is deliberately narrow: a GPU-local **framebuffer
physical page** identified by the exact PTE UVM wrote into the GMMU page
tables when the allocation was mapped. It is not a GDDR6X channel/bank/row
coordinate (that is G3), not a host-visible PCIe address, and never a bare
number — every PA is qualified by GPU UUID, aperture, page size, raw PTE
words, address-space IDs, and mapped/unmapped timestamps.

## Method

A read-only eBPF observer (port of the frozen REMU `gpu_va_pa_mapping`
component) attaches kprobe/kretprobe pairs to three functions of the
unmodified `nvidia.ko`/`nvidia-uvm.ko` and records, at definition level,
the PTE payload RM hands to UVM for every external allocation mapped by the
target process:

- `uvm_api_map_external_allocation` (VA range + GPU UUID),
- `nvUvmInterfaceGetExternalAllocPtes` (PTE payload with the PA),
- `uvm_api_free` (mapping teardown).

A pre-allocation gate guarantees probes are attached before any CUDA
allocation exists; the child then runs the complete G1.5 TensorRT workload
under observation. Everything is fail-closed: truncated payloads, gaps,
overlaps, non-local pages, nonzero statuses, ordering violations, lost
events, or UUID mismatches void the run. Nothing on the driver side is
modified and no other GPU process is observed or stopped.

The probe set, struct layouts, and AD102 GMMU v2 PTE decode are pinned by a
contract (`tools/g2_observer/pte_contract.json`,
`contract_sha256=59ede44a…`) to driver 580.95.05, kernel 6.8.0-136-generic,
and the exact module SHA-256 hashes; any driver upgrade invalidates it.

See [tools/g2_observer/README.md](../tools/g2_observer/README.md) for the
implementation and per-file roles.

## Validation layers

### 1. Scratch allocations — device and VMM APIs

Fresh gated scratch allocations (one `cudaMalloc` "device" probe and one
`cuMemCreate`/`cuMemMap` "VMM" probe per GPU) must return complete valid
local-VIDEO PTEs and survive a single-bit XOR/Hamming/restore closeout.

Result on 2026-09-15: **6/6 pass** (3 GPUs × 2 APIs), 4/4 PTEs each,
5/5 local-VIDEO samples, zero lost events. The device-API and VMM-API
paths produced consistent PA evidence, and per-GPU PA spaces were shown to
be independent.

### 2. VMM alias double-mapping

One CUDA VMM physical allocation is mapped at two reserved VA ranges. Both
VA ranges must decode page-by-page to the same PA pages, the reverse
mapping must be one-to-many (two VA pages per PA page), and the alias must
be proven semantically at runtime (pattern write through one VA read back
through the other; one-bit XOR through one observed and restored through
the other).

Result: **3/3 pass** (`G2_VMM_ALIAS_DOUBLE_MAPPING_OBSERVED`), zero
failures. Example (GPU 2): primary `0x757c9ac00000…` and secondary
`0x757c9b40000…` both resolved to the same PA pages
`0x14be00000/0x14c000000/0x14c200000/0x14c400000`.

This validates that the observer reports the actual hardware translation
rather than a per-allocation bookkeeping value: two different VAs of one
physical object yield one PA, and one PA reverse-resolves to both VAs.

### 3. Full TensorRT workload map

The complete G1.5 runner (engine deserialize → context → bindings →
clean/injected inference → registry snapshot → hold → teardown) runs under
the observer in gated mode. Every allocation registered by the
AllocationRegistry must be covered by a contiguous, valid, local-VIDEO PTE
payload with byte-exact tiling; every user-space ALLOCATED/FREE event must
correlate with kernel map/free events; the mapping must be alive across
the hold interval and gone within the teardown window.

Result on 2026-09-15: **3/3 pass**
(`G2_TENSORRT_LOCAL_PA_FULL_COVERAGE_OBSERVED`), runs
`run_trt_gpu0_1789464035224792109`, `run_trt_gpu1_1789464040494190705`,
`run_trt_gpu2_1789464045632008178`:

| GPU | UUID (prefix) | registry allocations | covered bytes | PA pages | map rows | lost events |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `073312bc` | 7/7 | 26,428,428 (= registry total) | 14 | 18 | 0 |
| 1 | `c9686b85` | 7/7 | 26,428,428 | 14 | 18 | 0 |
| 2 | `b94a373b` | 7/7 | 26,428,428 | 14 | 18 | 0 |

119 kernel events per GPU; the kernel GPU UUID matched the nvidia-smi
device UUID on every run. The per-run maps are aggregated by
`aggregate_va_pa_map.py` into the canonical
`artifacts/g2/gpu_va_pa_map.csv` (excluded from Git; VAs and PAs are
run-specific).

## Findings

1. **RM packs sibling allocations into shared physical pages.** The seven
   registered cudaMalloc allocations were mapped by only three external
   allocations: the 24 MiB engine-weight object, one 2 MiB page holding
   `trt-internal-1/2/3` **and** the `prob`/`index` bindings (five logical
   allocations on one GMMU page, one framebuffer PA page), and one 2 MiB
   page for the `data` binding. Allocation-level injection targeting must
   therefore be page-aware: a page-granular physical event can touch
   several logical allocations at once.
2. **The VA→PA relation looked affine within each observed window** (GPU 0:
   constant delta `0x77277d100000` across all pages of the run). This is a
   fresh-epoch allocation-policy artifact, not a contract: the offset
   differs per run and per GPU, and nothing in the toolchain relies on it.
3. **Per-GPU PA namespaces are independent**: zero PA-page overlap across
   the three GPUs, with each GPU's window at a different offset
   (0x1f000000 / 0x205e00000 / 0x26e400000) depending on co-tenant
   residency on the shared server. A PA without its GPU UUID is
   meaningless.
4. **All registered allocations used 2 MiB GMMU pages**; 64 KiB and 4 KiB
   pages appeared only in unregistered context/module noise allocations.

## Lookup interface and rejection cases

`tools/g2_observer/va_pa_lookup.py` answers forward
`(GPU, VA) → PA page + in-page offset` and reverse
`(GPU, PA page) → VA page(s)` queries against the canonical map, wired
into `make -C tools/g2_observer check`. Its self-test (18 cases, all
passing on the accepted map) pins the accept/rejection contract:

| Query | Outcome |
| --- | --- |
| forward: VA at a mapped page base | resolved, with allocation, buffer ID, semantic label |
| forward: mid-page VA | resolved to the same PA page + in-page offset |
| reverse: PA page with one allocation | one VA page |
| reverse: shared PA page | one-to-many (5 VA mappings on the packed page) |
| forward: VA past the mapped window | rejected |
| forward: VA on the wrong GPU | rejected |
| forward: stale VA captured from an earlier run | rejected |
| forward: inactive allocation at snapshot | rejected unless `--include-inactive` |
| any row with aperture ≠ VIDEO or invalid PTE | never yields a PA |
| corrupt row (extent ≠ page size, bad page size) | excluded from lookups and reported |
| one VA page claimed at two PAs | whole map refused |
| reverse: unaligned PA input | rejected (PA claims exist only at page granularity) |

Exit codes: 0 answered, 2 rejected, 1 map/usage error — so campaign
drivers can consume rejections programmatically.

## Validity boundary

- Observation happens at **map time**: probes must attach before the
  allocation exists (enforced by the pre-allocation gate). Arbitrary-time
  re-query of a live mapping is not provided; a GMMU page-table dump
  walker remains a possible independent cross-check (deferred).
- Maps are **run-scoped**: VAs and PAs are never reused across runs, and
  the lookup rejects addresses from earlier runs by construction.
- Only external RM allocations on the `cudaMalloc` device path and CUDA
  VMM are observed. Managed-memory migrations are out of scope for G2.
- Coverage is **page-granular**: sub-page physical layout is not claimed.
- TensorRT internal allocations keep the label `TENSORRT_INTERNAL_UNKNOWN`;
  the map gives their physical pages, not their semantic content.
- The PA is a framebuffer physical page on the named GPU. Translating it
  to GDDR6X channel/bank/row coordinates is future G3 work and is not
  claimed here.

## Reproduce

```bash
make -C tools/g2_observer all check
scripts/run_g1_5_validation.sh                        # prebuild the runner

sudo scripts/run_g2_observer_probe.sh --api device    # layer 1
sudo scripts/run_g2_observer_probe.sh --api vmm       # layer 1
sudo scripts/run_g2_observer_probe.sh --api alias     # layer 2
sudo scripts/run_g2_observer_probe.sh --api tensorrt  # layer 3 + aggregate map

tools/g2_observer/va_pa_lookup.py --self-test
tools/g2_observer/va_pa_lookup.py --gpu 0 --forward 0x...
tools/g2_observer/va_pa_lookup.py --gpu 0 --reverse 0x...
```

Orchestrator changes can be verified without a GPU re-run by replaying a
captured run offline:

```bash
tools/g2_observer/replay_trt_run.py [run_dir ...]
```
