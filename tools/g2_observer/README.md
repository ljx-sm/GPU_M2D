# G2 Observer — read-only eBPF GPU VA→PA mapping

This component implements the G2 primary route: establishing

```text
(device, allocation, GPU VA page) <-> (GPU UUID, aperture, local framebuffer PA, page size)
```

by passively observing the PTE payload the NVIDIA Resource Manager hands
to the UVM driver when a CUDA allocation is mapped into the GPU virtual
address space. It is a port of the validated REMU `gpu_va_pa_mapping`
observer (frozen 2026-09-08) and keeps the BPF program, event ABI, and
fail-closed PTE contract functionally identical.

## Why this method

A CUDA device pointer is a virtual address. On RTX 4090 there is no
fixed arithmetic VA→PA relation and no public CUDA interface exposes the
framebuffer physical address (see `docs/G2_PLATFORM_AUDIT.md`). The
translation the GMMU hardware will actually use exists in plaintext only
while RM passes it to UVM:

```text
cudaMalloc
  -> UVM ioctl: uvm_api_map_external_allocation        (VA range + GPU UUID)
  -> RM call:    nvUvmInterfaceGetExternalAllocPtes     (PTE payload with PA)
  -> UVM writes those PTEs into the GMMU page tables
  -> cudaFree:   uvm_api_free                           (mapping teardown)
```

The observer attaches read-only kprobe/kretprobe pairs to those three
functions on the already-loaded, unmodified modules, filtered to a
single target TGID, and records the complete PTE payload. The captured
PTEs are the entries UVM writes into hardware, so a decoded
`(valid, aperture=VIDEO)` PTE is the definition-level ground truth for
the local framebuffer physical page of the corresponding GPU VA page.

No driver state, PTE, or UVM object is modified. No other GPU process is
observed or stopped. Unloading the probes leaves the system untouched.

## Provenance

Ported from `/data1/luojx/REMU/gpu_va_pa_mapping/02_ebpf/` (BCC tracer,
contract, decoder, orchestrator, scratch harness; frozen evidence
manifests `PHASE3_*`, `G2P_*`, `G3P1_*`). On that host and driver the
method achieved, with independent validators:

- cudaMalloc and CUDA VMM scratch: 3/3 fresh epochs each, full PTE
  coverage, single-bit XOR/Hamming/restore closure;
- TensorRT runtime: 3/3 fresh epochs, 7/7 registered allocations with
  100% local-VIDEO PA coverage (GPU 0).

## Files

| File | Role |
| --- | --- |
| `pte_contract.json` | Version-pinned probe symbols, struct layouts, AD102 GMMU v2 PTE format, module SHA-256 |
| `pte_decoder.py` | Fail-closed PTE decoder (unknown sizes/high words never yield local PA) |
| `g2_observer.py` | BCC tracer: 3 kprobe/kretprobe pairs, full PTE capture (<=64 per query batch) |
| `g2_scratch_harness.cu` | Gated scratch allocation harness (device/VMM) with XOR closeout |
| `run_g2_scratch_probe.py` | Orchestrator: contract check, gate protocol, coverage/lifetime validation |

## Contract discipline

`load_and_validate_contract` refuses to run unless the architecture,
kernel release, `modinfo` driver version, both module SHA-256 hashes,
referenced source hashes, `/proc/kallsyms` symbol presence, and the
open-kernel-module git commit all match the frozen contract. Any driver
upgrade invalidates the contract: re-derive offsets against the new
source before observing.

## Usage

```bash
make -C tools/g2_observer all check   # build harness + self-check

# one GPU, one API (needs sudo for kprobe attach only)
sudo /usr/bin/python3 tools/g2_observer/run_g2_scratch_probe.py --api device --device 0

# all GPUs via the wrapper
sudo scripts/run_g2_observer_probe.sh --api device
sudo scripts/run_g2_observer_probe.sh --api vmm
```

Outputs land in `artifacts/g2/observer/run_*/` (`events.csv` raw event
stream, `samples.csv` requested-VA→PA rows, `summary.json` fail-closed
status, `harness.log`). Artifacts are excluded from Git because VAs and
PAs are allocation-specific.

## Validity boundary

- Observation happens at map time: probes must be attached before the
  allocation exists (the orchestrator enforces this with a pre-allocation
  gate). Arbitrary-time re-query of an existing mapping is not provided;
  a page-table dump/walker remains the independent cross-check.
- Only `cudaMalloc`-path (external RM allocation) and CUDA VMM mappings
  are observed. Managed-memory migrations are out of scope for G2.
- The PA is claimed only together with its GPU UUID, aperture, page
  size, PTE flags, and mapping epoch; it is never a bare number.
- Known-good on GPU 0 under driver 580.95.05; GPU 1/2 validation is part
  of the G2 acceptance run in this repository.
