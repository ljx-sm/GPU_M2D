# G1.5 ResNet-50 INT8 Integration

Status: **PASS** on 2026-09-14.

## Scope

G1.5 connects the G1 semantic mapper and CUDA XOR injector to the existing
TensorRT 8.6.1 ResNet-50 INT8 PTQ engine and a fixed labeled RESISC45 sample.
It does not infer GPU physical addresses or GDDR6X coordinates.

The three public TensorRT bindings are registered as exact semantic regions:

| Binding | Role | Runtime type |
| --- | --- | --- |
| `data` | input | FP32 |
| `prob` | output probability | FP32 |
| `index` | output class | INT32 |

The engine uses INT8 internally even though its public input/output contract is
FP32/INT32.

TensorRT-owned device allocations are observed through `IGpuAllocator`. Their
address, size, allocation phase, and active state are recorded, but their
semantic label remains `TENSORRT_INTERNAL_UNKNOWN`. No unsupported layer or
weight attribution is made.

All observed device allocations now share one run-scoped `AllocationRegistry`.
It provides the allocation-level invariant needed by the later physical-address
stage:

```text
(device, allocation ID, byte offset, bit) <-> (device, active GPU VA, bit)
```

The registry requires nonzero bounded ranges and unique lifetime-stable IDs,
rejects overlapping active ranges on the same GPU, qualifies reverse lookup by
CUDA device, and excludes inactive allocations. A numerical VA may be reused
only after the previous allocation is deactivated and receives a new allocation
ID. Therefore semantic knowledge is optional for physical fault injection, but
allocation ownership, bounds, device, and lifetime are not.

## Injection protocol

For each GPU:

1. Deserialize the clean engine and create its execution context.
2. Register the four observed TensorRT allocations and all three binding buffers
   in one lifetime-aware allocation registry.
3. Validate first/last-byte allocation-to-VA and VA-to-allocation round trips
   for every active allocation.
4. Run clean inference on evaluation sample 0.
5. Restore the complete input buffer.
6. Map `data` element 0, bit 0 to its process-visible GPU VA.
7. Reverse-check that VA against its active allocation ID and byte offset, then
   XOR that bit with the CUDA injector.
8. Copy back and compare the entire input allocation against the one-bit
   expected buffer.
9. Reverse-map the injected GPU VA/bit to the original Tensor element/bit.
10. Run injected inference and classify the top-1 result.

Runtime CSV artifacts are written under `artifacts/g1_5/` and intentionally
excluded from Git because GPU VAs are allocation-specific.

## Run

```bash
scripts/run_g1_5_validation.sh
```

## Results

Before integration, the unmodified prior stage-8 runner was executed on the
same fixed sample on devices 0, 1, and 2. Every device produced class 7 with
probability `0.73922801`, matching target label 7.

The integrated G1.5 runner then passed on all three devices for input `data`
element 0, bit 0:

```text
before=159 after=158 clean_class=7 injected_class=7
outcome=BENIGN_TOP1 numeric_output_changed=0
```

The complete 602,112-byte input allocation was compared with the expected
one-bit-different buffer. The forward and reverse semantic mappings agreed.
Five repeated runs on each GPU (15 total) produced the same semantic result.

A boundary case targeting the final FP32 input element and its final bit also
passed on all devices:

```text
element=150527 bit=31 byte_offset=602111
before=191 after=63 clean_class=7 injected_class=7
outcome=BENIGN_TOP1 numeric_output_changed=1
```

This demonstrates that top-1 behavior can remain benign while the numerical
output changes. Requests for bit 32 or element 150528 were rejected before
injection as out of range.

TensorRT made four observed internal allocations on every device: one during
engine deserialization and three during execution-context creation. Their
sizes and phases were identical across the three GPUs:

```text
23,995,396 + 22,528 + 2,048 + 1,806,336 = 25,826,308 bytes
```

All four were active at injection time and remain classified as
`TENSORRT_INTERNAL_UNKNOWN`. Together with the three exact public bindings, the
runtime registry contained seven active, non-overlapping allocations on each
GPU. This inventory is not treated as a serialized engine offset map or a
Tensor element map.

The allocation registry unit test covers first/last-bit round trips, exact and
unknown semantic labels, invalid offsets/bits, unregistered addresses,
same-device overlap rejection, cross-device VA disambiguation, stale-address
rejection after deallocation, and safe VA reuse under a new allocation ID.

CUDA Compute Sanitizer `memcheck` reported zero errors for the complete G1.5
runner on each of the three devices. The original G1 unit/integration tests
also remained at 100% pass after adding INT32 binding support.

## Classification used in G1.5

- `BENIGN_TOP1`: injected inference completes and retains the clean top-1
  class.
- `SDC_TOP1`: injected inference completes with a different valid top-1 class.
- `numeric_output_changed`: probability differs by more than `1e-6`, even if
  top-1 remains unchanged.

Full campaign-level DUE handling and retry policy remain part of the later
reliability-evaluation stage. A runner/setup failure is currently reported as
`GPU_M2D_G1_5_FAIL`, not silently classified as a memory-induced DUE.
