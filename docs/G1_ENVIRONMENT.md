# G1 Environment Baseline

Captured on 2026-09-14 on host `memlab-gpu`.

## Hardware

| GPU | Model | Memory | PCI bus | Compute capability |
| --- | --- | ---: | --- | --- |
| 0 | NVIDIA GeForce RTX 4090 | 24,564 MiB | `0000:16:00.0` | 8.9 |
| 1 | NVIDIA GeForce RTX 4090 | 24,564 MiB | `0000:27:00.0` | 8.9 |
| 2 | NVIDIA GeForce RTX 4090 | 24,564 MiB | `0000:38:00.0` | 8.9 |

The GPUs have no NVLink path. `nvidia-smi topo -m` reports `NODE` between each
pair and NUMA node 0 for all three devices.

## Software

| Component | Version |
| --- | --- |
| NVIDIA driver | 580.95.05 |
| CUDA Toolkit | 12.4 (`nvcc` 12.4.131) |
| CMake | 3.22.1 |
| GCC/G++ | 11.4.0 |
| OS/kernel | Ubuntu 22.04, Linux 6.8.0-136-generic x86_64 |

The base `python3` is 3.13.13 and does not currently expose PyTorch, TensorRT,
or NumPy. G1 uses C++/CUDA directly. This Python state is therefore recorded
but is not a G1 blocker. TensorRT integration must reuse or explicitly select
the prior REMU environment rather than silently installing new packages.

To capture a fresh machine-readable console snapshot, run:

```bash
scripts/collect_environment.sh
```
