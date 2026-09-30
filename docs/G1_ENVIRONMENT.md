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

## Later toolchain additions (G2–G7, recorded 2026-09-29)

The hardware, driver 580.95.05 and kernel above are unchanged. The G2
observer contract pins that driver/kernel pair, so any upgrade
invalidates it. Components added after G1:

| Component | Location / version | Used by |
| --- | --- | --- |
| System Python for the eBPF orchestrators | `/usr/bin/python3` (3.10.13, BCC) | G2–G5/G7 observer, campaign orchestrator (run via sudo wrapper) |
| TensorRT | 8.6.1, `/data1/luojx/REMU/.local/deps/TensorRT-8.6.1/include` + `.../tensorrt-8.6.1/tensorrt_libs` | G1.5 runner (`build-g1.5/`), G7 engine build/eval |
| OpenCV | `/data1/luojx/REMU/.local/deps/conda` (opencv4) | G1.5 runner preprocessing |
| cuDNN | 8.9.7 under `/data1/luojx/REMU/.local/deps/` | G7 engine build/eval |
| Python ML env | `/data1/luojx/miniforge3/envs/vit_fault` (torch 2.5.1+cu124, timm 1.0.26, onnx 1.17.0, cv2 4.10.0, `tensorrt_bindings` 8.6.1, matplotlib) | G7 prep, campaign figures |
| ModelOpt (isolated) | `/data1/luojx/REMU/.local/deps/modelopt-venv` (ModelOpt 0.47.0) | G7 explicit Q/DQ quantization only |

The base `python3` (3.13.13) still runs the pure-stdlib offline tools and
self-tests. Wiring details are in `tools/g7_prep/README.md` (Environment).
