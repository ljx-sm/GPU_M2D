# Existing ResNet-50 / RESISC45 Assets

G1 does not duplicate these large artifacts. Later TensorRT integration should
reuse them from their current locations, verify their recorded hashes, and only
copy or link an artifact into this project when a self-contained experiment
requires it.

## Selected workload

- Model: ResNet-50
- Precision: INT8 PTQ
- Dataset: RESISC45
- Primary engine:
  `/data1/luojx/REMU/artifacts/stage8/engines/paper_priority/resnet50_resisc45_int8_ptq.engine`
- Original REMU comparison engine:
  `/data1/luojx/REMU/artifacts/stage8/engines/original_repo_int8/resnet50_resisc45.engine`
- Source weights:
  `/data1/luojx/REMU/artifacts/stage8/source_assets/original_repo/resnet50_resisc45.wts`
- Complete dataset:
  `/data1/luojx/datasets/REMU_stage8/RESISC45/raw/NWPU-RESISC45`
- Author-sample evaluation split:
  `/data1/luojx/datasets/REMU_stage8/RESISC45/splits/original_repo_1000_eval.csv`
- Independent INT8 calibration split:
  `/data1/luojx/datasets/REMU_stage8/RESISC45/splits/ptq_calibration_900_balanced`

## G1 read-only verification

The existing assets were not copied or modified. The following checks passed
on 2026-09-14:

| Asset | Size/count | SHA256 |
| --- | ---: | --- |
| Primary INT8 PTQ engine | 26,663,572 bytes | `1d54083037286b7c7c08eecefdb08d69e66bfe48d17e53b4b64d5b91ae50fad9` |
| Original-repository INT8 engine | 25,451,044 bytes | `65604e33f95a95d4480bf42a39228c38a6afa91b4faa9abb01e828c9aec01b81` |
| Source weights | 251,296,401 bytes | `732ae8980d5d259e547883cf42ddb20998b680e5d7c5fd1c42ff4e2f38a9e694` |
| Complete RESISC45 | 45 class directories / 31,500 JPEG files | See prior full-file manifest |
| Author-sample evaluation CSV | 1 header + 1,000 records | See prior stage-8 manifest |
| PTQ calibration CSV | 900 records | See prior stage-8 manifest |

Prior validation records report 45 classes, 31,500 decoded images, 1,000
hash-matched author sample images, and no calibration/evaluation overlap. The
prior INT8 PTQ engine obtained 95.3% on the recovered 1,000-image protocol;
the original repository INT8 engine obtained 94.2%. These figures reproduce
the available protocol but do not reconstruct the paper's undisclosed split.

Before formal experiments, the selected files and dataset split must be
revalidated against the manifests under
`/data1/luojx/REMU/reproduce_manifest/build/`.

## G7 assets (ImageNet-1K, six models; added 2026-09-29)

G7 builds its own assets outside the repository. Nothing large is copied
in. The pipeline that produced them is
[tools/g7_prep/README.md](../tools/g7_prep/README.md).

- Dataset: `/data1/luojx/datasets/imagenet1k`. It holds all 50,000 val
  images, each cross-checked against the official `val_map.txt`.
  Splits (seed 7, disjoint, fail-closed checked) are under `splits/`:
  - `g7_calib_1000_perclass1.csv` (calibration, 1000 × 1);
  - `g7_eval_10000_perclass10.csv` (evaluation, 1000 × 10);
  - `g7_splits_manifest.json`.
- Models: `/data1/luojx/g7_models/<timm-name>/`. Each directory holds:
  - `model_meta.json`, the sole source of preprocessing facts;
  - the timm weights;
  - `model.onnx` and `model_qdq.onnx` (explicit Q/DQ);
  - `clean.engine`, the v2 head-quantized INT8 engine;
  - `engine_summary.json` and `eval/clean_summary.json`.

  The v1 FP32-head engines are archived under `v1_fp32head/`, and the raw
  downloads are kept in `/data1/luojx/g7_models/incoming/`.

The v2 engines the G7-v2 campaigns ran are listed below. The on-disk
sha256 values were re-checked on 2026-09-29 and match each workload's
bootstrap record. R is the frozen injection surface in bytes.

| Model (timm name) | Engine size | Engine SHA256 (prefix) | R (B) | Bootstrap run |
| --- | ---: | --- | ---: | --- |
| `resnet50` | 28,965,356 | `cc516d3afcda` | 28,832,268 | `run_bootstrap_gpu0_1790310201539047669` |
| `mobilenetv3_large_100` | 10,245,156 | `860f36c1ff3b` | 9,087,912 (scoped; full 9,336,232) | `run_bootstrap_gpu0_1790566515139607618` |
| `efficientnet_b0` | 11,464,244 | `fad423ace879` | 17,144,232 | `run_bootstrap_gpu0_1790587477121124681` |
| `deit_small_patch16_224` | 24,346,876 | `f226170b14ae` | 26,010,832 | `run_bootstrap_gpu0_1790614692011387809` |
| `swin_tiny_patch4_window7_224` | 37,200,580 | `ace6aad4c63c` | 43,799,616 | `run_bootstrap_gpu0_1790645259232989747` |
| `vit_base_patch16_224` | 89,610,748 | `a360a1935e38` | 93,867,728 | `run_bootstrap_gpu0_1790669078233737614` |

Bootstrap runs live under `artifacts/g7/campaign/` (untracked). Full
hashes are in each run's `bootstrap.json`. R is larger than the engine
size because it also counts the TRT runtime pools and I/O bindings.
