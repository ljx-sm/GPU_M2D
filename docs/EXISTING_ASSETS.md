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
