#!/usr/bin/env python3
"""GPU_M2D G7: build the TensorRT 8.6.1 INT8 PTQ engines for the six
ImageNet-1k models.

Methodology inherited from REMU tests/stage13/build_stage13_int8_engine.py:
entropy calibration (IInt8EntropyCalibrator2, batch 1), explicit batch,
4 GiB builder workspace, calibration cache keyed on the full calibration
identity, byte-atomic engine write, binding contract [data, prob, index].

G7 differences (all parameterized, no second source of truth):
  - calibration set  = g7_calib_1000_perclass1.csv (1000 classes x 1);
  - mean/std/interpolation come from each model's model_meta.json
    (written by download_models.py);
  - square resize to 224 with the model's own interpolation
    (bicubic -> INTER_CUBIC, bilinear -> INTER_LINEAR).

Environment wiring (same as run_stage13_int8_build_one.sh) -- the wrapper
build_g7_engines.sh sets it; running this file directly requires:

  export LD_LIBRARY_PATH=/data1/luojx/REMU/.local/deps/tensorrt-8.6.1/tensorrt_libs:/data1/luojx/REMU/.local/deps/cudnn-8.9.7.29/nvidia/cudnn/lib:/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  export CUDA_VISIBLE_DEVICES=<physical gpu>

Run under the vit_fault python (has tensorrt_bindings 8.6.1 + torch).
Outputs into <weights-root>/<timm-name>/: clean.engine,
engine_summary.json, calibration_<id>.cache.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import tensorrt as trt
except ModuleNotFoundError:
    import tensorrt_bindings as trt

WEIGHTS_ROOT = Path("/data1/luojx/g7_models")
CALIBRATION_CSV = Path(
    "/data1/luojx/datasets/imagenet1k/splits/g7_calib_1000_perclass1.csv"
)
MODELS = [
    "resnet50",
    "mobilenetv3_large_100",
    "efficientnet_b0",
    "vit_base_patch16_224",
    "deit_small_patch16_224",
    "swin_tiny_patch4_window7_224",
]
INPUT_SHAPE = (1, 3, 224, 224)
WORKSPACE_BYTES = 4 * 1024**3
INTERPOLATION_CV = {"bicubic": cv2.INTER_CUBIC, "bilinear": cv2.INTER_LINEAR}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    atomic_bytes(path, value.encode("utf-8"))


class CudaRuntime:
    MEMCPY_HOST_TO_DEVICE = 1
    MEMCPY_DEVICE_TO_HOST = 2

    def __init__(self) -> None:
        self.library = ctypes.CDLL("libcudart.so.12")
        self.library.cudaSetDevice.argtypes = [ctypes.c_int]
        self.library.cudaSetDevice.restype = ctypes.c_int
        self.library.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.library.cudaMalloc.restype = ctypes.c_int
        self.library.cudaFree.argtypes = [ctypes.c_void_p]
        self.library.cudaFree.restype = ctypes.c_int
        self.library.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.library.cudaMemcpy.restype = ctypes.c_int
        self.check(self.library.cudaSetDevice(0), "cudaSetDevice(0)")

    @staticmethod
    def check(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"{operation} failed with CUDA status {status}")

    def malloc(self, size: int) -> int:
        pointer = ctypes.c_void_p()
        self.check(self.library.cudaMalloc(ctypes.byref(pointer), size), "cudaMalloc")
        if pointer.value is None:
            raise RuntimeError("cudaMalloc returned a null pointer")
        return int(pointer.value)

    def copy_host_to_device(self, destination: int, source: np.ndarray) -> None:
        self.check(
            self.library.cudaMemcpy(
                ctypes.c_void_p(destination),
                ctypes.c_void_p(source.ctypes.data),
                source.nbytes,
                self.MEMCPY_HOST_TO_DEVICE,
            ),
            "cudaMemcpy(host_to_device)",
        )

    def copy_device_to_host(self, destination: np.ndarray, source: int) -> None:
        self.check(
            self.library.cudaMemcpy(
                ctypes.c_void_p(destination.ctypes.data),
                ctypes.c_void_p(source),
                destination.nbytes,
                self.MEMCPY_DEVICE_TO_HOST,
            ),
            "cudaMemcpy(device_to_host)",
        )

    def free(self, pointer: int) -> None:
        self.check(self.library.cudaFree(ctypes.c_void_p(pointer)), "cudaFree")


def preprocess_image(
    image_path: str,
    mean: np.ndarray,
    std: np.ndarray,
    interpolation: int,
) -> np.ndarray:
    """The G7 preprocessing contract (identical in calibration, clean
    evaluation, and later the fault-injection runner): square resize to
    224 with the model's own interpolation, RGB, /255, mean/std."""
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"cannot read image: {image_path}")
    image_bgr = cv2.resize(image_bgr, (224, 224), interpolation=interpolation)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    image_rgb *= np.float32(1.0 / 255.0)
    image_rgb = (image_rgb - mean) / std
    return np.ascontiguousarray(image_rgb.transpose(2, 0, 1)[None, ...])


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        image_paths: list[str],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        interpolation: int,
        cache_path: Path,
    ) -> None:
        super().__init__()
        self.image_paths = image_paths
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.interpolation = interpolation
        self.cache_path = cache_path
        self.next_image = 0
        self.cuda = CudaRuntime()
        self.device_input = self.cuda.malloc(int(np.prod(INPUT_SHAPE)) * 4)

    def __del__(self) -> None:
        pointer = getattr(self, "device_input", 0)
        if pointer:
            try:
                self.cuda.free(pointer)
            except Exception:
                pass
            self.device_input = 0

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names: list[str]) -> list[int] | None:
        if names != ["data"]:
            raise RuntimeError(f"unexpected calibration bindings: {names}")
        if self.next_image >= len(self.image_paths):
            return None
        image_path = self.image_paths[self.next_image]
        host = preprocess_image(image_path, self.mean, self.std, self.interpolation)
        self.cuda.copy_host_to_device(self.device_input, host)
        self.next_image += 1
        if (
            self.next_image == 1
            or self.next_image % 100 == 0
            or self.next_image == len(self.image_paths)
        ):
            print(
                f"int8_calibrated_images={self.next_image}/{len(self.image_paths)}",
                flush=True,
            )
        return [self.device_input]

    def read_calibration_cache(self) -> bytes | None:
        if self.cache_path.is_file():
            cache = self.cache_path.read_bytes()
            if cache:
                print(f"int8_calibration_cache=HIT path={self.cache_path}", flush=True)
                return cache
        print(f"int8_calibration_cache=MISS path={self.cache_path}", flush=True)
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        atomic_bytes(self.cache_path, bytes(cache))
        print(
            f"int8_calibration_cache=WRITTEN bytes={len(cache)} path={self.cache_path}",
            flush=True,
        )


def read_calibration_paths() -> list[str]:
    with CALIBRATION_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1000 or set(rows[0]) != {"path", "label"}:
        raise RuntimeError("G7 calibration split contract changed")
    paths = [row["path"] for row in rows]
    if len(set(paths)) != 1000 or any(not Path(path).is_file() for path in paths):
        raise RuntimeError("G7 calibration files are incomplete or duplicated")
    return paths


def gpu_identity(physical_gpu: int) -> dict[str, object]:
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            "--query-gpu=index,uuid,name,pci.bus_id,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    fields = [value.strip() for value in query.split(",")]
    if len(fields) != 6:
        raise RuntimeError(f"unexpected nvidia-smi GPU identity: {query}")
    return dict(zip(("index", "uuid", "name", "pci_bus_id", "compute_cap", "driver"), fields))


def parse_network(onnx_path: Path, logger):
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parsing failed:\n" + "\n".join(errors))
    if network.num_inputs != 1 or network.num_outputs != 2:
        raise RuntimeError("parsed network does not expose one input and two outputs")
    input_tensor = network.get_input(0)
    outputs = {network.get_output(index).name for index in range(2)}
    if (
        input_tensor.name != "data"
        or tuple(input_tensor.shape) != INPUT_SHAPE
        or input_tensor.dtype != trt.float32
        or outputs != {"prob", "index"}
    ):
        raise RuntimeError("parsed TensorRT interface contract changed")
    return builder, network


def engine_bindings(engine) -> list[dict[str, object]]:
    bindings = []
    for index in range(engine.num_bindings):
        bindings.append(
            {
                "index": index,
                "name": engine.get_binding_name(index),
                "is_input": bool(engine.binding_is_input(index)),
                "dtype": str(engine.get_binding_dtype(index)),
                "shape": list(engine.get_binding_shape(index)),
            }
        )
    return bindings


def zero_input_smoke(engine_bytes: bytes, logger) -> dict[str, object]:
    """One zero-image execute on the fresh engine: catches a broken
    tactic/kernel selection immediately instead of at eval time."""
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("INT8 engine failed immediate deserialization")
    context = engine.create_execution_context()
    cuda = CudaRuntime()
    bindings = [
        cuda.malloc(3 * 224 * 224 * 4),
        cuda.malloc(4),
        cuda.malloc(4),
    ]
    data = np.zeros(INPUT_SHAPE, dtype=np.float32)
    probability_host = np.zeros((1,), dtype=np.float32)
    index_host = np.full((1,), -1, dtype=np.int32)
    try:
        cuda.copy_host_to_device(bindings[0], data)
        if not context.execute_v2(bindings):
            raise RuntimeError("zero-input smoke execute_v2 returned false")
        cuda.copy_device_to_host(probability_host, bindings[1])
        cuda.copy_device_to_host(index_host, bindings[2])
    finally:
        for pointer in bindings:
            cuda.free(pointer)
    probability = float(probability_host[0])
    index = int(index_host[0])
    valid = 0.0 <= probability <= 1.0 and 0 <= index < 1000
    if not valid:
        raise RuntimeError(
            f"zero-input smoke produced an invalid output: prob={probability} idx={index}"
        )
    return {"prob": probability, "index": index, "valid": True}


def build(name: str, physical_gpu: int, parse_only: bool = False) -> Path | None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must contain exactly the requested physical GPU"
        )
    model_dir = WEIGHTS_ROOT / name
    onnx_path = model_dir / "model.onnx"
    onnx_summary = json.loads((model_dir / "onnx_summary.json").read_text(encoding="utf-8"))
    onnx_sha256 = sha256_file(onnx_path)
    if onnx_summary.get("status") != "PASS" or onnx_sha256 != onnx_summary["onnx_sha256"]:
        raise RuntimeError(f"ONNX identity mismatch: {name}")

    meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
    mean = tuple(float(value) for value in meta["mean"])
    std = tuple(float(value) for value in meta["std"])
    interpolation_name = str(meta["interpolation"])
    if interpolation_name not in INTERPOLATION_CV:
        raise RuntimeError(f"{name}: unknown interpolation {interpolation_name}")
    interpolation = INTERPOLATION_CV[interpolation_name]

    calibration_paths = read_calibration_paths()
    calibration_csv_sha256 = sha256_file(CALIBRATION_CSV)
    calibration_identity = hashlib.sha256(
        json.dumps(
            {
                "onnx_sha256": onnx_sha256,
                "calibration_csv_sha256": calibration_csv_sha256,
                "mean": mean,
                "std": std,
                "interpolation": interpolation_name,
                "tensorrt": trt.__version__,
                "int8_calibrator": "IInt8EntropyCalibrator2",
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    build_identity = hashlib.sha256(
        json.dumps(
            {
                "calibration_identity": calibration_identity,
                "workspace_bytes": WORKSPACE_BYTES,
                "builder_flags": ["INT8"],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    cache_path = model_dir / f"calibration_{calibration_identity[:16]}.cache"
    engine_path = model_dir / "clean.engine"
    summary_path = model_dir / "engine_summary.json"
    if not parse_only and summary_path.is_file() and engine_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            prior.get("status") == "PASS"
            and prior.get("build_identity") == build_identity
            and prior.get("engine_sha256") == sha256_file(engine_path)
            and prior.get("engine_size_bytes") == engine_path.stat().st_size
            and Path(str(prior.get("calibration_cache", ""))) == cache_path
            and cache_path.is_file()
        ):
            print(
                f"{name}: int8_engine_build=SKIP reason=eligible_existing_engine",
                flush=True,
            )
            return summary_path

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder, network = parse_network(onnx_path, logger)
    if parse_only:
        print(
            f"{name}: tensorrt_onnx_parse=PASS layers={network.num_layers} "
            f"inputs={network.num_inputs} outputs={network.num_outputs}",
            flush=True,
        )
        return None
    if not builder.platform_has_fast_int8:
        raise RuntimeError("selected GPU does not report fast INT8 support")

    calibrator = EntropyCalibrator(
        calibration_paths, mean, std, interpolation, cache_path
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator

    started = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT returned no serialized INT8 Engine")
    engine_bytes = bytes(serialized)
    if not engine_bytes:
        raise RuntimeError("TensorRT returned an empty serialized INT8 Engine")
    atomic_bytes(engine_path, engine_bytes)

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None or engine.has_implicit_batch_dimension:
        raise RuntimeError("INT8 engine failed deserialization sanity")
    contract = engine_bindings(engine)
    if [item["name"] for item in contract] != ["data", "prob", "index"]:
        raise RuntimeError(f"engine binding order changed: {contract}")
    smoke = zero_input_smoke(engine_bytes, logger)

    summary = {
        "schema_version": 1,
        "status": "PASS",
        "stage": "g7_imagenet1k",
        "model_key": name,
        "precision": "int8_ptq",
        "onnx_path": str(onnx_path),
        "onnx_sha256": onnx_sha256,
        "weights_sha256": onnx_summary["weights_sha256"],
        "calibration_manifest": str(CALIBRATION_CSV),
        "calibration_manifest_sha256": calibration_csv_sha256,
        "calibration_images": len(calibration_paths),
        "calibration_cache": str(cache_path),
        "calibration_cache_sha256": sha256_file(cache_path),
        "calibration_identity": calibration_identity,
        "preprocessing": {
            "mean": list(mean),
            "std": list(std),
            "interpolation": interpolation_name,
            "input_size": meta["input_size"],
        },
        "build_identity": build_identity,
        "workspace_bytes": WORKSPACE_BYTES,
        "builder_flags": ["INT8"],
        "engine_path": str(engine_path),
        "engine_sha256": sha256_file(engine_path),
        "engine_size_bytes": len(engine_bytes),
        "engine_bindings": contract,
        "zero_input_smoke": smoke,
        "build_elapsed_seconds": time.time() - started,
        "physical_gpu": gpu_identity(physical_gpu),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "tensorrt": trt.__version__,
        },
    }
    atomic_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"{name}: int8_engine_build=PASS size_bytes={len(engine_bytes)} "
        f"elapsed_seconds={summary['build_elapsed_seconds']:.1f} "
        f"smoke_prob={smoke['prob']:.6f}",
        flush=True,
    )
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models", nargs="*", default=MODELS, help="subset of model names, default all six"
    )
    parser.add_argument("--device", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--parse-only", action="store_true")
    args = parser.parse_args()
    if not args.models:
        parser.error("no models selected")
    for name in args.models:
        build(name, args.device, parse_only=args.parse_only)
    print(f"g7_int8_engine_build={'PARSE' if args.parse_only else 'PASS'} "
          f"models={len(args.models)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
