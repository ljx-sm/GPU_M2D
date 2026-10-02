#include "gpu_m2d/allocation_registry.hpp"
#include "gpu_m2d/device_bit_injector.hpp"
#include "gpu_m2d/l2_probe.hpp"
#include "gpu_m2d/residency_map.hpp"
#include "gpu_m2d/tensor_mapping.hpp"

#include <NvInfer.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr int kInputHeight = 224;
constexpr int kInputWidth = 224;
constexpr int kClassCount = 45;
constexpr float kProbabilityTolerance = 1.0e-6F;
// G4-T2 post-restore sanity inference: INT8 execution is deterministic for
// a fixed engine and input, so a fully restored run must reproduce the
// clean output; the tolerance only absorbs float reduction wobble.
constexpr float kSanityTolerance = 1.0e-4F;
const std::array<float, 3> kMean{{0.485F, 0.456F, 0.406F}};
const std::array<float, 3> kStd{{0.229F, 0.224F, 0.225F}};

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::lock_guard<std::mutex> lock(mutex_);
            std::cerr << "TensorRT: " << message << '\n';
        }
    }

private:
    std::mutex mutex_;
};

template <typename T>
struct TrtDeleter {
    void operator()(T* object) const noexcept {
        delete object;
    }
};

// Polls for a gate file's appearance. Used by the observer pre-allocation
// gate and by the G4-T2 injection release gate; the orchestrator creates
// the file only after its side of the handshake is complete.
bool wait_for_file(const std::string& path, int timeout_seconds) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
    while (std::chrono::steady_clock::now() < deadline) {
        if (access(path.c_str(), F_OK) == 0) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    return false;
}

// One reverse-chain XOR target selected by the G4-T2 orchestrator from the
// per-run dual-addressing snapshot (GDDR PA page -> VA -> allocation byte).
struct InjectionTarget {
    std::string target_id;
    std::string allocation_id;
    std::size_t byte_offset{0};
    unsigned bit_in_byte{0};
    std::uintptr_t expected_gpu_va{0};
};

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream stream(line);
    while (std::getline(stream, field, ',')) {
        fields.push_back(field);
    }
    if (!line.empty() && line.back() == ',') {
        fields.emplace_back();
    }
    return fields;
}

// Work file: "target_id,allocation_id,byte_offset,bit_in_byte,expected_gpu_va"
// with a header row and optional '#' comment lines. expected_gpu_va is the
// chain check: it must equal the live registry's forward mapping, so any
// drift between the orchestrator's snapshot and this process is refused.
std::vector<InjectionTarget> read_injection_work(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open injection work file: " + path);
    }
    std::vector<InjectionTarget> targets;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line.front() == '#' ||
            line.rfind("target_id,", 0) == 0) {
            continue;
        }
        const std::vector<std::string> fields = split_csv_line(line);
        if (fields.size() != 5) {
            throw std::runtime_error("malformed injection work row: " + line);
        }
        InjectionTarget target;
        target.target_id = fields[0];
        target.allocation_id = fields[1];
        target.byte_offset = std::stoull(fields[2]);
        target.bit_in_byte = std::stoul(fields[3]);
        target.expected_gpu_va = std::stoull(fields[4], nullptr, 16);
        if (target.target_id.empty() || target.allocation_id.empty() ||
            target.bit_in_byte > 7 || target.expected_gpu_va == 0) {
            throw std::runtime_error("invalid injection work row: " + line);
        }
        targets.push_back(std::move(target));
    }
    if (targets.empty()) {
        throw std::runtime_error("injection work file has no targets: " + path);
    }
    return targets;
}

// Optional G2 observer instrumentation. When a gate file is configured the
// runner blocks before creating any CUDA context (the eBPF probes attach in
// that window), then reports every AllocationRegistry lifetime transition as
// a GPU_M2D_EVENT line so the orchestrator can correlate user-space
// allocations with kernel MAP/PTE/FREE events. The registry behavior is
// identical with and without the observer.
struct ObserverEmitter {
    bool enabled{false};
    std::string gate_file;
    int hold_seconds{0};
    int gate_timeout_seconds{30};

    static std::uint64_t wall_time_ns() {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count());
    }

    static std::uint64_t monotonic_time_ns() {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
    }

    bool wait_for_gate() const {
        return wait_for_file(gate_file, gate_timeout_seconds);
    }

    void event(const char* name) const {
        if (!enabled) {
            return;
        }
        std::printf("GPU_M2D_EVENT,event=%s,pid=%d,tgid=%d,wall_time_ns=%" PRIu64
                    ",monotonic_ns=%" PRIu64 "\n",
                    name, static_cast<int>(getpid()), static_cast<int>(getpid()),
                    wall_time_ns(), monotonic_time_ns());
        std::fflush(stdout);
    }

    // Same lifecycle line format with extra comma-free key=value fields
    // (used by the G4-T2 per-target events).
    void event_with(const char* name, const std::string& fields) const {
        if (!enabled) {
            return;
        }
        std::printf("GPU_M2D_EVENT,event=%s,%s,pid=%d,tgid=%d,wall_time_ns=%"
                    PRIu64 ",monotonic_ns=%" PRIu64 "\n",
                    name, fields.c_str(), static_cast<int>(getpid()),
                    static_cast<int>(getpid()), wall_time_ns(), monotonic_time_ns());
        std::fflush(stdout);
    }

    void allocated(const std::string& allocation_id, const char* allocation_api,
                   std::uintptr_t base_va, std::size_t size_bytes,
                   const std::string& allocation_phase,
                   const std::string& semantic_label,
                   const std::string& cuda_buffer_id) const {
        if (!enabled) {
            return;
        }
        std::printf("GPU_M2D_EVENT,event=ALLOCATED,allocation_api=%s,pid=%d,tgid=%d,"
                    "allocation_id=%s,base_va=0x%" PRIxPTR ",size_bytes=%zu,"
                    "allocation_phase=%s,semantic_label=%s,cuda_buffer_id=%s,"
                    "wall_time_ns=%" PRIu64 ",monotonic_ns=%" PRIu64 "\n",
                    allocation_api, static_cast<int>(getpid()), static_cast<int>(getpid()),
                    allocation_id.c_str(), base_va, size_bytes, allocation_phase.c_str(),
                    semantic_label.c_str(), cuda_buffer_id.c_str(), wall_time_ns(),
                    monotonic_time_ns());
        std::fflush(stdout);
    }

    void freed(const std::string& allocation_id) const {
        if (!enabled) {
            return;
        }
        std::printf("GPU_M2D_EVENT,event=FREE,allocation_id=%s,wall_time_ns=%" PRIu64
                    ",monotonic_ns=%" PRIu64 "\n",
                    allocation_id.c_str(), wall_time_ns(), monotonic_time_ns());
        std::fflush(stdout);
    }
};

struct TrackedAllocation {
    std::uintptr_t gpu_va{0};
    std::string allocation_id;
    bool active{false};
};

class TrackingGpuAllocator final : public nvinfer1::IGpuAllocator {
public:
    TrackingGpuAllocator(gpu_m2d::AllocationRegistry& registry, int device_id,
                         const ObserverEmitter* observer = nullptr)
        : registry_(registry), device_id_(device_id), observer_(observer) {}

    void set_phase(std::string phase) {
        std::lock_guard<std::mutex> lock(mutex_);
        phase_ = std::move(phase);
    }

    void* allocate(std::uint64_t size, std::uint64_t alignment,
                   nvinfer1::AllocatorFlags) noexcept override {
        if (size == 0 || size > std::numeric_limits<std::size_t>::max() ||
            alignment > std::numeric_limits<std::size_t>::max()) {
            return nullptr;
        }

        void* memory = nullptr;
        if (cudaMalloc(&memory, static_cast<std::size_t>(size)) != cudaSuccess) {
            return nullptr;
        }

        std::string allocation_id;
        std::string phase;
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            allocation_id = "trt-internal-" + std::to_string(next_id_++);
            phase = phase_;
        } catch (...) {
            cudaFree(memory);
            return nullptr;
        }

        try {
            registry_.add_allocation(gpu_m2d::AllocationDescriptor{
                allocation_id,
                device_id_,
                reinterpret_cast<std::uintptr_t>(memory),
                static_cast<std::size_t>(size),
                static_cast<std::size_t>(alignment),
                "TensorRT IGpuAllocator",
                "TENSORRT_INTERNAL_UNKNOWN",
                phase,
                "TensorRT allocate callback to matching deallocate callback",
                true,
            });
            std::lock_guard<std::mutex> lock(mutex_);
            records_.push_back(TrackedAllocation{
                reinterpret_cast<std::uintptr_t>(memory), allocation_id, true});
            if (observer_ != nullptr) {
                observer_->allocated(allocation_id, "tensorrt-igpu-allocator",
                                     reinterpret_cast<std::uintptr_t>(memory),
                                     static_cast<std::size_t>(size), phase,
                                     "TENSORRT_INTERNAL_UNKNOWN", "");
            }
        } catch (...) {
            static_cast<void>(registry_.deactivate_allocation(allocation_id));
            cudaFree(memory);
            return nullptr;
        }
        return memory;
    }

    void free(void* memory) noexcept override {
        static_cast<void>(deallocate(memory));
    }

    bool deallocate(void* memory) noexcept override {
        if (memory == nullptr) {
            return true;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(memory);
        auto record = std::find_if(
            records_.rbegin(), records_.rend(),
            [address](const TrackedAllocation& candidate) {
                return candidate.gpu_va == address && candidate.active;
            });
        if (record == records_.rend() || cudaFree(memory) != cudaSuccess) {
            return false;
        }
        record->active = false;
        static_cast<void>(registry_.deactivate_allocation(record->allocation_id));
        if (observer_ != nullptr) {
            observer_->freed(record->allocation_id);
        }
        return true;
    }

private:
    gpu_m2d::AllocationRegistry& registry_;
    int device_id_{-1};
    const ObserverEmitter* observer_{nullptr};
    mutable std::mutex mutex_;
    std::vector<TrackedAllocation> records_;
    std::string phase_{"runtime_setup"};
    std::size_t next_id_{0};
};

class DeviceBuffer {
public:
    DeviceBuffer(std::size_t size_bytes,
                 gpu_m2d::AllocationRegistry& registry,
                 std::string allocation_id,
                 int device_id,
                 std::string semantic_label)
        : size_bytes_(size_bytes), registry_(&registry),
          allocation_id_(std::move(allocation_id)) {
        check_cuda(cudaMalloc(&pointer_, size_bytes_), "cudaMalloc binding");
        try {
            registry_->add_allocation(gpu_m2d::AllocationDescriptor{
                allocation_id_,
                device_id,
                reinterpret_cast<std::uintptr_t>(pointer_),
                size_bytes_,
                0,
                "G1.5 runner",
                std::move(semantic_label),
                "allocate_bindings",
                "runner cudaMalloc to runner cudaFree",
                true,
            });
        } catch (...) {
            cudaFree(pointer_);
            pointer_ = nullptr;
            throw;
        }
    }

    ~DeviceBuffer() {
        release();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : pointer_(other.pointer_), size_bytes_(other.size_bytes_),
          registry_(other.registry_), allocation_id_(std::move(other.allocation_id_)) {
        other.pointer_ = nullptr;
        other.size_bytes_ = 0;
        other.registry_ = nullptr;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            pointer_ = other.pointer_;
            size_bytes_ = other.size_bytes_;
            registry_ = other.registry_;
            allocation_id_ = std::move(other.allocation_id_);
            other.pointer_ = nullptr;
            other.size_bytes_ = 0;
            other.registry_ = nullptr;
        }
        return *this;
    }

    void* get() const noexcept { return pointer_; }
    std::size_t size_bytes() const noexcept { return size_bytes_; }
    const std::string& allocation_id() const noexcept { return allocation_id_; }

private:
    void release() noexcept {
        if (pointer_ != nullptr && cudaFree(pointer_) == cudaSuccess &&
            registry_ != nullptr) {
            static_cast<void>(registry_->deactivate_allocation(allocation_id_));
        }
        pointer_ = nullptr;
        size_bytes_ = 0;
        registry_ = nullptr;
    }

    static void check_cuda(cudaError_t status, const char* operation) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string(operation) + " failed: " +
                                     cudaGetErrorString(status));
        }
    }

    void* pointer_{nullptr};
    std::size_t size_bytes_{0};
    gpu_m2d::AllocationRegistry* registry_{nullptr};
    std::string allocation_id_;
};

class CudaStream {
public:
    CudaStream() {
        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
    }

    ~CudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const noexcept { return stream_; }

private:
    static void check_cuda(cudaError_t status, const char* operation) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string(operation) + " failed: " +
                                     cudaGetErrorString(status));
        }
    }

    cudaStream_t stream_{nullptr};
};

struct Options {
    std::string engine_path;
    std::string sample_csv;
    std::string output_prefix;
    std::size_t sample_index{0};
    std::size_t element_index{0};
    std::size_t element_bit_index{0};
    int device{0};
    std::string observer_gate;
    int hold_seconds{0};
    int gate_timeout_seconds{30};
    // G4-T2 gated dual-addressing injection mode: both paths must be given
    // together. The runner writes its gate-time allocation registry, waits
    // for the release file, then executes the work file's reverse-chain
    // XOR targets instead of the fixed element/bit self-injection.
    std::string injection_work;
    std::string injection_release;
    int injection_gate_timeout_seconds{0};
    // G5 campaign mode: like the T2 gate, but the work file carries N
    // trials of fault-model sites; per trial the runner flips ALL sites,
    // runs the full evaluation pass with the faults held in place,
    // classifies every image against the clean pass, restores byte-exactly
    // and proves no residue with a sanity inference. Mutually exclusive
    // with the T2 mode.
    std::string campaign_work;
    std::string campaign_release;
    int campaign_gate_timeout_seconds{0};
    // G7 workload parameterization. Defaults keep the G5 RESISC45 behavior
    // byte-identical; a G7 invocation overrides them from the model's
    // model_meta.json (the single source of preprocessing truth) via the
    // orchestrator.
    int class_count{kClassCount};
    std::string preprocess_mode{"legacy"};
    int resize_scale{224};
    int resize_interpolation{cv::INTER_CUBIC};
    std::array<float, 3> canonical_mean{kMean};
    std::array<float, 3> canonical_std{kStd};
    // Debug: write the sample-index image's preprocessed float tensor to
    // this path and exit (pure CPU, before any engine load or CUDA call)
    // so the canonical preprocessing can be diffed bit-exactly against the
    // python contract.
    std::string dump_preprocessed;
    // Restart-protocol host-image cache directory (empty = disabled): the
    // campaign prelude's preprocessed CHW float buffer is written here
    // keyed by every input that determines its bytes, so a relaunched
    // segment skips the ~3 min CPU re-preprocessing of the 10K pass.
    // Pure optimization -- any mismatch falls back to fresh work.
    std::string image_cache_dir;
    // G8 L2 residency probe pass (docs/G8_CACHE_FAULT_PLAN.md §4; T0 hook):
    // after the full strict clean pass, a second FAULTLESS pass probes the
    // L2 residency of every registered allocation at image boundaries
    // (every l2_probe_every images, before the image's input is staged),
    // times each inference with CUDA events (probe time excluded), and
    // must reproduce the clean pass bit-identically. Measure-only:
    // mutually exclusive with the injection and campaign modes.
    std::string l2_probe_out;
    std::size_t l2_probe_every{1};
    std::size_t l2_probe_unit{128};
    std::uint32_t l2_probe_threshold{0};
    int l2_probe_per_sm{16};
    // Staggered sub-sampling: sweep s probes units u with
    // u % l2_probe_stride == s % l2_probe_stride (bounds the sweep's own
    // L2 fill traffic for surfaces larger than L2); reverse probes in
    // descending unit order (contamination detector).
    std::size_t l2_probe_stride{1};
    bool l2_probe_reverse{false};
    // G8-T1: flip the probe direction for each unit's successive
    // observations; build and write the per-unit residency map; repeat the
    // probed pass N times in this process (same-process stability).
    bool l2_probe_alternate{false};
    bool l2_probe_map{false};
    std::size_t l2_probe_passes{1};
};

struct Sample {
    std::string path;
    int target{-1};
};

struct BindingInfo {
    int index{-1};
    std::string name;
    bool is_input{false};
    nvinfer1::DataType trt_dtype{nvinfer1::DataType::kFLOAT};
    gpu_m2d::DType dtype{gpu_m2d::DType::kFloat32};
    std::vector<std::size_t> shape;
    std::size_t element_count{0};
    std::size_t size_bytes{0};
};

struct Prediction {
    float probability{0.0F};
    std::int32_t class_index{-1};
};

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(status));
    }
}

std::string query_buffer_id(const void* pointer) {
    unsigned long long buffer_id = 0;
    if (cuPointerGetAttribute(&buffer_id, CU_POINTER_ATTRIBUTE_BUFFER_ID,
                              reinterpret_cast<CUdeviceptr>(pointer)) == CUDA_SUCCESS &&
        buffer_id != 0) {
        return std::to_string(buffer_id);
    }
    return "";
}

std::array<float, 3> parse_rgb_triple(const std::string& value,
                                      const char* flag_name) {
    std::array<std::string, 3> parts{};
    std::size_t position = 0;
    for (int index = 0; index < 3; ++index) {
        const std::size_t comma = value.find(',', position);
        if (index < 2 && comma == std::string::npos) {
            throw std::invalid_argument(std::string(flag_name) +
                                        " expects R,G,B: " + value);
        }
        parts[static_cast<std::size_t>(index)] =
            value.substr(position, comma == std::string::npos
                                       ? std::string::npos
                                       : comma - position);
        position = comma + 1;
    }
    std::array<float, 3> triple{};
    for (int index = 0; index < 3; ++index) {
        // stod (decimal -> double -> float) mirrors numpy's
        // np.asarray([...], dtype=np.float32) bit-exactly
        triple[static_cast<std::size_t>(index)] =
            static_cast<float>(std::stod(parts[static_cast<std::size_t>(index)]));
    }
    return triple;
}

int parse_interpolation(const std::string& value) {
    if (value == "bicubic") {
        return cv::INTER_CUBIC;
    }
    if (value == "bilinear") {
        return cv::INTER_LINEAR;
    }
    throw std::invalid_argument(
        "--interp must be bicubic or bilinear (timm vocabulary): " + value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for argument: " + key);
        }
        const std::string value = argv[++index];
        if (key == "--engine") {
            options.engine_path = value;
        } else if (key == "--sample-csv") {
            options.sample_csv = value;
        } else if (key == "--sample-index") {
            options.sample_index = std::stoull(value);
        } else if (key == "--device") {
            options.device = std::stoi(value);
        } else if (key == "--element") {
            options.element_index = std::stoull(value);
        } else if (key == "--bit") {
            options.element_bit_index = std::stoull(value);
        } else if (key == "--observer-gate") {
            options.observer_gate = value;
        } else if (key == "--hold-seconds") {
            options.hold_seconds = std::stoi(value);
            if (options.hold_seconds < 0) {
                throw std::invalid_argument("--hold-seconds must be >= 0");
            }
        } else if (key == "--gate-timeout-seconds") {
            options.gate_timeout_seconds = std::stoi(value);
            if (options.gate_timeout_seconds <= 0) {
                throw std::invalid_argument("--gate-timeout-seconds must be > 0");
            }
        } else if (key == "--injection-work") {
            options.injection_work = value;
        } else if (key == "--injection-release") {
            options.injection_release = value;
        } else if (key == "--injection-gate-timeout-seconds") {
            options.injection_gate_timeout_seconds = std::stoi(value);
            if (options.injection_gate_timeout_seconds <= 0) {
                throw std::invalid_argument("--injection-gate-timeout-seconds must be > 0");
            }
        } else if (key == "--campaign-work") {
            options.campaign_work = value;
        } else if (key == "--campaign-release") {
            options.campaign_release = value;
        } else if (key == "--campaign-gate-timeout-seconds") {
            options.campaign_gate_timeout_seconds = std::stoi(value);
            if (options.campaign_gate_timeout_seconds <= 0) {
                throw std::invalid_argument("--campaign-gate-timeout-seconds must be > 0");
            }
        } else if (key == "--output-prefix") {
            options.output_prefix = value;
        } else if (key == "--class-count") {
            options.class_count = std::stoi(value);
            if (options.class_count <= 0) {
                throw std::invalid_argument("--class-count must be > 0");
            }
        } else if (key == "--preprocess") {
            options.preprocess_mode = value;
            if (value != "legacy" && value != "canonical") {
                throw std::invalid_argument(
                    "--preprocess must be legacy or canonical: " + value);
            }
        } else if (key == "--resize-scale") {
            options.resize_scale = std::stoi(value);
            if (options.resize_scale < 224) {
                throw std::invalid_argument(
                    "--resize-scale must cover the 224 crop");
            }
        } else if (key == "--interp") {
            options.resize_interpolation = parse_interpolation(value);
        } else if (key == "--mean") {
            options.canonical_mean = parse_rgb_triple(value, "--mean");
        } else if (key == "--std") {
            options.canonical_std = parse_rgb_triple(value, "--std");
        } else if (key == "--dump-preprocessed") {
            options.dump_preprocessed = value;
        } else if (key == "--l2-probe-out") {
            options.l2_probe_out = value;
        } else if (key == "--l2-probe-every") {
            options.l2_probe_every = std::stoull(value);
            if (options.l2_probe_every == 0) {
                throw std::invalid_argument("--l2-probe-every must be > 0");
            }
        } else if (key == "--l2-probe-unit") {
            options.l2_probe_unit = std::stoull(value);
            if (options.l2_probe_unit != 32 && options.l2_probe_unit != 128) {
                throw std::invalid_argument("--l2-probe-unit must be 32 or 128");
            }
        } else if (key == "--l2-probe-threshold") {
            const unsigned long threshold = std::stoul(value);
            if (threshold == 0 || threshold > 0xffffUL) {
                throw std::invalid_argument(
                    "--l2-probe-threshold must be in 1..65535 cycles");
            }
            options.l2_probe_threshold = static_cast<std::uint32_t>(threshold);
        } else if (key == "--l2-probe-stride") {
            options.l2_probe_stride = std::stoull(value);
            if (options.l2_probe_stride == 0) {
                throw std::invalid_argument("--l2-probe-stride must be > 0");
            }
        } else if (key == "--l2-probe-reverse") {
            if (value != "0" && value != "1") {
                throw std::invalid_argument("--l2-probe-reverse must be 0 or 1");
            }
            options.l2_probe_reverse = value == "1";
        } else if (key == "--l2-probe-alternate") {
            if (value != "0" && value != "1") {
                throw std::invalid_argument("--l2-probe-alternate must be 0 or 1");
            }
            options.l2_probe_alternate = value == "1";
        } else if (key == "--l2-probe-map") {
            if (value != "0" && value != "1") {
                throw std::invalid_argument("--l2-probe-map must be 0 or 1");
            }
            options.l2_probe_map = value == "1";
        } else if (key == "--l2-probe-passes") {
            options.l2_probe_passes = std::stoull(value);
            if (options.l2_probe_passes == 0) {
                throw std::invalid_argument("--l2-probe-passes must be > 0");
            }
        } else if (key == "--l2-probe-per-sm") {
            options.l2_probe_per_sm = std::stoi(value);
            if (options.l2_probe_per_sm <= 0) {
                throw std::invalid_argument("--l2-probe-per-sm must be > 0");
            }
        } else if (key == "--image-cache-dir") {
            options.image_cache_dir = value;
            if (value.empty()) {
                throw std::invalid_argument("--image-cache-dir needs a path");
            }
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }

    if (options.engine_path.empty() || options.sample_csv.empty() ||
        options.output_prefix.empty()) {
        throw std::invalid_argument(
            "required arguments: --engine PATH --sample-csv PATH "
            "--output-prefix PATH [--sample-index N --device N --element N --bit N] "
            "[--observer-gate PATH --hold-seconds N --gate-timeout-seconds N] "
            "[--injection-work PATH --injection-release PATH "
            "--injection-gate-timeout-seconds N] "
            "[--campaign-work PATH --campaign-release PATH "
            "--campaign-gate-timeout-seconds N] "
            "[--class-count N --preprocess legacy|canonical --resize-scale N "
            "--interp bicubic|bilinear --mean R,G,B --std R,G,B "
            "--dump-preprocessed PATH] "
            "[--image-cache-dir DIR] "
            "[--l2-probe-out PREFIX --l2-probe-threshold CYCLES "
            "--l2-probe-every N --l2-probe-unit 32|128 "
            "--l2-probe-per-sm N --l2-probe-stride N "
            "--l2-probe-reverse 0|1 --l2-probe-alternate 0|1 "
            "--l2-probe-map 0|1 --l2-probe-passes N]");
    }
    if (options.injection_work.empty() != options.injection_release.empty()) {
        throw std::invalid_argument(
            "--injection-work and --injection-release must be given together");
    }
    if (options.campaign_work.empty() != options.campaign_release.empty()) {
        throw std::invalid_argument(
            "--campaign-work and --campaign-release must be given together");
    }
    if (!options.campaign_work.empty() && !options.injection_work.empty()) {
        throw std::invalid_argument(
            "--campaign-* and --injection-* are mutually exclusive modes");
    }
    if (!options.l2_probe_out.empty()) {
        if (!options.campaign_work.empty() || !options.injection_work.empty()) {
            throw std::invalid_argument(
                "--l2-probe-out is a measure-only mode; it excludes "
                "--campaign-* and --injection-*");
        }
        if (options.l2_probe_threshold == 0) {
            throw std::invalid_argument(
                "--l2-probe-out needs --l2-probe-threshold (G8-T0 calibration)");
        }
    }
    return options;
}

std::vector<char> read_binary_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open engine: " + path);
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size <= 0) {
        throw std::runtime_error("engine file is empty: " + path);
    }
    input.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<std::size_t>(size));
    input.read(bytes.data(), size);
    if (!input) {
        throw std::runtime_error("cannot read complete engine: " + path);
    }
    return bytes;
}

Sample read_sample(const std::string& csv_path, std::size_t sample_index) {
    std::ifstream input(csv_path);
    if (!input) {
        throw std::runtime_error("cannot open sample CSV: " + csv_path);
    }

    std::string line;
    std::size_t current_index = 0;
    bool first_line = true;
    while (std::getline(input, line)) {
        // tolerate CRLF rows (the G7 split CSVs are csv-module written);
        // without this the header check fails and every index shifts by one
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (first_line) {
            first_line = false;
            if (line == "path,label") {
                continue;
            }
        }
        if (current_index++ != sample_index) {
            continue;
        }
        const std::size_t comma = line.find(',');
        if (comma == std::string::npos) {
            throw std::runtime_error("invalid sample CSV row: " + line);
        }
        return Sample{line.substr(0, comma), std::stoi(line.substr(comma + 1))};
    }
    throw std::out_of_range("sample index is outside the CSV");
}

// Preprocessing policy. Legacy = the G5 RESISC45 squash resize (kept
// byte-identical; G5 comparability). Canonical = the G7 v2 contract
// (tools/g7_prep/build_g7_int8_engine.py:preprocess_image): aspect-
// preserving shorter-edge resize to resize_scale (torchvision int()
// truncation of the longer edge), INTER_AREA on downscale / the model's
// own interpolation on upscale, torchvision-exact round() center crop,
// BGR->RGB, float32 /255, float32 mean/std, CHW.
struct PreprocessSpec {
    bool canonical{false};
    int resize_scale{224};
    int resize_interpolation{cv::INTER_CUBIC};
    std::array<float, 3> mean{kMean};  // RGB order
    std::array<float, 3> std{kStd};    // RGB order
};

std::vector<float> preprocess(const std::string& path,
                              const PreprocessSpec& spec) {
    const cv::Mat source = cv::imread(path, cv::IMREAD_COLOR);
    if (source.empty()) {
        throw std::runtime_error("cannot decode image: " + path);
    }

    const int plane = kInputHeight * kInputWidth;
    std::vector<float> chw(3U * static_cast<std::size_t>(plane));

    if (!spec.canonical) {
        cv::Mat image;
        cv::resize(source, image, cv::Size(kInputWidth, kInputHeight), 0, 0,
                   cv::INTER_CUBIC);
        image.convertTo(image, CV_32FC3, 1.0 / 255.0);
        cv::subtract(image, cv::Scalar(kMean[2], kMean[1], kMean[0]), image);
        cv::divide(image, cv::Scalar(kStd[2], kStd[1], kStd[0]), image);
        for (int row = 0; row < kInputHeight; ++row) {
            const cv::Vec3f* pixels = image.ptr<cv::Vec3f>(row);
            for (int column = 0; column < kInputWidth; ++column) {
                const int offset = row * kInputWidth + column;
                chw[static_cast<std::size_t>(offset)] = pixels[column][2];
                chw[static_cast<std::size_t>(plane + offset)] = pixels[column][1];
                chw[static_cast<std::size_t>(2 * plane + offset)] = pixels[column][0];
            }
        }
        return chw;
    }

    // Canonical v2, statement-for-statement the python contract. Both call
    // the same OpenCV C++ routines underneath (cv::INTER_AREA here IS
    // cv2.INTER_AREA), so the output is bit-identical when the arithmetic
    // order and dtypes match exactly.
    const int height = source.rows;
    const int width = source.cols;
    int new_width = 0;
    int new_height = 0;
    if (width <= height) {
        new_width = spec.resize_scale;
        new_height = static_cast<int>(
            static_cast<double>(spec.resize_scale) * height / width);
    } else {
        new_width = static_cast<int>(
            static_cast<double>(spec.resize_scale) * width / height);
        new_height = spec.resize_scale;
    }
    const int resize_interpolation =
        (new_width < width || new_height < height) ? cv::INTER_AREA
                                                   : spec.resize_interpolation;
    cv::Mat resized;
    cv::resize(source, resized, cv::Size(new_width, new_height), 0, 0,
               resize_interpolation);
    // torchvision CenterCrop uses int(round(...)); python round() is
    // round-half-to-EVEN, which is std::nearbyint under the default FE
    // rounding mode (std::round would differ on every .5)
    const int top = std::max(
        0, static_cast<int>(std::nearbyint((new_height - kInputHeight) / 2.0)));
    const int left = std::max(
        0, static_cast<int>(std::nearbyint((new_width - kInputWidth) / 2.0)));
    const cv::Mat cropped = resized(
        cv::Range(top, top + kInputHeight), cv::Range(left, left + kInputWidth));
    cv::Mat rgb;
    cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);

    // float32 chain in the python order: *np.float32(1/255), then
    // (x - mean) / std -- separate statements so the compiler cannot
    // contract them into an FMA (numpy runs each ufunc separately too)
    const float inv_255 = static_cast<float>(1.0 / 255.0);
    for (int row = 0; row < kInputHeight; ++row) {
        const cv::Vec3b* pixels = rgb.ptr<cv::Vec3b>(row);
        for (int column = 0; column < kInputWidth; ++column) {
            const int offset = row * kInputWidth + column;
            for (int channel = 0; channel < 3; ++channel) {
                float value =
                    static_cast<float>(pixels[column][channel]) * inv_255;
                value = (value - spec.mean[channel]) / spec.std[channel];
                chw[static_cast<std::size_t>(channel * plane + offset)] = value;
            }
        }
    }
    return chw;
}

// ---- G5 campaign host-image cache (restart protocol) ------------------
// The campaign prelude preprocesses every evaluation image on the host
// (~3 min CPU for the G7 10K ImageNet pass) before the pre-allocation
// gate, and a process-fatal trial restart relaunches that whole prelude.
// This cache stores the preprocessed CHW float32 buffer on disk, keyed
// by every input that determines its bytes: the sample CSV content, the
// full preprocessing spec, the image count and per-image tensor size,
// and the runner BINARY itself (a rebuild against a different OpenCV
// could preprocess differently). Pure optimization: any miss, mismatch,
// or I/O error falls back to fresh preprocessing and rewrites the file;
// the gated window, the clean pass, and the trial loop are untouched.
constexpr unsigned char kImageCacheMagic[8] = {'G', 'M', '2', 'D',
                                               'I', 'M', 'G', 'C'};
constexpr uint32_t kImageCacheVersion = 1;

class Sha256 {
public:
    Sha256() { reset(); }
    void reset() {
        state_[0] = 0x6a09e667u; state_[1] = 0xbb67ae85u;
        state_[2] = 0x3c6ef372u; state_[3] = 0xa54ff53au;
        state_[4] = 0x510e527fu; state_[5] = 0x9b05688cu;
        state_[6] = 0x1f83d9abu; state_[7] = 0x5be0cd19u;
        bits_ = 0;
        fill_ = 0;
    }
    void update(const void* data, std::size_t size) {
        const unsigned char* p = static_cast<const unsigned char*>(data);
        bits_ += static_cast<uint64_t>(size) * 8u;
        while (size > 0) {
            const std::size_t take = std::min(size, sizeof(buffer_) - fill_);
            std::memcpy(buffer_ + fill_, p, take);
            fill_ += take;
            p += take;
            size -= take;
            if (fill_ == sizeof(buffer_)) {
                compress(buffer_);
                fill_ = 0;
            }
        }
    }
    void update_u32(uint32_t value) {
        unsigned char b[4] = {static_cast<unsigned char>(value >> 24),
                              static_cast<unsigned char>(value >> 16),
                              static_cast<unsigned char>(value >> 8),
                              static_cast<unsigned char>(value)};
        update(b, sizeof(b));
    }
    void update_u64(uint64_t value) {
        unsigned char b[8];
        for (int i = 0; i < 8; ++i) {
            b[i] = static_cast<unsigned char>(value >> (56 - 8 * i));
        }
        update(b, sizeof(b));
    }
    void update_float(float value) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        update_u32(bits);
    }
    // finalizes this hash object in place
    std::array<unsigned char, 32> digest() {
        const uint64_t bits = bits_;  // padding must not extend the length
        const unsigned char one = 0x80;
        update(&one, 1);
        const unsigned char zero = 0x00;
        while (fill_ != 56) {
            update(&zero, 1);
        }
        unsigned char length[8];
        for (int i = 0; i < 8; ++i) {
            length[i] = static_cast<unsigned char>(bits >> (56 - 8 * i));
        }
        update(length, sizeof(length));
        std::array<unsigned char, 32> out{};
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 4; ++j) {
                out[static_cast<std::size_t>(4 * i + j)] =
                    static_cast<unsigned char>(state_[i] >> (24 - 8 * j));
            }
        }
        return out;
    }

private:
    static uint32_t rotr(uint32_t x, int n) {
        return (x >> n) | (x << (32 - n));
    }
    void compress(const unsigned char* p) {
        static constexpr uint32_t K[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
            0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
            0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
            0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
            0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
            0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
            0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
            0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(p[4 * i]) << 24) |
                   (static_cast<uint32_t>(p[4 * i + 1]) << 16) |
                   (static_cast<uint32_t>(p[4 * i + 2]) << 8) |
                   static_cast<uint32_t>(p[4 * i + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^
                                (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^
                                (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + S1 + ch + K[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }
    uint32_t state_[8];
    uint64_t bits_;
    unsigned char buffer_[64];
    std::size_t fill_;
};

std::string to_hex(const std::array<unsigned char, 32>& digest) {
    static const char kDigits[] = "0123456789abcdef";
    std::string out(64, '0');
    for (std::size_t i = 0; i < digest.size(); ++i) {
        out[2 * i] = kDigits[digest[i] >> 4];
        out[2 * i + 1] = kDigits[digest[i] & 0x0F];
    }
    return out;
}

std::array<unsigned char, 32> sha256_file(const std::string& path,
                                           const char* what) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error(std::string("cannot open ") + what + ": " +
                                 path);
    }
    Sha256 hash;
    std::vector<char> chunk(1u << 20);
    while (file) {
        file.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const std::streamsize got = file.gcount();
        if (got > 0) {
            hash.update(chunk.data(), static_cast<std::size_t>(got));
        }
    }
    if (file.bad()) {
        throw std::runtime_error(std::string("read error on ") + what + ": " +
                                 path);
    }
    return hash.digest();
}

// Field encoding (the offline python cross-check mirrors it exactly):
// tag | u32 schema | u32 have-self | self[32]? | csv[32] | u64 count |
// u64 floats/image | u32 canonical | u32 scale | u32 interpolation |
// 3x f32 mean bits | 3x f32 std bits
std::array<unsigned char, 32> image_cache_key(
    const Options& options, const PreprocessSpec& spec,
    std::size_t image_count, std::size_t floats_per_image) {
    const std::array<unsigned char, 32> csv =
        sha256_file(options.sample_csv, "sample CSV");
    char self_path[4096] = {};
    const ssize_t self_length =
        readlink("/proc/self/exe", self_path, sizeof(self_path) - 1);
    bool have_self = self_length > 0;
    const std::array<unsigned char, 32> self =
        have_self ? sha256_file(std::string(self_path, static_cast<std::size_t>(
                                                          self_length)),
                                 "runner binary")
                  : std::array<unsigned char, 32>{};
    Sha256 hash;
    static const char kTag[] = "gpu-m2d g5 host image cache key v1";
    hash.update(kTag, sizeof(kTag) - 1);
    hash.update_u32(1);
    hash.update_u32(have_self ? 1u : 0u);
    if (have_self) {
        hash.update(self.data(), self.size());
    }
    hash.update(csv.data(), csv.size());
    hash.update_u64(image_count);
    hash.update_u64(floats_per_image);
    hash.update_u32(spec.canonical ? 1u : 0u);
    hash.update_u32(static_cast<uint32_t>(spec.resize_scale));
    hash.update_u32(static_cast<uint32_t>(spec.resize_interpolation));
    for (int i = 0; i < 3; ++i) {
        hash.update_float(spec.mean[static_cast<std::size_t>(i)]);
    }
    for (int i = 0; i < 3; ++i) {
        hash.update_float(spec.std[static_cast<std::size_t>(i)]);
    }
    return hash.digest();
}

// Returns true and fills `out` with the cached preprocessed buffer on a
// verified hit; false (out cleared) on any miss -- the caller then
// preprocesses fresh. Every failure path is a fallback, never an abort.
bool load_image_cache(const std::string& cache_dir,
                      const std::array<unsigned char, 32>& key,
                      std::size_t image_count, std::size_t floats_per_image,
                      std::vector<float>& out) {
    const std::string path = cache_dir + "/g5img_" + to_hex(key).substr(0, 16) +
                             ".bin";
    std::cout << "image-cache: key=" << to_hex(key) << '\n';
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cout << "image-cache: no cache file yet (" << path << ")\n";
        return false;
    }
    try {
        unsigned char magic[8];
        unsigned char file_key[32];
        unsigned char blob_sha[32];
        auto read_exact = [&file](void* dst, std::size_t size,
                                  const char* what) {
            file.read(static_cast<char*>(dst),
                      static_cast<std::streamsize>(size));
            if (file.gcount() != static_cast<std::streamsize>(size)) {
                throw std::runtime_error(std::string("truncated ") + what);
            }
        };
        read_exact(magic, sizeof(magic), "magic");
        if (std::memcmp(magic, kImageCacheMagic, sizeof(magic)) != 0) {
            throw std::runtime_error("bad magic");
        }
        uint32_t version = 0;
        read_exact(&version, sizeof(version), "version");
        if (version != kImageCacheVersion) {
            throw std::runtime_error("cache version mismatch");
        }
        read_exact(file_key, sizeof(file_key), "key");
        if (std::memcmp(file_key, key.data(), key.size()) != 0) {
            throw std::runtime_error("key mismatch");
        }
        uint64_t count = 0;
        uint64_t per_image = 0;
        read_exact(&count, sizeof(count), "image count");
        read_exact(&per_image, sizeof(per_image), "tensor size");
        if (count != image_count || per_image != floats_per_image) {
            throw std::runtime_error("count/tensor-size mismatch");
        }
        read_exact(blob_sha, sizeof(blob_sha), "blob checksum");
        const std::size_t total_bytes =
            image_count * floats_per_image * sizeof(float);
        out.assign(image_count * floats_per_image, 0.0F);
        Sha256 hasher;
        std::vector<char> chunk(1u << 20);
        std::size_t done = 0;
        while (done < total_bytes) {
            const std::size_t take = std::min(chunk.size(), total_bytes - done);
            read_exact(chunk.data(), take, "blob");
            hasher.update(chunk.data(), take);
            std::memcpy(reinterpret_cast<char*>(out.data()) + done, chunk.data(),
                        take);
            done += take;
        }
        file.get();
        if (!file.eof()) {
            throw std::runtime_error("trailing bytes after the blob");
        }
        const std::array<unsigned char, 32> have = hasher.digest();
        if (std::memcmp(have.data(), blob_sha, sizeof(blob_sha)) != 0) {
            throw std::runtime_error("blob checksum mismatch");
        }
        std::cout << "image-cache: HIT " << path << " (" << image_count
                  << " images, " << (total_bytes >> 20) << " MiB)\n";
        return true;
    } catch (const std::exception& error) {
        std::cout << "image-cache: miss (" << error.what()
                  << "), preprocessing fresh\n";
        out.clear();
        return false;
    }
}

// Best effort: a failure to write is reported and ignored (the next
// segment simply preprocesses fresh again).
void store_image_cache(const std::string& cache_dir,
                       const std::array<unsigned char, 32>& key,
                       std::size_t image_count, std::size_t floats_per_image,
                       const std::vector<float>& blob) {
    const std::string path = cache_dir + "/g5img_" + to_hex(key).substr(0, 16) +
                             ".bin";
    const std::string tmp = path + ".tmp." + std::to_string(::getpid());
    try {
        std::filesystem::create_directories(cache_dir);
        Sha256 hasher;
        hasher.update(blob.data(), blob.size() * sizeof(float));
        const std::array<unsigned char, 32> blob_sha = hasher.digest();
        std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
        if (!file) {
            throw std::runtime_error("cannot create " + tmp);
        }
        file.write(reinterpret_cast<const char*>(kImageCacheMagic),
                   sizeof(kImageCacheMagic));
        const uint32_t version = kImageCacheVersion;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        file.write(reinterpret_cast<const char*>(key.data()), key.size());
        const uint64_t count = image_count;
        const uint64_t per_image = floats_per_image;
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
        file.write(reinterpret_cast<const char*>(&per_image), sizeof(per_image));
        file.write(reinterpret_cast<const char*>(blob_sha.data()),
                   blob_sha.size());
        const std::size_t total = blob.size() * sizeof(float);
        const char* raw = reinterpret_cast<const char*>(blob.data());
        std::size_t done = 0;
        while (done < total) {
            const std::size_t take = std::min<std::size_t>(1u << 20, total - done);
            file.write(raw + done, static_cast<std::streamsize>(take));
            done += take;
        }
        file.flush();
        if (!file) {
            throw std::runtime_error("write failed on " + tmp);
        }
        file.close();
        std::filesystem::rename(tmp, path);
        std::cout << "image-cache: stored " << path << " (" << image_count
                  << " images, " << (total >> 20) << " MiB)\n";
    } catch (const std::exception& error) {
        std::error_code ignored;
        std::filesystem::remove(tmp, ignored);
        std::cout << "image-cache: store failed (" << error.what()
                  << "), continuing without a cache\n";
    }
}

gpu_m2d::DType to_gpu_m2d_dtype(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            return gpu_m2d::DType::kFloat32;
        case nvinfer1::DataType::kINT8:
            return gpu_m2d::DType::kInt8;
        case nvinfer1::DataType::kINT32:
            return gpu_m2d::DType::kInt32;
        default:
            throw std::invalid_argument("unsupported TensorRT binding dtype in G1.5");
    }
}

std::vector<std::size_t> shape_from_dims(const nvinfer1::Dims& dims) {
    if (dims.nbDims <= 0) {
        throw std::invalid_argument("scalar or invalid TensorRT binding shape");
    }
    std::vector<std::size_t> shape;
    shape.reserve(static_cast<std::size_t>(dims.nbDims));
    for (int index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] <= 0) {
            throw std::invalid_argument("dynamic TensorRT binding shape is unsupported in G1.5");
        }
        shape.push_back(static_cast<std::size_t>(dims.d[index]));
    }
    return shape;
}

std::size_t checked_element_count(const std::vector<std::size_t>& shape) {
    std::size_t count = 1;
    for (const std::size_t dimension : shape) {
        if (dimension > std::numeric_limits<std::size_t>::max() / count) {
            throw std::overflow_error("TensorRT binding element count overflow");
        }
        count *= dimension;
    }
    return count;
}

std::vector<BindingInfo> inspect_bindings(const nvinfer1::ICudaEngine& engine) {
    // G5 engines are implicit-batch (maxBatchSize contract); G7 explicit
    // Q/DQ engines carry the batch dim inside every binding shape. Both
    // expose the same data/prob/index contract, and for both the
    // per-binding element counts are checked against the runner's tensors
    // after inspection.
    if (engine.hasImplicitBatchDimension() && engine.getMaxBatchSize() < 1) {
        throw std::invalid_argument("implicit-batch engine has maxBatchSize < 1");
    }

    std::vector<BindingInfo> bindings;
    bindings.reserve(static_cast<std::size_t>(engine.getNbBindings()));
    for (int index = 0; index < engine.getNbBindings(); ++index) {
        BindingInfo binding;
        binding.index = index;
        binding.name = engine.getBindingName(index);
        binding.is_input = engine.bindingIsInput(index);
        binding.trt_dtype = engine.getBindingDataType(index);
        binding.dtype = to_gpu_m2d_dtype(binding.trt_dtype);
        binding.shape = shape_from_dims(engine.getBindingDimensions(index));
        binding.element_count = checked_element_count(binding.shape);
        binding.size_bytes = binding.element_count *
                             gpu_m2d::dtype_size_bytes(binding.dtype);
        bindings.push_back(std::move(binding));
    }
    return bindings;
}

const BindingInfo& find_binding(const std::vector<BindingInfo>& bindings,
                                const std::string& name) {
    const auto binding = std::find_if(
        bindings.begin(), bindings.end(),
        [&name](const BindingInfo& candidate) { return candidate.name == name; });
    if (binding == bindings.end()) {
        throw std::runtime_error("required binding is absent: " + name);
    }
    return *binding;
}

gpu_m2d::TensorDescriptor make_descriptor(
    const BindingInfo& binding,
    const gpu_m2d::DevicePointerInfo& pointer,
    int device) {
    return gpu_m2d::TensorDescriptor{
        binding.name,
        binding.is_input ? "input" : "output",
        binding.shape,
        {},
        binding.dtype,
        "TensorRT contiguous binding",
        pointer.gpu_va,
        pointer.allocation_size_bytes,
        "trt-binding-" + binding.name + "-gpu-" + std::to_string(device),
        "runner cudaMalloc to runner cudaFree",
    };
}

// G5 implicit-batch engines use enqueue(batchSize, ...); G7 explicit
// Q/DQ engines carry the batch dimension inside the binding shapes and
// use enqueueV2. Both paths feed the identical data/prob/index contract.
bool enqueue_inference(nvinfer1::IExecutionContext& context,
                       std::vector<void*>& binding_pointers,
                       cudaStream_t stream) {
    if (context.getEngine().hasImplicitBatchDimension()) {
        return context.enqueue(1, binding_pointers.data(), stream, nullptr);
    }
    return context.enqueueV2(binding_pointers.data(), stream, nullptr);
}

Prediction run_inference(nvinfer1::IExecutionContext& context,
                         std::vector<void*>& binding_pointers,
                         int probability_index,
                         int class_index,
                         CudaStream& stream,
                         std::int32_t class_count) {
    if (!enqueue_inference(context, binding_pointers, stream.get())) {
        throw std::runtime_error("TensorRT enqueue returned false");
    }

    Prediction prediction;
    check_cuda(cudaMemcpyAsync(&prediction.probability,
                               binding_pointers[probability_index],
                               sizeof(prediction.probability),
                               cudaMemcpyDeviceToHost, stream.get()),
               "copy probability to host");
    check_cuda(cudaMemcpyAsync(&prediction.class_index,
                               binding_pointers[class_index],
                               sizeof(prediction.class_index),
                               cudaMemcpyDeviceToHost, stream.get()),
               "copy class index to host");
    check_cuda(cudaStreamSynchronize(stream.get()), "synchronize inference stream");
    if (!std::isfinite(prediction.probability) || prediction.class_index < 0 ||
        prediction.class_index >= class_count) {
        throw std::runtime_error("TensorRT inference produced an invalid output");
    }
    return prediction;
}

std::string shape_string(const std::vector<std::size_t>& shape) {
    std::ostringstream output;
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            output << 'x';
        }
        output << shape[index];
    }
    return output.str();
}

std::string hex_address(std::uintptr_t address) {
    std::ostringstream output;
    output << "0x" << std::hex << address;
    return output.str();
}

void write_mapping_snapshot(const std::string& path,
                            const gpu_m2d::MappingSnapshot& snapshot,
                            int device) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write mapping snapshot: " + path);
    }
    output << "run_id,device,tensor_name,tensor_type,shape,dtype,layout,gpu_va,"
              "allocation_size_bytes,required_bytes,allocation_id,lifetime\n";
    for (const auto& tensor : snapshot.tensors()) {
        output << snapshot.run_id() << ',' << device << ',' << tensor.name << ','
               << tensor.tensor_type << ',' << shape_string(tensor.shape) << ','
               << gpu_m2d::dtype_name(tensor.dtype) << ',' << tensor.layout << ','
               << hex_address(tensor.allocation_base_gpu_va) << ','
               << tensor.allocation_size_bytes << ','
               << gpu_m2d::tensor_required_bytes(tensor) << ','
               << tensor.allocation_id << ',' << tensor.lifetime << '\n';
    }
}

void validate_registry_round_trips(
    const gpu_m2d::AllocationRegistry& registry) {
    for (const auto& allocation : registry.allocations()) {
        if (!allocation.active) {
            continue;
        }
        const std::array<std::size_t, 2> offsets{
            0, allocation.size_bytes - 1};
        const std::array<std::uint8_t, 2> bits{0, 7};
        for (std::size_t index = 0; index < offsets.size(); ++index) {
            const auto forward = registry.allocation_bit_to_gpu_va(
                allocation.allocation_id, offsets[index], bits[index]);
            const auto reverse = registry.gpu_va_to_allocation_bit(
                allocation.device_id, forward.gpu_va, forward.bit_in_byte);
            if (reverse.allocation_id != allocation.allocation_id ||
                reverse.byte_offset != offsets[index] ||
                reverse.bit_in_byte != bits[index]) {
                throw std::runtime_error(
                    "allocation registry forward/reverse mismatch");
            }
        }
    }
}

void write_allocation_registry(
    const std::string& path,
    const gpu_m2d::AllocationRegistry& registry) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write allocation registry: " + path);
    }
    output << "run_id,allocation_id,device,gpu_va,size_bytes,alignment_bytes,owner,"
              "allocation_phase,lifetime,active_at_injection,semantic_label\n";
    for (const auto& allocation : registry.allocations()) {
        output << registry.run_id() << ',' << allocation.allocation_id << ','
               << allocation.device_id << ','
               << hex_address(allocation.base_gpu_va) << ','
               << allocation.size_bytes << ',' << allocation.alignment_bytes << ','
               << allocation.owner << ',' << allocation.allocation_phase << ','
               << allocation.lifetime << ',' << (allocation.active ? 1 : 0) << ','
               << allocation.semantic_label << '\n';
    }
}

void write_result(const std::string& path,
                  const Options& options,
                  const Sample& sample,
                  const std::string& run_id,
                  const gpu_m2d::TensorBitAddress& mapped,
                  const gpu_m2d::BitFlipResult& flip,
                  const Prediction& clean,
                  const Prediction& injected) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write injection result: " + path);
    }

    const bool top1_changed = clean.class_index != injected.class_index;
    const bool numeric_changed = top1_changed ||
        std::fabs(clean.probability - injected.probability) > kProbabilityTolerance;
    const std::string outcome = top1_changed ? "SDC_TOP1" : "BENIGN_TOP1";

    output << "run_id,device,image,target,tensor_name,element_index,element_bit_index,"
              "byte_offset,bit_in_byte,xor_mask,gpu_va,before,after,clean_class,"
              "clean_probability,injected_class,injected_probability,top1_outcome,"
              "numeric_output_changed,clean_correct,injected_correct\n";
    output << run_id << ',' << options.device << ',' << sample.path << ','
           << sample.target << ',' << mapped.tensor_name << ','
           << mapped.element_index << ',' << mapped.element_bit_index << ','
           << mapped.byte_offset << ',' << static_cast<unsigned>(mapped.bit_in_byte)
           << ',' << static_cast<unsigned>(mapped.xor_mask) << ','
           << hex_address(mapped.gpu_va) << ',' << static_cast<unsigned>(flip.before)
           << ',' << static_cast<unsigned>(flip.after) << ',' << clean.class_index
           << ',' << std::setprecision(9) << clean.probability << ','
           << injected.class_index << ',' << injected.probability << ',' << outcome
           << ',' << (numeric_changed ? 1 : 0) << ','
           << (clean.class_index == sample.target ? 1 : 0) << ','
           << (injected.class_index == sample.target ? 1 : 0) << '\n';
}

// One row per reverse-chain XOR target of a G4-T2 run, plus the shared
// injected/sanity inference outcomes (repeated per row for self-contained
// CSV consumption).
struct G4TargetRecord {
    InjectionTarget target;
    std::string semantic_label;
    gpu_m2d::BitFlipResult flip;
    bool guard_bytes_unchanged{false};
    bool reverse_map_ok{false};
    bool restored_byte_ok{false};
    bool restore_guard_ok{false};
    std::vector<std::uint8_t> before_full;  // pristine pre-injection snapshot
};

void write_g4_result(const std::string& path, const Options& options,
                     const Sample& sample, const std::string& run_id,
                     const std::vector<G4TargetRecord>& records,
                     const Prediction& clean, const Prediction& injected,
                     const std::string& injected_outcome, bool injected_valid,
                     const Prediction& sanity) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write G4 injection result: " + path);
    }
    const bool sanity_matches_clean =
        sanity.class_index == clean.class_index &&
        std::fabs(sanity.probability - clean.probability) <= kSanityTolerance;
    output << "run_id,device,image,target,target_id,allocation_id,semantic_label,"
              "byte_offset,bit_in_byte,xor_mask,gpu_va,expected_gpu_va,before,after,"
              "guard_bytes_unchanged,reverse_map_ok,restored_byte_ok,restore_guard_ok,"
              "clean_class,clean_probability,injected_class,injected_probability,"
              "injected_outcome,sanity_class,sanity_probability,sanity_matches_clean\n";
    for (const G4TargetRecord& record : records) {
        output << run_id << ',' << options.device << ',' << sample.path << ','
               << sample.target << ',' << record.target.target_id << ','
               << record.target.allocation_id << ',' << record.semantic_label << ','
               << record.target.byte_offset << ',' << record.target.bit_in_byte << ','
               << static_cast<unsigned>(record.flip.xor_mask) << ','
               << hex_address(record.flip.gpu_va) << ','
               << hex_address(record.target.expected_gpu_va) << ','
               << static_cast<unsigned>(record.flip.before) << ','
               << static_cast<unsigned>(record.flip.after) << ','
               << (record.guard_bytes_unchanged ? 1 : 0) << ','
               << (record.reverse_map_ok ? 1 : 0) << ','
               << (record.restored_byte_ok ? 1 : 0) << ','
               << (record.restore_guard_ok ? 1 : 0) << ',' << clean.class_index << ','
               << std::setprecision(9) << clean.probability << ','
               << (injected_valid ? std::to_string(injected.class_index)
                                  : std::string("NA"))
               << ',';
        if (injected_valid) {
            output << injected.probability;
        } else {
            output << "NA";
        }
        output << ',' << injected_outcome << ',' << sanity.class_index << ','
               << sanity.probability << ',' << (sanity_matches_clean ? 1 : 0)
               << '\n';
    }
}

// ---------------------------------------------------------------------------
// G5 campaign mode (docs/G5_FAULT_MODEL.md §6)
// ---------------------------------------------------------------------------

// One fault-model site of one trial (same chain contract as the T2
// targets, plus its trial/event/site coordinates for the result join).
struct CampaignSite {
    InjectionTarget target;
    std::size_t trial_index{0};
    std::size_t event_index{0};
    std::size_t site_index{0};
};

// Campaign work file:
// "trial_index,event_index,site_index,target_id,allocation_id,byte_offset,
//  bit_in_byte,expected_gpu_va" grouped into consecutive trials starting
// at 0. Returns one site list per trial.
std::vector<std::vector<CampaignSite>> read_campaign_work(
    const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open campaign work file: " + path);
    }
    std::vector<std::vector<CampaignSite>> trials;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line.front() == '#' ||
            line.rfind("trial_index,", 0) == 0) {
            continue;
        }
        const std::vector<std::string> fields = split_csv_line(line);
        if (fields.size() != 8) {
            throw std::runtime_error("malformed campaign work row: " + line);
        }
        CampaignSite site;
        site.trial_index = std::stoull(fields[0]);
        site.event_index = std::stoull(fields[1]);
        site.site_index = std::stoull(fields[2]);
        site.target.target_id = fields[3];
        site.target.allocation_id = fields[4];
        site.target.byte_offset = std::stoull(fields[5]);
        site.target.bit_in_byte = std::stoul(fields[6]);
        site.target.expected_gpu_va = std::stoull(fields[7], nullptr, 16);
        if (site.target.target_id.empty() || site.target.allocation_id.empty() ||
            site.target.bit_in_byte > 7 || site.target.expected_gpu_va == 0) {
            throw std::runtime_error("invalid campaign work row: " + line);
        }
        // Trials must arrive contiguous from 0: each row either continues
        // the current trial or opens trials.size().
        if (site.trial_index != trials.size() &&
            (trials.empty() || site.trial_index != trials.size() - 1)) {
            throw std::runtime_error(
                "campaign work trials must be contiguous from 0 (row: " +
                line + ")");
        }
        if (site.trial_index == trials.size()) {
            trials.emplace_back();
        }
        trials.back().push_back(std::move(site));
    }
    if (trials.empty()) {
        throw std::runtime_error("campaign work file has no sites: " + path);
    }
    for (const std::vector<CampaignSite>& trial : trials) {
        if (trial.empty()) {
            throw std::runtime_error("campaign work has an empty trial");
        }
    }
    return trials;
}

std::vector<Sample> read_all_samples(const std::string& csv_path) {
    std::ifstream input(csv_path);
    if (!input) {
        throw std::runtime_error("cannot open sample CSV: " + csv_path);
    }
    std::vector<Sample> samples;
    std::string line;
    bool first_line = true;
    while (std::getline(input, line)) {
        // tolerate CRLF rows (see read_sample)
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (first_line) {
            first_line = false;
            if (line == "path,label") {
                continue;
            }
        }
        if (line.empty()) {
            continue;
        }
        const std::size_t comma = line.find(',');
        if (comma == std::string::npos) {
            throw std::runtime_error("invalid sample CSV row: " + line);
        }
        samples.push_back(Sample{line.substr(0, comma),
                                 std::stoi(line.substr(comma + 1))});
    }
    if (samples.empty()) {
        throw std::runtime_error("sample CSV has no rows: " + csv_path);
    }
    return samples;
}

// One evaluation-image outcome. An invalid numeric output is an honest DUE
// datum of the injected faults (not a tool failure); enqueue/copy failures
// still throw fail-closed.
struct ImageOutcome {
    bool valid{false};
    float probability{0.0F};
    std::int32_t class_index{-1};
};

// One evaluation image under the campaign's HELD faults. The runner
// re-stages the input bytes for every image, so faults inside the input
// binding are re-applied after the copy (write-through of a persistent
// fault for the pass); the engine rewrites the output bindings on every
// enqueue, so faults there are re-applied after the enqueue, before the
// readback. TRT-internal regions are never re-staged by the runner: their
// flips ride along until the engine itself rewrites the cell (weights are
// never rewritten; engine-owned scratch is honest soft-upset semantics).
ImageOutcome run_campaign_image(nvinfer1::IExecutionContext& context,
                                std::vector<void*>& binding_pointers,
                                int input_index, const float* input_host,
                                std::size_t input_bytes, int probability_index,
                                int class_index, CudaStream& stream,
                                std::int32_t class_count,
                                void* input_base, std::size_t input_size,
                                const std::vector<InjectionTarget>& input_sites,
                                void* probability_base,
                                std::size_t probability_size,
                                const std::vector<InjectionTarget>& probability_sites,
                                void* class_base, std::size_t class_size,
                                const std::vector<InjectionTarget>& class_sites) {
    check_cuda(cudaMemcpyAsync(binding_pointers[input_index], input_host,
                               input_bytes, cudaMemcpyHostToDevice,
                               stream.get()),
               "copy eval image to device");
    check_cuda(cudaStreamSynchronize(stream.get()),
               "synchronize input copy");
    for (const InjectionTarget& site : input_sites) {
        static_cast<void>(gpu_m2d::flip_device_bit(
            input_base, input_size, site.byte_offset,
            static_cast<std::uint8_t>(site.bit_in_byte)));
    }
    if (!enqueue_inference(context, binding_pointers, stream.get())) {
        throw std::runtime_error("TensorRT enqueue returned false");
    }
    check_cuda(cudaStreamSynchronize(stream.get()),
               "synchronize inference stream");
    for (const InjectionTarget& site : probability_sites) {
        static_cast<void>(gpu_m2d::flip_device_bit(
            probability_base, probability_size, site.byte_offset,
            static_cast<std::uint8_t>(site.bit_in_byte)));
    }
    for (const InjectionTarget& site : class_sites) {
        static_cast<void>(gpu_m2d::flip_device_bit(
            class_base, class_size, site.byte_offset,
            static_cast<std::uint8_t>(site.bit_in_byte)));
    }

    ImageOutcome outcome;
    check_cuda(cudaMemcpyAsync(&outcome.probability,
                               binding_pointers[probability_index],
                               sizeof(outcome.probability),
                               cudaMemcpyDeviceToHost, stream.get()),
               "copy probability to host");
    check_cuda(cudaMemcpyAsync(&outcome.class_index,
                               binding_pointers[class_index],
                               sizeof(outcome.class_index),
                               cudaMemcpyDeviceToHost, stream.get()),
               "copy class index to host");
    check_cuda(cudaStreamSynchronize(stream.get()), "synchronize readback");
    outcome.valid = std::isfinite(outcome.probability) &&
                    outcome.class_index >= 0 &&
                    outcome.class_index < class_count;
    return outcome;
}

// Per-site result row. guard_bytes_unchanged comes from the trial's
// allocation-level pre-pass guard compare; restored_byte_ok means "the
// cell was not rewritten between flip and restore" (unflip.after ==
// flip.before) -- legitimately false for cells the engine owns and
// rewrites during the pass (output bindings, scratch); restore_check
// states which restore verification applied to the site's allocation:
//   exact                        full allocation compared byte-exact
//   mismatch:N                   informational TRT-internal mismatch (the
//                                engine rewrote scratch/scratch-adjacent
//                                bytes; sanity inference is the residue
//                                proof)
//   skipped:engine-owned-output   output binding: the engine rewrites the
//                                cell every enqueue, no stable baseline
struct G5SiteRecord {
    CampaignSite site;
    std::string semantic_label;
    gpu_m2d::BitFlipResult flip;
    bool reverse_map_ok{false};
    bool alloc_guard_ok{false};
    bool restored_byte_ok{false};
    std::string restore_check{"pending"};
};

// One trial's summary row (site flips are joined per site in
// g5_site_result.csv; per-image rows live in g5_image_detail.csv).
struct G5TrialRecord {
    std::size_t trial_index{0};
    std::size_t site_count{0};
    std::size_t event_count{0};
    std::size_t images_total{0};
    std::size_t images_evaluated{0};
    std::size_t images_benign{0};
    std::size_t images_sdc_numeric{0};
    std::size_t images_sdc_top1{0};
    std::size_t images_invalid{0};
    std::string injected_outcome;
    std::int32_t sanity_class{-1};
    float sanity_probability{0.0F};
    bool sanity_matches_clean{false};
    std::size_t restore_alloc_exact{0};
    std::size_t restore_alloc_mismatch{0};
    std::size_t restore_alloc_skipped{0};
    std::size_t restore_mismatch_bytes{0};
};

// One evaluation image of one trial (the DUE abort marks the remaining
// images of the trial evaluated=0/IMAGE_DUE).
struct G5ImageRow {
    std::size_t trial_index{0};
    std::size_t image_index{0};
    bool evaluated{false};
    std::int32_t clean_class{-1};
    float clean_probability{0.0F};
    bool have_injected{false};
    std::int32_t injected_class{-1};
    float injected_probability{0.0F};
    const char* outcome{""};
};

// The three campaign result CSVs are written INCREMENTALLY: the streams
// open before the first trial and every COMPLETED trial's rows are
// appended and flushed, so a process-fatal fault (a flip in TRT runtime
// control state surfaces as a CUDA illegal memory access) that kills the
// runner mid-trial cannot lose the completed trials' data. The
// orchestrator's restart protocol resumes the level in a fresh process,
// counts the dying trial as PROCESS_FATAL, and merges the segments -- so
// a trial is on disk iff it reached TRIAL_END.
struct G5ResultStreams {
    std::ofstream site;
    std::ofstream trial;
    std::ofstream image;
    std::size_t flushed_sites{0};
    std::size_t flushed_trials{0};
    std::size_t flushed_images{0};
    std::string run_id;
    int device{0};

    G5ResultStreams(const std::string& prefix, const Options& options,
                    const std::string& run_id_)
        : run_id(run_id_), device(options.device) {
        const std::string site_path = prefix + "_g5_site_result.csv";
        const std::string trial_path = prefix + "_g5_trial_result.csv";
        const std::string image_path = prefix + "_g5_image_detail.csv";
        site.open(site_path, std::ios::out | std::ios::trunc);
        trial.open(trial_path, std::ios::out | std::ios::trunc);
        image.open(image_path, std::ios::out | std::ios::trunc);
        if (!site || !trial || !image) {
            throw std::runtime_error("cannot open G5 result CSVs at " +
                                     prefix);
        }
        site << "run_id,device,trial_index,event_index,site_index,target_id,"
                "allocation_id,semantic_label,byte_offset,bit_in_byte,xor_mask,"
                "gpu_va,expected_gpu_va,before,after,guard_bytes_unchanged,"
                "reverse_map_ok,restored_byte_ok,restore_check\n";
        trial << "run_id,device,trial_index,site_count,event_count,images_total,"
                 "images_evaluated,images_benign,images_sdc_numeric,"
                 "images_sdc_top1,images_invalid,injected_outcome,sanity_class,"
                 "sanity_probability,sanity_matches_clean,restore_alloc_exact,"
                 "restore_alloc_mismatch,restore_alloc_skipped,"
                 "restore_mismatch_bytes\n"
              << std::setprecision(9);
        image << "run_id,device,trial_index,image_index,evaluated,clean_class,"
                 "injected_class,clean_probability,injected_probability,outcome\n"
              << std::setprecision(9);
    }

    void append_site(const G5SiteRecord& record) {
        site << run_id << ',' << device << ','
             << record.site.trial_index << ',' << record.site.event_index
             << ',' << record.site.site_index << ','
             << record.site.target.target_id << ','
             << record.site.target.allocation_id << ','
             << record.semantic_label << ','
             << record.site.target.byte_offset << ','
             << record.site.target.bit_in_byte << ','
             << static_cast<unsigned>(record.flip.xor_mask) << ','
             << hex_address(record.flip.gpu_va) << ','
             << hex_address(record.site.target.expected_gpu_va) << ','
             << static_cast<unsigned>(record.flip.before) << ','
             << static_cast<unsigned>(record.flip.after) << ','
             << (record.alloc_guard_ok ? 1 : 0) << ','
             << (record.reverse_map_ok ? 1 : 0) << ','
             << (record.restored_byte_ok ? 1 : 0) << ','
             << record.restore_check << '\n';
    }

    void append_trial(const G5TrialRecord& record) {
        trial << run_id << ',' << device << ','
              << record.trial_index << ',' << record.site_count << ','
              << record.event_count << ',' << record.images_total << ','
              << record.images_evaluated << ',' << record.images_benign
              << ',' << record.images_sdc_numeric << ','
              << record.images_sdc_top1 << ',' << record.images_invalid
              << ',' << record.injected_outcome << ','
              << record.sanity_class << ',' << record.sanity_probability
              << ',' << (record.sanity_matches_clean ? 1 : 0) << ','
              << record.restore_alloc_exact << ','
              << record.restore_alloc_mismatch << ','
              << record.restore_alloc_skipped << ','
              << record.restore_mismatch_bytes << '\n';
    }

    void append_image(const G5ImageRow& row) {
        image << run_id << ',' << device << ',' << row.trial_index
              << ',' << row.image_index << ',' << (row.evaluated ? 1 : 0)
              << ',' << row.clean_class << ','
              << (row.have_injected ? std::to_string(row.injected_class)
                                    : std::string("NA"))
              << ',' << row.clean_probability << ',';
        if (row.have_injected) {
            image << row.injected_probability;
        } else {
            image << "NA";
        }
        image << ',' << row.outcome << '\n';
    }

    // Append every not-yet-flushed record (exactly one completed trial's
    // worth) and flush all three streams to the OS: after this call the
    // data survives the death of this process.
    void flush_trial(const std::vector<G5SiteRecord>& sites,
                     const std::vector<G5TrialRecord>& trials,
                     const std::vector<G5ImageRow>& images) {
        for (std::size_t index = flushed_sites; index < sites.size();
             ++index) {
            append_site(sites[index]);
        }
        for (std::size_t index = flushed_trials; index < trials.size();
             ++index) {
            append_trial(trials[index]);
        }
        for (std::size_t index = flushed_images; index < images.size();
             ++index) {
            append_image(images[index]);
        }
        flushed_sites = sites.size();
        flushed_trials = trials.size();
        flushed_images = images.size();
        site.flush();
        trial.flush();
        image.flush();
        if (!site || !trial || !image) {
            throw std::runtime_error("flushing G5 result CSVs failed");
        }
    }
};

void write_g5_clean_pass(const std::string& path,
                         const std::vector<Sample>& samples,
                         const std::vector<Prediction>& clean) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write G5 clean pass: " + path);
    }
    output << "image_index,path,label,clean_class,clean_probability\n"
           << std::setprecision(9);
    for (std::size_t index = 0; index < samples.size(); ++index) {
        output << index << ',' << samples[index].path << ','
               << samples[index].target << ',' << clean[index].class_index
               << ',' << clean[index].probability << '\n';
    }
}

}  // namespace

// ---- G8 L2 residency probe pass (docs/G8_CACHE_FAULT_PLAN.md §4) ----
// FAULTLESS pass(es) over every evaluation image after the strict clean
// pass. Every `l2_probe_every` images, at the image boundary (before the
// image's input is staged), one probe sweep classifies the probed units of
// every registered allocation as L2-resident or not (stride > 1: a
// staggered 1/stride subset per sweep; alternate: each unit's successive
// observations flip probe direction). Each inference is timed with CUDA
// events around the enqueue (probe time excluded: it runs between images).
// Every output must equal the clean pass bit-for-bit -- the probe is
// read-only, so any mismatch is a neutrality failure.
// --l2-probe-passes N repeats the pass N times in the SAME process (G8-T1
// same-process stability); with N > 1 each pass writes under PREFIX_p<i>.
// Outputs per pass (prefix = --l2-probe-out [+ _p<i>]):
//   _ranges.csv      probed allocations (id, VA, bytes, units, label)
//   _images.csv      per image: inference ms, bit-identical flags
//   _boundaries.csv  per probe sweep x allocation: units, probed, L2 hits
//   _hist.csv        per allocation: latency histogram over all sweeps
//                    (16-cycle bins) -- in-situ check of the T0 threshold
//   with --l2-probe-map 1 (G8-T1 residency map):
//   _residency.bin   per-unit residency periods + per-image inference
//                    times + per-unit resident bytes (ResidencyMapBuilder)
//   _residency.json  metadata, ranges, T_total, R_eff_bits, bin sha256
std::string json_escape(const std::string& value) {
    std::string out;
    for (char c : value) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
            out += buf;
        } else {
            out += c;
        }
    }
    return out;
}

int run_l2_probe_pass(const Options& options,
                      nvinfer1::IExecutionContext& context,
                      std::vector<void*>& binding_pointers,
                      const BindingInfo& input_binding,
                      const BindingInfo& probability_binding,
                      const BindingInfo& class_binding,
                      CudaStream& stream,
                      const std::vector<Sample>& samples,
                      const std::vector<float>& inputs,
                      const std::vector<Prediction>& clean,
                      const gpu_m2d::AllocationRegistry& registry,
                      ObserverEmitter& observer) {
    std::vector<gpu_m2d::AllocationDescriptor> active;
    for (const gpu_m2d::AllocationDescriptor& d : registry.allocations()) {
        if (d.active) {
            active.push_back(d);
        }
    }
    std::sort(active.begin(), active.end(),
              [](const auto& a, const auto& b) { return a.base_gpu_va < b.base_gpu_va; });
    std::vector<gpu_m2d::L2ProbeRange> ranges;
    for (const gpu_m2d::AllocationDescriptor& d : active) {
        ranges.push_back({d.allocation_id,
                          reinterpret_cast<const void*>(d.base_gpu_va),
                          d.size_bytes});
    }
    gpu_m2d::L2Prober prober(ranges, options.l2_probe_unit,
                             options.l2_probe_threshold,
                             options.l2_probe_per_sm,
                             options.l2_probe_stride, options.l2_probe_reverse,
                             options.l2_probe_alternate);
    const std::vector<std::uint16_t> unit_bytes = prober.unit_resident_bytes();
    std::uint64_t surface_bits = 0;
    for (std::uint16_t b : unit_bytes) {
        surface_bits += 8ull * b;
    }
    std::vector<std::uint64_t> range_first(prober.range_count(), 0);
    for (std::size_t i = 1; i < prober.range_count(); ++i) {
        range_first[i] = range_first[i - 1] + prober.range_units(i - 1);
    }
    const std::uint64_t window =
        static_cast<std::uint64_t>(options.l2_probe_every) * prober.stride();

    constexpr std::size_t kBinCycles = 16;
    constexpr std::size_t kBins = 4096 / kBinCycles;  // last bin = overflow
    cudaEvent_t infer_start = nullptr;
    cudaEvent_t infer_stop = nullptr;
    check_cuda(cudaEventCreate(&infer_start), "create inference start event");
    check_cuda(cudaEventCreate(&infer_stop), "create inference stop event");
    void* input_device = binding_pointers[input_binding.index];
    std::size_t mismatches_all = 0;

    for (std::size_t pass = 0; pass < options.l2_probe_passes; ++pass) {
        const std::string prefix =
            options.l2_probe_passes > 1
                ? options.l2_probe_out + "_p" + std::to_string(pass)
                : options.l2_probe_out;
        {
            std::ofstream out(prefix + "_ranges.csv");
            if (!out) {
                throw std::runtime_error("cannot write " + prefix + "_ranges.csv");
            }
            out << "allocation_id,gpu_va,size_bytes,units,unit_bytes,"
                   "allocation_phase,semantic_label\n";
            for (std::size_t i = 0; i < prober.range_count(); ++i) {
                out << active[i].allocation_id << ','
                    << hex_address(active[i].base_gpu_va) << ','
                    << active[i].size_bytes << ',' << prober.range_units(i) << ','
                    << prober.unit_bytes() << ',' << active[i].allocation_phase
                    << ',' << active[i].semantic_label << '\n';
            }
        }
        std::ofstream images_csv(prefix + "_images.csv");
        std::ofstream boundaries_csv(prefix + "_boundaries.csv");
        if (!images_csv || !boundaries_csv) {
            throw std::runtime_error("cannot write L2 probe outputs: " + prefix);
        }
        images_csv << "image_index,infer_ms,probability_bit_identical,"
                      "class_identical\n";
        boundaries_csv << "image_index,probe_ms,reverse,allocation_id,units,"
                          "probed,l2_hits\n";
        std::vector<std::vector<std::uint64_t>> histogram(
            prober.range_count(), std::vector<std::uint64_t>(kBins, 0));
        std::unique_ptr<gpu_m2d::ResidencyMapBuilder> map;
        if (options.l2_probe_map) {
            map = std::make_unique<gpu_m2d::ResidencyMapBuilder>(
                prober.total_units(), static_cast<std::uint32_t>(samples.size()));
        }
        std::vector<double> image_ms(samples.size(), 0.0);

        observer.event("L2_PROBE_PASS_BEGIN");
        std::size_t mismatches = 0;
        std::size_t sweeps = 0;
        double probe_ms_total = 0.0;
        for (std::size_t image = 0; image < samples.size(); ++image) {
            if (image % options.l2_probe_every == 0) {
                const float probe_ms = prober.probe(stream.get(), sweeps);
                probe_ms_total += probe_ms;
                ++sweeps;
                const std::vector<std::uint64_t> hits = prober.hits_per_range();
                const std::vector<std::uint64_t> probed = prober.probed_per_range();
                const std::vector<std::uint16_t>& latency = prober.latencies();
                for (std::size_t r = 0; r < prober.range_count(); ++r) {
                    boundaries_csv << image << ',' << probe_ms << ','
                                   << (prober.last_reverse() ? 1 : 0) << ','
                                   << prober.range(r).id << ','
                                   << prober.range_units(r) << ',' << probed[r]
                                   << ',' << hits[r] << '\n';
                }
                std::size_t r = 0;
                for (std::uint64_t u = prober.last_phase(); u < prober.total_units();
                     u += prober.stride()) {
                    while (r + 1 < prober.range_count() && u >= range_first[r + 1]) {
                        ++r;
                    }
                    histogram[r][std::min<std::size_t>(latency[u] / kBinCycles,
                                                       kBins - 1)]++;
                    if (map) {
                        map->observe(u, static_cast<std::uint32_t>(image),
                                     latency[u] < prober.threshold_cycles());
                    }
                }
            }
            check_cuda(cudaMemcpyAsync(input_device,
                                       &inputs[image * input_binding.element_count],
                                       input_binding.size_bytes,
                                       cudaMemcpyHostToDevice, stream.get()),
                       "copy probe-pass image to device");
            check_cuda(cudaEventRecord(infer_start, stream.get()),
                       "record inference start");
            if (!enqueue_inference(context, binding_pointers, stream.get())) {
                throw std::runtime_error("TensorRT enqueue returned false");
            }
            check_cuda(cudaEventRecord(infer_stop, stream.get()),
                       "record inference stop");
            Prediction got;
            check_cuda(cudaMemcpyAsync(&got.probability,
                                       binding_pointers[probability_binding.index],
                                       sizeof(got.probability),
                                       cudaMemcpyDeviceToHost, stream.get()),
                       "copy probability to host");
            check_cuda(cudaMemcpyAsync(&got.class_index,
                                       binding_pointers[class_binding.index],
                                       sizeof(got.class_index),
                                       cudaMemcpyDeviceToHost, stream.get()),
                       "copy class index to host");
            check_cuda(cudaStreamSynchronize(stream.get()), "synchronize probe pass");
            float infer_ms = 0.0F;
            check_cuda(cudaEventElapsedTime(&infer_ms, infer_start, infer_stop),
                       "inference elapsed time");
            image_ms[image] = infer_ms;
            const bool prob_same = std::memcmp(&got.probability,
                                               &clean[image].probability,
                                               sizeof(float)) == 0;
            const bool class_same = got.class_index == clean[image].class_index;
            if (!prob_same || !class_same) {
                ++mismatches;
            }
            images_csv << image << ',' << infer_ms << ',' << (prob_same ? 1 : 0)
                       << ',' << (class_same ? 1 : 0) << '\n';
        }
        observer.event("L2_PROBE_PASS_END");
        mismatches_all += mismatches;

        {
            std::ofstream out(prefix + "_hist.csv");
            if (!out) {
                throw std::runtime_error("cannot write " + prefix + "_hist.csv");
            }
            out << "allocation_id,bin_lo_cycles,count\n";
            for (std::size_t r = 0; r < prober.range_count(); ++r) {
                for (std::size_t b = 0; b < kBins; ++b) {
                    if (histogram[r][b] != 0) {
                        out << prober.range(r).id << ',' << b * kBinCycles << ','
                            << histogram[r][b] << '\n';
                    }
                }
            }
        }
        double infer_ms_total = 0.0;
        for (double ms : image_ms) {
            infer_ms_total += ms;
        }
        double r_eff_bits = -1.0;
        std::uint64_t periods = 0;
        if (map) {
            map->finish();
            r_eff_bits = map->r_eff_bits(image_ms, unit_bytes);
            periods = map->periods().size();
            const std::string bin_path = prefix + "_residency.bin";
            map->write_binary(bin_path, static_cast<std::uint32_t>(prober.unit_bytes()),
                              image_ms, unit_bytes);
            const std::string bin_sha =
                to_hex(sha256_file(bin_path, "residency map"));
            std::ofstream json(prefix + "_residency.json");
            if (!json) {
                throw std::runtime_error("cannot write " + prefix + "_residency.json");
            }
            json.precision(17);
            json << "{\n"
                 << "  \"schema\": \"gpu-m2d.g8.residency-map.v1\",\n"
                 << "  \"bin_file\": \"" << json_escape(bin_path) << "\",\n"
                 << "  \"bin_sha256\": \"" << bin_sha << "\",\n"
                 << "  \"engine\": \"" << json_escape(options.engine_path) << "\",\n"
                 << "  \"sample_csv\": \"" << json_escape(options.sample_csv) << "\",\n"
                 << "  \"device\": " << options.device << ",\n"
                 << "  \"pass_index\": " << pass << ",\n"
                 << "  \"passes_in_process\": " << options.l2_probe_passes << ",\n"
                 << "  \"unit_bytes\": " << prober.unit_bytes() << ",\n"
                 << "  \"probe_every\": " << options.l2_probe_every << ",\n"
                 << "  \"stride\": " << prober.stride() << ",\n"
                 << "  \"alternate\": " << (prober.alternate() ? "true" : "false") << ",\n"
                 << "  \"observation_window_images\": " << window << ",\n"
                 << "  \"threshold_cycles\": " << prober.threshold_cycles() << ",\n"
                 << "  \"probes_per_sm\": " << options.l2_probe_per_sm << ",\n"
                 << "  \"images\": " << samples.size() << ",\n"
                 << "  \"units\": " << prober.total_units() << ",\n"
                 << "  \"observed_units\": " << map->observed_units() << ",\n"
                 << "  \"periods\": " << periods << ",\n"
                 << "  \"sweeps\": " << sweeps << ",\n"
                 << "  \"neutral_mismatches\": " << mismatches << ",\n"
                 << "  \"surface_bits\": " << surface_bits << ",\n"
                 << "  \"t_total_ms\": " << infer_ms_total << ",\n"
                 << "  \"r_eff_bits\": " << r_eff_bits << ",\n"
                 << "  \"ranges\": [\n";
            for (std::size_t i = 0; i < prober.range_count(); ++i) {
                json << "    {\"allocation_id\": \""
                     << json_escape(active[i].allocation_id) << "\", \"gpu_va\": \""
                     << hex_address(active[i].base_gpu_va) << "\", \"size_bytes\": "
                     << active[i].size_bytes << ", \"first_unit\": "
                     << range_first[i] << ", \"units\": " << prober.range_units(i)
                     << ", \"allocation_phase\": \""
                     << json_escape(active[i].allocation_phase)
                     << "\", \"semantic_label\": \""
                     << json_escape(active[i].semantic_label) << "\"}"
                     << (i + 1 < prober.range_count() ? "," : "") << '\n';
            }
            json << "  ]\n}\n";
        }
        const double images = static_cast<double>(samples.size());
        std::cout << (mismatches == 0 ? "GPU_M2D_L2_PROBE_PASS"
                                      : "GPU_M2D_L2_PROBE_FAIL")
                  << " pass=" << pass << " images=" << samples.size()
                  << " mismatches=" << mismatches << " sweeps=" << sweeps
                  << " every=" << options.l2_probe_every
                  << " unit=" << prober.unit_bytes()
                  << " units=" << prober.total_units()
                  << " ranges=" << prober.range_count()
                  << " threshold=" << prober.threshold_cycles()
                  << " stride=" << prober.stride()
                  << " reverse=" << (prober.reverse() ? 1 : 0)
                  << " alternate=" << (prober.alternate() ? 1 : 0)
                  << " infer_ms_total=" << infer_ms_total
                  << " infer_ms_mean=" << infer_ms_total / images
                  << " probe_ms_mean=" << (sweeps ? probe_ms_total / sweeps : 0.0);
        if (map) {
            std::cout << " periods=" << periods << " r_eff_bits=" << r_eff_bits
                      << " surface_bits=" << surface_bits;
        }
        std::cout << '\n';
    }
    cudaEventDestroy(infer_start);
    cudaEventDestroy(infer_stop);
    return mismatches_all == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    try {
        std::setvbuf(stdout, nullptr, _IOLBF, 0);
        const Options options = parse_options(argc, argv);

        ObserverEmitter observer;
        observer.enabled = !options.observer_gate.empty();
        observer.gate_file = options.observer_gate;
        observer.hold_seconds = options.hold_seconds;
        observer.gate_timeout_seconds = options.gate_timeout_seconds;

        // CPU-only preparation happens before the gate so the gated window
        // contains nothing but CUDA work.
        PreprocessSpec preprocess_spec;
        preprocess_spec.canonical = options.preprocess_mode == "canonical";
        preprocess_spec.resize_scale = options.resize_scale;
        preprocess_spec.resize_interpolation = options.resize_interpolation;
        preprocess_spec.mean = options.canonical_mean;
        preprocess_spec.std = options.canonical_std;
        const Sample sample = read_sample(options.sample_csv, options.sample_index);
        const std::vector<float> input =
            preprocess(sample.path, preprocess_spec);
        // Preprocessing parity escape hatch: dump the sample-index image's
        // tensor and exit before any engine load or CUDA call, so the
        // canonical C++ port can be diffed bit-exactly against the python
        // contract offline.
        if (!options.dump_preprocessed.empty()) {
            std::ofstream dump(options.dump_preprocessed, std::ios::binary);
            if (!dump) {
                throw std::runtime_error("cannot write preprocessed dump: " +
                                         options.dump_preprocessed);
            }
            dump.write(reinterpret_cast<const char*>(input.data()),
                       static_cast<std::streamsize>(input.size() *
                                                    sizeof(float)));
            std::cout << "GPU_M2D_PREPROCESS_DUMP_PASS"
                      << " image=" << sample.path
                      << " floats=" << input.size() << '\n';
            return 0;
        }
        const std::vector<char> engine_bytes = read_binary_file(options.engine_path);
        // G5/G7 campaign: stage every evaluation image's preprocessed input
        // on the host BEFORE the gate (CPU-only work; the gated window
        // stays CUDA-only). ~600 MB host for the G5 1000-image pass,
        // ~5.6 GiB for the G7 10000-image pass. With --image-cache-dir the
        // buffer comes from the on-disk cache when its key matches (the
        // restart protocol's relaunches then skip the ~3 min re-preprocess).
        std::vector<Sample> campaign_samples;
        std::vector<float> campaign_input;
        const bool full_pass_mode =
            !options.campaign_work.empty() || !options.l2_probe_out.empty();
        if (full_pass_mode) {
            campaign_samples = read_all_samples(options.sample_csv);
            if (options.sample_index >= campaign_samples.size()) {
                throw std::out_of_range(
                    "sanity sample index is outside the campaign CSV");
            }
            const std::size_t floats_per_image = input.size();
            bool from_cache = false;
            std::array<unsigned char, 32> cache_key{};
            if (!options.image_cache_dir.empty()) {
                cache_key = image_cache_key(options, preprocess_spec,
                                            campaign_samples.size(),
                                            floats_per_image);
                from_cache = load_image_cache(options.image_cache_dir, cache_key,
                                              campaign_samples.size(),
                                              floats_per_image, campaign_input);
            }
            if (!from_cache) {
                campaign_input.reserve(campaign_samples.size() * floats_per_image);
                for (const Sample& eval_sample : campaign_samples) {
                    const std::vector<float> staged =
                        preprocess(eval_sample.path, preprocess_spec);
                    campaign_input.insert(campaign_input.end(), staged.begin(),
                                          staged.end());
                }
                if (!options.image_cache_dir.empty()) {
                    store_image_cache(options.image_cache_dir, cache_key,
                                      campaign_samples.size(), floats_per_image,
                                      campaign_input);
                }
            }
        }
        const std::string run_id =
            "g1_5-gpu" + std::to_string(options.device) + "-sample" +
            std::to_string(options.sample_index) + "-element" +
            std::to_string(options.element_index) + "-bit" +
            std::to_string(options.element_bit_index);

        // The stale-gate check must run before WAIT_PRE_ALLOC_GATE is
        // printed: the orchestrator creates the gate as soon as it sees
        // that marker, so checking afterwards races with it.
        if (observer.enabled && access(observer.gate_file.c_str(), F_OK) == 0) {
            throw std::runtime_error("observer gate already exists: " +
                                     observer.gate_file);
        }
        // Same race discipline for the T2 release file: the orchestrator
        // creates it only after the work file is complete, so a file that
        // exists this early is stale from an earlier attempt.
        if (!options.injection_release.empty() &&
            access(options.injection_release.c_str(), F_OK) == 0) {
            throw std::runtime_error("injection release file already exists: " +
                                     options.injection_release);
        }
        // And the same discipline for the G5 campaign release file.
        if (!options.campaign_release.empty() &&
            access(options.campaign_release.c_str(), F_OK) == 0) {
            throw std::runtime_error("campaign release file already exists: " +
                                     options.campaign_release);
        }
        observer.event("PROCESS_READY");
        observer.event("WAIT_PRE_ALLOC_GATE");
        if (observer.enabled) {
            std::fflush(stdout);
            if (!observer.wait_for_gate()) {
                throw std::runtime_error("observer pre-allocation gate timed out");
            }
        }
        observer.event("PRE_ALLOC_GATE_OPEN");

        observer.event("CONTEXT_BEGIN");
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (options.device < 0 || options.device >= device_count) {
            throw std::out_of_range("requested CUDA device does not exist");
        }
        check_cuda(cudaSetDevice(options.device), "cudaSetDevice");

        cudaDeviceProp device_properties{};
        check_cuda(cudaGetDeviceProperties(&device_properties, options.device),
                   "cudaGetDeviceProperties");
        char pci_bus_id[32]{};
        check_cuda(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), options.device),
                   "cudaDeviceGetPCIBusId");
        observer.event("CONTEXT_READY");

        observer.event("RUNTIME_BEGIN");
        TrtLogger logger;
        gpu_m2d::AllocationRegistry allocation_registry(run_id);
        TrackingGpuAllocator allocator(allocation_registry, options.device, &observer);
        std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> runtime(
            nvinfer1::createInferRuntime(logger));
        if (!runtime) {
            throw std::runtime_error("createInferRuntime returned null");
        }
        runtime->setGpuAllocator(&allocator);

        allocator.set_phase("deserialize_engine");
        std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> engine(
            runtime->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size()));
        if (!engine) {
            throw std::runtime_error("TensorRT engine deserialization failed");
        }

        allocator.set_phase("create_execution_context");
        std::unique_ptr<nvinfer1::IExecutionContext,
                        TrtDeleter<nvinfer1::IExecutionContext>> context(
            engine->createExecutionContext());
        if (!context) {
            throw std::runtime_error("TensorRT execution context creation failed");
        }

        const std::vector<BindingInfo> binding_info = inspect_bindings(*engine);
        const BindingInfo& input_binding = find_binding(binding_info, "data");
        const BindingInfo& probability_binding = find_binding(binding_info, "prob");
        const BindingInfo& class_binding = find_binding(binding_info, "index");
        if (!input_binding.is_input || probability_binding.is_input ||
            class_binding.is_input || input_binding.dtype != gpu_m2d::DType::kFloat32 ||
            probability_binding.dtype != gpu_m2d::DType::kFloat32 ||
            class_binding.dtype != gpu_m2d::DType::kInt32 ||
            input_binding.element_count != input.size() ||
            probability_binding.element_count != 1 || class_binding.element_count != 1) {
            throw std::runtime_error("unexpected ResNet-50 binding contract");
        }

        std::vector<DeviceBuffer> buffers;
        buffers.reserve(binding_info.size());
        std::vector<void*> binding_pointers(binding_info.size(), nullptr);
        gpu_m2d::MappingSnapshot mapping(run_id);
        for (const BindingInfo& binding : binding_info) {
            const std::string allocation_id =
                "trt-binding-" + binding.name + "-gpu-" +
                std::to_string(options.device);
            buffers.emplace_back(binding.size_bytes, allocation_registry,
                                 allocation_id, options.device,
                                 "TENSOR:" + binding.name);
            binding_pointers[static_cast<std::size_t>(binding.index)] =
                buffers.back().get();
            const auto pointer = gpu_m2d::inspect_device_pointer(
                buffers.back().get(), buffers.back().size_bytes());
            if (pointer.device_id != options.device) {
                throw std::runtime_error("TensorRT binding allocated on unexpected GPU");
            }
            mapping.add_tensor(make_descriptor(binding, pointer, options.device));
            observer.allocated(allocation_id, "cudaMalloc-binding",
                               reinterpret_cast<std::uintptr_t>(buffers.back().get()),
                               buffers.back().size_bytes(), "allocate_bindings",
                               "TENSOR:" + binding.name,
                               query_buffer_id(buffers.back().get()));
        }
        validate_registry_round_trips(allocation_registry);

        write_mapping_snapshot(options.output_prefix + "_mapping.csv", mapping,
                               options.device);
        observer.event("BINDINGS_READY");

        CudaStream stream;
        Prediction clean{};
        std::vector<Prediction> campaign_clean;
        if (!full_pass_mode) {
            allocator.set_phase("clean_inference");
            observer.event("CLEAN_INFERENCE_BEGIN");
            check_cuda(cudaMemcpy(binding_pointers[input_binding.index], input.data(),
                                  input_binding.size_bytes, cudaMemcpyHostToDevice),
                       "copy clean input to device");
            clean = run_inference(
                *context, binding_pointers, probability_binding.index,
                class_binding.index, stream, options.class_count);
            observer.event("CLEAN_INFERENCE_END");
        } else {
            // G5 clean pass: every image, strict -- an invalid output in the
            // CLEAN pass is a tool failure, not a fault datum (no faults
            // exist yet). Every trial's images are classified against these
            // records.
            allocator.set_phase("clean_pass");
            observer.event("CLEAN_PASS_BEGIN");
            void* clean_probability_base =
                binding_pointers[probability_binding.index];
            void* clean_class_base = binding_pointers[class_binding.index];
            for (std::size_t image_index = 0;
                 image_index < campaign_samples.size(); ++image_index) {
                const ImageOutcome outcome = run_campaign_image(
                    *context, binding_pointers, input_binding.index,
                    &campaign_input[image_index * input_binding.element_count],
                    input_binding.size_bytes, probability_binding.index,
                    class_binding.index, stream, options.class_count,
                    binding_pointers[input_binding.index],
                    input_binding.size_bytes, {}, clean_probability_base,
                    probability_binding.size_bytes, {}, clean_class_base,
                    class_binding.size_bytes, {});
                if (!outcome.valid) {
                    throw std::runtime_error(
                        "clean pass image " + std::to_string(image_index) +
                        " produced an invalid output");
                }
                campaign_clean.push_back(
                    Prediction{outcome.probability, outcome.class_index});
            }
            write_g5_clean_pass(options.output_prefix + "_g5_clean_pass.csv",
                                campaign_samples, campaign_clean);
            observer.event("CLEAN_PASS_END");
        }
        if (!options.l2_probe_out.empty()) {
            allocator.set_phase("l2_probe_pass");
            return run_l2_probe_pass(options, *context, binding_pointers,
                                     input_binding, probability_binding,
                                     class_binding, stream, campaign_samples,
                                     campaign_input, campaign_clean,
                                     allocation_registry, observer);
        }

        Prediction injected;
        gpu_m2d::TensorBitAddress mapped{};
        gpu_m2d::BitFlipResult flip{};
        std::vector<G4TargetRecord> g4_records;
        std::string g4_injected_outcome;
        bool g4_injected_valid = false;
        std::size_t campaign_trials_run = 0;
        std::size_t campaign_sites_total = 0;
        std::size_t campaign_images = 0;
        if (!options.campaign_work.empty()) {
        // ---- G5 campaign: the orchestrator has observed every PTE, built
        // the per-run dual-addressing snapshot, sampled every trial from
        // the frozen fault model on that snapshot, and written the work
        // file; it is authoritative about WHERE to flip. Per trial: flip
        // ALL sites, run the full evaluation pass with the faults held in
        // place, classify every image against the clean pass, restore
        // byte-exactly, then a strict sanity inference must reproduce the
        // clean output (no residue). The gate/registry skeleton is the
        // G4-T2 one; the per-trial events (TRIAL_*/SITE_*) are new.
        allocator.set_phase("g5_campaign");
        write_allocation_registry(options.output_prefix + "_allocations.csv",
                                  allocation_registry);
        observer.event("ALLOCATION_REGISTRY_GATE_WRITTEN");
        observer.event("CAMPAIGN_GATE_WAIT");
        if (observer.enabled) {
            std::fflush(stdout);
        }
        if (!wait_for_file(options.campaign_release,
                           options.campaign_gate_timeout_seconds > 0
                               ? options.campaign_gate_timeout_seconds
                               : options.gate_timeout_seconds)) {
            throw std::runtime_error("campaign release gate timed out: " +
                                     options.campaign_release);
        }
        observer.event("CAMPAIGN_WORK_BEGIN");
        const std::vector<std::vector<CampaignSite>> campaign_trials =
            read_campaign_work(options.campaign_work);
        campaign_trials_run = campaign_trials.size();
        if (campaign_trials_run == 0) {
            throw std::runtime_error("campaign work file has no trials");
        }

        // Live allocation reference for every allocation the work touches.
        struct AllocationRef {
            void* base{nullptr};
            std::size_t size_bytes{0};
            std::string semantic_label;
        };
        std::map<std::string, AllocationRef> allocation_refs;
        for (const gpu_m2d::AllocationDescriptor& descriptor :
             allocation_registry.allocations()) {
            if (descriptor.active) {
                allocation_refs[descriptor.allocation_id] = AllocationRef{
                    reinterpret_cast<void*>(descriptor.base_gpu_va),
                    descriptor.size_bytes, descriptor.semantic_label};
            }
        }

        // Pre-flight chain check for EVERY site of EVERY trial: the live
        // registry must resolve each (allocation, byte, bit) to exactly the
        // orchestrator's expected VA, so any drift between the sampled
        // snapshot and this process refuses before the first flip.
        for (const std::vector<CampaignSite>& trial : campaign_trials) {
            for (const CampaignSite& site : trial) {
                const auto forward =
                    allocation_registry.allocation_bit_to_gpu_va(
                        site.target.allocation_id, site.target.byte_offset,
                        static_cast<std::uint8_t>(site.target.bit_in_byte));
                if (forward.gpu_va != site.target.expected_gpu_va ||
                    allocation_refs.find(site.target.allocation_id) ==
                        allocation_refs.end()) {
                    throw std::runtime_error(
                        "campaign site " + site.target.target_id +
                        ": live chain disagrees with the sampled snapshot");
                }
            }
        }

        void* input_base = binding_pointers[input_binding.index];
        void* probability_base = binding_pointers[probability_binding.index];
        void* class_base = binding_pointers[class_binding.index];
        const std::string input_allocation_id =
            "trt-binding-" + input_binding.name + "-gpu-" +
            std::to_string(options.device);
        const std::string probability_allocation_id =
            "trt-binding-" + probability_binding.name + "-gpu-" +
            std::to_string(options.device);
        const std::string class_allocation_id =
            "trt-binding-" + class_binding.name + "-gpu-" +
            std::to_string(options.device);
        const float* sanity_input =
            &campaign_input[options.sample_index * input_binding.element_count];
        const std::size_t images_total = campaign_samples.size();
        campaign_images = images_total;

        std::vector<G5SiteRecord> g5_site_records;
        std::vector<G5TrialRecord> g5_trial_records;
        std::vector<G5ImageRow> g5_image_rows;
        G5ResultStreams g5_results(options.output_prefix, options, run_id);

        for (std::size_t trial_index = 0; trial_index < campaign_trials.size();
             ++trial_index) {
            const std::vector<CampaignSite>& sites = campaign_trials[trial_index];
            observer.event_with("TRIAL_BEGIN",
                                "trial_index=" + std::to_string(trial_index) +
                                    ",sites=" + std::to_string(sites.size()));

            // Touched allocations of this trial, unique, first-seen order.
            std::vector<std::string> touched;
            for (const CampaignSite& site : sites) {
                if (std::find(touched.begin(), touched.end(),
                              site.target.allocation_id) == touched.end()) {
                    touched.push_back(site.target.allocation_id);
                }
            }

            // ONE pristine snapshot per touched allocation (~26 MiB total;
            // per-site full snapshots would need ~5 GB at L5). Weights and
            // engine scratch hold their pre-trial content; the input binding
            // is deterministic because the pass re-stages it per image.
            std::map<std::string, std::vector<std::uint8_t>> pristine;
            for (const std::string& allocation_id : touched) {
                const AllocationRef& ref = allocation_refs[allocation_id];
                std::vector<std::uint8_t> bytes(ref.size_bytes);
                check_cuda(cudaMemcpy(bytes.data(), ref.base, ref.size_bytes,
                                      cudaMemcpyDeviceToHost),
                           "snapshot allocation before campaign flips");
                pristine.emplace(allocation_id, std::move(bytes));
            }

            // Flip every site in work order.
            std::vector<G5SiteRecord> records;
            records.reserve(sites.size());
            for (const CampaignSite& site : sites) {
                const AllocationRef& ref =
                    allocation_refs[site.target.allocation_id];
                G5SiteRecord record;
                record.site = site;
                record.semantic_label = ref.semantic_label;
                record.flip = gpu_m2d::flip_device_bit(
                    ref.base, ref.size_bytes, site.target.byte_offset,
                    static_cast<std::uint8_t>(site.target.bit_in_byte));
                if (record.flip.gpu_va != site.target.expected_gpu_va) {
                    throw std::runtime_error(
                        "campaign site " + site.target.target_id +
                        ": flipped VA drifted from the sampled snapshot");
                }
                try {
                    const auto reversed =
                        allocation_registry.gpu_va_to_allocation_bit(
                            options.device, record.flip.gpu_va,
                            record.flip.bit_in_byte);
                    record.reverse_map_ok =
                        reversed.allocation_id == site.target.allocation_id &&
                        reversed.byte_offset == site.target.byte_offset &&
                        reversed.bit_in_byte == site.target.bit_in_byte;
                } catch (const std::exception&) {
                    record.reverse_map_ok = false;
                }
                if (!record.reverse_map_ok) {
                    throw std::runtime_error("campaign site " +
                                             site.target.target_id +
                                             ": reverse chain check failed");
                }
                observer.event_with(
                    "SITE_FLIPPED",
                    "trial_index=" + std::to_string(trial_index) +
                        ",event_index=" + std::to_string(site.event_index) +
                        ",site_index=" + std::to_string(site.site_index) +
                        ",target_id=" + site.target.target_id +
                        ",allocation_id=" + site.target.allocation_id +
                        ",gpu_va=" + hex_address(record.flip.gpu_va) +
                        ",before=" + std::to_string(record.flip.before) +
                        ",after=" + std::to_string(record.flip.after) +
                        ",xor_mask=" +
                        std::to_string(record.flip.xor_mask));
                records.push_back(std::move(record));
            }

            // Allocation-level guard compare: no engine run has happened
            // since the pristine snapshot, so every touched allocation must
            // equal pristine ^ accumulated site masks byte-exactly (covers
            // input binding, output bindings AND TRT-internal regions --
            // the strong flip-side check for every site of the trial).
            for (const std::string& allocation_id : touched) {
                std::vector<std::uint8_t> expected = pristine[allocation_id];
                for (const G5SiteRecord& record : records) {
                    if (record.site.target.allocation_id == allocation_id) {
                        expected[record.site.target.byte_offset] ^=
                            record.flip.xor_mask;
                    }
                }
                std::vector<std::uint8_t> live(
                    allocation_refs[allocation_id].size_bytes);
                check_cuda(cudaMemcpy(
                               live.data(),
                               allocation_refs[allocation_id].base, live.size(),
                               cudaMemcpyDeviceToHost),
                           "verify full allocation after campaign flips");
                if (live != expected) {
                    throw std::runtime_error(
                        "campaign guard compare failed for allocation " +
                        allocation_id);
                }
                for (G5SiteRecord& record : records) {
                    if (record.site.target.allocation_id == allocation_id) {
                        record.alloc_guard_ok = true;
                    }
                }
            }

            // Held-fault site lists for the pass (see run_campaign_image):
            // input faults re-applied after every per-image copy, output
            // faults after every enqueue; TRT-internal sites are never
            // re-applied (weights persist; scratch is soft-upset).
            std::vector<InjectionTarget> input_sites;
            std::vector<InjectionTarget> probability_sites;
            std::vector<InjectionTarget> class_sites;
            for (const G5SiteRecord& record : records) {
                const std::string& allocation_id =
                    record.site.target.allocation_id;
                if (allocation_id == input_allocation_id) {
                    input_sites.push_back(record.site.target);
                } else if (allocation_id == probability_allocation_id) {
                    probability_sites.push_back(record.site.target);
                } else if (allocation_id == class_allocation_id) {
                    class_sites.push_back(record.site.target);
                }
            }

            // ---- injected evaluation pass with the faults held: every
            // image compared against the campaign's clean records. The
            // first invalid output is an honest DUE and aborts the rest of
            // the trial's images (marked evaluated=0/IMAGE_DUE).
            std::size_t images_evaluated = 0;
            std::size_t count_benign = 0;
            std::size_t count_numeric = 0;
            std::size_t count_top1 = 0;
            std::size_t count_invalid = 0;
            std::size_t last_evaluated = 0;
            for (std::size_t image_index = 0; image_index < images_total;
                 ++image_index) {
                const Prediction& clean_record = campaign_clean[image_index];
                G5ImageRow row;
                row.trial_index = trial_index;
                row.image_index = image_index;
                row.evaluated = true;
                row.clean_class = clean_record.class_index;
                row.clean_probability = clean_record.probability;
                const ImageOutcome outcome = run_campaign_image(
                    *context, binding_pointers, input_binding.index,
                    &campaign_input[image_index * input_binding.element_count],
                    input_binding.size_bytes, probability_binding.index,
                    class_binding.index, stream, options.class_count,
                    input_base,
                    input_binding.size_bytes, input_sites, probability_base,
                    probability_binding.size_bytes, probability_sites,
                    class_base, class_binding.size_bytes, class_sites);
                images_evaluated++;
                last_evaluated = image_index;
                row.have_injected = outcome.valid;
                if (outcome.valid) {
                    row.injected_class = outcome.class_index;
                    row.injected_probability = outcome.probability;
                    const bool top1_changed =
                        outcome.class_index != clean_record.class_index;
                    const bool numeric_changed =
                        top1_changed ||
                        std::fabs(outcome.probability -
                                  clean_record.probability) >
                            kProbabilityTolerance;
                    if (top1_changed) {
                        row.outcome = "IMAGE_SDC_TOP1";
                        count_top1++;
                    } else if (numeric_changed) {
                        row.outcome = "IMAGE_SDC_NUMERIC";
                        count_numeric++;
                    } else {
                        row.outcome = "IMAGE_BENIGN";
                        count_benign++;
                    }
                    g5_image_rows.push_back(std::move(row));
                } else {
                    row.outcome = "IMAGE_DUE";
                    count_invalid++;
                    g5_image_rows.push_back(std::move(row));
                    for (std::size_t remaining = image_index + 1;
                         remaining < images_total; ++remaining) {
                        G5ImageRow skipped;
                        skipped.trial_index = trial_index;
                        skipped.image_index = remaining;
                        skipped.evaluated = false;
                        skipped.clean_class =
                            campaign_clean[remaining].class_index;
                        skipped.clean_probability =
                            campaign_clean[remaining].probability;
                        skipped.outcome = "IMAGE_DUE";
                        count_invalid++;
                        g5_image_rows.push_back(std::move(skipped));
                    }
                    break;
                }
            }
            std::string trial_outcome = "BENIGN";
            if (count_invalid > 0) {
                trial_outcome = "DUE_INVALID_OUTPUT";
            } else if (count_top1 > 0) {
                trial_outcome = "SDC_TOP1";
            } else if (count_numeric > 0) {
                trial_outcome = "SDC_NUMERIC";
            }
            observer.event_with(
                "TRIAL_INJECTED_END",
                "trial_index=" + std::to_string(trial_index) +
                    ",outcome=" + trial_outcome +
                    ",evaluated=" + std::to_string(images_evaluated) +
                    ",benign=" + std::to_string(count_benign) +
                    ",sdc_numeric=" + std::to_string(count_numeric) +
                    ",sdc_top1=" + std::to_string(count_top1) +
                    ",invalid=" + std::to_string(count_invalid));

            // ---- restore every site in reverse work order.
            for (auto record_it = records.rbegin(); record_it != records.rend();
                 ++record_it) {
                G5SiteRecord& record = *record_it;
                const AllocationRef& ref =
                    allocation_refs[record.site.target.allocation_id];
                const auto unflip = gpu_m2d::flip_device_bit(
                    ref.base, ref.size_bytes, record.site.target.byte_offset,
                    static_cast<std::uint8_t>(record.site.target.bit_in_byte));
                if (unflip.gpu_va != record.flip.gpu_va) {
                    throw std::runtime_error(
                        "campaign site " + record.site.target.target_id +
                        ": restore VA drifted");
                }
                // False here means the byte CHANGED between flip and
                // restore (the engine rewrote it mid-pass): expected only
                // for engine-owned regions, and for the input binding whose
                // bytes the pass legitimately re-stages -- the binding's
                // authoritative restore proof is the exact full compare
                // against the last staged image below.
                record.restored_byte_ok = unflip.after == record.flip.before;
                observer.event_with(
                    "SITE_RESTORED",
                    "trial_index=" + std::to_string(trial_index) +
                        ",target_id=" + record.site.target.target_id +
                        ",allocation_id=" +
                        record.site.target.allocation_id +
                        ",byte=" + std::to_string(unflip.after));
            }

            std::size_t restore_alloc_exact = 0;
            std::size_t restore_alloc_mismatch = 0;
            std::size_t restore_alloc_skipped = 0;
            std::size_t restore_mismatch_bytes = 0;

            // Input binding: the runner stages every byte, so the exact
            // expected content is the last image the pass evaluated --
            // site bytes must have returned to their staged values and no
            // other byte may differ. This is a fail-closed check.
            {
                const std::uint8_t* last_staged =
                    reinterpret_cast<const std::uint8_t*>(
                        &campaign_input[last_evaluated *
                                        input_binding.element_count]);
                std::vector<std::uint8_t> live(input_binding.size_bytes);
                check_cuda(cudaMemcpy(live.data(), input_base, live.size(),
                                      cudaMemcpyDeviceToHost),
                           "verify input binding after restore");
                std::size_t mismatched = 0;
                for (std::size_t offset = 0; offset < live.size(); ++offset) {
                    if (live[offset] != last_staged[offset]) {
                        mismatched++;
                    }
                }
                if (mismatched != 0) {
                    throw std::runtime_error(
                        "input binding restore mismatch: " +
                        std::to_string(mismatched) + " bytes differ");
                }
                restore_alloc_exact++;
                for (G5SiteRecord& record : records) {
                    if (record.site.target.allocation_id ==
                        input_allocation_id) {
                        record.restore_check = "exact";
                    }
                }
            }

            for (const std::string& allocation_id : touched) {
                if (allocation_id == input_allocation_id) {
                    continue;
                }
                if (allocation_id == probability_allocation_id ||
                    allocation_id == class_allocation_id) {
                    // Engine-owned output: rewritten by every enqueue, so no
                    // stable post-restore baseline exists; the flip itself
                    // was proven by the pre-pass guard compare.
                    restore_alloc_skipped++;
                    for (G5SiteRecord& record : records) {
                        if (record.site.target.allocation_id ==
                            allocation_id) {
                            record.restore_check =
                                "skipped:engine-owned-output";
                        }
                    }
                    continue;
                }
                // TRT-internal: informational compare against pristine.
                // Weights regions return byte-exact; engine scratch may
                // legitimately differ (the engine rewrote it during the
                // pass -- soft-upset semantics, recorded, not fatal). The
                // behavioral no-residue proof is the sanity inference.
                const AllocationRef& ref = allocation_refs[allocation_id];
                std::vector<std::uint8_t> live(ref.size_bytes);
                check_cuda(cudaMemcpy(live.data(), ref.base, ref.size_bytes,
                                      cudaMemcpyDeviceToHost),
                           "verify TRT-internal allocation after restore");
                const std::vector<std::uint8_t>& reference =
                    pristine[allocation_id];
                std::size_t mismatched = 0;
                for (std::size_t offset = 0; offset < live.size(); ++offset) {
                    if (live[offset] != reference[offset]) {
                        mismatched++;
                    }
                }
                if (mismatched == 0) {
                    restore_alloc_exact++;
                } else {
                    restore_alloc_mismatch++;
                    restore_mismatch_bytes += mismatched;
                }
                for (G5SiteRecord& record : records) {
                    if (record.site.target.allocation_id == allocation_id) {
                        record.restore_check =
                            mismatched == 0
                                ? "exact"
                                : "mismatch:" + std::to_string(mismatched);
                    }
                }
            }

            // ---- strict sanity inference: with every fault restored the
            // engine must reproduce the sanity image's clean output
            // (INT8 execution is deterministic; the tolerance only absorbs
            // float reduction wobble).
            check_cuda(cudaMemcpy(input_base, sanity_input,
                                  input_binding.size_bytes,
                                  cudaMemcpyHostToDevice),
                       "stage sanity image for sanity inference");
            observer.event_with("TRIAL_SANITY_BEGIN",
                                "trial_index=" + std::to_string(trial_index));
            const Prediction sanity = run_inference(
                *context, binding_pointers, probability_binding.index,
                class_binding.index, stream, options.class_count);
            const bool sanity_matches =
                sanity.class_index ==
                    campaign_clean[options.sample_index].class_index &&
                std::fabs(sanity.probability -
                          campaign_clean[options.sample_index].probability) <=
                    kSanityTolerance;
            if (!sanity_matches) {
                throw std::runtime_error(
                    "campaign trial " + std::to_string(trial_index) +
                    ": post-restore sanity inference diverged from clean");
            }
            observer.event_with(
                "TRIAL_SANITY_END",
                "trial_index=" + std::to_string(trial_index) +
                    ",class=" + std::to_string(sanity.class_index) +
                    ",probability=" +
                    std::to_string(sanity.probability) + ",matches_clean=1");

            std::size_t event_count = 0;
            for (const CampaignSite& site : sites) {
                event_count = std::max(event_count, site.event_index + 1);
            }
            G5TrialRecord trial_record;
            trial_record.trial_index = trial_index;
            trial_record.site_count = sites.size();
            trial_record.event_count = event_count;
            trial_record.images_total = images_total;
            trial_record.images_evaluated = images_evaluated;
            trial_record.images_benign = count_benign;
            trial_record.images_sdc_numeric = count_numeric;
            trial_record.images_sdc_top1 = count_top1;
            trial_record.images_invalid = count_invalid;
            trial_record.injected_outcome = trial_outcome;
            trial_record.sanity_class = sanity.class_index;
            trial_record.sanity_probability = sanity.probability;
            trial_record.sanity_matches_clean = sanity_matches;
            trial_record.restore_alloc_exact = restore_alloc_exact;
            trial_record.restore_alloc_mismatch = restore_alloc_mismatch;
            trial_record.restore_alloc_skipped = restore_alloc_skipped;
            trial_record.restore_mismatch_bytes = restore_mismatch_bytes;
            g5_trial_records.push_back(std::move(trial_record));
            campaign_sites_total += sites.size();
            g5_site_records.insert(
                g5_site_records.end(), std::make_move_iterator(records.begin()),
                std::make_move_iterator(records.end()));
            observer.event_with("TRIAL_END",
                                "trial_index=" + std::to_string(trial_index) +
                                    ",outcome=" + trial_outcome);
            // Durable per-trial flush: this trial's rows reach the OS now
            // (G5ResultStreams); a later process-fatal trial cannot
            // destroy them.
            g5_results.flush_trial(g5_site_records, g5_trial_records,
                                   g5_image_rows);
        }

        // Final no-op append + stream-state check: every trial was already
        // flushed at its TRIAL_END (G5ResultStreams).
        g5_results.flush_trial(g5_site_records, g5_trial_records,
                               g5_image_rows);
        // Final registry at the same path as the gate-time one: the
        // orchestrator snapshotted the gate file when the gate opened and
        // diffs it against this final content (stability check).
        write_allocation_registry(options.output_prefix + "_allocations.csv",
                                  allocation_registry);
        observer.event("CAMPAIGN_WORK_END");
        } else if (options.injection_work.empty()) {
        allocator.set_phase("injected_inference");
        observer.event("INJECTED_INFERENCE_BEGIN");
        check_cuda(cudaMemcpy(binding_pointers[input_binding.index], input.data(),
                              input_binding.size_bytes, cudaMemcpyHostToDevice),
                   "reset input before injection");
        mapped = mapping.tensor_bit_to_gpu_va(
            input_binding.name, options.element_index, options.element_bit_index);
        flip = gpu_m2d::flip_device_bit(
            binding_pointers[input_binding.index], input_binding.size_bytes,
            mapped.byte_offset, mapped.bit_in_byte);
        if (flip.gpu_va != mapped.gpu_va || flip.xor_mask != mapped.xor_mask) {
            throw std::runtime_error("mapper/injector disagreement");
        }

        std::vector<std::uint8_t> expected(input_binding.size_bytes);
        std::memcpy(expected.data(), input.data(), input_binding.size_bytes);
        expected[mapped.byte_offset] ^= mapped.xor_mask;
        std::vector<std::uint8_t> observed(input_binding.size_bytes);
        check_cuda(cudaMemcpy(observed.data(), binding_pointers[input_binding.index],
                              observed.size(), cudaMemcpyDeviceToHost),
                   "verify complete injected input buffer");
        if (observed != expected) {
            throw std::runtime_error("injected input or non-target guard byte mismatch");
        }

        const auto reversed = mapping.gpu_va_to_tensor_bit(
            flip.gpu_va, flip.bit_in_byte);
        if (reversed.tensor_name != input_binding.name ||
            reversed.element_index != options.element_index ||
            reversed.element_bit_index != options.element_bit_index) {
            throw std::runtime_error("real TensorRT input reverse mapping mismatch");
        }
        const auto allocation_reversed =
            allocation_registry.gpu_va_to_allocation_bit(
                options.device, flip.gpu_va, flip.bit_in_byte);
        const std::string input_allocation_id =
            "trt-binding-" + input_binding.name + "-gpu-" +
            std::to_string(options.device);
        if (allocation_reversed.allocation_id != input_allocation_id ||
            allocation_reversed.byte_offset != mapped.byte_offset ||
            allocation_reversed.bit_in_byte != mapped.bit_in_byte) {
            throw std::runtime_error(
                "real TensorRT input allocation reverse mapping mismatch");
        }
        const auto allocation_forward =
            allocation_registry.allocation_bit_to_gpu_va(
                input_allocation_id, mapped.byte_offset, mapped.bit_in_byte);
        if (allocation_forward.gpu_va != flip.gpu_va) {
            throw std::runtime_error(
                "real TensorRT input allocation forward mapping mismatch");
        }

        injected = run_inference(
            *context, binding_pointers, probability_binding.index,
            class_binding.index, stream, options.class_count);
        observer.event("INJECTED_INFERENCE_END");
        write_result(options.output_prefix + "_result.csv", options, sample, run_id,
                     mapped, flip, clean, injected);
        write_allocation_registry(options.output_prefix + "_allocations.csv",
                                  allocation_registry);
        } else {
        // ---- G4-T2: gated dual-addressing XOR through the observed chain.
        // The orchestrator has observed every PTE, built the per-run
        // snapshot (TensorBit-VA-PA-GDDR), and selected reverse-chain
        // targets; the work file is authoritative about WHERE to flip.
        allocator.set_phase("g4_injection");
        write_allocation_registry(options.output_prefix + "_allocations.csv",
                                  allocation_registry);
        observer.event("ALLOCATION_REGISTRY_GATE_WRITTEN");
        observer.event("INJECTION_GATE_WAIT");
        if (observer.enabled) {
            std::fflush(stdout);
        }
        if (!wait_for_file(options.injection_release,
                           options.injection_gate_timeout_seconds > 0
                               ? options.injection_gate_timeout_seconds
                               : options.gate_timeout_seconds)) {
            throw std::runtime_error("injection release gate timed out: " +
                                     options.injection_release);
        }
        observer.event("INJECTION_WORK_BEGIN");
        const std::vector<InjectionTarget> targets =
            read_injection_work(options.injection_work);

        // Pre-flight chain check: every work target must resolve in the
        // LIVE registry to the orchestrator's expected VA, so any drift
        // between the snapshot and this process refuses before any flip.
        for (const InjectionTarget& target : targets) {
            const auto forward = allocation_registry.allocation_bit_to_gpu_va(
                target.allocation_id, target.byte_offset,
                static_cast<std::uint8_t>(target.bit_in_byte));
            if (forward.gpu_va != target.expected_gpu_va) {
                throw std::runtime_error(
                    "work target " + target.target_id + ": registry forward VA " +
                    hex_address(forward.gpu_va) + " != expected " +
                    hex_address(target.expected_gpu_va));
            }
        }

        for (const InjectionTarget& target : targets) {
            observer.event_with("TARGET_BEGIN",
                                "target_id=" + target.target_id +
                                    ",allocation_id=" + target.allocation_id);
            const auto descriptors = allocation_registry.allocations();
            const auto descriptor = std::find_if(
                descriptors.begin(), descriptors.end(),
                [&target](const gpu_m2d::AllocationDescriptor& candidate) {
                    return candidate.allocation_id == target.allocation_id &&
                           candidate.active;
                });
            if (descriptor == descriptors.end()) {
                throw std::runtime_error("work target allocation is not active: " +
                                         target.allocation_id);
            }
            void* base = reinterpret_cast<void*>(descriptor->base_gpu_va);

            G4TargetRecord record;
            record.target = target;
            record.semantic_label = descriptor->semantic_label;
            record.before_full.resize(descriptor->size_bytes);
            check_cuda(cudaMemcpy(record.before_full.data(), base,
                                  descriptor->size_bytes, cudaMemcpyDeviceToHost),
                       "snapshot allocation before injection");
            record.flip = gpu_m2d::flip_device_bit(
                base, descriptor->size_bytes, target.byte_offset,
                static_cast<std::uint8_t>(target.bit_in_byte));
            if (record.flip.gpu_va != target.expected_gpu_va) {
                throw std::runtime_error("flipped VA drifted from the work target");
            }

            std::vector<std::uint8_t> after_full(descriptor->size_bytes);
            check_cuda(cudaMemcpy(after_full.data(), base, descriptor->size_bytes,
                                  cudaMemcpyDeviceToHost),
                       "verify full allocation after injection");
            std::vector<std::uint8_t> expected_full = record.before_full;
            expected_full[target.byte_offset] ^= record.flip.xor_mask;
            record.guard_bytes_unchanged = after_full == expected_full;

            try {
                const auto reversed =
                    allocation_registry.gpu_va_to_allocation_bit(
                        options.device, record.flip.gpu_va,
                        record.flip.bit_in_byte);
                record.reverse_map_ok =
                    reversed.allocation_id == target.allocation_id &&
                    reversed.byte_offset == target.byte_offset &&
                    reversed.bit_in_byte == record.flip.bit_in_byte;
            } catch (const std::exception&) {
                record.reverse_map_ok = false;
            }
            if (!record.guard_bytes_unchanged || !record.reverse_map_ok) {
                throw std::runtime_error("target verification failed: " +
                                         target.target_id);
            }
            observer.event_with(
                "TARGET_FLIPPED",
                "target_id=" + target.target_id +
                    ",allocation_id=" + target.allocation_id +
                    ",gpu_va=" + hex_address(record.flip.gpu_va) +
                    ",before=" + std::to_string(record.flip.before) +
                    ",after=" + std::to_string(record.flip.after) +
                    ",xor_mask=" + std::to_string(record.flip.xor_mask));
            g4_records.push_back(std::move(record));
        }

        allocator.set_phase("injected_inference");
        observer.event("INJECTED_INFERENCE_BEGIN");
        try {
            injected = run_inference(
                *context, binding_pointers, probability_binding.index,
                class_binding.index, stream, options.class_count);
            g4_injected_valid = true;
        } catch (const std::exception&) {
            // An invalid numeric output is an honest DUE outcome of the
            // injected faults, not a tool failure; the restore and sanity
            // phases below still verify the device state.
            g4_injected_valid = false;
        }
        observer.event("INJECTED_INFERENCE_END");
        if (!g4_injected_valid) {
            g4_injected_outcome = "DUE_INVALID_OUTPUT";
        } else if (injected.class_index != clean.class_index) {
            g4_injected_outcome = "SDC_TOP1";
        } else if (std::fabs(injected.probability - clean.probability) >
                   kProbabilityTolerance) {
            g4_injected_outcome = "SDC_NUMERIC";
        } else {
            g4_injected_outcome = "BENIGN";
        }
        observer.event_with("INJECTED_OUTCOME", "outcome=" + g4_injected_outcome);

        observer.event("RESTORE_BEGIN");
        for (auto record_it = g4_records.rbegin(); record_it != g4_records.rend();
             ++record_it) {
            G4TargetRecord& record = *record_it;
            const auto descriptors = allocation_registry.allocations();
            const auto descriptor = std::find_if(
                descriptors.begin(), descriptors.end(),
                [&record](const gpu_m2d::AllocationDescriptor& candidate) {
                    return candidate.allocation_id == record.target.allocation_id &&
                           candidate.active;
                });
            if (descriptor == descriptors.end()) {
                throw std::runtime_error("restore target allocation is not active: " +
                                         record.target.allocation_id);
            }
            void* base = reinterpret_cast<void*>(descriptor->base_gpu_va);
            const auto unflip = gpu_m2d::flip_device_bit(
                base, descriptor->size_bytes, record.target.byte_offset,
                static_cast<std::uint8_t>(record.target.bit_in_byte));
            record.restored_byte_ok = unflip.after == record.flip.before;
            std::vector<std::uint8_t> restored_full(descriptor->size_bytes);
            check_cuda(cudaMemcpy(restored_full.data(), base, descriptor->size_bytes,
                                  cudaMemcpyDeviceToHost),
                       "verify full allocation after restore");
            record.restore_guard_ok = restored_full == record.before_full;
            if (!record.restored_byte_ok || !record.restore_guard_ok) {
                throw std::runtime_error("target restore failed: " +
                                         record.target.target_id);
            }
            observer.event_with("TARGET_RESTORED",
                                "target_id=" + record.target.target_id +
                                    ",allocation_id=" +
                                    record.target.allocation_id +
                                    ",byte=" + std::to_string(unflip.after));
        }
        observer.event("RESTORE_END");

        // With every fault restored the engine must reproduce the clean
        // output exactly (INT8 inference is deterministic for a fixed
        // engine and input): this is the no-residue check.
        observer.event("SANITY_INFERENCE_BEGIN");
        const Prediction sanity = run_inference(
            *context, binding_pointers, probability_binding.index,
            class_binding.index, stream, options.class_count);
        observer.event("SANITY_INFERENCE_END");
        if (sanity.class_index != clean.class_index ||
            std::fabs(sanity.probability - clean.probability) > kSanityTolerance) {
            throw std::runtime_error(
                "post-restore sanity inference diverged from the clean run");
        }
        write_g4_result(options.output_prefix + "_g4_result.csv", options, sample,
                        run_id, g4_records, clean, injected, g4_injected_outcome,
                        g4_injected_valid, sanity);
        write_allocation_registry(options.output_prefix + "_allocations.csv",
                                  allocation_registry);
        observer.event("INJECTION_WORK_END");
        }
        observer.event("SNAPSHOT_READY");

        if (!options.campaign_work.empty()) {
            std::cout << "GPU_M2D_G5_CAMPAIGN_PASS"
                      << " device=" << options.device
                      << " gpu=\"" << device_properties.name << "\""
                      << " pci_bus_id=" << pci_bus_id
                      << " trials=" << campaign_trials_run
                      << " sites_total=" << campaign_sites_total
                      << " images=" << campaign_images << '\n';
        } else if (!options.injection_work.empty()) {
            std::cout << "GPU_M2D_G4_T2_PASS"
                      << " device=" << options.device
                      << " gpu=\"" << device_properties.name << "\""
                      << " targets=" << g4_records.size();
            for (const G4TargetRecord& record : g4_records) {
                std::cout << " " << record.target.target_id
                          << "=alloc:" << record.target.allocation_id
                          << ",va=" << hex_address(record.flip.gpu_va)
                          << ",byte_offset=" << record.target.byte_offset
                          << ",bit=" << record.target.bit_in_byte
                          << ",before=" << static_cast<unsigned>(record.flip.before)
                          << ",after=" << static_cast<unsigned>(record.flip.after);
            }
            std::cout << " injected_outcome=" << g4_injected_outcome
                      << " restored=" << g4_records.size()
                      << " sanity=clean\n";
        } else {
        const bool top1_changed = clean.class_index != injected.class_index;
        const bool numeric_changed = top1_changed ||
            std::fabs(clean.probability - injected.probability) >
                kProbabilityTolerance;
        std::cout << "GPU_M2D_G1_5_PASS"
                  << " device=" << options.device
                  << " gpu=\"" << device_properties.name << "\""
                  << " pci_bus_id=" << pci_bus_id
                  << " tensor=" << mapped.tensor_name
                  << " element=" << mapped.element_index
                  << " bit=" << mapped.element_bit_index
                  << " gpu_va=" << hex_address(mapped.gpu_va)
                  << " before=" << static_cast<unsigned>(flip.before)
                  << " after=" << static_cast<unsigned>(flip.after)
                  << " clean_class=" << clean.class_index
                  << " injected_class=" << injected.class_index
                  << " outcome=" << (top1_changed ? "SDC_TOP1" : "BENIGN_TOP1")
                  << " numeric_output_changed=" << (numeric_changed ? 1 : 0)
                  << '\n';
        }

        allocator.set_phase("destroy_runtime_objects");
        if (observer.enabled && options.hold_seconds > 0) {
            observer.event("HOLD_BEGIN");
            std::this_thread::sleep_for(std::chrono::seconds(options.hold_seconds));
            observer.event("HOLD_END");
        }
        observer.event("TEARDOWN_BEGIN");
        // Explicit teardown in the natural destruction order so the observer
        // sees every allocation free before PROCESS_END. The binding FREE
        // events are emitted here because DeviceBuffer destruction goes
        // through no observer hook.
        for (const BindingInfo& binding : binding_info) {
            const std::string allocation_id =
                "trt-binding-" + binding.name + "-gpu-" +
                std::to_string(options.device);
            if (std::any_of(buffers.begin(), buffers.end(),
                            [&allocation_id](const DeviceBuffer& buffer) {
                                return buffer.allocation_id() == allocation_id;
                            })) {
                observer.freed(allocation_id);
            }
        }
        buffers.clear();
        context.reset();
        engine.reset();
        runtime.reset();
        observer.event("PROCESS_END");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GPU_M2D_G1_5_FAIL: " << error.what() << '\n';
        return 1;
    }
}
