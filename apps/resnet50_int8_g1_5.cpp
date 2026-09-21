#include "gpu_m2d/allocation_registry.hpp"
#include "gpu_m2d/device_bit_injector.hpp"
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
            "--campaign-gate-timeout-seconds N]");
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

std::vector<float> preprocess(const std::string& path) {
    const cv::Mat source = cv::imread(path, cv::IMREAD_COLOR);
    if (source.empty()) {
        throw std::runtime_error("cannot decode image: " + path);
    }

    cv::Mat image;
    cv::resize(source, image, cv::Size(kInputWidth, kInputHeight), 0, 0,
               cv::INTER_CUBIC);
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);
    cv::subtract(image, cv::Scalar(kMean[2], kMean[1], kMean[0]), image);
    cv::divide(image, cv::Scalar(kStd[2], kStd[1], kStd[0]), image);

    const int plane = kInputHeight * kInputWidth;
    std::vector<float> chw(3U * static_cast<std::size_t>(plane));
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
    if (!engine.hasImplicitBatchDimension() || engine.getMaxBatchSize() < 1) {
        throw std::invalid_argument("G1.5 expects the existing implicit-batch engine");
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

Prediction run_inference(nvinfer1::IExecutionContext& context,
                         std::vector<void*>& binding_pointers,
                         int probability_index,
                         int class_index,
                         CudaStream& stream) {
    if (!context.enqueue(1, binding_pointers.data(), stream.get(), nullptr)) {
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
        prediction.class_index >= kClassCount) {
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
    if (!context.enqueue(1, binding_pointers.data(), stream.get(), nullptr)) {
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
                    outcome.class_index < kClassCount;
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

void write_g5_site_result(const std::string& path, const Options& options,
                          const std::string& run_id,
                          const std::vector<G5SiteRecord>& records) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write G5 site result: " + path);
    }
    output << "run_id,device,trial_index,event_index,site_index,target_id,"
              "allocation_id,semantic_label,byte_offset,bit_in_byte,xor_mask,"
              "gpu_va,expected_gpu_va,before,after,guard_bytes_unchanged,"
              "reverse_map_ok,restored_byte_ok,restore_check\n";
    for (const G5SiteRecord& record : records) {
        output << run_id << ',' << options.device << ','
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
}

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

void write_g5_trial_result(const std::string& path, const Options& options,
                           const std::string& run_id,
                           const std::vector<G5TrialRecord>& records) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write G5 trial result: " + path);
    }
    output << "run_id,device,trial_index,site_count,event_count,images_total,"
              "images_evaluated,images_benign,images_sdc_numeric,"
              "images_sdc_top1,images_invalid,injected_outcome,sanity_class,"
              "sanity_probability,sanity_matches_clean,restore_alloc_exact,"
              "restore_alloc_mismatch,restore_alloc_skipped,"
              "restore_mismatch_bytes\n";
    for (const G5TrialRecord& record : records) {
        output << run_id << ',' << options.device << ','
               << record.trial_index << ',' << record.site_count << ','
               << record.event_count << ',' << record.images_total << ','
               << record.images_evaluated << ',' << record.images_benign
               << ',' << record.images_sdc_numeric << ','
               << record.images_sdc_top1 << ',' << record.images_invalid
               << ',' << record.injected_outcome << ','
               << record.sanity_class << ',' << std::setprecision(9)
               << record.sanity_probability << ','
               << (record.sanity_matches_clean ? 1 : 0) << ','
               << record.restore_alloc_exact << ','
               << record.restore_alloc_mismatch << ','
               << record.restore_alloc_skipped << ','
               << record.restore_mismatch_bytes << '\n';
    }
}

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

void write_g5_image_detail(const std::string& path, const Options& options,
                           const std::string& run_id,
                           const std::vector<G5ImageRow>& rows) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write G5 image detail: " + path);
    }
    output << "run_id,device,trial_index,image_index,evaluated,clean_class,"
              "injected_class,clean_probability,injected_probability,outcome\n"
           << std::setprecision(9);
    for (const G5ImageRow& row : rows) {
        output << run_id << ',' << options.device << ',' << row.trial_index
               << ',' << row.image_index << ',' << (row.evaluated ? 1 : 0)
               << ',' << row.clean_class << ','
               << (row.have_injected
                       ? std::to_string(row.injected_class)
                       : std::string("NA"))
               << ',' << row.clean_probability << ',';
        if (row.have_injected) {
            output << row.injected_probability;
        } else {
            output << "NA";
        }
        output << ',' << row.outcome << '\n';
    }
}

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
        const Sample sample = read_sample(options.sample_csv, options.sample_index);
        const std::vector<float> input = preprocess(sample.path);
        const std::vector<char> engine_bytes = read_binary_file(options.engine_path);
        // G5 campaign: stage every evaluation image's preprocessed input on
        // the host BEFORE the gate (CPU-only work; the gated window stays
        // CUDA-only). ~600 MB host for the 1000-image pass.
        std::vector<Sample> campaign_samples;
        std::vector<float> campaign_input;
        if (!options.campaign_work.empty()) {
            campaign_samples = read_all_samples(options.sample_csv);
            if (options.sample_index >= campaign_samples.size()) {
                throw std::out_of_range(
                    "sanity sample index is outside the campaign CSV");
            }
            campaign_input.reserve(campaign_samples.size() * input.size());
            for (const Sample& eval_sample : campaign_samples) {
                const std::vector<float> staged = preprocess(eval_sample.path);
                campaign_input.insert(campaign_input.end(), staged.begin(),
                                      staged.end());
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
        if (options.campaign_work.empty()) {
            allocator.set_phase("clean_inference");
            observer.event("CLEAN_INFERENCE_BEGIN");
            check_cuda(cudaMemcpy(binding_pointers[input_binding.index], input.data(),
                                  input_binding.size_bytes, cudaMemcpyHostToDevice),
                       "copy clean input to device");
            clean = run_inference(
                *context, binding_pointers, probability_binding.index,
                class_binding.index, stream);
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
                    class_binding.index, stream,
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
                    class_binding.index, stream, input_base,
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
                class_binding.index, stream);
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
        }

        write_g5_site_result(options.output_prefix + "_g5_site_result.csv",
                             options, run_id, g5_site_records);
        write_g5_trial_result(options.output_prefix + "_g5_trial_result.csv",
                              options, run_id, g5_trial_records);
        write_g5_image_detail(options.output_prefix + "_g5_image_detail.csv",
                              options, run_id, g5_image_rows);
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
            class_binding.index, stream);
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
                class_binding.index, stream);
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
            class_binding.index, stream);
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
