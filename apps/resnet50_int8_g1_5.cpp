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
            "--injection-gate-timeout-seconds N]");
    }
    if (options.injection_work.empty() != options.injection_release.empty()) {
        throw std::invalid_argument(
            "--injection-work and --injection-release must be given together");
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
        allocator.set_phase("clean_inference");
        observer.event("CLEAN_INFERENCE_BEGIN");
        check_cuda(cudaMemcpy(binding_pointers[input_binding.index], input.data(),
                              input_binding.size_bytes, cudaMemcpyHostToDevice),
                   "copy clean input to device");
        const Prediction clean = run_inference(
            *context, binding_pointers, probability_binding.index,
            class_binding.index, stream);
        observer.event("CLEAN_INFERENCE_END");

        Prediction injected;
        gpu_m2d::TensorBitAddress mapped{};
        gpu_m2d::BitFlipResult flip{};
        std::vector<G4TargetRecord> g4_records;
        std::string g4_injected_outcome;
        bool g4_injected_valid = false;
        if (options.injection_work.empty()) {
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

        if (!options.injection_work.empty()) {
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
