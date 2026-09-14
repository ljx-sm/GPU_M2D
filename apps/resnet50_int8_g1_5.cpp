#include "gpu_m2d/allocation_registry.hpp"
#include "gpu_m2d/device_bit_injector.hpp"
#include "gpu_m2d/tensor_mapping.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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
#include <utility>
#include <vector>

namespace {

constexpr int kInputHeight = 224;
constexpr int kInputWidth = 224;
constexpr int kClassCount = 45;
constexpr float kProbabilityTolerance = 1.0e-6F;
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

struct TrackedAllocation {
    std::uintptr_t gpu_va{0};
    std::string allocation_id;
    bool active{false};
};

class TrackingGpuAllocator final : public nvinfer1::IGpuAllocator {
public:
    TrackingGpuAllocator(gpu_m2d::AllocationRegistry& registry, int device_id)
        : registry_(registry), device_id_(device_id) {}

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
        return true;
    }

private:
    gpu_m2d::AllocationRegistry& registry_;
    int device_id_{-1};
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
            "--output-prefix PATH [--sample-index N --device N --element N --bit N]");
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

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
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

        const Sample sample = read_sample(options.sample_csv, options.sample_index);
        const std::vector<float> input = preprocess(sample.path);
        const std::vector<char> engine_bytes = read_binary_file(options.engine_path);
        const std::string run_id =
            "g1_5-gpu" + std::to_string(options.device) + "-sample" +
            std::to_string(options.sample_index) + "-element" +
            std::to_string(options.element_index) + "-bit" +
            std::to_string(options.element_bit_index);

        TrtLogger logger;
        gpu_m2d::AllocationRegistry allocation_registry(run_id);
        TrackingGpuAllocator allocator(allocation_registry, options.device);
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
        }
        validate_registry_round_trips(allocation_registry);

        write_mapping_snapshot(options.output_prefix + "_mapping.csv", mapping,
                               options.device);

        CudaStream stream;
        allocator.set_phase("clean_inference");
        check_cuda(cudaMemcpy(binding_pointers[input_binding.index], input.data(),
                              input_binding.size_bytes, cudaMemcpyHostToDevice),
                   "copy clean input to device");
        const Prediction clean = run_inference(
            *context, binding_pointers, probability_binding.index,
            class_binding.index, stream);

        allocator.set_phase("injected_inference");
        check_cuda(cudaMemcpy(binding_pointers[input_binding.index], input.data(),
                              input_binding.size_bytes, cudaMemcpyHostToDevice),
                   "reset input before injection");
        const auto mapped = mapping.tensor_bit_to_gpu_va(
            input_binding.name, options.element_index, options.element_bit_index);
        const auto flip = gpu_m2d::flip_device_bit(
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

        const Prediction injected = run_inference(
            *context, binding_pointers, probability_binding.index,
            class_binding.index, stream);
        write_result(options.output_prefix + "_result.csv", options, sample, run_id,
                     mapped, flip, clean, injected);
        write_allocation_registry(options.output_prefix + "_allocations.csv",
                                  allocation_registry);

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

        allocator.set_phase("destroy_runtime_objects");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GPU_M2D_G1_5_FAIL: " << error.what() << '\n';
        return 1;
    }
}
