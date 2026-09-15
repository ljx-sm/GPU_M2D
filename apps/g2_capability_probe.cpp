#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

struct Options {
    int device{0};
    std::size_t allocation_bytes{2U * 1024U * 1024U};
};

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(status));
    }
}

void check_driver(CUresult status, const char* operation) {
    if (status == CUDA_SUCCESS) {
        return;
    }
    const char* name = nullptr;
    const char* description = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &description);
    throw std::runtime_error(std::string(operation) + " failed: " +
                             (name == nullptr ? "UNKNOWN" : name) + " (" +
                             (description == nullptr ? "no description" : description) +
                             ")");
}

std::string driver_status(CUresult status) {
    const char* name = nullptr;
    cuGetErrorName(status, &name);
    return name == nullptr ? "UNKNOWN" : name;
}

std::size_t parse_size(const std::string& text, const char* option) {
    std::size_t consumed = 0;
    const unsigned long long value = std::stoull(text, &consumed, 10);
    if (consumed != text.size() || value == 0 ||
        value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string("invalid value for ") + option);
    }
    return static_cast<std::size_t>(value);
}

int parse_int(const std::string& text, const char* option) {
    std::size_t consumed = 0;
    const long value = std::stol(text, &consumed, 10);
    if (consumed != text.size() || value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string("invalid value for ") + option);
    }
    return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + option);
        }
        const std::string value = argv[++index];
        if (option == "--device") {
            options.device = parse_int(value, "--device");
        }
        else if (option == "--allocation-bytes") {
            options.allocation_bytes = parse_size(value, "--allocation-bytes");
        }
        else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    return options;
}

class DeviceAllocation {
public:
    explicit DeviceAllocation(std::size_t size) : size_(size) {
        check_cuda(cudaMalloc(&pointer_, size_), "cudaMalloc probe allocation");
    }

    ~DeviceAllocation() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    void* get() const noexcept { return pointer_; }
    std::size_t size() const noexcept { return size_; }

private:
    void* pointer_{nullptr};
    std::size_t size_{0};
};

void print_attribute(CUdevice device, CUdevice_attribute attribute,
                     const char* name) {
    int value = -1;
    const CUresult status = cuDeviceGetAttribute(&value, attribute, device);
    std::cout << name << "_status=" << driver_status(status) << '\n';
    if (status == CUDA_SUCCESS) {
        std::cout << name << '=' << value << '\n';
    }
}

template <typename T>
CUresult query_pointer_attribute(T* output, CUpointer_attribute attribute,
                                 const void* pointer) {
    return cuPointerGetAttribute(output, attribute,
                                 reinterpret_cast<CUdeviceptr>(pointer));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        check_driver(cuInit(0), "cuInit");

        int device_count = 0;
        check_driver(cuDeviceGetCount(&device_count), "cuDeviceGetCount");
        if (options.device < 0 || options.device >= device_count) {
            throw std::out_of_range("requested CUDA device does not exist");
        }
        check_cuda(cudaSetDevice(options.device), "cudaSetDevice");

        CUdevice device = 0;
        check_driver(cuDeviceGet(&device, options.device), "cuDeviceGet");
        char device_name[256]{};
        check_driver(cuDeviceGetName(device_name, sizeof(device_name), device),
                     "cuDeviceGetName");

        char pci_bus_id[32]{};
        check_cuda(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id),
                                         options.device),
                   "cudaDeviceGetPCIBusId");
        std::cout << "probe=GPU_M2D_G2_CAPABILITY\n"
                  << "device=" << options.device << '\n'
                  << "device_name=" << device_name << '\n'
                  << "pci_bus_id=" << pci_bus_id << '\n';

        print_attribute(device,
                        CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                        "vmm_supported");
        print_attribute(
            device, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
            "vmm_posix_fd_supported");
        print_attribute(
            device, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
            "gpudirect_rdma_with_vmm_supported");
        print_attribute(device, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED,
                        "gpudirect_rdma_supported");
        print_attribute(device, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
                        "dma_buf_supported");

        CUmemAllocationProp properties{};
        properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        properties.location.id = options.device;
        properties.requestedHandleTypes =
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        std::size_t minimum_granularity = 0;
        const CUresult granularity_status = cuMemGetAllocationGranularity(
            &minimum_granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        std::cout << "vmm_minimum_granularity_status="
                  << driver_status(granularity_status) << '\n';
        if (granularity_status == CUDA_SUCCESS) {
            std::cout << "vmm_minimum_granularity_bytes="
                      << minimum_granularity << '\n';
        }

        DeviceAllocation allocation(options.allocation_bytes);
        check_cuda(cudaMemset(allocation.get(), 0xA5, allocation.size()),
                   "initialize probe allocation");
        check_cuda(cudaDeviceSynchronize(), "synchronize probe allocation");
        std::cout << "allocation_gpu_va=0x" << std::hex
                  << reinterpret_cast<std::uintptr_t>(allocation.get()) << std::dec
                  << '\n'
                  << "allocation_bytes=" << allocation.size() << '\n';

        unsigned int memory_type = 0;
        CUresult status = query_pointer_attribute(
            &memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, allocation.get());
        std::cout << "pointer_memory_type_status=" << driver_status(status) << '\n';
        if (status == CUDA_SUCCESS) {
            std::cout << "pointer_memory_type=" << memory_type << '\n';
        }

        int pointer_device = -1;
        status = query_pointer_attribute(
            &pointer_device, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, allocation.get());
        std::cout << "pointer_device_status=" << driver_status(status) << '\n';
        if (status == CUDA_SUCCESS) {
            std::cout << "pointer_device=" << pointer_device << '\n';
        }

        unsigned long long buffer_id = 0;
        status = query_pointer_attribute(
            &buffer_id, CU_POINTER_ATTRIBUTE_BUFFER_ID, allocation.get());
        std::cout << "pointer_buffer_id_status=" << driver_status(status) << '\n';
        if (status == CUDA_SUCCESS) {
            std::cout << "pointer_buffer_id=" << buffer_id << '\n';
        }

        CUdeviceptr range_start = 0;
        status = query_pointer_attribute(
            &range_start, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, allocation.get());
        std::cout << "pointer_range_start_status=" << driver_status(status) << '\n';
        if (status == CUDA_SUCCESS) {
            std::cout << "pointer_range_start=0x" << std::hex << range_start
                      << std::dec << '\n';
        }

        std::size_t range_size = 0;
        status = query_pointer_attribute(
            &range_size, CU_POINTER_ATTRIBUTE_RANGE_SIZE, allocation.get());
        std::cout << "pointer_range_size_status=" << driver_status(status) << '\n';
        if (status == CUDA_SUCCESS) {
            std::cout << "pointer_range_size=" << range_size << '\n';
        }

        unsigned int rdma_capable = 0;
        status = query_pointer_attribute(
            &rdma_capable, CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE,
            allocation.get());
        std::cout << "pointer_gpudirect_rdma_capable_status="
                  << driver_status(status) << '\n';
        if (status == CUDA_SUCCESS) {
            std::cout << "pointer_gpudirect_rdma_capable=" << rdma_capable << '\n';
        }

        CUDA_POINTER_ATTRIBUTE_P2P_TOKENS tokens{};
        status = query_pointer_attribute(
            &tokens, CU_POINTER_ATTRIBUTE_P2P_TOKENS, allocation.get());
        std::cout << "pointer_p2p_tokens_status=" << driver_status(status) << '\n';
        if (status == CUDA_SUCCESS) {
            std::cout << "pointer_p2p_token_nonzero="
                      << (tokens.p2pToken != 0 ? 1 : 0) << '\n'
                      << "pointer_va_space_token_nonzero="
                      << (tokens.vaSpaceToken != 0 ? 1 : 0) << '\n';
        }

        const bool public_pa_path_available =
            rdma_capable != 0 && status == CUDA_SUCCESS;
        std::cout << "public_gpudirect_pa_candidate="
                  << (public_pa_path_available ? 1 : 0) << '\n'
                  << "GPU_M2D_G2_CAPABILITY_PASS\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "GPU_M2D_G2_CAPABILITY_FAIL: " << error.what() << '\n';
        return 1;
    }
}
