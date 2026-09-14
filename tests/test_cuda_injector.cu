#include "gpu_m2d/device_bit_injector.hpp"
#include "gpu_m2d/tensor_mapping.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception>
void require_throws(const std::function<void()>& action,
                    const std::string& message) {
    try {
        action();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

int parse_device(int argc, char** argv) {
    if (argc == 1) {
        return 0;
    }
    if (argc != 3 || std::string(argv[1]) != "--device") {
        throw std::invalid_argument("usage: test_cuda_injector [--device ID]");
    }
    return std::stoi(argv[2]);
}

gpu_m2d::TensorDescriptor make_descriptor(
    std::string name,
    gpu_m2d::DType dtype,
    std::vector<std::size_t> shape,
    const gpu_m2d::DevicePointerInfo& pointer,
    std::string allocation_id) {
    return gpu_m2d::TensorDescriptor{
        std::move(name),
        "controlled_test_tensor",
        std::move(shape),
        {},
        dtype,
        "contiguous",
        pointer.gpu_va,
        pointer.allocation_size_bytes,
        std::move(allocation_id),
        "cudaMalloc-to-cudaFree",
    };
}

void validate_int8_mapping_and_flip(int device) {
    const std::vector<std::uint8_t> baseline{
        0x00, 0x55, 0xAA, 0xFF, 0x13, 0x80, 0x7F, 0x01,
    };
    std::uint8_t* device_data = nullptr;
    check_cuda(cudaMalloc(&device_data, baseline.size()), "cudaMalloc INT8 buffer");

    try {
        const auto pointer =
            gpu_m2d::inspect_device_pointer(device_data, baseline.size());
        require(pointer.device_id == device, "CUDA pointer device mismatch");

        gpu_m2d::MappingSnapshot snapshot("g1-int8-device-" +
                                          std::to_string(device));
        snapshot.add_tensor(make_descriptor(
            "controlled.int8", gpu_m2d::DType::kInt8, {2, 4}, pointer,
            "cuda-int8-device-" + std::to_string(device)));

        const std::array<std::pair<std::size_t, std::size_t>, 5> cases{{
            {0, 0}, {1, 7}, {2, 2}, {5, 6}, {baseline.size() - 1, 3},
        }};
        for (const auto& test_case : cases) {
            check_cuda(cudaMemcpy(device_data, baseline.data(), baseline.size(),
                                  cudaMemcpyHostToDevice),
                       "reset INT8 device buffer");

            const auto mapped = snapshot.tensor_bit_to_gpu_va(
                "controlled.int8", test_case.first, test_case.second);
            const auto result = gpu_m2d::flip_device_bit(
                device_data, baseline.size(), mapped.byte_offset,
                mapped.bit_in_byte);

            std::vector<std::uint8_t> observed(baseline.size());
            check_cuda(cudaMemcpy(observed.data(), device_data, observed.size(),
                                  cudaMemcpyDeviceToHost),
                       "copy INT8 device buffer after injection");

            auto expected = baseline;
            expected[mapped.byte_offset] ^= mapped.xor_mask;
            require(observed == expected,
                    "target or guard byte mismatch after INT8 injection");
            require(result.gpu_va == mapped.gpu_va,
                    "injector and semantic mapper GPU VA mismatch");
            require(result.before == baseline[mapped.byte_offset],
                    "injector before byte mismatch");
            require(result.after == expected[mapped.byte_offset],
                    "injector after byte mismatch");

            const auto reversed = snapshot.gpu_va_to_tensor_bit(
                result.gpu_va, result.bit_in_byte);
            require(reversed.element_index == test_case.first,
                    "INT8 reverse element mismatch after injection");
            require(reversed.element_bit_index == test_case.second,
                    "INT8 reverse bit mismatch after injection");
        }

        require_throws<std::out_of_range>(
            [&] {
                gpu_m2d::flip_device_bit(device_data, baseline.size(),
                                         baseline.size(), 0);
            },
            "out-of-range device byte offset was accepted");
        require_throws<std::out_of_range>(
            [&] {
                gpu_m2d::flip_device_bit(device_data, baseline.size(), 0, 8);
            },
            "out-of-range device bit was accepted");
    } catch (...) {
        cudaFree(device_data);
        throw;
    }
    check_cuda(cudaFree(device_data), "cudaFree INT8 buffer");
}

void validate_fp32_cross_byte_mapping(int device) {
    const std::array<std::uint8_t, 8> baseline{{
        0x00, 0x55, 0xAA, 0xFF, 0x13, 0x24, 0x42, 0x7F,
    }};
    void* device_data = nullptr;
    check_cuda(cudaMalloc(&device_data, baseline.size()), "cudaMalloc FP32 buffer");

    try {
        const auto pointer =
            gpu_m2d::inspect_device_pointer(device_data, baseline.size());
        gpu_m2d::MappingSnapshot snapshot("g1-fp32-device-" +
                                          std::to_string(device));
        snapshot.add_tensor(make_descriptor(
            "controlled.fp32", gpu_m2d::DType::kFloat32, {2}, pointer,
            "cuda-fp32-device-" + std::to_string(device)));

        check_cuda(cudaMemcpy(device_data, baseline.data(), baseline.size(),
                              cudaMemcpyHostToDevice),
                   "initialize FP32 byte pattern");
        const auto mapped =
            snapshot.tensor_bit_to_gpu_va("controlled.fp32", 1, 31);
        require(mapped.byte_offset == 7 && mapped.bit_in_byte == 7,
                "FP32 element-bit to byte-bit mapping mismatch");
        const auto result = gpu_m2d::flip_device_bit(
            device_data, baseline.size(), mapped.byte_offset,
            mapped.bit_in_byte);
        require(result.before == 0x7F && result.after == 0xFF,
                "FP32 sign-bit byte transition mismatch");

        const auto reversed = snapshot.gpu_va_to_tensor_bit(
            result.gpu_va, result.bit_in_byte);
        require(reversed.element_index == 1 && reversed.element_bit_index == 31,
                "FP32 reverse mapping mismatch");
    } catch (...) {
        cudaFree(device_data);
        throw;
    }
    check_cuda(cudaFree(device_data), "cudaFree FP32 buffer");
}

void validate_new_snapshot_after_reallocation(int device) {
    constexpr std::size_t kBytes = 16;
    void* first = nullptr;
    void* second = nullptr;
    check_cuda(cudaMalloc(&first, kBytes), "cudaMalloc first lifetime");
    const auto first_info = gpu_m2d::inspect_device_pointer(first, kBytes);
    check_cuda(cudaFree(first), "cudaFree first lifetime");

    check_cuda(cudaMalloc(&second, kBytes), "cudaMalloc second lifetime");
    try {
        const auto second_info = gpu_m2d::inspect_device_pointer(second, kBytes);
        gpu_m2d::MappingSnapshot second_snapshot(
            "g1-reallocation-device-" + std::to_string(device));
        second_snapshot.add_tensor(make_descriptor(
            "reallocated.tensor", gpu_m2d::DType::kUInt8, {kBytes}, second_info,
            "cuda-reallocation-device-" + std::to_string(device)));
        const auto mapped = second_snapshot.tensor_bit_to_gpu_va(
            "reallocated.tensor", kBytes - 1, 7);
        require(mapped.gpu_va == second_info.gpu_va + kBytes - 1,
                "reallocated snapshot did not use current GPU VA");

        std::cout << "GPU_M2D_G1_REALLOCATION device=" << device
                  << " old_gpu_va=0x" << std::hex << first_info.gpu_va
                  << " new_gpu_va=0x" << second_info.gpu_va << std::dec << '\n';
    } catch (...) {
        cudaFree(second);
        throw;
    }
    check_cuda(cudaFree(second), "cudaFree second lifetime");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int device = parse_device(argc, argv);
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device < 0 || device >= device_count) {
            throw std::out_of_range("requested CUDA device does not exist");
        }
        check_cuda(cudaSetDevice(device), "cudaSetDevice");

        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, device),
                   "cudaGetDeviceProperties");
        validate_int8_mapping_and_flip(device);
        validate_fp32_cross_byte_mapping(device);
        validate_new_snapshot_after_reallocation(device);

        std::cout << "GPU_M2D_G1_CUDA_PASS device=" << device
                  << " name=\"" << properties.name << "\""
                  << " compute_capability=" << properties.major << '.'
                  << properties.minor << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GPU_M2D_G1_CUDA_FAIL: " << error.what() << '\n';
        return 1;
    }
}
