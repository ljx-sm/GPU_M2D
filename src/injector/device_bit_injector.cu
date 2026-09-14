#include "gpu_m2d/device_bit_injector.hpp"

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace gpu_m2d {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorName(status)
            << " (" << cudaGetErrorString(status) << ')';
    throw std::runtime_error(message.str());
}

__global__ void xor_byte_kernel(std::uint8_t* address, std::uint8_t mask) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *address ^= mask;
    }
}

}  // namespace

DevicePointerInfo inspect_device_pointer(const void* device_ptr,
                                         std::size_t allocation_size_bytes) {
    if (device_ptr == nullptr) {
        throw std::invalid_argument("device pointer must not be null");
    }
    if (allocation_size_bytes == 0) {
        throw std::invalid_argument("allocation size must not be zero");
    }

    cudaPointerAttributes attributes{};
    check_cuda(cudaPointerGetAttributes(&attributes, device_ptr),
               "cudaPointerGetAttributes");
    if (attributes.type != cudaMemoryTypeDevice) {
        throw std::invalid_argument("pointer is not CUDA device memory");
    }

    return DevicePointerInfo{
        reinterpret_cast<std::uintptr_t>(device_ptr),
        allocation_size_bytes,
        attributes.device,
    };
}

BitFlipResult flip_device_bit(void* device_ptr,
                              std::size_t allocation_size_bytes,
                              std::size_t byte_offset,
                              std::uint8_t bit_in_byte) {
    const DevicePointerInfo pointer_info =
        inspect_device_pointer(device_ptr, allocation_size_bytes);
    if (byte_offset >= allocation_size_bytes) {
        throw std::out_of_range("device byte offset is outside the allocation");
    }
    if (bit_in_byte >= 8) {
        throw std::out_of_range("bit_in_byte must be in [0, 7]");
    }

    auto* target = static_cast<std::uint8_t*>(device_ptr) + byte_offset;
    const std::uint8_t mask = static_cast<std::uint8_t>(1U << bit_in_byte);
    std::uint8_t before = 0;
    std::uint8_t after = 0;

    check_cuda(cudaMemcpy(&before, target, sizeof(before), cudaMemcpyDeviceToHost),
               "copy target byte before injection");
    xor_byte_kernel<<<1, 1>>>(target, mask);
    check_cuda(cudaGetLastError(), "launch XOR bit-flip kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize XOR bit-flip kernel");
    check_cuda(cudaMemcpy(&after, target, sizeof(after), cudaMemcpyDeviceToHost),
               "copy target byte after injection");

    const std::uint8_t expected = static_cast<std::uint8_t>(before ^ mask);
    if (after != expected) {
        throw std::runtime_error("device bit-flip postcondition failed");
    }

    return BitFlipResult{
        pointer_info.gpu_va + byte_offset,
        byte_offset,
        bit_in_byte,
        mask,
        before,
        after,
    };
}

}  // namespace gpu_m2d
