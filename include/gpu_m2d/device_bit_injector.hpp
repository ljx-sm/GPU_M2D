#ifndef GPU_M2D_DEVICE_BIT_INJECTOR_HPP
#define GPU_M2D_DEVICE_BIT_INJECTOR_HPP

#include <cstddef>
#include <cstdint>

namespace gpu_m2d {

struct DevicePointerInfo {
    std::uintptr_t gpu_va{0};
    std::size_t allocation_size_bytes{0};
    int device_id{-1};
};

struct BitFlipResult {
    std::uintptr_t gpu_va{0};
    std::size_t byte_offset{0};
    std::uint8_t bit_in_byte{0};
    std::uint8_t xor_mask{0};
    std::uint8_t before{0};
    std::uint8_t after{0};
};

DevicePointerInfo inspect_device_pointer(const void* device_ptr,
                                         std::size_t allocation_size_bytes);

BitFlipResult flip_device_bit(void* device_ptr,
                              std::size_t allocation_size_bytes,
                              std::size_t byte_offset,
                              std::uint8_t bit_in_byte);

}  // namespace gpu_m2d

#endif  // GPU_M2D_DEVICE_BIT_INJECTOR_HPP
