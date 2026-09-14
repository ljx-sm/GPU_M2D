#ifndef GPU_M2D_ALLOCATION_REGISTRY_HPP
#define GPU_M2D_ALLOCATION_REGISTRY_HPP

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace gpu_m2d {

struct AllocationDescriptor {
    std::string allocation_id;
    int device_id{-1};
    std::uintptr_t base_gpu_va{0};
    std::size_t size_bytes{0};
    std::size_t alignment_bytes{0};
    std::string owner;
    std::string semantic_label;
    std::string allocation_phase;
    std::string lifetime;
    bool active{true};
};

struct AllocationBitAddress {
    std::string allocation_id;
    int device_id{-1};
    std::size_t byte_offset{0};
    std::uint8_t bit_in_byte{0};
    std::uint8_t xor_mask{0};
    std::uintptr_t gpu_va{0};
    std::string owner;
    std::string semantic_label;
};

// A run-scoped registry of CUDA device allocations. Reverse lookup only returns
// active allocations, preventing a stale or already-freed GPU VA from being
// accepted as an injection target.
class AllocationRegistry {
public:
    explicit AllocationRegistry(std::string run_id);

    const std::string& run_id() const noexcept;
    std::vector<AllocationDescriptor> allocations() const;

    void add_allocation(AllocationDescriptor descriptor);
    bool deactivate_allocation(std::string_view allocation_id);

    AllocationBitAddress allocation_bit_to_gpu_va(
        std::string_view allocation_id,
        std::size_t byte_offset,
        std::uint8_t bit_in_byte) const;

    AllocationBitAddress gpu_va_to_allocation_bit(
        int device_id,
        std::uintptr_t gpu_byte_va,
        std::uint8_t bit_in_byte) const;

private:
    std::string run_id_;
    mutable std::mutex mutex_;
    std::vector<AllocationDescriptor> allocations_;
};

}  // namespace gpu_m2d

#endif  // GPU_M2D_ALLOCATION_REGISTRY_HPP
