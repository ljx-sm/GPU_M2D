#include "gpu_m2d/allocation_registry.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gpu_m2d {
namespace {

std::uintptr_t inclusive_end(const AllocationDescriptor& descriptor) {
    const std::size_t final_offset = descriptor.size_bytes - 1;
    if (final_offset >
        std::numeric_limits<std::uintptr_t>::max() - descriptor.base_gpu_va) {
        throw std::overflow_error("allocation GPU virtual address range overflows");
    }
    return descriptor.base_gpu_va + static_cast<std::uintptr_t>(final_offset);
}

void validate_bit(std::uint8_t bit_in_byte) {
    if (bit_in_byte >= 8) {
        throw std::out_of_range("bit_in_byte must be in [0, 7]");
    }
}

AllocationBitAddress make_address(const AllocationDescriptor& descriptor,
                                  std::size_t byte_offset,
                                  std::uint8_t bit_in_byte,
                                  std::uintptr_t gpu_va) {
    return AllocationBitAddress{
        descriptor.allocation_id,
        descriptor.device_id,
        byte_offset,
        bit_in_byte,
        static_cast<std::uint8_t>(1U << bit_in_byte),
        gpu_va,
        descriptor.owner,
        descriptor.semantic_label,
    };
}

}  // namespace

AllocationRegistry::AllocationRegistry(std::string run_id)
    : run_id_(std::move(run_id)) {
    if (run_id_.empty()) {
        throw std::invalid_argument("allocation registry run_id must not be empty");
    }
}

const std::string& AllocationRegistry::run_id() const noexcept {
    return run_id_;
}

std::vector<AllocationDescriptor> AllocationRegistry::allocations() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocations_;
}

void AllocationRegistry::add_allocation(AllocationDescriptor descriptor) {
    if (descriptor.allocation_id.empty()) {
        throw std::invalid_argument("allocation_id must not be empty");
    }
    if (descriptor.device_id < 0) {
        throw std::invalid_argument("device_id must not be negative");
    }
    if (descriptor.base_gpu_va == 0) {
        throw std::invalid_argument("GPU virtual address must not be zero");
    }
    if (descriptor.size_bytes == 0) {
        throw std::invalid_argument("allocation size must not be zero");
    }
    if (descriptor.owner.empty() || descriptor.semantic_label.empty() ||
        descriptor.allocation_phase.empty() || descriptor.lifetime.empty()) {
        throw std::invalid_argument("allocation provenance fields must not be empty");
    }
    if (!descriptor.active) {
        throw std::invalid_argument("new allocation must be active");
    }

    const std::uintptr_t descriptor_end = inclusive_end(descriptor);
    std::lock_guard<std::mutex> lock(mutex_);
    for (const AllocationDescriptor& existing : allocations_) {
        if (existing.allocation_id == descriptor.allocation_id) {
            throw std::invalid_argument("duplicate allocation_id: " +
                                        descriptor.allocation_id);
        }
        if (!existing.active || existing.device_id != descriptor.device_id) {
            continue;
        }
        const std::uintptr_t existing_end = inclusive_end(existing);
        const bool overlaps = descriptor.base_gpu_va <= existing_end &&
                              existing.base_gpu_va <= descriptor_end;
        if (overlaps) {
            throw std::invalid_argument(
                "active allocation GPU virtual address ranges overlap");
        }
    }
    allocations_.push_back(std::move(descriptor));
}

bool AllocationRegistry::deactivate_allocation(
    std::string_view allocation_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto allocation = std::find_if(
        allocations_.begin(), allocations_.end(),
        [allocation_id](const AllocationDescriptor& candidate) {
            return candidate.allocation_id == allocation_id;
        });
    if (allocation == allocations_.end() || !allocation->active) {
        return false;
    }
    allocation->active = false;
    return true;
}

AllocationBitAddress AllocationRegistry::allocation_bit_to_gpu_va(
    std::string_view allocation_id,
    std::size_t byte_offset,
    std::uint8_t bit_in_byte) const {
    validate_bit(bit_in_byte);
    std::lock_guard<std::mutex> lock(mutex_);
    const auto allocation = std::find_if(
        allocations_.begin(), allocations_.end(),
        [allocation_id](const AllocationDescriptor& candidate) {
            return candidate.allocation_id == allocation_id;
        });
    if (allocation == allocations_.end()) {
        throw std::out_of_range("unknown allocation_id: " +
                                std::string(allocation_id));
    }
    if (!allocation->active) {
        throw std::out_of_range("allocation is inactive: " +
                                std::string(allocation_id));
    }
    if (byte_offset >= allocation->size_bytes) {
        throw std::out_of_range("allocation byte offset is out of range");
    }
    const std::uintptr_t gpu_va = allocation->base_gpu_va +
                                  static_cast<std::uintptr_t>(byte_offset);
    return make_address(*allocation, byte_offset, bit_in_byte, gpu_va);
}

AllocationBitAddress AllocationRegistry::gpu_va_to_allocation_bit(
    int device_id,
    std::uintptr_t gpu_byte_va,
    std::uint8_t bit_in_byte) const {
    if (device_id < 0) {
        throw std::invalid_argument("device_id must not be negative");
    }
    validate_bit(bit_in_byte);

    std::lock_guard<std::mutex> lock(mutex_);
    for (const AllocationDescriptor& allocation : allocations_) {
        if (!allocation.active || allocation.device_id != device_id ||
            gpu_byte_va < allocation.base_gpu_va) {
            continue;
        }
        const std::uintptr_t difference = gpu_byte_va - allocation.base_gpu_va;
        if (difference >= allocation.size_bytes) {
            continue;
        }
        const std::size_t byte_offset = static_cast<std::size_t>(difference);
        return make_address(allocation, byte_offset, bit_in_byte, gpu_byte_va);
    }
    throw std::out_of_range(
        "GPU virtual address is not in an active allocation on this device");
}

}  // namespace gpu_m2d
