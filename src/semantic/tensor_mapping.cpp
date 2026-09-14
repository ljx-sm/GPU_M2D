#include "gpu_m2d/tensor_mapping.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gpu_m2d {
namespace {

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs,
                             std::string_view context) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(std::string(context) + " overflows size_t");
    }
    return lhs * rhs;
}

std::uintptr_t checked_address_add(std::uintptr_t base, std::size_t offset,
                                   std::string_view context) {
    if (offset > std::numeric_limits<std::uintptr_t>::max() - base) {
        throw std::overflow_error(std::string(context) + " overflows uintptr_t");
    }
    return base + static_cast<std::uintptr_t>(offset);
}

const TensorDescriptor& find_tensor(
    const std::vector<TensorDescriptor>& tensors,
    std::string_view tensor_name) {
    const auto it = std::find_if(
        tensors.begin(), tensors.end(),
        [tensor_name](const TensorDescriptor& descriptor) {
            return descriptor.name == tensor_name;
        });
    if (it == tensors.end()) {
        throw std::out_of_range("unknown tensor: " + std::string(tensor_name));
    }
    return *it;
}

}  // namespace

std::string_view dtype_name(DType dtype) {
    switch (dtype) {
        case DType::kUInt8:
            return "uint8";
        case DType::kInt8:
            return "int8";
        case DType::kFloat32:
            return "float32";
        case DType::kInt32:
            return "int32";
    }
    throw std::invalid_argument("unsupported dtype");
}

std::size_t dtype_size_bytes(DType dtype) {
    switch (dtype) {
        case DType::kUInt8:
        case DType::kInt8:
            return 1;
        case DType::kFloat32:
        case DType::kInt32:
            return 4;
    }
    throw std::invalid_argument("unsupported dtype");
}

std::size_t tensor_element_count(const TensorDescriptor& descriptor) {
    if (descriptor.shape.empty()) {
        throw std::invalid_argument("tensor shape must not be empty");
    }

    std::size_t count = 1;
    for (const std::size_t dimension : descriptor.shape) {
        if (dimension == 0) {
            throw std::invalid_argument("zero-sized tensor dimensions are not supported in G1");
        }
        count = checked_multiply(count, dimension, "tensor element count");
    }
    return count;
}

std::size_t tensor_required_bytes(const TensorDescriptor& descriptor) {
    return checked_multiply(tensor_element_count(descriptor),
                            dtype_size_bytes(descriptor.dtype),
                            "tensor byte size");
}

bool has_contiguous_strides(const TensorDescriptor& descriptor) {
    if (descriptor.strides.empty()) {
        return true;
    }
    if (descriptor.strides.size() != descriptor.shape.size()) {
        return false;
    }

    std::size_t expected_stride = 1;
    for (std::size_t index = descriptor.shape.size(); index-- > 0;) {
        if (descriptor.strides[index] != expected_stride) {
            return false;
        }
        expected_stride = checked_multiply(expected_stride,
                                           descriptor.shape[index],
                                           "contiguous stride");
    }
    return true;
}

MappingSnapshot::MappingSnapshot(std::string run_id)
    : run_id_(std::move(run_id)) {
    if (run_id_.empty()) {
        throw std::invalid_argument("mapping snapshot run_id must not be empty");
    }
}

const std::string& MappingSnapshot::run_id() const noexcept {
    return run_id_;
}

const std::vector<TensorDescriptor>& MappingSnapshot::tensors() const noexcept {
    return tensors_;
}

void MappingSnapshot::add_tensor(TensorDescriptor descriptor) {
    if (descriptor.name.empty()) {
        throw std::invalid_argument("tensor name must not be empty");
    }
    if (descriptor.allocation_id.empty()) {
        throw std::invalid_argument("allocation_id must not be empty");
    }
    if (descriptor.lifetime.empty()) {
        throw std::invalid_argument("tensor lifetime must not be empty");
    }
    if (descriptor.allocation_base_gpu_va == 0) {
        throw std::invalid_argument("GPU virtual address must not be zero");
    }
    if (!has_contiguous_strides(descriptor)) {
        throw std::invalid_argument("G1 only supports contiguous tensors");
    }

    const std::size_t required_bytes = tensor_required_bytes(descriptor);
    if (descriptor.allocation_size_bytes < required_bytes) {
        throw std::invalid_argument("allocation is smaller than the tensor byte range");
    }
    const std::uintptr_t descriptor_end = checked_address_add(
        descriptor.allocation_base_gpu_va, required_bytes,
        "tensor address range");

    for (const TensorDescriptor& existing : tensors_) {
        if (existing.name == descriptor.name) {
            throw std::invalid_argument("duplicate tensor name: " + descriptor.name);
        }
        const std::uintptr_t existing_end = checked_address_add(
            existing.allocation_base_gpu_va, tensor_required_bytes(existing),
            "existing tensor address range");
        const bool overlaps = descriptor.allocation_base_gpu_va < existing_end &&
                              existing.allocation_base_gpu_va < descriptor_end;
        if (overlaps) {
            throw std::invalid_argument("tensor GPU virtual address ranges overlap");
        }
    }

    tensors_.push_back(std::move(descriptor));
}

TensorBitAddress MappingSnapshot::tensor_bit_to_gpu_va(
    std::string_view tensor_name,
    std::size_t element_index,
    std::size_t element_bit_index) const {
    const TensorDescriptor& descriptor = find_tensor(tensors_, tensor_name);
    const std::size_t element_count = tensor_element_count(descriptor);
    if (element_index >= element_count) {
        throw std::out_of_range("tensor element index is out of range");
    }

    const std::size_t element_size = dtype_size_bytes(descriptor.dtype);
    const std::size_t bits_per_element = checked_multiply(element_size, 8,
                                                          "element bit width");
    if (element_bit_index >= bits_per_element) {
        throw std::out_of_range("tensor bit index is out of range for dtype");
    }

    const std::size_t byte_offset = checked_multiply(
        element_index, element_size, "element byte offset") +
        element_bit_index / 8;
    const std::uint8_t bit_in_byte =
        static_cast<std::uint8_t>(element_bit_index % 8);

    return TensorBitAddress{
        descriptor.name,
        element_index,
        element_bit_index,
        byte_offset,
        bit_in_byte,
        static_cast<std::uint8_t>(1U << bit_in_byte),
        checked_address_add(descriptor.allocation_base_gpu_va, byte_offset,
                            "mapped GPU virtual address"),
    };
}

TensorBitAddress MappingSnapshot::gpu_va_to_tensor_bit(
    std::uintptr_t gpu_byte_va,
    std::uint8_t bit_in_byte) const {
    if (bit_in_byte >= 8) {
        throw std::out_of_range("bit_in_byte must be in [0, 7]");
    }

    for (const TensorDescriptor& descriptor : tensors_) {
        const std::size_t required_bytes = tensor_required_bytes(descriptor);
        const std::uintptr_t end = checked_address_add(
            descriptor.allocation_base_gpu_va, required_bytes,
            "tensor reverse-lookup range");
        if (gpu_byte_va < descriptor.allocation_base_gpu_va || gpu_byte_va >= end) {
            continue;
        }

        const std::size_t byte_offset = static_cast<std::size_t>(
            gpu_byte_va - descriptor.allocation_base_gpu_va);
        const std::size_t element_size = dtype_size_bytes(descriptor.dtype);
        const std::size_t element_index = byte_offset / element_size;
        const std::size_t byte_in_element = byte_offset % element_size;
        const std::size_t element_bit_index = byte_in_element * 8 + bit_in_byte;

        return TensorBitAddress{
            descriptor.name,
            element_index,
            element_bit_index,
            byte_offset,
            bit_in_byte,
            static_cast<std::uint8_t>(1U << bit_in_byte),
            gpu_byte_va,
        };
    }

    throw std::out_of_range("GPU virtual address is not present in this snapshot");
}

}  // namespace gpu_m2d
