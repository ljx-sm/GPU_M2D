#ifndef GPU_M2D_TENSOR_MAPPING_HPP
#define GPU_M2D_TENSOR_MAPPING_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace gpu_m2d {

enum class DType {
    kUInt8,
    kInt8,
    kFloat32,
};

std::string_view dtype_name(DType dtype);
std::size_t dtype_size_bytes(DType dtype);

struct TensorDescriptor {
    std::string name;
    std::string tensor_type;
    std::vector<std::size_t> shape;
    // Row-major strides in elements. Empty means canonical contiguous strides.
    std::vector<std::size_t> strides;
    DType dtype{DType::kUInt8};
    std::string layout{"contiguous"};
    std::uintptr_t allocation_base_gpu_va{0};
    std::size_t allocation_size_bytes{0};
    std::string allocation_id;
    std::string lifetime;
};

struct TensorBitAddress {
    std::string tensor_name;
    std::size_t element_index{0};
    std::size_t element_bit_index{0};
    std::size_t byte_offset{0};
    std::uint8_t bit_in_byte{0};
    std::uint8_t xor_mask{0};
    std::uintptr_t gpu_va{0};
};

class MappingSnapshot {
public:
    explicit MappingSnapshot(std::string run_id);

    const std::string& run_id() const noexcept;
    const std::vector<TensorDescriptor>& tensors() const noexcept;

    void add_tensor(TensorDescriptor descriptor);

    TensorBitAddress tensor_bit_to_gpu_va(
        std::string_view tensor_name,
        std::size_t element_index,
        std::size_t element_bit_index) const;

    TensorBitAddress gpu_va_to_tensor_bit(
        std::uintptr_t gpu_byte_va,
        std::uint8_t bit_in_byte) const;

private:
    std::string run_id_;
    std::vector<TensorDescriptor> tensors_;
};

std::size_t tensor_element_count(const TensorDescriptor& descriptor);
std::size_t tensor_required_bytes(const TensorDescriptor& descriptor);
bool has_contiguous_strides(const TensorDescriptor& descriptor);

}  // namespace gpu_m2d

#endif  // GPU_M2D_TENSOR_MAPPING_HPP
