#include "gpu_m2d/tensor_mapping.hpp"

#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

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

gpu_m2d::TensorDescriptor make_tensor(
    std::string name,
    gpu_m2d::DType dtype,
    std::vector<std::size_t> shape,
    std::uintptr_t base,
    std::size_t allocation_size) {
    return gpu_m2d::TensorDescriptor{
        std::move(name),
        "weight",
        std::move(shape),
        {},
        dtype,
        "contiguous",
        base,
        allocation_size,
        "unit-allocation",
        "unit-test-lifetime",
    };
}

void run_tests() {
    using gpu_m2d::DType;
    using gpu_m2d::MappingSnapshot;

    MappingSnapshot snapshot("g1-unit-run");
    auto int8_tensor = make_tensor("layer1.weight", DType::kInt8, {2, 2},
                                   0x100000, 4);
    int8_tensor.strides = {2, 1};
    snapshot.add_tensor(int8_tensor);
    snapshot.add_tensor(make_tensor("layer2.weight", DType::kFloat32, {2},
                                    0x200000, 8));
    snapshot.add_tensor(make_tensor("output.index", DType::kInt32, {1},
                                    0x300000, 4));

    require(snapshot.run_id() == "g1-unit-run", "run_id was not preserved");
    require(snapshot.tensors().size() == 3, "tensor registry size mismatch");
    require(gpu_m2d::tensor_element_count(int8_tensor) == 4,
            "INT8 element count mismatch");
    require(gpu_m2d::tensor_required_bytes(int8_tensor) == 4,
            "INT8 byte count mismatch");
    require(gpu_m2d::has_contiguous_strides(int8_tensor),
            "canonical strides rejected");

    const auto int8_forward =
        snapshot.tensor_bit_to_gpu_va("layer1.weight", 3, 7);
    require(int8_forward.gpu_va == 0x100003, "INT8 forward GPU VA mismatch");
    require(int8_forward.byte_offset == 3, "INT8 byte offset mismatch");
    require(int8_forward.bit_in_byte == 7, "INT8 byte bit mismatch");
    require(int8_forward.xor_mask == 0x80, "INT8 XOR mask mismatch");

    const auto int8_reverse =
        snapshot.gpu_va_to_tensor_bit(int8_forward.gpu_va,
                                      int8_forward.bit_in_byte);
    require(int8_reverse.tensor_name == "layer1.weight",
            "INT8 reverse tensor mismatch");
    require(int8_reverse.element_index == 3, "INT8 reverse element mismatch");
    require(int8_reverse.element_bit_index == 7,
            "INT8 reverse element bit mismatch");

    const auto fp32_forward =
        snapshot.tensor_bit_to_gpu_va("layer2.weight", 1, 31);
    require(fp32_forward.gpu_va == 0x200007, "FP32 forward GPU VA mismatch");
    require(fp32_forward.byte_offset == 7, "FP32 byte offset mismatch");
    require(fp32_forward.bit_in_byte == 7, "FP32 byte bit mismatch");

    const auto fp32_reverse =
        snapshot.gpu_va_to_tensor_bit(fp32_forward.gpu_va,
                                      fp32_forward.bit_in_byte);
    require(fp32_reverse.element_index == 1, "FP32 reverse element mismatch");
    require(fp32_reverse.element_bit_index == 31,
            "FP32 reverse element bit mismatch");

    const auto int32_forward =
        snapshot.tensor_bit_to_gpu_va("output.index", 0, 30);
    require(int32_forward.gpu_va == 0x300003, "INT32 forward GPU VA mismatch");
    require(int32_forward.bit_in_byte == 6, "INT32 byte bit mismatch");
    const auto int32_reverse = snapshot.gpu_va_to_tensor_bit(
        int32_forward.gpu_va, int32_forward.bit_in_byte);
    require(int32_reverse.element_index == 0 &&
                int32_reverse.element_bit_index == 30,
            "INT32 reverse mapping mismatch");

    require_throws<std::out_of_range>(
        [&] { snapshot.tensor_bit_to_gpu_va("missing", 0, 0); },
        "unknown tensor was accepted");
    require_throws<std::out_of_range>(
        [&] { snapshot.tensor_bit_to_gpu_va("layer1.weight", 4, 0); },
        "out-of-range element was accepted");
    require_throws<std::out_of_range>(
        [&] { snapshot.tensor_bit_to_gpu_va("layer1.weight", 0, 8); },
        "out-of-range INT8 bit was accepted");
    require_throws<std::out_of_range>(
        [&] { snapshot.gpu_va_to_tensor_bit(0x900000, 0); },
        "unknown GPU VA was accepted");
    require_throws<std::out_of_range>(
        [&] { snapshot.gpu_va_to_tensor_bit(0x100000, 8); },
        "out-of-range byte bit was accepted");

    require_throws<std::invalid_argument>(
        [&] {
            auto overlap = make_tensor("overlap", DType::kUInt8, {2},
                                       0x100002, 2);
            snapshot.add_tensor(overlap);
        },
        "overlapping tensor range was accepted");

    require_throws<std::invalid_argument>(
        [] {
            MappingSnapshot local("noncontiguous");
            auto tensor = make_tensor("view", DType::kUInt8, {2, 2},
                                      0x400000, 8);
            tensor.strides = {3, 1};
            local.add_tensor(tensor);
        },
        "non-contiguous tensor was accepted in G1");

    require_throws<std::invalid_argument>(
        [] {
            MappingSnapshot local("undersized");
            local.add_tensor(make_tensor("short", DType::kFloat32, {2},
                                         0x500000, 7));
        },
        "undersized allocation was accepted");
}

}  // namespace

int main() {
    try {
        run_tests();
        std::cout << "GPU_M2D_G1_SEMANTIC_PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GPU_M2D_G1_SEMANTIC_FAIL: " << error.what() << '\n';
        return 1;
    }
}
