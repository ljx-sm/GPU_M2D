#include "gpu_m2d/allocation_registry.hpp"

#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

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

gpu_m2d::AllocationDescriptor make_allocation(
    std::string id, int device, std::uintptr_t base, std::size_t size,
    std::string semantic_label = "TENSORRT_INTERNAL_UNKNOWN") {
    return gpu_m2d::AllocationDescriptor{
        std::move(id), device, base, size, 256, "unit-test",
        std::move(semantic_label), "unit-test-phase", "unit-test-lifetime", true};
}

void run_tests() {
    gpu_m2d::AllocationRegistry registry("allocation-registry-unit-run");
    registry.add_allocation(
        make_allocation("binding-data", 0, 0x100000, 16, "TENSOR:data"));
    registry.add_allocation(
        make_allocation("trt-internal-0", 0, 0x200000, 32));

    require(registry.run_id() == "allocation-registry-unit-run",
            "run_id was not preserved");
    require(registry.allocations().size() == 2,
            "allocation inventory size mismatch");

    const auto first = registry.allocation_bit_to_gpu_va("binding-data", 0, 0);
    require(first.gpu_va == 0x100000 && first.xor_mask == 0x01,
            "first-bit forward lookup mismatch");
    require(first.semantic_label == "TENSOR:data",
            "exact semantic label was not preserved");

    const auto last = registry.allocation_bit_to_gpu_va("binding-data", 15, 7);
    require(last.gpu_va == 0x10000f && last.xor_mask == 0x80,
            "last-bit forward lookup mismatch");
    const auto reversed = registry.gpu_va_to_allocation_bit(0, last.gpu_va, 7);
    require(reversed.allocation_id == "binding-data" &&
                reversed.byte_offset == 15 && reversed.bit_in_byte == 7,
            "allocation reverse lookup mismatch");

    const auto unknown_semantic =
        registry.gpu_va_to_allocation_bit(0, 0x200011, 3);
    require(unknown_semantic.allocation_id == "trt-internal-0" &&
                unknown_semantic.byte_offset == 17 &&
                unknown_semantic.semantic_label == "TENSORRT_INTERNAL_UNKNOWN",
            "semantically unknown allocation did not remain addressable");

    require_throws<std::out_of_range>(
        [&] { registry.allocation_bit_to_gpu_va("binding-data", 16, 0); },
        "out-of-range allocation offset was accepted");
    require_throws<std::out_of_range>(
        [&] { registry.allocation_bit_to_gpu_va("binding-data", 0, 8); },
        "out-of-range forward bit was accepted");
    require_throws<std::out_of_range>(
        [&] { registry.gpu_va_to_allocation_bit(0, 0x100000, 8); },
        "out-of-range reverse bit was accepted");
    require_throws<std::out_of_range>(
        [&] { registry.allocation_bit_to_gpu_va("missing", 0, 0); },
        "unknown allocation ID was accepted");
    require_throws<std::out_of_range>(
        [&] { registry.gpu_va_to_allocation_bit(0, 0x900000, 0); },
        "unregistered GPU VA was accepted");

    require_throws<std::invalid_argument>(
        [&] { registry.add_allocation(
                  make_allocation("overlap", 0, 0x10000f, 2)); },
        "same-device active overlap was accepted");
    registry.add_allocation(make_allocation("adjacent", 0, 0x100010, 16));

    // Numeric CUDA VAs are meaningful only together with a device/context.
    registry.add_allocation(
        make_allocation("same-va-other-device", 1, 0x100000, 16));
    require(registry.gpu_va_to_allocation_bit(1, 0x100005, 2).allocation_id ==
                "same-va-other-device",
            "device-qualified reverse lookup was not disambiguated");
    require_throws<std::out_of_range>(
        [&] { registry.gpu_va_to_allocation_bit(2, 0x100005, 2); },
        "wrong-device GPU VA was accepted");

    require(registry.deactivate_allocation("binding-data"),
            "active allocation was not deactivated");
    require(!registry.deactivate_allocation("binding-data"),
            "already-inactive allocation was deactivated twice");
    require(!registry.deactivate_allocation("missing"),
            "unknown allocation was reported as deactivated");
    require_throws<std::out_of_range>(
        [&] { registry.allocation_bit_to_gpu_va("binding-data", 0, 0); },
        "inactive allocation ID remained forward-addressable");
    require_throws<std::out_of_range>(
        [&] { registry.gpu_va_to_allocation_bit(0, 0x100005, 0); },
        "stale GPU VA remained reverse-addressable");

    registry.add_allocation(
        make_allocation("binding-data-reused", 0, 0x100000, 16,
                        "TENSOR:data-next-lifetime"));
    require(registry.gpu_va_to_allocation_bit(0, 0x100005, 0).allocation_id ==
                "binding-data-reused",
            "new lifetime did not own a reused GPU VA");
    require_throws<std::invalid_argument>(
        [&] { registry.add_allocation(
                  make_allocation("binding-data", 0, 0x300000, 4)); },
        "historical allocation ID was reused");

    require_throws<std::invalid_argument>(
        [&] { registry.add_allocation(
                  make_allocation("zero-size", 0, 0x400000, 0)); },
        "zero-size allocation was accepted");
    require_throws<std::invalid_argument>(
        [&] { registry.add_allocation(
                  make_allocation("zero-va", 0, 0, 1)); },
        "zero GPU VA was accepted");
    require_throws<std::overflow_error>(
        [&] {
            registry.add_allocation(make_allocation(
                "overflow", 0, std::numeric_limits<std::uintptr_t>::max(), 2));
        },
        "overflowing GPU VA range was accepted");
}

}  // namespace

int main() {
    try {
        run_tests();
        std::cout << "GPU_M2D_ALLOCATION_REGISTRY_PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GPU_M2D_ALLOCATION_REGISTRY_FAIL: " << error.what() << '\n';
        return 1;
    }
}
