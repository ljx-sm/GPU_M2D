#ifndef GPU_M2D_RESIDENCY_MAP_HPP
#define GPU_M2D_RESIDENCY_MAP_HPP

// G8 L2 residency map (docs/G8_CACHE_FAULT_PLAN.md §3.2, §3.4, §4).
//
// Turns the per-unit hit/miss observations of an L2 probe pass into
// residency PERIODS: maximal runs of images [start, end] (inclusive image
// indices) during which a unit (32-B sector or 128-B line) was resident.
// Pure host C++ (no CUDA), unit-tested in tests/test_residency_map.cpp.
//
// Observation rule (a probe at the boundary BEFORE image b):
//   - observations of one unit arrive in strictly increasing b;
//   - a hit opens a period at b, or at image 0 when it is the unit's first
//     observation (the images before a unit's first observation are
//     unobserved and take the first observation's state);
//   - a miss closes an open period at b - 1;
//   - finish() closes every open period at the last image.
// So a hit at b covers b up to the image before the unit's next miss
// observation: with probes every k images and stride s, a unit's state is
// known to a resolution of k * s images.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gpu_m2d {

struct ResidencyPeriod {
    std::uint32_t start{0};  // first resident image (inclusive)
    std::uint32_t end{0};    // last resident image (inclusive)
};

class ResidencyMapBuilder {
public:
    ResidencyMapBuilder(std::uint64_t units, std::uint32_t images);

    void observe(std::uint64_t unit, std::uint32_t image, bool resident);
    void finish();

    std::uint64_t units() const noexcept { return units_; }
    std::uint32_t images() const noexcept { return images_; }
    bool finished() const noexcept { return finished_; }
    // After finish(): periods of unit u are
    // periods()[offsets()[u] .. offsets()[u + 1]), sorted by start.
    const std::vector<std::uint64_t>& offsets() const;
    const std::vector<ResidencyPeriod>& periods() const;
    std::uint64_t observed_units() const noexcept { return observed_units_; }

    // Total resident inference time of each unit: the sum of
    // image_time[i] over the images in its periods (plan §3.2 T_l).
    std::vector<double> resident_time(const std::vector<double>& image_time) const;
    // Time-averaged resident bits (plan §3.2):
    //   R_eff_bits = sum_l(bits_l * T_l) / T_total,
    // with bits_l = 8 * unit_resident_bytes[l] and T_total the sum of all
    // image times.
    double r_eff_bits(const std::vector<double>& image_time,
                      const std::vector<std::uint16_t>& unit_resident_bytes) const;

    // Writes the binary map (format in tools/g8_cache/residency_map.py):
    // magic "G8RMAP01", u32 version 1, u32 unit_bytes, u64 units,
    // u64 images, u64 periods, f64 image_time[images],
    // u16 unit_resident_bytes[units], u64 offsets[units + 1],
    // u32 (start, end)[periods]; little-endian.
    void write_binary(const std::string& path, std::uint32_t unit_bytes,
                      const std::vector<double>& image_time,
                      const std::vector<std::uint16_t>& unit_resident_bytes) const;

private:
    static constexpr std::uint32_t kNone = 0xffffffffu;
    struct Tagged {
        std::uint32_t unit;
        ResidencyPeriod period;
    };
    void require_finished() const;

    std::uint64_t units_;
    std::uint32_t images_;
    bool finished_{false};
    std::uint64_t observed_units_{0};
    std::vector<std::uint32_t> open_start_;  // kNone = no open period
    std::vector<std::uint32_t> last_seen_;   // kNone = never observed
    std::vector<Tagged> closed_;
    std::vector<std::uint64_t> offsets_;
    std::vector<ResidencyPeriod> periods_;
};

}  // namespace gpu_m2d

#endif  // GPU_M2D_RESIDENCY_MAP_HPP
