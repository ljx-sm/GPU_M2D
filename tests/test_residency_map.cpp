#include "gpu_m2d/residency_map.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

bool periods_equal(const gpu_m2d::ResidencyMapBuilder& m, std::uint64_t unit,
                   const std::vector<gpu_m2d::ResidencyPeriod>& want) {
    const auto& off = m.offsets();
    const auto& per = m.periods();
    if (off[unit + 1] - off[unit] != want.size()) {
        return false;
    }
    for (std::size_t i = 0; i < want.size(); ++i) {
        const auto& p = per[off[unit] + i];
        if (p.start != want[i].start || p.end != want[i].end) {
            return false;
        }
    }
    return true;
}

template <typename F>
bool throws(F f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    using gpu_m2d::ResidencyMapBuilder;
    // 5 units, 10 images, observed every image (k = 1, stride 1).
    {
        ResidencyMapBuilder m(5, 10);
        for (std::uint32_t i = 0; i < 10; ++i) {
            m.observe(0, i, true);               // always resident
            m.observe(1, i, false);              // never resident
            m.observe(2, i, i < 3 || i >= 7);    // two periods
            m.observe(3, i, i >= 4);             // enters at 4
            // unit 4: never observed
        }
        m.finish();
        expect(periods_equal(m, 0, {{0, 9}}), "always resident -> [0,9]");
        expect(periods_equal(m, 1, {}), "never resident -> no period");
        expect(periods_equal(m, 2, {{0, 2}, {7, 9}}), "leave and return");
        expect(periods_equal(m, 3, {{4, 9}}), "enters at 4");
        expect(periods_equal(m, 4, {}), "unobserved -> no period");
        expect(m.observed_units() == 4, "observed units");

        const std::vector<double> t(10, 1.0);
        const auto rt = m.resident_time(t);
        expect(rt[0] == 10.0 && rt[1] == 0.0 && rt[2] == 6.0 && rt[3] == 6.0,
               "resident time, uniform images");
        // bits: units 0..3 full 32-B sectors, unit 4 partial 8 B
        const std::vector<std::uint16_t> bytes{32, 32, 32, 32, 8};
        const double r = m.r_eff_bits(t, bytes);
        // (256*10 + 256*6 + 256*6) / 10 = 563.2
        expect(std::fabs(r - 563.2) < 1e-9, "R_eff uniform times");

        // non-uniform times: image 0 lasts 11 units, others 1 (total 20)
        std::vector<double> t2(10, 1.0);
        t2[0] = 11.0;
        const auto rt2 = m.resident_time(t2);
        expect(rt2[2] == 16.0 && rt2[3] == 6.0, "resident time, weighted images");
        // (256*20 + 256*16 + 256*6) / 20 = 537.6
        expect(std::fabs(m.r_eff_bits(t2, bytes) - 537.6) < 1e-9,
               "R_eff weighted times");
    }
    // stride 4: unit 0 observed at 0,4,8; unit 1 at 1,5,9 (staggered).
    {
        ResidencyMapBuilder m(2, 10);
        m.observe(0, 0, false);
        m.observe(1, 1, true);   // first observation hit -> back to image 0
        m.observe(0, 4, true);
        m.observe(1, 5, false);  // closes [0,4]
        m.observe(0, 8, true);
        m.observe(1, 9, true);   // reopens at 9
        m.finish();
        expect(periods_equal(m, 0, {{4, 9}}), "stride: opens at observation");
        expect(periods_equal(m, 1, {{0, 4}, {9, 9}}),
               "stride: backward extension and close before miss");
    }
    // misuse is rejected
    {
        ResidencyMapBuilder m(1, 4);
        m.observe(0, 2, true);
        expect(throws([&] { m.observe(0, 2, true); }), "repeat image rejected");
        expect(throws([&] { m.observe(0, 1, true); }), "decreasing image rejected");
        expect(throws([&] { m.observe(1, 3, true); }), "unit out of range rejected");
        expect(throws([&] { (void)m.periods(); }), "periods before finish rejected");
        m.finish();
        expect(throws([&] { m.observe(0, 3, true); }), "observe after finish rejected");
        expect(throws([] { ResidencyMapBuilder bad(0, 4); }), "zero units rejected");
    }
    if (failures == 0) {
        std::printf("GPU_M2D_RESIDENCY_MAP_TEST_PASS\n");
        return 0;
    }
    return 1;
}
