#ifndef GPU_M2D_L2_PROBE_HPP
#define GPU_M2D_L2_PROBE_HPP

// Host API for the G8 L2 residency probe (docs/G8_CACHE_FAULT_PLAN.md §4).
// The kernel itself lives in l2_probe_kernels.cuh; this header is plain C++
// so the TensorRT runner (a .cpp translation unit) can drive it.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

namespace gpu_m2d {

struct L2ProbeRange {
    std::string id;          // allocation id, for reporting
    const void* base{nullptr};
    std::size_t bytes{0};
};

// Probes every unit (128-B line or 32-B sector) of a fixed set of device
// ranges, one load per unit, and classifies each as an L2 hit when its
// latency is below `threshold_cycles` (calibrated by G8-T0). Each probe is
// one warp with ONE active lane (per-line resolution); `probes_per_sm`
// warps per SM run concurrently (G8-T0: 16 keeps hits and misses separated
// on RTX 4090, threshold 440 cycles).
class L2Prober {
public:
    // stride > 1: each probe() sweeps only units u with
    // u % stride == sweep_index % stride (staggered sub-sampling that bounds
    // the sweep's own fill traffic); reverse: probe in descending order.
    // alternate: the direction flips for each unit's successive
    // observations, i.e. sweep s probes in reverse iff (s / stride) is odd
    // (overrides `reverse`; averages out the in-sweep order effect T0
    // measured on ViT-B scratch).
    // external_output: a caller-owned device buffer of at least
    // required_output_bytes(ranges, unit_bytes) bytes for the latencies
    // (the runner passes a REGISTERED allocation so the G2/G5 ledger
    // covers it); nullptr = the prober cudaMallocs its own.
    L2Prober(std::vector<L2ProbeRange> ranges, std::size_t unit_bytes,
             std::uint32_t threshold_cycles, int probes_per_sm,
             std::size_t stride = 1, bool reverse = false,
             bool alternate = false, void* external_output = nullptr,
             std::size_t external_output_bytes = 0);

    // Units of `ranges` at `unit_bytes` (sum over ranges) and the output
    // buffer size a prober over them needs.
    static std::uint64_t count_units(const std::vector<L2ProbeRange>& ranges,
                                     std::size_t unit_bytes);
    static std::size_t required_output_bytes(
        const std::vector<L2ProbeRange>& ranges, std::size_t unit_bytes) {
        return static_cast<std::size_t>(count_units(ranges, unit_bytes)) *
               sizeof(std::uint16_t);
    }
    ~L2Prober();
    L2Prober(const L2Prober&) = delete;
    L2Prober& operator=(const L2Prober&) = delete;

    // Enqueues one sweep on `stream` (phase = sweep_index % stride), waits
    // for it, copies the latencies back, and returns the probe kernel's GPU
    // time in ms.
    float probe(cudaStream_t stream, std::uint64_t sweep_index = 0);

    std::size_t range_count() const noexcept { return ranges_.size(); }
    const L2ProbeRange& range(std::size_t index) const { return ranges_[index]; }
    std::uint64_t range_units(std::size_t index) const { return units_[index]; }
    std::uint64_t total_units() const noexcept { return total_units_; }
    std::size_t unit_bytes() const noexcept { return unit_bytes_; }
    std::uint32_t threshold_cycles() const noexcept { return threshold_; }
    std::size_t stride() const noexcept { return stride_; }
    bool reverse() const noexcept { return reverse_; }
    bool alternate() const noexcept { return alternate_; }
    // Direction actually used by the last probe().
    bool last_reverse() const noexcept { return last_reverse_; }
    // Bytes of each unit that lie inside its range (unit_bytes except for
    // partial units at a range edge), in global unit order.
    std::vector<std::uint16_t> unit_resident_bytes() const;
    // Phase of the last probe(): the probed units are u % stride == phase.
    std::uint64_t last_phase() const noexcept { return phase_; }
    bool probed_last(std::uint64_t unit) const noexcept {
        return unit % stride_ == phase_;
    }

    // Results of the last probe(): per-unit latency (cycles, 16-bit
    // saturated) in range order -- valid only for probed_last() units --
    // and per-range probed-unit and hit counts.
    const std::vector<std::uint16_t>& latencies() const noexcept {
        return host_latency_;
    }
    std::vector<std::uint64_t> hits_per_range() const;
    std::vector<std::uint64_t> probed_per_range() const;
    std::uint64_t range_first_unit(std::size_t index) const {
        return first_unit_[index];
    }

private:
    std::vector<L2ProbeRange> ranges_;
    std::vector<std::uint64_t> units_;
    std::vector<std::uint64_t> first_unit_;
    std::uint64_t total_units_{0};
    std::size_t unit_bytes_{0};
    std::uint32_t threshold_{0};
    std::size_t stride_{1};
    bool reverse_{false};
    bool alternate_{false};
    bool last_reverse_{false};
    std::uint64_t phase_{0};
    int blocks_{0};
    int block_threads_{0};
    std::uint16_t* device_latency_{nullptr};
    bool owns_output_{true};
    std::vector<std::uint16_t> host_latency_;
    std::vector<std::uint16_t> compact_;
    cudaEvent_t start_{nullptr};
    cudaEvent_t stop_{nullptr};
};

}  // namespace gpu_m2d

#endif  // GPU_M2D_L2_PROBE_HPP
