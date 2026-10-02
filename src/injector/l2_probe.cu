#include "gpu_m2d/l2_probe.hpp"

#include "gpu_m2d/l2_probe_kernels.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace gpu_m2d {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorName(status)
            << " (" << cudaGetErrorString(status) << ')';
    throw std::runtime_error(message.str());
}

std::uint64_t units_overlapping(const void* base, std::size_t bytes,
                                std::size_t unit) {
    const auto start = reinterpret_cast<std::uintptr_t>(base);
    const std::uintptr_t first = start & ~static_cast<std::uintptr_t>(unit - 1);
    const std::uintptr_t end = start + bytes;
    return (end - first + unit - 1) / unit;
}

}  // namespace

L2Prober::L2Prober(std::vector<L2ProbeRange> ranges, std::size_t unit_bytes,
                   std::uint32_t threshold_cycles, int probes_per_sm,
                   std::size_t stride, bool reverse, bool alternate)
    : ranges_(std::move(ranges)),
      unit_bytes_(unit_bytes),
      threshold_(threshold_cycles),
      stride_(stride),
      reverse_(reverse),
      alternate_(alternate) {
    if (stride_ == 0) {
        throw std::invalid_argument("L2 probe stride must be >= 1");
    }
    if (unit_bytes_ != l2probe::kLineBytes && unit_bytes_ != l2probe::kSectorBytes) {
        throw std::invalid_argument("L2 probe unit must be 128 or 32 bytes");
    }
    if (ranges_.empty() ||
        ranges_.size() > static_cast<std::size_t>(l2probe::kMaxRanges)) {
        throw std::invalid_argument("L2 probe needs 1..32 ranges");
    }
    if (threshold_ == 0 || probes_per_sm <= 0) {
        throw std::invalid_argument("L2 probe threshold/parallelism must be > 0");
    }
    // Ranges must not share a unit: a shared unit would be probed twice
    // per sweep (the second probe would always hit) and double-counted.
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> spans;
    for (const L2ProbeRange& range : ranges_) {
        if (range.base == nullptr || range.bytes == 0) {
            throw std::invalid_argument("L2 probe range " + range.id + " is empty");
        }
        const auto start = reinterpret_cast<std::uintptr_t>(range.base);
        spans.emplace_back(start & ~static_cast<std::uintptr_t>(unit_bytes_ - 1),
                           start + range.bytes);
        first_unit_.push_back(total_units_);
        units_.push_back(units_overlapping(range.base, range.bytes, unit_bytes_));
        total_units_ += units_.back();
    }
    std::sort(spans.begin(), spans.end());
    for (std::size_t i = 1; i < spans.size(); ++i) {
        if (spans[i].first < spans[i - 1].second) {
            throw std::invalid_argument(
                "L2 probe ranges overlap at probe-unit granularity");
        }
    }

    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    int sms = 0;
    check_cuda(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device),
               "query SM count");
    // One active lane per warp (l2_probe_kernels.cuh): a probe is a warp.
    const int threads_per_sm = probes_per_sm * 32;
    block_threads_ = std::min(threads_per_sm, 512);
    blocks_ = std::max(1, sms * threads_per_sm / block_threads_);

    check_cuda(cudaMalloc(&device_latency_, total_units_ * sizeof(std::uint16_t)),
               "allocate L2 probe output");
    host_latency_.resize(total_units_);
    check_cuda(cudaEventCreate(&start_), "create probe start event");
    check_cuda(cudaEventCreate(&stop_), "create probe stop event");
}

L2Prober::~L2Prober() {
    if (start_ != nullptr) {
        cudaEventDestroy(start_);
    }
    if (stop_ != nullptr) {
        cudaEventDestroy(stop_);
    }
    if (device_latency_ != nullptr) {
        cudaFree(device_latency_);
    }
}

float L2Prober::probe(cudaStream_t stream, std::uint64_t sweep_index) {
    phase_ = sweep_index % stride_;
    l2probe::ProbeRanges params{};
    params.count = static_cast<int>(ranges_.size());
    params.total_lines = total_units_;
    params.unit = unit_bytes_;
    params.stride = stride_;
    params.phase = phase_;
    last_reverse_ = alternate_ ? ((sweep_index / stride_) % 2 == 1) : reverse_;
    params.reverse = last_reverse_ ? 1 : 0;
    params.lanes = 1;
    for (std::size_t i = 0; i < ranges_.size(); ++i) {
        params.r[i].base = static_cast<const std::uint8_t*>(ranges_[i].base);
        params.r[i].bytes = ranges_[i].bytes;
        params.r[i].first_line = first_unit_[i];
        params.r[i].lines = units_[i];
    }
    check_cuda(cudaEventRecord(start_, stream), "record probe start");
    l2probe::probe_latency_kernel<<<blocks_, block_threads_,
                                    block_threads_ * sizeof(std::uint32_t),
                                    stream>>>(params, device_latency_);
    check_cuda(cudaGetLastError(), "launch L2 probe");
    check_cuda(cudaEventRecord(stop_, stream), "record probe stop");
    check_cuda(cudaMemcpyAsync(host_latency_.data(), device_latency_,
                               total_units_ * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
               "copy probe latencies");
    check_cuda(cudaStreamSynchronize(stream), "synchronize probe");
    float ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&ms, start_, stop_), "probe elapsed time");
    return ms;
}

std::vector<std::uint64_t> L2Prober::hits_per_range() const {
    std::vector<std::uint64_t> hits(ranges_.size(), 0);
    for (std::size_t i = 0; i < ranges_.size(); ++i) {
        const std::uint64_t first = first_unit_[i];
        const std::uint64_t count = units_[i];
        std::uint64_t h = 0;
        for (std::uint64_t u = first; u < first + count; ++u) {
            if (probed_last(u)) {
                h += host_latency_[u] < threshold_ ? 1 : 0;
            }
        }
        hits[i] = h;
    }
    return hits;
}

std::vector<std::uint16_t> L2Prober::unit_resident_bytes() const {
    std::vector<std::uint16_t> bytes(total_units_, 0);
    for (std::size_t i = 0; i < ranges_.size(); ++i) {
        const auto base = reinterpret_cast<std::uintptr_t>(ranges_[i].base);
        const std::uintptr_t end = base + ranges_[i].bytes;
        const std::uintptr_t aligned =
            base & ~static_cast<std::uintptr_t>(unit_bytes_ - 1);
        for (std::uint64_t k = 0; k < units_[i]; ++k) {
            const std::uintptr_t lo = std::max(aligned + k * unit_bytes_, base);
            const std::uintptr_t hi = std::min(aligned + (k + 1) * unit_bytes_, end);
            bytes[first_unit_[i] + k] = static_cast<std::uint16_t>(hi - lo);
        }
    }
    return bytes;
}

std::vector<std::uint64_t> L2Prober::probed_per_range() const {
    std::vector<std::uint64_t> probed(ranges_.size(), 0);
    for (std::size_t i = 0; i < ranges_.size(); ++i) {
        std::uint64_t n = 0;
        for (std::uint64_t u = first_unit_[i]; u < first_unit_[i] + units_[i];
             ++u) {
            n += probed_last(u) ? 1 : 0;
        }
        probed[i] = n;
    }
    return probed;
}

}  // namespace gpu_m2d
