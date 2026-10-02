#include "gpu_m2d/residency_map.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace gpu_m2d {

ResidencyMapBuilder::ResidencyMapBuilder(std::uint64_t units, std::uint32_t images)
    : units_(units), images_(images) {
    if (units_ == 0 || images_ == 0) {
        throw std::invalid_argument("residency map needs units > 0 and images > 0");
    }
    if (units_ >= kNone) {
        throw std::invalid_argument("residency map supports < 2^32 - 1 units");
    }
    open_start_.assign(units_, kNone);
    last_seen_.assign(units_, kNone);
}

void ResidencyMapBuilder::observe(std::uint64_t unit, std::uint32_t image,
                                  bool resident) {
    if (finished_) {
        throw std::logic_error("residency map: observe() after finish()");
    }
    if (unit >= units_ || image >= images_) {
        throw std::out_of_range("residency map: unit or image out of range");
    }
    std::uint32_t& last = last_seen_[unit];
    const bool first = last == kNone;
    if (!first && image <= last) {
        throw std::logic_error(
            "residency map: observations of a unit must strictly increase");
    }
    last = image;
    if (first) {
        ++observed_units_;
    }
    std::uint32_t& open = open_start_[unit];
    if (resident) {
        if (open == kNone) {
            open = first ? 0u : image;
        }
    } else if (open != kNone) {
        closed_.push_back({static_cast<std::uint32_t>(unit), {open, image - 1}});
        open = kNone;
    }
}

void ResidencyMapBuilder::finish() {
    if (finished_) {
        return;
    }
    for (std::uint64_t u = 0; u < units_; ++u) {
        if (open_start_[u] != kNone) {
            closed_.push_back({static_cast<std::uint32_t>(u),
                               {open_start_[u], images_ - 1}});
        }
    }
    // Periods of one unit were appended in increasing start order, so a
    // stable sort by unit keeps them sorted by start within each unit.
    std::stable_sort(closed_.begin(), closed_.end(),
                     [](const Tagged& a, const Tagged& b) { return a.unit < b.unit; });
    offsets_.assign(units_ + 1, 0);
    periods_.clear();
    periods_.reserve(closed_.size());
    std::uint64_t u = 0;
    for (const Tagged& t : closed_) {
        while (u <= t.unit) {
            offsets_[u++] = periods_.size();
        }
        periods_.push_back(t.period);
    }
    while (u <= units_) {
        offsets_[u++] = periods_.size();
    }
    closed_.clear();
    closed_.shrink_to_fit();
    open_start_.clear();
    open_start_.shrink_to_fit();
    finished_ = true;
}

void ResidencyMapBuilder::require_finished() const {
    if (!finished_) {
        throw std::logic_error("residency map: finish() has not been called");
    }
}

const std::vector<std::uint64_t>& ResidencyMapBuilder::offsets() const {
    require_finished();
    return offsets_;
}

const std::vector<ResidencyPeriod>& ResidencyMapBuilder::periods() const {
    require_finished();
    return periods_;
}

std::vector<double> ResidencyMapBuilder::resident_time(
    const std::vector<double>& image_time) const {
    require_finished();
    if (image_time.size() != images_) {
        throw std::invalid_argument("residency map: image_time size != images");
    }
    std::vector<double> prefix(images_ + 1, 0.0);
    for (std::uint32_t i = 0; i < images_; ++i) {
        prefix[i + 1] = prefix[i] + image_time[i];
    }
    std::vector<double> t(units_, 0.0);
    for (std::uint64_t u = 0; u < units_; ++u) {
        double sum = 0.0;
        for (std::uint64_t p = offsets_[u]; p < offsets_[u + 1]; ++p) {
            sum += prefix[periods_[p].end + 1] - prefix[periods_[p].start];
        }
        t[u] = sum;
    }
    return t;
}

double ResidencyMapBuilder::r_eff_bits(
    const std::vector<double>& image_time,
    const std::vector<std::uint16_t>& unit_resident_bytes) const {
    if (unit_resident_bytes.size() != units_) {
        throw std::invalid_argument("residency map: unit_resident_bytes size != units");
    }
    const std::vector<double> t = resident_time(image_time);
    double total = 0.0;
    for (double x : image_time) {
        total += x;
    }
    if (total <= 0.0) {
        throw std::invalid_argument("residency map: total inference time <= 0");
    }
    double weighted = 0.0;
    for (std::uint64_t u = 0; u < units_; ++u) {
        weighted += 8.0 * unit_resident_bytes[u] * t[u];
    }
    return weighted / total;
}

namespace {

template <typename T>
void put(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

}  // namespace

void ResidencyMapBuilder::write_binary(
    const std::string& path, std::uint32_t unit_bytes,
    const std::vector<double>& image_time,
    const std::vector<std::uint16_t>& unit_resident_bytes) const {
    require_finished();
    if (image_time.size() != images_ || unit_resident_bytes.size() != units_) {
        throw std::invalid_argument("residency map: array sizes do not match");
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot write residency map: " + path);
    }
    out.write("G8RMAP01", 8);
    put(out, static_cast<std::uint32_t>(1));
    put(out, unit_bytes);
    put(out, units_);
    put(out, static_cast<std::uint64_t>(images_));
    put(out, static_cast<std::uint64_t>(periods_.size()));
    out.write(reinterpret_cast<const char*>(image_time.data()),
              static_cast<std::streamsize>(image_time.size() * sizeof(double)));
    out.write(reinterpret_cast<const char*>(unit_resident_bytes.data()),
              static_cast<std::streamsize>(unit_resident_bytes.size() *
                                           sizeof(std::uint16_t)));
    out.write(reinterpret_cast<const char*>(offsets_.data()),
              static_cast<std::streamsize>(offsets_.size() * sizeof(std::uint64_t)));
    for (const ResidencyPeriod& p : periods_) {
        put(out, p.start);
        put(out, p.end);
    }
    if (!out) {
        throw std::runtime_error("residency map write failed: " + path);
    }
}

}  // namespace gpu_m2d
