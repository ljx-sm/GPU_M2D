// GPU_M2D G2 VMM alias harness for the eBPF PTE observer.
//
// The harness is started by the observer orchestrator and blocks before
// any CUDA context or allocation exists. Once the gate file appears
// (probes are attached by then) it creates ONE CUDA VMM physical
// allocation and maps it at TWO distinct reserved VA ranges (the alias
// required by the G2 acceptance plan: one physical object, two VAs).
// It then proves the alias semantically at runtime (pattern written
// through the primary VA must be readable through the secondary VA,
// and the one-bit XOR closeout performed through the primary VA must
// be observed through the secondary VA), holds both mappings, and
// tears everything down. Timeline markers use GPU_M2D_EVENT lines.
//
// Derived from the GPU_M2D scratch harness (tools/g2_observer/
// g2_scratch_harness.cu), which is a port of the REMU harness frozen
// 2026-09-08.

#include <cuda.h>
#include <cuda_runtime.h>

#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <time.h>
#include <vector>

struct HarnessConfig {
    int device = 0;
    size_t size_bytes = 8ULL * 1024ULL * 1024ULL;
    int hold_seconds = 2;
    std::string gate_file;
    int gate_timeout_seconds = 30;
};

struct AliasMapping {
    size_t size_bytes = 0;
    CUdeviceptr primary_va = 0;
    CUdeviceptr secondary_va = 0;
    CUmemGenericAllocationHandle physical_handle = 0;
    size_t granularity = 0;
};

static uint64_t wall_time_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}

static uint64_t monotonic_time_ns()
{
    struct timespec value {};
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        std::perror("clock_gettime");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL +
           static_cast<uint64_t>(value.tv_nsec);
}

static long parse_long(const char *value, const char *option)
{
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!value[0] || !end || *end != '\0') {
        std::fprintf(stderr, "Invalid value for %s: %s\n", option, value);
        std::exit(EXIT_FAILURE);
    }
    return parsed;
}

static void driver_check(CUresult result, const char *operation)
{
    if (result != CUDA_SUCCESS) {
        const char *name = "unknown";
        const char *description = "unknown";
        cuGetErrorName(result, &name);
        cuGetErrorString(result, &description);
        std::fprintf(stderr,
                     "GPU_M2D_ERROR,operation=%s,driver_code=%d,driver_name=%s,driver_error=%s\n",
                     operation,
                     static_cast<int>(result),
                     name,
                     description);
        std::exit(EXIT_FAILURE);
    }
}

static void cuda_check(cudaError_t result, const char *operation)
{
    if (result != cudaSuccess) {
        std::fprintf(stderr,
                     "GPU_M2D_ERROR,operation=%s,cuda_code=%d,cuda_error=%s\n",
                     operation,
                     static_cast<int>(result),
                     cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

static void emit_event(const char *event)
{
    std::printf("GPU_M2D_EVENT,event=%s,pid=%d,tgid=%d,wall_time_ns=%" PRIu64
                ",monotonic_ns=%" PRIu64 "\n",
                event,
                static_cast<int>(getpid()),
                static_cast<int>(getpid()),
                wall_time_ns(),
                monotonic_time_ns());
}

static std::string format_uuid(const cudaUUID_t &uuid)
{
    char output[37];
    const unsigned char *bytes = reinterpret_cast<const unsigned char *>(uuid.bytes);
    std::snprintf(output,
                  sizeof(output),
                  "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  bytes[0], bytes[1], bytes[2], bytes[3],
                  bytes[4], bytes[5], bytes[6], bytes[7],
                  bytes[8], bytes[9], bytes[10], bytes[11],
                  bytes[12], bytes[13], bytes[14], bytes[15]);
    return output;
}

static long required_value(int argc, char **argv, int &index, const std::string &option)
{
    if (index + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", option.c_str());
        std::exit(EXIT_FAILURE);
    }
    return parse_long(argv[++index], option.c_str());
}

static HarnessConfig parse_config(int argc, char **argv)
{
    HarnessConfig config;
    for (int index = 1; index < argc; ++index) {
        const std::string option(argv[index]);
        if (option == "--device") {
            config.device = static_cast<int>(required_value(argc, argv, index, option));
        }
        else if (option == "--size-mib") {
            const long size_mib = required_value(argc, argv, index, option);
            if (size_mib <= 0 || size_mib > 64)
                std::exit(EXIT_FAILURE);
            config.size_bytes = static_cast<size_t>(size_mib) * 1024ULL * 1024ULL;
        }
        else if (option == "--hold-seconds") {
            config.hold_seconds = static_cast<int>(required_value(argc, argv, index, option));
            if (config.hold_seconds < 0)
                std::exit(EXIT_FAILURE);
        }
        else if (option == "--gate-file") {
            if (index + 1 >= argc)
                std::exit(EXIT_FAILURE);
            config.gate_file = argv[++index];
        }
        else if (option == "--gate-timeout-seconds") {
            config.gate_timeout_seconds = static_cast<int>(required_value(argc, argv, index, option));
            if (config.gate_timeout_seconds <= 0)
                std::exit(EXIT_FAILURE);
        }
        else if (option == "--help") {
            std::printf("Usage: %s --gate-file PATH [--device N] [--size-mib N] "
                        "[--hold-seconds N] [--gate-timeout-seconds N]\n",
                        argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        else {
            std::fprintf(stderr, "Unknown option: %s\n", option.c_str());
            std::exit(EXIT_FAILURE);
        }
    }
    if (config.gate_file.empty())
        std::exit(EXIT_FAILURE);
    return config;
}

static bool wait_for_gate(const std::string &path, int timeout_seconds)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
    while (std::chrono::steady_clock::now() < deadline) {
        if (access(path.c_str(), F_OK) == 0)
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    return false;
}

static size_t round_up(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

__global__ static void write_pattern(unsigned char *base, size_t size_bytes, unsigned char seed)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size_bytes) {
        base[index] = static_cast<unsigned char>(seed + index);
    }
}

__global__ static void xor_one_byte(unsigned char *base, size_t offset, unsigned char mask)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        base[offset] ^= mask;
}

static AliasMapping build_alias(int device, size_t requested_size)
{
    driver_check(cuInit(0), "cuInit");
    CUmemAllocationProp properties {};
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = device;
    properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

    AliasMapping mapping;
    driver_check(cuMemGetAllocationGranularity(&mapping.granularity,
                                               &properties,
                                               CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                 "cuMemGetAllocationGranularity");
    mapping.size_bytes = round_up(requested_size, mapping.granularity);

    // ONE physical allocation...
    driver_check(cuMemCreate(&mapping.physical_handle, mapping.size_bytes, &properties, 0),
                 "cuMemCreate");
    // ...mapped at TWO independent reserved VA ranges.
    driver_check(cuMemAddressReserve(&mapping.primary_va, mapping.size_bytes,
                                     mapping.granularity, 0, 0),
                 "cuMemAddressReserve(primary)");
    driver_check(cuMemAddressReserve(&mapping.secondary_va, mapping.size_bytes,
                                     mapping.granularity, 0, 0),
                 "cuMemAddressReserve(secondary)");
    driver_check(cuMemMap(mapping.primary_va, mapping.size_bytes, 0, mapping.physical_handle, 0),
                 "cuMemMap(primary)");
    driver_check(cuMemMap(mapping.secondary_va, mapping.size_bytes, 0, mapping.physical_handle, 0),
                 "cuMemMap(secondary)");
    CUmemAccessDesc access {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    driver_check(cuMemSetAccess(mapping.primary_va, mapping.size_bytes, &access, 1),
                 "cuMemSetAccess(primary)");
    driver_check(cuMemSetAccess(mapping.secondary_va, mapping.size_bytes, &access, 1),
                 "cuMemSetAccess(secondary)");
    return mapping;
}

static void release_alias(AliasMapping &mapping)
{
    driver_check(cuMemUnmap(mapping.primary_va, mapping.size_bytes), "cuMemUnmap(primary)");
    driver_check(cuMemUnmap(mapping.secondary_va, mapping.size_bytes), "cuMemUnmap(secondary)");
    driver_check(cuMemRelease(mapping.physical_handle), "cuMemRelease");
    driver_check(cuMemAddressFree(mapping.primary_va, mapping.size_bytes), "cuMemAddressFree(primary)");
    driver_check(cuMemAddressFree(mapping.secondary_va, mapping.size_bytes), "cuMemAddressFree(secondary)");
}

// The runtime alias proof: a device-side pattern write through the
// primary VA must be bit-identical when read back through the secondary
// VA, and a one-bit XOR performed through the primary VA must be
// observed (and restored) through the secondary VA.
static bool verify_alias_semantics(const AliasMapping &mapping)
{
    constexpr size_t target_offset = 4096;
    constexpr unsigned int bit_index = 3;
    constexpr unsigned char mask = static_cast<unsigned char>(1U << bit_index);
    const size_t size = mapping.size_bytes;
    const unsigned char *primary = reinterpret_cast<unsigned char *>(mapping.primary_va);
    const unsigned char *secondary = reinterpret_cast<unsigned char *>(mapping.secondary_va);

    const unsigned int blocks = static_cast<unsigned int>((size + 255) / 256);
    write_pattern<<<blocks, 256>>>(const_cast<unsigned char *>(primary), size, 0x5A);
    cuda_check(cudaGetLastError(), "write_pattern launch");
    cuda_check(cudaDeviceSynchronize(), "write_pattern synchronize");

    std::vector<unsigned char> via_secondary(size);
    cuda_check(cudaMemcpy(via_secondary.data(), secondary, size, cudaMemcpyDeviceToHost),
               "cudaMemcpy(alias secondary D2H)");
    size_t pattern_mismatches = 0;
    for (size_t index = 0; index < size; ++index) {
        if (via_secondary[index] != static_cast<unsigned char>(0x5A + index))
            ++pattern_mismatches;
    }

    const unsigned char before_via_secondary = via_secondary[target_offset];
    xor_one_byte<<<1, 1>>>(const_cast<unsigned char *>(primary), target_offset, mask);
    cuda_check(cudaGetLastError(), "xor_one_byte(forward) launch");
    cuda_check(cudaDeviceSynchronize(), "xor_one_byte(forward) synchronize");
    std::vector<unsigned char> mutated_via_secondary(size);
    cuda_check(cudaMemcpy(mutated_via_secondary.data(), secondary, size, cudaMemcpyDeviceToHost),
               "cudaMemcpy(alias mutated D2H)");

    xor_one_byte<<<1, 1>>>(const_cast<unsigned char *>(primary), target_offset, mask);
    cuda_check(cudaGetLastError(), "xor_one_byte(restore) launch");
    cuda_check(cudaDeviceSynchronize(), "xor_one_byte(restore) synchronize");
    std::vector<unsigned char> restored_via_secondary(size);
    cuda_check(cudaMemcpy(restored_via_secondary.data(), secondary, size, cudaMemcpyDeviceToHost),
               "cudaMemcpy(alias restored D2H)");

    size_t forward_changed_bytes = 0;
    size_t final_changed_bytes = 0;
    for (size_t index = 0; index < size; ++index) {
        if (mutated_via_secondary[index] != via_secondary[index])
            ++forward_changed_bytes;
        if (restored_via_secondary[index] != via_secondary[index])
            ++final_changed_bytes;
    }
    const unsigned char after_via_secondary = mutated_via_secondary[target_offset];

    const bool pass = pattern_mismatches == 0 && forward_changed_bytes == 1 &&
                      final_changed_bytes == 0 &&
                      after_via_secondary == (before_via_secondary ^ mask);
    std::printf("GPU_M2D_ALIAS,status=%s,pattern_mismatches=%zu,"
                "target_offset=%zu,bit_index=%u,mask=0x%02x,"
                "before_secondary=0x%02x,after_secondary=0x%02x,"
                "forward_changed_bytes=%zu,final_changed_bytes=%zu,size_bytes=%zu\n",
                pass ? "PASS" : "FAIL",
                pattern_mismatches,
                target_offset,
                bit_index,
                static_cast<unsigned int>(mask),
                before_via_secondary,
                after_via_secondary,
                forward_changed_bytes,
                final_changed_bytes,
                size);
    return pass;
}

static void print_alias_samples(const std::string &root_id, const AliasMapping &mapping)
{
    const size_t offsets[] = {
        0,
        4ULL * 1024ULL,
        64ULL * 1024ULL,
        2ULL * 1024ULL * 1024ULL,
        4ULL * 1024ULL * 1024ULL,
    };
    for (size_t index = 0; index < sizeof(offsets) / sizeof(offsets[0]); ++index) {
        if (offsets[index] >= mapping.size_bytes)
            continue;
        std::printf("GPU_M2D_SAMPLE,allocation_id=%s-%s,index=%zu,offset_bytes=%zu,va=0x%llx\n",
                    root_id.c_str(),
                    "primary",
                    index,
                    offsets[index],
                    static_cast<unsigned long long>(mapping.primary_va + offsets[index]));
        std::printf("GPU_M2D_SAMPLE,allocation_id=%s-%s,index=%zu,offset_bytes=%zu,va=0x%llx\n",
                    root_id.c_str(),
                    "secondary",
                    index,
                    offsets[index],
                    static_cast<unsigned long long>(mapping.secondary_va + offsets[index]));
    }
}

int main(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const HarnessConfig config = parse_config(argc, argv);
    if (access(config.gate_file.c_str(), F_OK) == 0) {
        std::fprintf(stderr, "Gate file already exists; refusing stale gate: %s\n", config.gate_file.c_str());
        return EXIT_FAILURE;
    }

    emit_event("PROCESS_READY");
    emit_event("WAIT_PRE_ALLOC_GATE");
    std::fflush(stdout);
    if (!wait_for_gate(config.gate_file, config.gate_timeout_seconds)) {
        std::fprintf(stderr, "GPU_M2D_ERROR,operation=wait_for_pre_alloc_gate,error=timeout\n");
        return EXIT_FAILURE;
    }
    emit_event("PRE_ALLOC_GATE_OPEN");

    emit_event("CONTEXT_BEGIN");
    driver_check(cuInit(0), "cuInit(context)");
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (config.device < 0 || config.device >= device_count)
        return EXIT_FAILURE;
    cuda_check(cudaSetDevice(config.device), "cudaSetDevice");
    cuda_check(cudaFree(nullptr), "cudaFree(context initialization)");
    cudaDeviceProp device_properties {};
    cuda_check(cudaGetDeviceProperties(&device_properties, config.device), "cudaGetDeviceProperties");
    emit_event("CONTEXT_READY");

    emit_event("ALLOCATION_BEGIN");
    const uint64_t allocation_time = wall_time_ns();
    AliasMapping mapping = build_alias(config.device, config.size_bytes);
    const std::string root_id = "g2-alias-" + std::to_string(config.device) + "-pid-" +
                                std::to_string(static_cast<int>(getpid())) + "-ns-" +
                                std::to_string(allocation_time);
    const std::string uuid_text = format_uuid(device_properties.uuid);
    std::printf("GPU_M2D_EVENT,event=ALLOCATED,allocation_api=vmm-alias-primary,pid=%d,tgid=%d,"
                "device=%d,gpu_uuid=%s,allocation_id=%s-primary,wall_time_ns=%" PRIu64
                ",monotonic_ns=%" PRIu64 ",base_va=0x%llx,size_bytes=%zu\n",
                static_cast<int>(getpid()), static_cast<int>(getpid()), config.device,
                uuid_text.c_str(), root_id.c_str(), wall_time_ns(), monotonic_time_ns(),
                static_cast<unsigned long long>(mapping.primary_va), mapping.size_bytes);
    std::printf("GPU_M2D_EVENT,event=ALLOCATED,allocation_api=vmm-alias-secondary,pid=%d,tgid=%d,"
                "device=%d,gpu_uuid=%s,allocation_id=%s-secondary,wall_time_ns=%" PRIu64
                ",monotonic_ns=%" PRIu64 ",base_va=0x%llx,size_bytes=%zu\n",
                static_cast<int>(getpid()), static_cast<int>(getpid()), config.device,
                uuid_text.c_str(), root_id.c_str(), wall_time_ns(), monotonic_time_ns(),
                static_cast<unsigned long long>(mapping.secondary_va), mapping.size_bytes);
    print_alias_samples(root_id, mapping);

    emit_event("ACCESS_BEGIN");
    const bool alias_passed = verify_alias_semantics(mapping);
    emit_event("ACCESS_END");

    std::printf("GPU_M2D_EVENT,event=HOLD_BEGIN,allocation_id=%s,timestamp_ns=%" PRIu64
                ",hold_seconds=%d\n",
                root_id.c_str(), wall_time_ns(), config.hold_seconds);
    std::fflush(stdout);
    std::this_thread::sleep_for(std::chrono::seconds(config.hold_seconds));
    std::printf("GPU_M2D_EVENT,event=HOLD_END,allocation_id=%s,timestamp_ns=%" PRIu64 "\n",
                root_id.c_str(), wall_time_ns());

    emit_event("FREE_BEGIN");
    release_alias(mapping);
    emit_event("FREE_END");
    return alias_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
