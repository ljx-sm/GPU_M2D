// GPU_M2D G9 integer-precision study (user decision 2026-10-23): a
// TensorRT 8.6.1 plugin that keeps a layer's weights PACKED in device
// memory as INT16 or INT4 (per-channel symmetric scales, FP32) and
// expands them to an FP32 weight tensor on every inference, so the stored
// bits a fault can hit are the INT16 / INT4 words themselves.
//
// Plugin "M2DPackedWeight" version "1". Fields:
//   bits    INT32[1]   16 or 4
//   dims    INT32[n]   weight tensor shape (the plugin's only output)
//   axis    INT32[1]   channel axis of the per-channel scales
//   scales  FLOAT32[C] per-channel scale (C = dims[axis])
//   packed  INT8[...]  INT16: little-endian words; INT4: two signed
//                      nibbles per byte, element 2i in the low nibble
// Input 0 is an anchor tensor (any FP32 tensor, unread) that orders the
// expansion inside each inference.
//
// Device memory: the packed words and the scales live in ONE buffer
// allocated through the IGpuAllocator TensorRT passes to
// attachToContext -- the same allocator the G1.5 runner tracks, so the
// buffer enters the allocation registry, the VA->PA snapshot and the
// residency map like any TensorRT allocation. During the builder's
// tactic timing (no execution context) a private cudaMalloc buffer is
// used instead and released at terminate().

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace m2d_g9 {

constexpr char const* kName = "M2DPackedWeight";
constexpr char const* kVersion = "1";

__global__ void expand(int32_t bits, int64_t n, int64_t inner, int32_t channels,
                       uint8_t const* packed, float const* scales, float* out) {
    for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n;
         i += int64_t(gridDim.x) * blockDim.x) {
        int32_t q;
        if (bits == 16) {
            q = int16_t(uint16_t(packed[2 * i]) | (uint16_t(packed[2 * i + 1]) << 8));
        } else {
            uint8_t byte = packed[i >> 1];
            int32_t nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
            q = nib >= 8 ? nib - 16 : nib;
        }
        int32_t c = int32_t((i / inner) % channels);
        out[i] = float(q) * scales[c];
    }
}

template <typename T>
void put(char*& p, T const& v) { std::memcpy(p, &v, sizeof(T)); p += sizeof(T); }
template <typename T>
T get(char const*& p) { T v; std::memcpy(&v, p, sizeof(T)); p += sizeof(T); return v; }

class PackedWeight final : public IPluginV2DynamicExt {
public:
    PackedWeight(int32_t bits, Dims dims, int32_t axis, std::vector<float> scales,
                 std::vector<uint8_t> packed)
        : mBits(bits), mDims(dims), mAxis(axis), mScales(std::move(scales)),
          mPacked(std::move(packed)) {}

    PackedWeight(void const* data, size_t length) {
        char const* p = static_cast<char const*>(data);
        mBits = get<int32_t>(p);
        mAxis = get<int32_t>(p);
        mDims.nbDims = get<int32_t>(p);
        for (int32_t i = 0; i < mDims.nbDims; ++i) mDims.d[i] = get<int32_t>(p);
        mScales.resize(get<int64_t>(p));
        std::memcpy(mScales.data(), p, mScales.size() * sizeof(float));
        p += mScales.size() * sizeof(float);
        mPacked.resize(get<int64_t>(p));
        std::memcpy(mPacked.data(), p, mPacked.size());
        (void)length;
    }

    // ---- IPluginV2
    AsciiChar const* getPluginType() const noexcept override { return kName; }
    AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {
        if (mPrivate) { cudaFree(mPrivate); mPrivate = nullptr; }
    }
    size_t getSerializationSize() const noexcept override {
        return 3 * sizeof(int32_t) + mDims.nbDims * sizeof(int32_t) + 2 * sizeof(int64_t) +
               mScales.size() * sizeof(float) + mPacked.size();
    }
    void serialize(void* buffer) const noexcept override {
        char* p = static_cast<char*>(buffer);
        put(p, mBits);
        put(p, mAxis);
        put(p, int32_t(mDims.nbDims));
        for (int32_t i = 0; i < mDims.nbDims; ++i) put(p, int32_t(mDims.d[i]));
        put(p, int64_t(mScales.size()));
        std::memcpy(p, mScales.data(), mScales.size() * sizeof(float));
        p += mScales.size() * sizeof(float);
        put(p, int64_t(mPacked.size()));
        std::memcpy(p, mPacked.data(), mPacked.size());
    }
    void destroy() noexcept override { terminate(); delete this; }
    void setPluginNamespace(AsciiChar const* ns) noexcept override { mNamespace = ns; }
    AsciiChar const* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

    // ---- IPluginV2Ext
    DataType getOutputDataType(int32_t, DataType const*, int32_t) const noexcept override {
        return DataType::kFLOAT;
    }
    void attachToContext(cudnnContext*, cublasContext*, IGpuAllocator* allocator) noexcept override {
        mAllocator = allocator;
        mDevice = static_cast<uint8_t*>(allocator->allocate(bufferBytes(), 256, 0));
        if (mDevice) upload(mDevice);
    }
    void detachFromContext() noexcept override {
        if (mAllocator && mDevice) mAllocator->deallocate(mDevice);
        mDevice = nullptr;
        mAllocator = nullptr;
    }

    // ---- IPluginV2DynamicExt
    IPluginV2DynamicExt* clone() const noexcept override {
        auto* c = new PackedWeight(mBits, mDims, mAxis, mScales, mPacked);
        c->setPluginNamespace(mNamespace.c_str());
        return c;
    }
    DimsExprs getOutputDimensions(int32_t, DimsExprs const*, int32_t,
                                  IExprBuilder& builder) noexcept override {
        DimsExprs out;
        out.nbDims = mDims.nbDims;
        for (int32_t i = 0; i < mDims.nbDims; ++i) out.d[i] = builder.constant(mDims.d[i]);
        return out;
    }
    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t,
                                   int32_t) noexcept override {
        return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }
    void configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
                         int32_t) noexcept override {}
    size_t getWorkspaceSize(PluginTensorDesc const*, int32_t, PluginTensorDesc const*,
                            int32_t) const noexcept override {
        return 0;
    }
    int32_t enqueue(PluginTensorDesc const*, PluginTensorDesc const*, void const* const*,
                    void* const* outputs, void*, cudaStream_t stream) noexcept override {
        uint8_t* base = mDevice;
        if (!base) {  // builder tactic timing: no execution context attached
            if (!mPrivate) {
                if (cudaMalloc(&mPrivate, bufferBytes()) != cudaSuccess) return 1;
                upload(mPrivate);
            }
            base = mPrivate;
        }
        int64_t n = elements();
        int32_t channels = mDims.d[mAxis];
        int64_t inner = 1;
        for (int32_t i = mAxis + 1; i < mDims.nbDims; ++i) inner *= mDims.d[i];
        float const* scales = reinterpret_cast<float const*>(base);
        uint8_t const* packed = base + scaleBytes();
        int32_t blocks = int32_t(std::min<int64_t>((n + 255) / 256, 4096));
        expand<<<blocks, 256, 0, stream>>>(mBits, n, inner, channels, packed, scales,
                                          static_cast<float*>(outputs[0]));
        return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
    }

private:
    int64_t elements() const {
        int64_t n = 1;
        for (int32_t i = 0; i < mDims.nbDims; ++i) n *= mDims.d[i];
        return n;
    }
    // scales first, padded to 256 B, then the packed words
    size_t scaleBytes() const { return (mScales.size() * sizeof(float) + 255) / 256 * 256; }
    size_t bufferBytes() const { return scaleBytes() + mPacked.size(); }
    void upload(uint8_t* dst) const {
        cudaMemcpy(dst, mScales.data(), mScales.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dst + scaleBytes(), mPacked.data(), mPacked.size(), cudaMemcpyHostToDevice);
    }

    int32_t mBits{16};
    Dims mDims{};
    int32_t mAxis{0};
    std::vector<float> mScales;
    std::vector<uint8_t> mPacked;
    std::string mNamespace;
    IGpuAllocator* mAllocator{nullptr};
    uint8_t* mDevice{nullptr};
    uint8_t* mPrivate{nullptr};
};

class PackedWeightCreator final : public IPluginCreator {
public:
    PackedWeightCreator() {
        mFields = {PluginField("bits", nullptr, PluginFieldType::kINT32, 1),
                   PluginField("dims", nullptr, PluginFieldType::kINT32, 0),
                   PluginField("axis", nullptr, PluginFieldType::kINT32, 1),
                   PluginField("scales", nullptr, PluginFieldType::kFLOAT32, 0),
                   PluginField("packed", nullptr, PluginFieldType::kINT8, 0)};
        mCollection.nbFields = int32_t(mFields.size());
        mCollection.fields = mFields.data();
    }
    AsciiChar const* getPluginName() const noexcept override { return kName; }
    AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &mCollection; }
    IPluginV2* createPlugin(AsciiChar const*, PluginFieldCollection const* fc) noexcept override {
        int32_t bits = 0, axis = 0;
        Dims dims{};
        std::vector<float> scales;
        std::vector<uint8_t> packed;
        for (int32_t i = 0; i < fc->nbFields; ++i) {
            PluginField const& f = fc->fields[i];
            std::string name = f.name;
            if (name == "bits") bits = *static_cast<int32_t const*>(f.data);
            else if (name == "axis") axis = *static_cast<int32_t const*>(f.data);
            else if (name == "dims") {
                dims.nbDims = f.length;
                for (int32_t k = 0; k < f.length; ++k) dims.d[k] = static_cast<int32_t const*>(f.data)[k];
            } else if (name == "scales") {
                auto const* s = static_cast<float const*>(f.data);
                scales.assign(s, s + f.length);
            } else if (name == "packed") {
                auto const* b = static_cast<uint8_t const*>(f.data);
                packed.assign(b, b + f.length);
            }
        }
        if ((bits != 16 && bits != 4) || dims.nbDims < 1 || axis < 0 || axis >= dims.nbDims ||
            int64_t(scales.size()) != dims.d[axis]) {
            return nullptr;
        }
        int64_t n = 1;
        for (int32_t i = 0; i < dims.nbDims; ++i) n *= dims.d[i];
        if (int64_t(packed.size()) != (bits == 16 ? 2 * n : (n + 1) / 2)) return nullptr;
        auto* p = new PackedWeight(bits, dims, axis, std::move(scales), std::move(packed));
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    IPluginV2* deserializePlugin(AsciiChar const*, void const* data, size_t length) noexcept override {
        auto* p = new PackedWeight(data, length);
        p->setPluginNamespace(mNamespace.c_str());
        return p;
    }
    void setPluginNamespace(AsciiChar const* ns) noexcept override { mNamespace = ns; }
    AsciiChar const* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
    std::vector<PluginField> mFields;
    PluginFieldCollection mCollection{};
    std::string mNamespace;
};

}  // namespace m2d_g9

using m2d_g9::PackedWeightCreator;

REGISTER_TENSORRT_PLUGIN(PackedWeightCreator);
