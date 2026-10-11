// TNet layer GEMM on int8 tensor cores with CUTLASS (compiled into the miner; no cuBLAS DLLs):
// Y (rows x n, int32, row-major) = X (rows x n, int8, row-major) * W, with W given as WT = W^T stored
// row-major (n x n), i.e. W in column-major order. Two kernels: Sm75 (Turing, mma.m8n8k16) and Sm80
// (Ampere, Ada, Blackwell GeForce: mma.m16n8k32 with cp.async stages), picked per device. Each comes in two forms: int32 out
// (the plain product) and int8 out with the requantization done in the epilogue (RequantOp).

#include "cp_tnet_gemm.h"

#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA

#include <cstdio>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

namespace {

using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;
using Epilogue = cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>;

using GemmSm75 = cutlass::gemm::device::Gemm<
    int8_t, RowMajor, int8_t, ColumnMajor, int32_t, RowMajor, int32_t, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75, cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>, Epilogue, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 2,
    16, 16>;

using GemmSm80 = cutlass::gemm::device::Gemm<
    int8_t, RowMajor, int8_t, ColumnMajor, int32_t, RowMajor, int32_t, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>, Epilogue, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 3,
    16, 16>;

// Epilogue: X = clamp((acc M + 2^23) >> 24, -128, 127) per element, exactly the consensus requantization (rq_requant in
// cp_tnet.cu does it as a separate pass). Writing int8 instead of int32 saves the product's round trip through memory.
template <int Count>
struct RequantOp {
    using ElementOutput = int8_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = int32_t;
    static int const kCount = Count;
    using FragmentOutput = cutlass::Array<int8_t, kCount>;
    using FragmentAccumulator = cutlass::Array<int32_t, kCount>;
    using ComputeFragment = cutlass::Array<int32_t, kCount>;
    static bool const kIsHeavy = false;

    struct Params {
        int32_t mult;
        CUTLASS_HOST_DEVICE Params() : mult(0) {}
        CUTLASS_HOST_DEVICE Params(int32_t m) : mult(m) {}
    };

    int32_t mult_;

    CUTLASS_HOST_DEVICE RequantOp(Params const& p) : mult_(p.mult) {}
    CUTLASS_HOST_DEVICE bool is_source_needed() const { return false; }
    CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}  // no split-k: every partition would need the int32 sum

    CUTLASS_HOST_DEVICE FragmentOutput operator()(FragmentAccumulator const& acc) const {
        FragmentOutput out;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kCount; ++i) {
            const long long t = ((long long)acc[i] * mult_ + (1LL << 23)) >> 24;
            out[i] = (int8_t)(t < -128 ? -128 : (t > 127 ? 127 : t));
        }
        return out;
    }
    CUTLASS_HOST_DEVICE FragmentOutput operator()(FragmentAccumulator const& acc, FragmentOutput const&) const {
        return (*this)(acc);
    }
};

// 256 x 128 blocks: 239 ns/ticket against 244 for 128 x 256 and 274 for 128 x 128 on a CMP 50HX (n = 8192, --bench)
using GemmSm75Rq = cutlass::gemm::device::Gemm<
    int8_t, RowMajor, int8_t, ColumnMajor, int8_t, RowMajor, int32_t, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75, cutlass::gemm::GemmShape<256, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>, RequantOp<16>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 2,
    16, 16>;

// 256 x 128 blocks, 3 stages: 193.1 ns/ticket against 195.1 for 128 x 256 on an RTX 3090 at a locked 1110 MHz
// (4 stages, 128 x 128, 256 x 64 and 128-deep k-tiles were slower)
using GemmSm80Rq = cutlass::gemm::device::Gemm<
    int8_t, RowMajor, int8_t, ColumnMajor, int8_t, RowMajor, int32_t, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80, cutlass::gemm::GemmShape<256, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>, RequantOp<16>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 3,
    16, 16>;

template <typename Gemm>
int run_rq(const int8_t* x, const int8_t* wt, int8_t* xo, int rows, int n, int mult, cudaStream_t stream) {
    typename Gemm::Arguments args({rows, n, n}, {x, n}, {wt, n}, {xo, n}, {xo, n}, {mult});
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 2;
    if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return 3;
    return gemm(stream) == cutlass::Status::kSuccess ? 0 : 4;
}

template <typename Gemm>
int run(const int8_t* x, const int8_t* wt, int32_t* y, int rows, int n, cudaStream_t stream) {
    typename Gemm::Arguments args({rows, n, n}, {x, n}, {wt, n}, {y, n}, {y, n}, {1, 0});
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 2;
    if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return 3;
    return gemm(stream) == cutlass::Status::kSuccess ? 0 : 4;
}

// The Sm80 kernel only has code where the build includes an sm_80+ target (or PTX the driver can lift).
template <typename Gemm>
bool has_code(int dev) {
    cudaFuncAttributes a;
    if (cudaFuncGetAttributes(&a, cutlass::Kernel<typename Gemm::GemmKernel>) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    (void)dev;
    return a.ptxVersion >= 80 || a.binaryVersion >= 80;
}

}  // namespace

extern "C" int cp_tnet_gemm_kind(int dev) {
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    const int sm = major * 10 + minor;
    if (sm >= 80 && has_code<GemmSm80>(dev)) return CP_TNET_GEMM_SM80;
    if (sm >= 75) return CP_TNET_GEMM_SM75;
    return CP_TNET_GEMM_NONE;
}

extern "C" const char* cp_tnet_gemm_name(int kind) {
    switch (kind) {
    case CP_TNET_GEMM_SM75: return "CUTLASS int8 TensorOp sm75 (mma.m8n8k16)";
    case CP_TNET_GEMM_SM80: return "CUTLASS int8 TensorOp sm80 (mma.m16n8k32, 3 stages)";
    default: return "none";
    }
}

extern "C" int cp_tnet_gemm(int kind, const int8_t* x, const int8_t* wt, int32_t* y, int rows, int n, cudaStream_t stream) {
    switch (kind) {
    case CP_TNET_GEMM_SM75: return run<GemmSm75>(x, wt, y, rows, n, stream);
    case CP_TNET_GEMM_SM80: return run<GemmSm80>(x, wt, y, rows, n, stream);
    default: return 1;
    }
}

extern "C" int cp_tnet_gemm_rq(int kind, const int8_t* x, const int8_t* wt, int8_t* xo, int rows, int n, int mult,
                               cudaStream_t stream) {
    switch (kind) {
    case CP_TNET_GEMM_SM75: return run_rq<GemmSm75Rq>(x, wt, xo, rows, n, mult, stream);
    case CP_TNET_GEMM_SM80: return run_rq<GemmSm80Rq>(x, wt, xo, rows, n, mult, stream);
    default: return 1;
    }
}

#endif  // CP_ENABLE_CUDA
