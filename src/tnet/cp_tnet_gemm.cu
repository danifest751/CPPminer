// TNet layer GEMM on int8 tensor cores with CUTLASS (compiled into the miner; no cuBLAS DLLs):
// Y (rows x n, int32, row-major) = X (rows x n, int8, row-major) * W, with W given as WT = W^T stored
// row-major (n x n), i.e. W in column-major order. Two kernels: Sm75 (Turing, mma.m8n8k16) and Sm80
// (Ampere, Ada, Blackwell GeForce: mma.m16n8k32 with cp.async stages), picked per device.

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

#endif  // CP_ENABLE_CUDA
