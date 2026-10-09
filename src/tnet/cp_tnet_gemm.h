// TNet layer GEMM on int8 tensor cores (CUTLASS), see cp_tnet_gemm.cu.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#define CP_TNET_GEMM_NONE 0
#define CP_TNET_GEMM_SM75 1
#define CP_TNET_GEMM_SM80 2

#ifdef __cplusplus
extern "C" {
#endif

/* The kernel to use on device `dev` (CP_TNET_GEMM_NONE below compute 7.5). */
int cp_tnet_gemm_kind(int dev);
const char* cp_tnet_gemm_name(int kind);
/* Y (rows x n, int32, row-major) = X (rows x n, int8, row-major) * W, WT = W^T row-major (n x n).
 * Returns 0 on success. */
int cp_tnet_gemm(int kind, const int8_t* x, const int8_t* wt, int32_t* y, int rows, int n, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
