// Independent sm75 INT8 MMA prototype: one warp computes one exact Pearl
// scattered 8x8 proof tile. Fragment mapping follows PTX ISA 8.5 m8n8k16.
// It deliberately trades inter-tile operand reuse for a small register fragment.
#pragma once

#include "cp_cutlass_jackpot.cuh"

#ifndef CP_MINIMAL_PREFETCH
#define CP_MINIMAL_PREFETCH 0
#endif
#ifndef CP_MINIMAL_CG
#define CP_MINIMAL_CG 0
#endif

namespace cp_research {
struct MinimalParams {
    const int8_t *a, *b;
    int m, n;
    uint32_t *dump;
    CpCutlassJackpotLaunch jackpot;
};

__device__ __forceinline__ uint32_t minimal_load(const int8_t *p)
{
    uint32_t value;
#if CP_MINIMAL_CG
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(p));
#else
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(value) : "l"(p));
#endif
    return value;
}

__global__ void minimal_kernel(MinimalParams p)
{
    const size_t tile = size_t(blockIdx.x) * 8 + threadIdx.x / 32;
    const size_t tiles = size_t(p.m) * p.n / 64;
    if (tile >= tiles) return; // whole-warp predicate
    const int lane = threadIdx.x % 32;
    const size_t cta = tile / 256;
    const int row_period = int(cta / (p.n / 128));
    const int col_period = int(cta % (p.n / 128));
    const int virtual_thread = int(tile % 256);
    int row, col;
    MmaLaneTile128x128::thread_cell_global(row_period * 128, col_period * 128,
                                         virtual_thread, row, col);
    const int group = lane / 4;
    const int scatter_a = (group & 3) + (group / 4) * 16;
    const int scatter_b = (group & 3) + (group / 4) * 32;
    const int8_t *a = p.a + size_t(row + scatter_a) * 4096 + (lane % 4) * 4;
    const int8_t *b = p.b + size_t(col + scatter_b) * 4096 + (lane % 4) * 4;
    int32_t c0 = 0, c1 = 0;
    uint32_t words[16] = {};
#if CP_MINIMAL_PREFETCH
    uint32_t next_a = minimal_load(a), next_b = minimal_load(b);
#endif
#pragma unroll 1
    for (int k = 0; k < 4096; k += 16) {
#if CP_MINIMAL_PREFETCH
        uint32_t av = next_a, bv = next_b;
        if (k + 16 < 4096) {
            next_a = minimal_load(a + k + 16);
            next_b = minimal_load(b + k + 16);
        }
#else
        uint32_t av = minimal_load(a + k), bv = minimal_load(b + k);
#endif
        asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                     "{%0, %1}, {%2}, {%3}, {%0, %1};"
                     : "+r"(c0), "+r"(c1) : "r"(av), "r"(bv));
        if ((k + 16) % 128 == 0) {
            uint32_t x = uint32_t(c0) ^ uint32_t(c1);
#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
                x ^= __shfl_xor_sync(0xffffffff, x, offset);
            const int step = k / 128;
            if (lane == 0) {
                if (p.dump) p.dump[size_t(step) * tiles + tile] = x;
#pragma unroll
                for (int i = 0; i < 16; ++i)
                    if ((step & 15) == i)
                        words[i] = cp_cutlass_rotl32(words[i], 13) ^ x;
            }
        }
    }
    if (lane == 0)
        cp_cutlass_jackpot_try(words, p.jackpot.d_a_key8, p.jackpot.bound,
                              row_period, col_period, virtual_thread,
                              p.jackpot.d_found, p.jackpot.d_out_t_rows,
                              p.jackpot.d_out_t_cols);
}

struct MinimalOp {
    MinimalParams params{};
    cutlass::Status initialize(int m, int n, int k, int, int, int8_t *a, int8_t *b,
                              uint32_t *dump, int, size_t,
                              const CpCutlassJackpotLaunch *jackpot) {
        if (m % 128 || n % 128 || k != 4096 || !jackpot)
            return cutlass::Status::kErrorInvalidProblem;
        params = {a,b,m,n,dump,*jackpot};
        return cutlass::Status::kSuccess;
    }
    cutlass::Status operator()() {
        const size_t tiles = size_t(params.m) * params.n / 64;
        minimal_kernel<<<unsigned((tiles+7)/8),256>>>(params);
        return cudaGetLastError() == cudaSuccess ? cutlass::Status::kSuccess
                                                : cutlass::Status::kErrorInternal;
    }
};
} // namespace cp_research
