// Lane-parallel Poseidon2 (Goldilocks, t=12) prototype: layout comparison vs 1 state/thread.
//   qlane test   - verify both layouts against the host reference permute()
//   qlane bench  - cycles per permutation for both layouts
//
// Lane layout: a warp holds 2 hashes, 16 lanes each (12 used, 4 zero). Every field element lives
// in one thread; the coupling (M4 blocks, external diffusion, internal row sum) goes through
// __shfl_sync within the 16-lane group. Compare with the 1-state-per-thread layout.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include "qpow/poseidon2.hpp"

typedef uint32_t u32;
typedef uint64_t u64;
#define DEV __device__ __forceinline__

DEV u64 mk64(u32 lo, u32 hi) { u64 r; asm("mov.b64 %0, {%1,%2};" : "=l"(r) : "r"(lo), "r"(hi)); return r; }
DEV void sp64(u64 x, u32& lo, u32& hi) { asm("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(x)); }

__constant__ u32 c_eps;

DEV u64 red128(u32 r0, u32 r1, u32 r2, u32 r3)
{
    u32 z0, z1;
    asm("{.reg .u32 t0, t1, rc, tc;\n\t"
        "mad.lo.cc.u32  t0, %2, %6, %3;\n\t"
        "madc.hi.cc.u32 t1, %2, %6, %4;\n\t"
        "addc.u32       rc, %5, 0;\n\t"
        "addc.u32       tc, t1, 0;\n\t"
        "sub.cc.u32     %0, t0, rc;\n\t"
        "subc.u32       %1, tc, 0;}"
        : "=r"(z0), "=r"(z1) : "r"(r2), "r"(r0), "r"(r1), "r"(r3), "r"(c_eps));
    return mk64(z0, z1);
}
DEV void mul128(u64 a, u64 b, u32& r0, u32& r1, u32& r2, u32& r3)
{
    u32 a0, a1, b0, b1; sp64(a, a0, a1); sp64(b, b0, b1);
    u64 P, Q, T;
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(P) : "r"(a0), "r"(b0));
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(T) : "r"(a0), "r"(b1));
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(Q) : "r"(a1), "r"(b1));
    u32 p0, p1, q0, q1, t0, t1; sp64(P, p0, p1); sp64(Q, q0, q1); sp64(T, t0, t1);
    asm("{.reg .u32 m0, m1, mc;\n\t"
        "mad.lo.cc.u32  m0, %3, %4, %5;\n\t"
        "madc.hi.cc.u32 m1, %3, %4, %6;\n\t"
        "addc.u32       mc, 0, 0;\n\t"
        "add.cc.u32     %0, %7, m0;\n\t"
        "addc.cc.u32    %1, %8, m1;\n\t"
        "addc.u32       %2, %9, mc;}"
        : "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(a1), "r"(b0), "r"(t0), "r"(t1), "r"(p1), "r"(q0), "r"(q1));
    r0 = p0;
}
DEV void sqr128(u64 a, u32& r0, u32& r1, u32& r2, u32& r3)
{
    u32 a0, a1; sp64(a, a0, a1);
    u64 P, Q, T;
    asm("mul.wide.u32 %0, %1, %1;" : "=l"(P) : "r"(a0));
    asm("mul.wide.u32 %0, %1, %1;" : "=l"(Q) : "r"(a1));
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(T) : "r"(a0), "r"(a1));
    u32 p0, p1, q0, q1, t0, t1; sp64(P, p0, p1); sp64(Q, q0, q1); sp64(T, t0, t1);
    u32 d0, d1, d2;
    asm("shl.b32 %0, %1, 1;" : "=r"(d0) : "r"(t0));
    asm("shf.l.wrap.b32 %0, %1, %2, 1;" : "=r"(d1) : "r"(t0), "r"(t1));
    asm("shr.b32 %0, %1, 31;" : "=r"(d2) : "r"(t1));
    asm("add.cc.u32  %0, %3, %6;\n\t addc.cc.u32 %1, %4, %7;\n\t addc.u32 %2, %5, %8;"
        : "=r"(r1), "=r"(r2), "=r"(r3) : "r"(p1), "r"(q0), "r"(q1), "r"(d0), "r"(d1), "r"(d2));
    r0 = p0;
}
DEV u64 gmul(u64 a, u64 b) { u32 r0, r1, r2, r3; mul128(a, b, r0, r1, r2, r3); return red128(r0, r1, r2, r3); }
DEV u64 gsqr(u64 a) { u32 r0, r1, r2, r3; sqr128(a, r0, r1, r2, r3); return red128(r0, r1, r2, r3); }
struct W { u32 lo, hi, t; };
DEV W w2(u64 a, u64 b) {
    W r; u32 a0, a1, b0, b1; sp64(a, a0, a1); sp64(b, b0, b1);
    asm("add.cc.u32 %0, %3, %5;\n\t addc.cc.u32 %1, %4, %6;\n\t addc.u32 %2, 0, 0;"
        : "=r"(r.lo), "=r"(r.hi), "=r"(r.t) : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    return r;
}
DEV u64 wred(W a) {
    u32 w0, w1, c;
    asm("mad.lo.cc.u32  %0, %3, %6, %4;\n\t madc.hi.u32 %1, %3, %6, 0;\n\t add.cc.u32 %1, %1, %5;\n\t addc.u32 %2, 0, 0;"
        : "=r"(w0), "=r"(w1), "=r"(c) : "r"(a.t), "r"(a.lo), "r"(a.hi), "r"(c_eps));
    u32 n = 0u - c;
    asm("add.cc.u32 %0, %0, %2;\n\t addc.u32 %1, %1, 0;" : "+r"(w0), "+r"(w1) : "r"(n));
    return mk64(w0, w1);
}
DEV u64 gadd(u64 a, u64 b) { return wred(w2(a, b)); }
DEV u64 canon(u64 a) { return a >= 0xFFFFFFFF00000001ull ? a - 0xFFFFFFFF00000001ull : a; }
DEV u64 sbox(u64 x) { u64 x2 = gsqr(x), x3 = gmul(x2, x), x4 = gsqr(x2); return gmul(x3, x4); }

__constant__ u64 c_rci[4][12];
__constant__ u64 c_rint[22];
__constant__ u64 c_rct[4][12];
__constant__ u64 c_diag[12];

// ---------------------------------------------------------------- 1 state / thread
DEV void ext_layer_1(u64 s[12])
{
#pragma unroll
    for (int b = 0; b < 3; b++) {
        u64* x = s + 4 * b;
        u64 t01 = gadd(x[0], x[1]), t23 = gadd(x[2], x[3]);
        u64 t0123 = gadd(t01, t23), t01123 = gadd(t0123, x[1]), t01233 = gadd(t0123, x[3]);
        u64 y0 = gadd(t01123, t01);
        u64 y1 = gadd(t01123, gadd(x[2], x[2]));
        u64 y2 = gadd(t01233, t23);
        u64 y3 = gadd(t01233, gadd(x[0], x[0]));
        x[0] = y0; x[1] = y1; x[2] = y2; x[3] = y3;
    }
    u64 sums[4];
#pragma unroll
    for (int j = 0; j < 4; j++) sums[j] = gadd(gadd(s[j], s[4 + j]), s[8 + j]);
#pragma unroll
    for (int i = 0; i < 12; i++) s[i] = gadd(s[i], sums[i & 3]);
}
DEV void perm1(u64 s[12])
{
    ext_layer_1(s);
#pragma unroll
    for (int r = 0; r < 4; r++) {
#pragma unroll
        for (int i = 0; i < 12; i++) s[i] = sbox(gadd(s[i], c_rci[r][i]));
        ext_layer_1(s);
    }
#pragma unroll 1
    for (int r = 0; r < 22; r++) {
        s[0] = sbox(gadd(s[0], c_rint[r]));
        u64 sum = s[0];
#pragma unroll
        for (int i = 1; i < 12; i++) sum = gadd(sum, s[i]);
#pragma unroll
        for (int i = 0; i < 12; i++) s[i] = gadd(gmul(s[i], c_diag[i]), sum);
    }
#pragma unroll
    for (int r = 0; r < 4; r++) {
#pragma unroll
        for (int i = 0; i < 12; i++) s[i] = sbox(gadd(s[i], c_rct[r][i]));
        ext_layer_1(s);
    }
}

// ---------------------------------------------------------------- lane-parallel (16-lane group)
DEV u64 sh16(u64 v, int src) {
    u32 lo = (u32)v, hi = (u32)(v >> 32);
    lo = __shfl_sync(0xffffffffu, lo, src, 16);
    hi = __shfl_sync(0xffffffffu, hi, src, 16);
    return mk64(lo, hi);
}
DEV u64 shx16(u64 v, int k) {
    u32 lo = (u32)v, hi = (u32)(v >> 32);
    lo = __shfl_xor_sync(0xffffffffu, lo, k, 16);
    hi = __shfl_xor_sync(0xffffffffu, hi, k, 16);
    return mk64(lo, hi);
}
// s is this thread's lane value (lane 0..11; lanes 12..15 hold 0)
DEV void m4_diff_lane(u64& s, int lane)
{
    const int b = lane & ~3;
    u64 x0 = sh16(s, b), x1 = sh16(s, b + 1), x2 = sh16(s, b + 2), x3 = sh16(s, b + 3);
    u64 t01 = gadd(x0, x1), t23 = gadd(x2, x3);
    u64 t0123 = gadd(t01, t23), t01123 = gadd(t0123, x1), t01233 = gadd(t0123, x3);
    u64 y0 = gadd(t01123, t01);
    u64 y1 = gadd(t01123, gadd(x2, x2));
    u64 y2 = gadd(t01233, t23);
    u64 y3 = gadd(t01233, gadd(x0, x0));
    const int j = lane & 3;
    u64 y = (j == 0) ? y0 : (j == 1) ? y1 : (j == 2) ? y2 : y3;
    u64 sumj = gadd(gadd(sh16(y, j), sh16(y, j + 4)), sh16(y, j + 8));
    s = (lane < 12) ? gadd(y, sumj) : 0ULL;
}
DEV void ext_round_lane(u64& s, int lane, const u64* rc)
{
    s = (lane < 12) ? sbox(gadd(s, rc[lane])) : 0ULL;
    m4_diff_lane(s, lane);
}
DEV void perm_lane(u64& s, int lane)
{
    m4_diff_lane(s, lane);
#pragma unroll
    for (int r = 0; r < 4; r++) ext_round_lane(s, lane, c_rci[r]);
#pragma unroll 1
    for (int r = 0; r < 22; r++) {
        u64 v = s;
        if (lane == 0) v = sbox(gadd(v, c_rint[r]));
        u64 sum = v;
        sum = gadd(sum, shx16(sum, 8));
        sum = gadd(sum, shx16(sum, 4));
        sum = gadd(sum, shx16(sum, 2));
        sum = gadd(sum, shx16(sum, 1));
        if (lane < 12) s = gadd(gmul(v, c_diag[lane]), sum);
    }
#pragma unroll
    for (int r = 0; r < 4; r++) ext_round_lane(s, lane, c_rct[r]);
}

// ---------------------------------------------------------------- kernels
__global__ void k_perm1(const u64* in, u64* out, int n, int reps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 s[12];
#pragma unroll
    for (int k = 0; k < 12; k++) s[k] = in[i * 12 + k];
    for (int r = 0; r < reps; r++) perm1(s);
#pragma unroll
    for (int k = 0; k < 12; k++) out[i * 12 + k] = s[k];
}
__global__ void k_perm_lane(const u64* in, u64* out, int nhash, int reps)
{
    const int t = threadIdx.x;          // 0..31
    const int g = t >> 4;               // group (hash within warp)
    const int lane = t & 15;
    const int h = blockIdx.x * (blockDim.x >> 4) + g;
    if (h >= nhash) return;
    u64 s = (lane < 12) ? in[h * 12 + lane] : 0ULL;
    for (int r = 0; r < reps; r++) perm_lane(s, lane);
    if (lane < 12) out[h * 12 + lane] = s;
}
__global__ void k_perm_lane_cyc(const u64* in, u64* out, int nhash, int reps, u64* cyc)
{
    long long t0 = clock64();
    const int t = threadIdx.x;
    const int g = t >> 4;
    const int lane = t & 15;
    const int h = blockIdx.x * (blockDim.x >> 4) + g;
    u64 s = (lane < 12) ? in[(h % nhash) * 12 + lane] : 0ULL;
    for (int r = 0; r < reps; r++) perm_lane(s, lane);
    if (lane < 12 && h < nhash) out[h * 12 + lane] = s;
    if (cyc) { __syncthreads(); if (threadIdx.x == 0) cyc[blockIdx.x] = (u64)(clock64() - t0); }
}
__global__ void k_perm1_cyc(const u64* in, u64* out, int n, int reps, u64* cyc)
{
    long long t0 = clock64();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    u64 s[12];
#pragma unroll
    for (int k = 0; k < 12; k++) s[k] = in[(i % n) * 12 + k];
    for (int r = 0; r < reps; r++) perm1(s);
    if (i < n)
#pragma unroll
        for (int k = 0; k < 12; k++) out[i * 12 + k] = s[k];
    if (cyc) { __syncthreads(); if (threadIdx.x == 0) cyc[blockIdx.x] = (u64)(clock64() - t0); }
}

__global__ void k_ext_lane(const u64* in, u64* out)
{
    const int t = threadIdx.x; const int g = t >> 4; const int lane = t & 15;
    u64 s = (lane < 12) ? in[g * 12 + lane] : 0ULL;
    m4_diff_lane(s, lane);
    if (lane < 12) out[g * 12 + lane] = s;
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static void up_consts()
{
    { u32 e = 0xFFFFFFFFu; CK(cudaMemcpyToSymbol(c_eps, &e, 4)); }
    CK(cudaMemcpyToSymbol(c_rci, qpow::RC_INITIAL, sizeof(qpow::RC_INITIAL)));
    CK(cudaMemcpyToSymbol(c_rint, qpow::RC_INTERNAL, sizeof(qpow::RC_INTERNAL)));
    CK(cudaMemcpyToSymbol(c_rct, qpow::RC_TERMINAL, sizeof(qpow::RC_TERMINAL)));
    CK(cudaMemcpyToSymbol(c_diag, qpow::MDS_DIAG, sizeof(qpow::MDS_DIAG)));
}

static int test()
{
    std::mt19937_64 rng(3);
    const int n = 64;
    std::vector<u64> in(n * 12), ref(n * 12), o1(n * 12, 0), ol(n * 12, 0);
    for (auto& v : in) { v = rng() % 0xFFFFFFFF00000001ull; if (!v) v = 1; }
    for (int i = 0; i < n; i++) { std::memcpy(&ref[i * 12], &in[i * 12], 96); qpow::permute(&ref[i * 12]); }
    u64 *di, *d1, *dl;
    CK(cudaMalloc(&di, 8 * n * 12)); CK(cudaMalloc(&d1, 8 * n * 12)); CK(cudaMalloc(&dl, 8 * n * 12));
    CK(cudaMemcpy(di, in.data(), 8 * n * 12, cudaMemcpyHostToDevice));
    k_perm1<<<(n + 127) / 128, 128>>>(di, d1, n, 1);
    int nhash = ((n + 1) / 2) * 2;
    k_perm_lane<<<(nhash + 1) / 2, 32>>>(di, dl, n, 1);
    CK(cudaGetLastError());
    CK(cudaMemcpy(o1.data(), d1, 8 * n * 12, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ol.data(), dl, 8 * n * 12, cudaMemcpyDeviceToHost));
    int bad1 = 0, badl = 0;
    for (int i = 0; i < n; i++)
        for (int k = 0; k < 12; k++) {
            u64 r = qpow::gf_canon(ref[i * 12 + k]);
            if (qpow::gf_canon(o1[i * 12 + k]) != r) bad1++;
            if (qpow::gf_canon(ol[i * 12 + k]) != r) badl++;
        }
    printf("1-thread mismatches: %d, lane mismatches: %d\n", bad1, badl);
    {
        u64 he[24], hh[24], heout[24];
        std::memcpy(he, in.data(), 24 * 8);
        std::memcpy(hh, he, 24 * 8);
        qpow::ext_layer(hh); qpow::ext_layer(hh + 12);
        u64 *de, *doe; CK(cudaMalloc(&de, 24 * 8)); CK(cudaMalloc(&doe, 24 * 8));
        CK(cudaMemcpy(de, he, 24 * 8, cudaMemcpyHostToDevice));
        k_ext_lane<<<1, 32>>>(de, doe); CK(cudaGetLastError());
        CK(cudaMemcpy(heout, doe, 24 * 8, cudaMemcpyDeviceToHost));
        int be = 0; for (int k = 0; k < 24; k++) if (qpow::gf_canon(heout[k]) != qpow::gf_canon(hh[k])) be++;
        printf("ext_layer mismatches: %d (ref0 %016llx lane0 %016llx)\n", be,
               (unsigned long long)qpow::gf_canon(hh[0]), (unsigned long long)qpow::gf_canon(heout[0]));
        cudaFree(de); cudaFree(doe);
    }
    if (badl) {
        printf("idx k   ref                1thr               lane\n");
        for (int k = 0; k < 12; k++)
            printf("0  %2d  %016llx  %016llx  %016llx\n", k,
                   (unsigned long long)qpow::gf_canon(ref[k]),
                   (unsigned long long)qpow::gf_canon(o1[k]),
                   (unsigned long long)qpow::gf_canon(ol[k]));
    }
    cudaFree(di); cudaFree(d1); cudaFree(dl);
    return bad1 + badl;
}

static void bench()
{
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    const int SM = prop.multiProcessorCount;
    const int reps = 1024;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    // 1 state / thread, 256-thread blocks
    int pb1 = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&pb1, k_perm1_cyc, 256, 0);
    const int g1 = SM * pb1, n1 = g1 * 256;
    std::vector<u64> in1(n1 * 12, 0x0123456789abcdefull);
    u64 *di1, *do1, *cyc;
    CK(cudaMalloc(&di1, 8 * (size_t)n1 * 12)); CK(cudaMalloc(&do1, 8 * (size_t)n1 * 12));
    CK(cudaMalloc(&cyc, 8 * g1)); CK(cudaMemcpy(di1, in1.data(), 8 * (size_t)n1 * 12, cudaMemcpyHostToDevice));
    k_perm1_cyc<<<g1, 256>>>(di1, do1, n1, reps, cyc); CK(cudaDeviceSynchronize());
    double best1 = 0;
    for (int it = 0; it < 3; it++) {
        cudaEventRecord(e0); k_perm1_cyc<<<g1, 256>>>(di1, do1, n1, reps, cyc); cudaEventRecord(e1);
        CK(cudaEventSynchronize(e1)); float ms; cudaEventElapsedTime(&ms, e0, e1);
        double pps = (double)n1 * reps / (ms * 1e-3); if (pps > best1) best1 = pps;
    }

    // lane-parallel, warp blocks (2 hashes per warp)
    int pbl = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&pbl, k_perm_lane_cyc, 32, 0);
    const int gl = SM * pbl, nh = gl * 2;
    std::vector<u64> inl(nh * 12, 0x0123456789abcdefull);
    u64 *dil, *dol;
    CK(cudaMalloc(&dil, 8 * (size_t)nh * 12)); CK(cudaMalloc(&dol, 8 * (size_t)nh * 12));
    CK(cudaMemcpy(dil, inl.data(), 8 * (size_t)nh * 12, cudaMemcpyHostToDevice));
    k_perm_lane_cyc<<<gl, 32>>>(dil, dol, nh, reps, cyc); CK(cudaDeviceSynchronize());
    double bestl = 0;
    for (int it = 0; it < 3; it++) {
        cudaEventRecord(e0); k_perm_lane_cyc<<<gl, 32>>>(dil, dol, nh, reps, cyc); cudaEventRecord(e1);
        CK(cudaEventSynchronize(e1)); float ms; cudaEventElapsedTime(&ms, e0, e1);
        double pps = (double)gl * 2 * reps / (ms * 1e-3); if (pps > bestl) bestl = pps;
    }

    cudaFuncAttributes fa1, fal;
    cudaFuncGetAttributes(&fa1, k_perm1_cyc); cudaFuncGetAttributes(&fal, k_perm_lane_cyc);
    printf("{\"perm1_Mperm_s\": %.1f, \"perm1_regs\": %d, \"perm1_bpsm\": %d, \"lane_Mperm_s\": %.1f, \"lane_regs\": %d, \"lane_bpsm\": %d, \"ratio_1_over_lane\": %.2f, \"SMs\": %d}\n",
           best1 / 1e6, fa1.numRegs, pb1, bestl / 1e6, fal.numRegs, pbl, best1 / bestl, SM);
}

int main(int argc, char** argv)
{
    up_consts();
    if (argc > 1 && !strcmp(argv[1], "test")) { int b = test(); printf(b ? "FAIL\n" : "PASS\n"); return b != 0; }
    if (argc > 1 && !strcmp(argv[1], "bench")) { bench(); return 0; }
    fprintf(stderr, "usage: qlane test | bench\n");
    return 2;
}
