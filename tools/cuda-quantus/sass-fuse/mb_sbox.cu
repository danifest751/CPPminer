// mb_sbox.cu: S-box (x^7 mod p) throughput, ptxas production code vs a hand-written SASS body.
// k_ptx runs the production sbox() from qk.inc on NCH independent chains per thread.
// k_hand has the same loop with xor placeholders; gen_sbox.py replaces them with hand SASS.
// Both are checked against a host reference (canonical mod p) and timed with clock64.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include "qk.inc"

#ifndef NCH
#define NCH 3
#endif
#define TPB 256

/* OP: 0 = sbox (x^7), 1 = gsqr (x^2), 2 = gmul(x, x) (x^2), 3 = gmul(gsqr(x), x) (x^3) */
template <int OP>
__device__ __forceinline__ u64 op_of(u64 x)
{
    if (OP == 1) return gsqr(x);
    if (OP == 2) return gmul(x, x);
    if (OP == 3) return gmul(gsqr(x), x);
    return sbox(x);
}

template <int OP>
__global__ void __launch_bounds__(TPB) k_ptx(u64* io, long long* cyc, int iters)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    u64 x[NCH];
#pragma unroll
    for (int i = 0; i < NCH; i++) x[i] = io[(size_t)t * NCH + i];
    __syncthreads();
    const long long t0 = clock64();
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
#pragma unroll
        for (int i = 0; i < NCH; i++) x[i] = op_of<OP>(x[i]);
    }
    const long long t1 = clock64();
#pragma unroll
    for (int i = 0; i < NCH; i++) io[(size_t)t * NCH + i] = x[i];
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}

extern "C" __global__ void __launch_bounds__(TPB) k_hand(u64* io, long long* cyc, int iters)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    u32 l[NCH], h[NCH];
#pragma unroll
    for (int i = 0; i < NCH; i++) { const u64 v = io[(size_t)t * NCH + i]; l[i] = (u32)v; h[i] = (u32)(v >> 32); }
    __syncthreads();
    const long long t0 = clock64();
#pragma unroll 1
    for (int it = 0; it < iters; it++) {
        /* placeholders: chain i low word xor 0x11110i00, high word xor 0x11110i01 (+ c_eps read so
         * the constant bank stays referenced) */
#define PH(I) \
        asm volatile("xor.b32 %0, %0, %1;" : "+r"(l[I]) : "n"(0x11110000 + (I) * 256)); \
        asm volatile("xor.b32 %0, %0, %1;" : "+r"(h[I]) : "n"(0x11110001 + (I) * 256));
        PH(0)
#if NCH > 1
        PH(1)
#endif
#if NCH > 2
        PH(2)
#endif
#if NCH > 3
        PH(3)
#endif
    }
    const long long t1 = clock64();
#pragma unroll
    for (int i = 0; i < NCH; i++) io[(size_t)t * NCH + i] = ((u64)h[i] << 32) | l[i];
    if (threadIdx.x == 0) cyc[blockIdx.x] = (t1 - t0) + (c_eps == 7 ? 1 : 0);
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static const u64 P = 0xFFFFFFFF00000001ull;
static u64 mulmod(u64 a, u64 b) { return (u64)(((unsigned __int128)a * b) % P); }
static int g_pow = 7;   /* env POW: reference exponent per iteration (debugging partial bodies) */
static u64 sbox_ref(u64 x)
{
    x %= P;
    u64 r = 1;
    for (int i = 0; i < g_pow; i++) r = mulmod(r, x);
    return r;
}

typedef void (*Kern)(u64*, long long*, int);

static void run(const char* name, Kern k, int sms, int bps, int check_iters, int time_iters)
{
    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, (const void*)k));
    int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)k, TPB, 0));
    const int b = bps < occ ? bps : occ;
    const int blocks = sms * b;
    const size_t n = (size_t)blocks * TPB * NCH;
    std::vector<u64> in(n), out(n);
    std::mt19937_64 rng(7);
    for (auto& v : in) v = rng();
    u64* d_io; long long* d_cyc;
    CK(cudaMalloc(&d_io, n * 8)); CK(cudaMalloc(&d_cyc, blocks * 8));
    // correctness
    CK(cudaMemcpy(d_io, in.data(), n * 8, cudaMemcpyHostToDevice));
    k<<<blocks, TPB>>>(d_io, d_cyc, check_iters); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out.data(), d_io, n * 8, cudaMemcpyDeviceToHost));
    size_t bad = 0, first = (size_t)-1;
    const size_t nchk = n < 65536 ? n : 65536;
    for (size_t i = 0; i < nchk; i++) {
        u64 r = in[i];
        for (int it = 0; it < check_iters; it++) r = sbox_ref(r);
        if (out[i] % P != r) { if (first == (size_t)-1) first = i; bad++; }
    }
    // speed
    double best = 0;
    std::vector<long long> hc(blocks);
    for (int rep = 0; rep < 5; rep++) {
        k<<<blocks, TPB>>>(d_io, d_cyc, time_iters); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hc.data(), d_cyc, blocks * 8, cudaMemcpyDeviceToHost));
        long long mx = 0; for (auto c : hc) if (c > mx) mx = c;
        const double per_sm = (double)b * TPB * NCH * time_iters / mx;
        if (per_sm > best) best = per_sm;
    }
    printf("%-6s regs=%d blocks/sm=%d (occ %d) check: %zu/%zu wrong%s  sbox/clk/SM=%.3f\n", name, fa.numRegs, b, occ,
           bad, nchk, bad ? " FIRST" : "", best);
    if (getenv("DBG"))
        for (int i = 0; i < 4; i++) {
            u64 r = in[i];
            for (int it = 0; it < check_iters; it++) r = sbox_ref(r);
            printf("       #%d in=%016llx got=%016llx want=%016llx\n", i, (unsigned long long)in[i],
                   (unsigned long long)out[i], (unsigned long long)r);
        }
    if (bad) printf("       first bad #%zu in=%016llx got=%016llx\n", first, (unsigned long long)in[first], (unsigned long long)out[first]);
    cudaFree(d_io); cudaFree(d_cyc);
}

int main(int argc, char** argv)
{
    const int bps = argc > 1 ? atoi(argv[1]) : 3;
    const int check_iters = argc > 2 ? atoi(argv[2]) : 8;
    const int time_iters = argc > 3 ? atoi(argv[3]) : 2048;
    if (getenv("POW")) g_pow = atoi(getenv("POW"));
    { u32 e = 0xFFFFFFFFu; CK(cudaMemcpyToSymbol(c_eps, &e, 4)); }
    int sms; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int op = getenv("OP") ? atoi(getenv("OP")) : 0;
    Kern kp = op == 1 ? k_ptx<1> : op == 2 ? k_ptx<2> : op == 3 ? k_ptx<3> : k_ptx<0>;
    run("ptxas", kp, sms, bps, check_iters, time_iters);
    run("hand", k_hand, sms, bps, check_iters, time_iters);
    return 0;
}
