// TNet v1 miner for the Requant coin (github.com/requant-network/requant, SPEC.md): the CUDA backend, on tensor cores with
// the CUTLASS int8 GEMM compiled into the miner (cp_tnet_gemm.cu; cuBLAS only as an optional cross-check). The network and
// the mining loop are shared with the OpenCL backend (cp_tnet_net.cpp, cp_tnet_common.hpp).
//
// Per nonce the input X_0 has B rows (B = 65536 on the test network); rows are independent, so they are
// processed in batches of --batch rows: expand the batch's rows of X_0, run L layers of int8 GEMM + integer
// requantization (in the GEMM's epilogue, or as a separate pass with CP_TNET_FUSED=0), hash every w-byte piece of
// every output row, compare with the 256-bit target. Kernels are
// byte-compatible with the Rust reference (crates/tnet) and were cross-checked against it on Turing and Ampere.

#include "cp_tnet.h"

#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <cmath>
#include <deque>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include "cp_tnet_common.hpp"
#include "cp_tnet_gemm.h"
#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
#include <cublas_v2.h>
#endif


// ---------------------------------------------------------------- SHA-256 (host and device)

__constant__ uint32_t rq_dK[64];

__host__ __device__ __forceinline__ uint32_t rq_rotr(uint32_t x, int s) { return (x >> s) | (x << (32 - s)); }

// fully unrolled on the GPU, so the message schedule stays in registers and K comes as immediates
#ifdef __CUDA_ARCH__
#define RQ_UNROLL _Pragma("unroll")
#else
#define RQ_UNROLL
#endif

__host__ __device__ __forceinline__ void rq_compress(uint32_t h[8], const uint32_t w0[16]) {
    uint32_t w[64];
    RQ_UNROLL
    for (int i = 0; i < 16; ++i) w[i] = w0[i];
    RQ_UNROLL
    for (int i = 16; i < 64; ++i)
        w[i] = w[i - 16] + (rq_rotr(w[i - 15], 7) ^ rq_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)) + w[i - 7] +
               (rq_rotr(w[i - 2], 17) ^ rq_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10));
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    RQ_UNROLL
    for (int i = 0; i < 64; ++i) {
#ifdef __CUDA_ARCH__
        const uint32_t k = rq_dK[i];
#else
        const uint32_t k = rqh_K[i];
#endif
        uint32_t t1 = hh + (rq_rotr(e, 6) ^ rq_rotr(e, 11) ^ rq_rotr(e, 25)) + ((e & f) ^ (~e & g)) + k + w[i];
        uint32_t t2 = (rq_rotr(a, 2) ^ rq_rotr(a, 13) ^ rq_rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}
__host__ __device__ __forceinline__ void rq_init(uint32_t h[8]) {
    h[0] = 0x6a09e667; h[1] = 0xbb67ae85; h[2] = 0x3c6ef372; h[3] = 0xa54ff53a;
    h[4] = 0x510e527f; h[5] = 0x9b05688c; h[6] = 0x1f83d9ab; h[7] = 0x5be0cd19;
}
__host__ __device__ __forceinline__ uint32_t rq_be(const uint8_t* b) {
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | b[3];
}


// ---------------------------------------------------------------- kernels

__global__ void rq_expand(RqPrefix px, uint32_t c0, size_t count, uint8_t* __restrict__ out) {
    const size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const uint32_t cc = c0 + (uint32_t)k;
    uint32_t w[16], h[8];
    for (int i = 0; i < 11; ++i) w[i] = px.pw[i];
    w[11] = (px.b44 << 24) | ((cc & 0xff) << 16) | (((cc >> 8) & 0xff) << 8) | ((cc >> 16) & 0xff);
    w[12] = ((cc >> 24) << 24) | (0x80u << 16);
    w[13] = 0; w[14] = 0; w[15] = 49 * 8;
    rq_init(h);
    rq_compress(h, w);
    // the digest's bytes in order: two 16-byte stores (out is 32-byte aligned)
    uint4* o = reinterpret_cast<uint4*>(out + 32 * k);
    o[0] = make_uint4(__byte_perm(h[0], 0, 0x0123), __byte_perm(h[1], 0, 0x0123), __byte_perm(h[2], 0, 0x0123),
                      __byte_perm(h[3], 0, 0x0123));
    o[1] = make_uint4(__byte_perm(h[4], 0, 0x0123), __byte_perm(h[5], 0, 0x0123), __byte_perm(h[6], 0, 0x0123),
                      __byte_perm(h[7], 0, 0x0123));
}

// WT[j n + k] = W[k n + j]
__global__ void rq_transpose(const int8_t* __restrict__ W, int8_t* __restrict__ WT, int n) {
    __shared__ int8_t tile[32][33];
    const int bx = blockIdx.x * 32, by = blockIdx.y * 32;
    for (int y = threadIdx.y; y < 32; y += 8) tile[y][threadIdx.x] = W[(size_t)(by + y) * n + bx + threadIdx.x];
    __syncthreads();
    for (int y = threadIdx.y; y < 32; y += 8) WT[(size_t)(bx + y) * n + by + threadIdx.x] = tile[threadIdx.x][y];
}

// X = clamp((Y M + 2^23) >> 24, -128, 127)
__device__ __forceinline__ signed char rq_q(int v, int M) {
    const long long t = ((long long)v * M + (1LL << 23)) >> 24;
    return (signed char)(t < -128 ? -128 : (t > 127 ? 127 : t));
}
__global__ void rq_requant(const int32_t* __restrict__ Y, int8_t* __restrict__ X, size_t count, int M) {
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count / 4) return;
    const int4 y = reinterpret_cast<const int4*>(Y)[t];
    char4 o;
    o.x = rq_q(y.x, M); o.y = rq_q(y.y, M); o.z = rq_q(y.z, M); o.w = rq_q(y.w, M);
    reinterpret_cast<char4*>(X)[t] = o;
}


// Winning tickets kept per scan (a pool's easy share target can give several in one batch).
#define RQ_MAX_HITS 16

// Ticket (r0 + i, c): SHA256(piece || 0x54 || hd || LE64 nonce || LE32 row || LE32 c) <= target; w % 64 == 0.
__global__ void rq_tickets(const int8_t* __restrict__ X, int n, int rows, int w, uint32_t r0, RqTail tail, RqTarget tg,
                           unsigned int* __restrict__ found, unsigned long long* __restrict__ hits) {
    const int per_row = n / w;
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)rows * per_row) return;
    const int i = (int)(t / per_row), c = (int)(t % per_row);
    const uint32_t row = r0 + (uint32_t)i;
    const uint4* piece = reinterpret_cast<const uint4*>(X + (size_t)i * n + (size_t)c * w);
    uint32_t h[8], wd[16];
    rq_init(h);
    for (int blk = 0; blk < w / 64; ++blk) {
        RQ_UNROLL
        for (int q = 0; q < 4; ++q) {
            const uint4 v = piece[blk * 4 + q];
            wd[4 * q] = __byte_perm(v.x, 0, 0x0123);
            wd[4 * q + 1] = __byte_perm(v.y, 0, 0x0123);
            wd[4 * q + 2] = __byte_perm(v.z, 0, 0x0123);
            wd[4 * q + 3] = __byte_perm(v.w, 0, 0x0123);
        }
        rq_compress(h, wd);
    }
    RQ_UNROLL
    for (int q = 0; q < 16; ++q) wd[q] = tail.wd[q];
    // bytes 40..51: nonce byte 7 | LE32 row | LE32 c | 0x80 0 0
    const uint32_t rb = __byte_perm(row, 0, 0x0123), cb = __byte_perm((uint32_t)c, 0, 0x0123);
    wd[10] = (wd[10] & 0xff000000u) | (rb >> 8);
    wd[11] = (rb << 24) | (cb >> 8);
    wd[12] = (cb << 24) | 0x00800000u;
    rq_compress(h, wd);
    for (int q = 0; q < 8; ++q) {
        if (h[q] < tg.t[q]) break;
        if (h[q] > tg.t[q]) return;
    }
    const unsigned k = atomicAdd(found, 1u);
    if (k < RQ_MAX_HITS) hits[k] = ((unsigned long long)i << 32) | (unsigned)c;
}


// ---------------------------------------------------------------- engine

// A tensor-core GEMM kernel against a CPU product on random int8 matrices (rows not a multiple of the tile, so
// the edges are covered too): it must agree exactly, or the miner would only find invalid tickets.
static bool gemm_check(int kind, int rows, int n) {
    if (kind == CP_TNET_GEMM_NONE) return false;
    std::mt19937 rng(12345);
    std::vector<int8_t> x((size_t)rows * n), wt((size_t)n * n);
    for (auto& v : x) v = (int8_t)(rng() & 0xff);
    for (auto& v : wt) v = (int8_t)(rng() & 0xff);
    int8_t *dx = nullptr, *dw = nullptr;
    int32_t* dy = nullptr;
    std::vector<int32_t> y((size_t)rows * n);
    bool ok = cudaMalloc(&dx, x.size()) == cudaSuccess && cudaMalloc(&dw, wt.size()) == cudaSuccess &&
              cudaMalloc(&dy, y.size() * 4) == cudaSuccess;
    if (ok) {
        cudaMemcpy(dx, x.data(), x.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dw, wt.data(), wt.size(), cudaMemcpyHostToDevice);
        ok = cp_tnet_gemm(kind, dx, dw, dy, rows, n, 0) == 0 &&
             cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dy);
    for (int r = 0; ok && r < rows; ++r)
        for (int j = 0; ok && j < n; ++j) {
            int32_t s = 0;
            for (int k = 0; k < n; ++k) s += (int32_t)x[(size_t)r * n + k] * wt[(size_t)j * n + k];
            ok = s == y[(size_t)r * n + j];
        }
    cudaGetLastError();
    return ok;
}

// The requantizing GEMM (requantization in the epilogue) against the CPU: product, then clamp((s M + 2^23) >> 24). Two
// multipliers: the network's order of magnitude, and one that saturates most outputs, so both clamps are exercised.
static bool rq_check(int kind, int rows, int n, int mult) {
    std::mt19937 rng(54321 + mult);
    std::vector<int8_t> x((size_t)rows * n), wt((size_t)n * n), xo((size_t)rows * n);
    for (auto& v : x) v = (int8_t)(rng() & 0xff);
    for (auto& v : wt) v = (int8_t)(rng() & 0xff);
    int8_t *dx = nullptr, *dw = nullptr, *dxo = nullptr;
    bool ok = cudaMalloc(&dx, x.size()) == cudaSuccess && cudaMalloc(&dw, wt.size()) == cudaSuccess &&
              cudaMalloc(&dxo, xo.size()) == cudaSuccess;
    if (ok) {
        cudaMemcpy(dx, x.data(), x.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dw, wt.data(), wt.size(), cudaMemcpyHostToDevice);
        ok = cp_tnet_gemm_rq(kind, dx, dw, dxo, rows, n, mult, 0) == 0 &&
             cudaMemcpy(xo.data(), dxo, xo.size(), cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dxo);
    for (int r = 0; ok && r < rows; ++r)
        for (int j = 0; ok && j < n; ++j) {
            int32_t s = 0;
            for (int k = 0; k < n; ++k) s += (int32_t)x[(size_t)r * n + k] * wt[(size_t)j * n + k];
            const long long t = ((long long)s * mult + (1LL << 23)) >> 24;
            ok = (int8_t)(t < -128 ? -128 : (t > 127 ? 127 : t)) == xo[(size_t)r * n + j];
        }
    cudaGetLastError();
    return ok;
}

// The GEMM kernel for `device`: the best for its architecture (CP_TNET_GEMM=sm75|sm80 overrides), checked against
// the CPU; if the sm80 kernel disagrees, the sm75 one (mma.m8n8k16 runs on every GPU since Turing).
static int pick_gemm(int device) {
    int kind = cp_tnet_gemm_kind(device);
    const char* force = getenv("CP_TNET_GEMM");
    if (force && !strcmp(force, "sm75")) kind = CP_TNET_GEMM_SM75;
    if (force && !strcmp(force, "sm80")) kind = CP_TNET_GEMM_SM80;
    if (kind != CP_TNET_GEMM_NONE && gemm_check(kind, 200, 512)) return kind;
    if (kind == CP_TNET_GEMM_SM80 && gemm_check(CP_TNET_GEMM_SM75, 200, 512)) {
        fprintf(stderr, "[tnet] the sm80 GEMM disagrees with the CPU on this GPU; using the sm75 kernel\n");
        return CP_TNET_GEMM_SM75;
    }
    return CP_TNET_GEMM_NONE;
}

struct RqEngine : RqDevice {
    int device = 0;
    int n = 0, L = 0, w = 0, mult = 0, batch = 0;
    // fused (RqDevice): requantization in the GEMM epilogue; CP_TNET_FUSED=0 or a failed check: a separate pass
    std::vector<int8_t*> WT;
    int8_t *X = nullptr, *X2 = nullptr;
    int32_t* Y = nullptr;
    uint8_t *dhd = nullptr, *tmp = nullptr;
    unsigned int* dfound = nullptr;
    unsigned long long* dhits = nullptr;
    std::vector<uint8_t> seed;
    int gemm = CP_TNET_GEMM_NONE;
#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
    cublasHandle_t cb = nullptr;  // CP_TNET_CUBLAS=1: cuBLAS instead of CUTLASS, for comparisons
#endif

    explicit RqEngine(int dev) : device(dev) {}

    std::string name() const override {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        return prop.name;
    }
    std::string kernel() const override { return cp_tnet_gemm_name(gemm); }
    bool failed() override { return cudaGetLastError() != cudaSuccess; }

    bool init(const RqWork& wk, int rows) override {
        n = wk.n; L = wk.L; w = wk.w; mult = wk.mult; batch = rows;
        if (n % 64 || w % 64 || n % w) { fprintf(stderr, "[tnet] unsupported parameters n=%d w=%d\n", n, w); return false; }
        const size_t nn = (size_t)n * n, bn = (size_t)batch * n;
        gemm = pick_gemm(device);
        if (gemm == CP_TNET_GEMM_NONE) {
            fprintf(stderr, "[tnet] no working int8 GEMM: needs an NVIDIA GPU of compute 7.5 (Turing) or newer\n");
            return false;
        }
        bool unfused = false;
#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
        const char* use_cublas = getenv("CP_TNET_CUBLAS");
        if (use_cublas && !strcmp(use_cublas, "1")) cublasCreate(&cb);
        unfused = cb != nullptr;
#endif
        if (fused < 0) {
            const char* env = getenv("CP_TNET_FUSED");
            fused = !unfused && !(env && !strcmp(env, "0"));
        }
        if (fused && !(rq_check(gemm, 200, 512, 2505) && rq_check(gemm, 200, 512, 300000))) {
            fprintf(stderr, "[tnet] the requantizing GEMM disagrees with the CPU on this GPU; requantizing in a separate pass\n");
            fused = 0;
        }
        WT.assign(L, nullptr);
        for (auto& p : WT) if (cudaMalloc(&p, nn) != cudaSuccess) return false;
        // the int32 product is kept only for the separate pass
        return cudaMalloc(&tmp, (nn + 31) / 32 * 32) == cudaSuccess && cudaMalloc(&X, (bn + 31) / 32 * 32 + 32) == cudaSuccess &&
               cudaMalloc(&X2, bn) == cudaSuccess && (fused || cudaMalloc(&Y, bn * 4) == cudaSuccess) &&
               cudaMalloc(&dhd, 32) == cudaSuccess && cudaMalloc(&dfound, 4) == cudaSuccess &&
               cudaMalloc(&dhits, 8 * RQ_MAX_HITS) == cudaSuccess;
    }

    ~RqEngine() override {
        for (auto p : WT) cudaFree(p);
        cudaFree(X);
        cudaFree(X2);
        cudaFree(Y);
        cudaFree(dhd);
        cudaFree(tmp);
        cudaFree(dfound);
        cudaFree(dhits);
#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
        if (cb) cublasDestroy(cb);
#endif
    }

    // W_l = expand(SHA256("abacus/tnet-w" || seed || LE32(l)), n^2), stored transposed.
    void set_epoch(const std::vector<uint8_t>& s) override {
        if (s == seed) return;
        const auto t0 = std::chrono::steady_clock::now();
        const size_t nn = (size_t)n * n, hashes = (nn + 31) / 32;
        for (int l = 0; l < L; ++l) {
            uint8_t sl[32];
            rq_w_seed(s, l, sl);
            rq_expand<<<(unsigned)((hashes + 255) / 256), 256>>>(rq_prefix(sl), 0, hashes, tmp);
            rq_transpose<<<dim3(n / 32, n / 32), dim3(32, 8)>>>((const int8_t*)tmp, WT[l], n);
        }
        cudaDeviceSynchronize();
        seed = s;
        printf("[tnet] epoch weights ready (%.2f s)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    // Rows [r0, r0 + rows) of nonce; appends the winning tickets (at most RQ_MAX_HITS) to `hits`.
    void scan(const RqWork& wk, uint64_t nonce, uint32_t r0, int rows, std::vector<RqHit>& hits) override {
        uint8_t xs[32];
        rq_x0_seed(wk.digest, nonce, xs);
        const size_t bn = (size_t)rows * n, hashes = (bn + 31) / 32;
        const uint32_t c0 = (uint32_t)((size_t)r0 * n / 32);
        cudaMemcpy(dhd, wk.digest.data(), 32, cudaMemcpyHostToDevice);
        cudaMemset(dfound, 0, 4);
        cudaEvent_t ev[4];
        if (profile) {
            for (auto& e : ev) cudaEventCreate(&e);
            cudaEventRecord(ev[0]);
        }
        rq_expand<<<(unsigned)((hashes + 255) / 256), 256>>>(rq_prefix(xs), c0, hashes, (uint8_t*)X);
        if (profile) cudaEventRecord(ev[1]);
        int8_t *cur = X, *nxt = X2;
        for (int l = 0; l < L; ++l) {
            if (fused) {
                cp_tnet_gemm_rq(gemm, cur, WT[l], nxt, rows, n, mult, 0);
                std::swap(cur, nxt);
                continue;
            }
            // row-major (rows x n) = cur * W_l, with W_l given transposed (WT, row-major)
#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
            if (cb) {
                const int32_t alpha = 1, beta = 0;
                cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, n, &alpha, WT[l], CUDA_R_8I, n, cur, CUDA_R_8I, n, &beta,
                             Y, CUDA_R_32I, n, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
            } else
#endif
                cp_tnet_gemm(gemm, cur, WT[l], Y, rows, n, 0);
            rq_requant<<<(unsigned)((bn / 4 + 255) / 256), 256>>>(Y, nxt, bn, mult);
            std::swap(cur, nxt);
        }
        RqTarget tg;
        for (int q = 0; q < 8; ++q) tg.t[q] = rq_be(wk.target.data() + 4 * q);
        const size_t tickets = (size_t)rows * (n / w);
        if (profile) cudaEventRecord(ev[2]);
        rq_tickets<<<(unsigned)((tickets + 127) / 128), 128>>>(cur, n, rows, w, r0, rq_tail(wk.digest.data(), nonce, w), tg,
                                                              dfound, dhits);
        if (profile) {
            cudaEventRecord(ev[3]);
            cudaEventSynchronize(ev[3]);
            float a = 0, b = 0, c = 0;
            cudaEventElapsedTime(&a, ev[0], ev[1]);
            cudaEventElapsedTime(&b, ev[1], ev[2]);
            cudaEventElapsedTime(&c, ev[2], ev[3]);
            ms_expand += a;
            ms_layers += b;
            ms_hash += c;
            ++scans;
            for (auto& e : ev) cudaEventDestroy(e);
        }
        unsigned int found = 0;
        cudaMemcpy(&found, dfound, 4, cudaMemcpyDeviceToHost);
        if (found == 0) return;
        const unsigned k = found < RQ_MAX_HITS ? found : RQ_MAX_HITS;
        unsigned long long packed[RQ_MAX_HITS];
        cudaMemcpy(packed, dhits, 8 * k, cudaMemcpyDeviceToHost);
        for (unsigned q = 0; q < k; ++q) {
            const uint32_t i = (uint32_t)(packed[q] >> 32);
            RqHit h;
            h.nonce = nonce;
            h.row = r0 + i;
            h.c = (uint32_t)(packed[q] & 0xFFFFFFFF);
            h.piece.resize(w);
            cudaMemcpy(h.piece.data(), cur + (size_t)i * n + (size_t)h.c * w, w, cudaMemcpyDeviceToHost);
            hits.push_back(std::move(h));
        }
    }
};

// Offline benchmark (rq_bench in cp_tnet_net.cpp) on CUDA device `device`.
extern "C" int cp_tnet_cuda_bench(int device, int batch, double seconds) {
    if (cudaSetDevice(device) != cudaSuccess) { fprintf(stderr, "[tnet] no CUDA device %d\n", device); return 1; }
    cudaMemcpyToSymbol(rq_dK, rqh_K, sizeof(rqh_K));
    return rq_bench([device] { return std::unique_ptr<RqDevice>(new RqEngine(device)); }, batch, seconds);
}

extern "C" int cp_tnet_cuda_selftest(int device) {
    int count = 0;
    const cudaError_t ce = cudaGetDeviceCount(&count);
    if (ce != cudaSuccess || count == 0 || device >= count || cudaSetDevice(device) != cudaSuccess) {
        printf("[tnet] selftest: no usable NVIDIA GPU %d (%s); install a current NVIDIA driver (R528 or newer)\n",
               device, ce != cudaSuccess ? cudaGetErrorString(ce) : "device not found");
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("[tnet] selftest on %s (compute %d.%d)\n", prop.name, prop.major, prop.minor);
    cudaMemcpyToSymbol(rq_dK, rqh_K, sizeof(rqh_K));
    // host SHA-256("abc")
    uint8_t d[32];
    host_sha256(std::vector<uint8_t>{'a', 'b', 'c'}, d);
    const bool abc = to_hex(d, 32) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    // device expansion block 5 of seed 0x03.. == host SHA256("abacus/expand" || seed || LE32 5)
    uint8_t seed[32];
    memset(seed, 3, 32);
    uint8_t* dout;
    cudaMalloc(&dout, 64);
    rq_expand<<<1, 2>>>(rq_prefix(seed), 5, 2, dout);
    uint8_t got[64];
    cudaMemcpy(got, dout, 64, cudaMemcpyDeviceToHost);
    cudaFree(dout);
    bool exp = true;
    for (uint32_t k = 0; k < 2; ++k) {
        std::vector<uint8_t> m((const uint8_t*)"abacus/expand", (const uint8_t*)"abacus/expand" + 13);
        m.insert(m.end(), seed, seed + 32);
        for (int q = 0; q < 4; ++q) m.push_back((uint8_t)((5 + k) >> (8 * q)));
        host_sha256(m, d);
        exp = exp && memcmp(d, got + 32 * k, 32) == 0;
    }
    // ticket hashing: SHA256(piece || 0x54 || hd || nonce || row || c) against a target equal to one ticket's hash
    // (it must be found) and one below it (it must not)
    bool tk = true;
    {
        const int tn = 512, tw = 256, trows = 4;
        const uint32_t r0 = 70000, ti = 3, tc = 1;
        const uint64_t tnonce = 0x0123456789abcdefULL;
        std::vector<int8_t> x((size_t)trows * tn);
        std::mt19937 rng(99);
        for (auto& v : x) v = (int8_t)(rng() & 0xff);
        uint8_t hd[32];
        for (int q = 0; q < 32; ++q) hd[q] = (uint8_t)(q * 7 + 1);
        std::vector<uint8_t> m(x.begin() + ti * tn + tc * tw, x.begin() + ti * tn + (tc + 1) * tw);
        m.push_back(0x54);
        m.insert(m.end(), hd, hd + 32);
        for (int q = 0; q < 8; ++q) m.push_back((uint8_t)(tnonce >> (8 * q)));
        for (int q = 0; q < 4; ++q) m.push_back((uint8_t)((r0 + ti) >> (8 * q)));
        for (int q = 0; q < 4; ++q) m.push_back((uint8_t)(tc >> (8 * q)));
        uint8_t want[32];
        host_sha256(m, want);
        int8_t* dx = nullptr;
        unsigned int* df = nullptr;
        unsigned long long* dh = nullptr;
        cudaMalloc(&dx, x.size());
        cudaMalloc(&df, 4);
        cudaMalloc(&dh, 8 * RQ_MAX_HITS);
        cudaMemcpy(dx, x.data(), x.size(), cudaMemcpyHostToDevice);
        for (int below = 0; below < 2; ++below) {
            RqTarget tg;
            for (int q = 0; q < 8; ++q) tg.t[q] = rq_be(want + 4 * q);
            if (below) tg.t[7] -= 1;  // so the hash must not end in 0 (checked below)
            cudaMemset(df, 0, 4);
            rq_tickets<<<1, 64>>>(dx, tn, trows, tw, r0, rq_tail(hd, tnonce, tw), tg, df, dh);
            unsigned int f = 0;
            unsigned long long got[RQ_MAX_HITS];
            cudaMemcpy(&f, df, 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(got, dh, 8 * RQ_MAX_HITS, cudaMemcpyDeviceToHost);
            bool seen = false;
            for (unsigned q = 0; q < f && q < RQ_MAX_HITS; ++q) seen |= got[q] == (((unsigned long long)ti << 32) | tc);
            tk = tk && (below ? !seen : seen) && (want[31] != 0);
        }
        cudaFree(dx);
        cudaFree(df);
        cudaFree(dh);
    }
    const int kind = pick_gemm(device);
    const bool mm = kind != CP_TNET_GEMM_NONE;
    const bool rq = mm && rq_check(kind, 200, 512, 2505) && rq_check(kind, 200, 512, 300000);
    printf("[tnet] selftest: sha256 %s, expansion %s, tickets %s, int8 GEMM %s (%s), requantizing GEMM %s, cuda %s\n",
           abc ? "ok" : "FAIL", exp ? "ok" : "FAIL", tk ? "ok" : "FAIL", mm ? "ok" : "FAIL", cp_tnet_gemm_name(kind),
           rq ? "ok" : "FAIL", cudaGetErrorString(cudaGetLastError()));
    return abc && exp && tk && mm && rq ? 0 : 1;
}

extern "C" int cp_tnet_cuda_solo(const char* rpc, const char* payee_hex, const char* worker, int device, int batch, double seconds,
                                 long long blocks) {
    if (cudaSetDevice(device) != cudaSuccess) { fprintf(stderr, "[tnet] no CUDA device %d\n", device); return 1; }
    cudaMemcpyToSymbol(rq_dK, rqh_K, sizeof(rqh_K));
    return rq_solo([device] { return std::unique_ptr<RqDevice>(new RqEngine(device)); }, rpc, payee_hex, worker, batch, seconds,
                   blocks, device);
}

#endif  // CP_ENABLE_CUDA
