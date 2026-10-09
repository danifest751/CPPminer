// Abacus backend (candidate A) for CPPminer: CUDA nonce search over the verifiable-algebra PoW.
//
// Encoding matches the Abacus prototype exactly (byte-for-byte), so a future solo miner can submit
// directly:
//   preheader = "abacus/ph" || chain_id(32) || version(u32 LE) || prev(32) || height(u64 LE)
//               || timestamp(u64 LE) || nonce(u64 LE)
//   seed      = SHA256("abacus/instance" || preheader)
//   A,B       = expand(seed, 2*n*n)   (SHA256 counter mode, 4 goldilocks elements per hash)
//   C         = A*B over Goldilocks (P = 2^64 - 2^32 + 1)
//   score     = SHA256("abacus/score" || preheader || C_le)   (accept: leading_zero_bits(score) >= bits)
//
// This is a mock/benchmark loop (no pool yet). It exercises the full attempt on the GPU matmul.

#include "cp_abacus.h"

#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>

#define GOLDI 0xFFFFFFFF00000001ULL
#define EPS   0xFFFFFFFFULL

// ---------------- host SHA-256 ----------------
namespace {
const uint32_t K256[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

void sha256(const uint8_t* msg, size_t len, uint8_t out[32]) {
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::vector<uint8_t> data(msg, msg + len);
    uint64_t bits = (uint64_t)len * 8;
    data.push_back(0x80);
    while (data.size() % 64 != 56) data.push_back(0);
    for (int i = 7; i >= 0; --i) data.push_back((uint8_t)(bits >> (8 * i)));
    for (size_t off = 0; off < data.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = ((uint32_t)data[off+4*i]<<24)|((uint32_t)data[off+4*i+1]<<16)|((uint32_t)data[off+4*i+2]<<8)|data[off+4*i+3];
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = (w[i-15]>>7|w[i-15]<<25)^(w[i-15]>>18|w[i-15]<<14)^(w[i-15]>>3);
            uint32_t s1 = (w[i-2]>>17|w[i-2]<<15)^(w[i-2]>>19|w[i-2]<<13)^(w[i-2]>>10);
            w[i] = w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = (e>>6|e<<26)^(e>>11|e<<21)^(e>>25|e<<7);
            uint32_t ch = (e&f)^((~e)&g);
            uint32_t t1 = hh+S1+ch+K256[i]+w[i];
            uint32_t S0 = (a>>2|a<<30)^(a>>13|a<<19)^(a>>22|a<<10);
            uint32_t maj = (a&b)^(a&c)^(b&c);
            uint32_t t2 = S0+maj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    for (int i = 0; i < 8; ++i) { out[4*i]=(uint8_t)(h[i]>>24); out[4*i+1]=(uint8_t)(h[i]>>16); out[4*i+2]=(uint8_t)(h[i]>>8); out[4*i+3]=(uint8_t)h[i]; }
}

void expand(const uint8_t* seed, size_t seed_len, uint64_t* out, size_t count) {
    static const char DOM[] = "abacus/expand";
    size_t dl = strlen(DOM), got = 0; uint32_t ctr = 0;
    while (got < count) {
        std::vector<uint8_t> b;
        b.insert(b.end(), DOM, DOM + dl);
        b.insert(b.end(), seed, seed + seed_len);
        b.push_back(ctr & 0xff); b.push_back((ctr>>8)&0xff); b.push_back((ctr>>16)&0xff); b.push_back((ctr>>24)&0xff);
        uint8_t hh[32]; sha256(b.data(), b.size(), hh);
        for (int k = 0; k < 4 && got < count; ++k) {
            uint64_t v = 0; for (int j = 0; j < 8; ++j) v |= (uint64_t)hh[k*8+j] << (8*j);
            out[got++] = v % GOLDI;
        }
        ctr++;
    }
}

void make_preheader(uint8_t* ph, size_t* ph_len, const uint8_t chain_id[32], uint32_t version, uint64_t height, uint64_t ts, uint64_t nonce) {
    static const char DOM[] = "abacus/ph";
    size_t o = 0; size_t dl = strlen(DOM);
    memcpy(ph+o, DOM, dl); o += dl;
    memcpy(ph+o, chain_id, 32); o += 32;
    for (int i=0;i<4;i++) ph[o++] = (uint8_t)(version >> (8*i));
    for (int i=0;i<32;i++) ph[o++] = 0;
    for (int i=0;i<8;i++) ph[o++] = (uint8_t)(height >> (8*i));
    for (int i=0;i<8;i++) ph[o++] = (uint8_t)(ts >> (8*i));
    for (int i=0;i<8;i++) ph[o++] = (uint8_t)(nonce >> (8*i));
    *ph_len = o;
}

uint32_t leading_zero_bits(const uint8_t s[32]) {
    uint32_t lead = 0;
    for (int i = 0; i < 32; ++i) {
        if (s[i] == 0) { lead += 8; continue; }
        uint8_t b = s[i]; uint32_t z = 0; while (!(b & 0x80)) { z++; b <<= 1; }
        lead += z; break;
    }
    return lead;
}

__host__ __device__ __forceinline__ uint64_t gl_add(uint64_t a, uint64_t b) {
    uint64_t s = a + b; if (s < a) s += EPS; if (s >= GOLDI) s -= GOLDI; return s;
}
__device__ __forceinline__ uint64_t gl_mul(uint64_t a, uint64_t b) {
    uint64_t lo = a * b, hi = __umul64hi(a, b);
    uint64_t hh = hi >> 32, hl = hi & 0xFFFFFFFFULL;
    uint64_t t0 = lo - hh; if (lo < hh) t0 -= EPS;
    uint64_t t1 = hl * EPS; uint64_t r = t0 + t1; if (r < t0) r += EPS; return r;
}

#define TS 16
__global__ void matmul_kernel(const uint64_t* A, const uint64_t* B, uint64_t* C, int n) {
    __shared__ uint64_t As[TS][TS], Bs[TS][TS];
    int row = blockIdx.y*TS + threadIdx.y, col = blockIdx.x*TS + threadIdx.x;
    uint64_t acc = 0;
    for (int t = 0; t < n; t += TS) {
        As[threadIdx.y][threadIdx.x] = A[row*n + (t+threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t+threadIdx.y)*n + col];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TS; ++k) acc = gl_add(acc, gl_mul(As[threadIdx.y][k], Bs[k][threadIdx.x]));
        __syncthreads();
    }
    C[row*n + col] = acc;
}
} // namespace

#endif // CP_ENABLE_CUDA

extern "C" int cp_abacus_cuda_mock(int n, int bits, int seconds, int device) {
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    if (n <= 0 || (n % TS) != 0) { fprintf(stderr, "[abacus] n must be a positive multiple of %d\n", TS); return 1; }
    if (cudaSetDevice(device) != cudaSuccess) { fprintf(stderr, "[abacus] cudaSetDevice(%d) failed\n", device); return 1; }

    const size_t nn = (size_t)n * n;
    std::vector<uint64_t> A(nn), B(nn), C(nn);
    uint64_t *dA = nullptr, *dB = nullptr, *dC = nullptr;
    if (cudaMalloc(&dA, nn*8)!=cudaSuccess || cudaMalloc(&dB, nn*8)!=cudaSuccess || cudaMalloc(&dC, nn*8)!=cudaSuccess) {
        fprintf(stderr, "[abacus] cudaMalloc failed\n"); return 1;
    }
    dim3 block(TS, TS), grid(n/TS, n/TS);

    uint8_t chain_id[32]; memset(chain_id, 0xAB, 32);
    std::vector<uint8_t> ph(128), seedbuf;
    uint64_t nonce = 0, attempts = 0, found = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (;;) {
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration<double>(now - t0).count() >= seconds) break;

        size_t phlen = 0;
        make_preheader(ph.data(), &phlen, chain_id, 1, 0, 0, nonce);
        // seed = SHA256("abacus/instance" || preheader)
        { std::vector<uint8_t> m; const char DOM[]="abacus/instance"; m.insert(m.end(),DOM,DOM+strlen(DOM)); m.insert(m.end(),ph.data(),ph.data()+phlen); uint8_t hh[32]; sha256(m.data(),m.size(),hh); seedbuf.assign(hh,hh+32); }
        expand(seedbuf.data(), 32, A.data(), nn);
        expand(seedbuf.data(), 32, B.data(), nn);

        cudaMemcpy(dA, A.data(), nn*8, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B.data(), nn*8, cudaMemcpyHostToDevice);
        matmul_kernel<<<grid, block>>>(dA, dB, dC, n);
        cudaMemcpy(C.data(), dC, nn*8, cudaMemcpyDeviceToHost);

        // score = SHA256("abacus/score" || preheader || C_le)
        std::vector<uint8_t> ms; const char DOMS[]="abacus/score"; ms.insert(ms.end(),DOMS,DOMS+strlen(DOMS));
        ms.insert(ms.end(), ph.data(), ph.data()+phlen);
        for (size_t i = 0; i < nn; ++i) { uint64_t x = C[i]; for (int j=0;j<8;j++) ms.push_back((uint8_t)(x >> (8*j))); }
        uint8_t sc[32]; sha256(ms.data(), ms.size(), sc);

        attempts++;
        if ((int)leading_zero_bits(sc) >= bits) { found++; printf("[abacus] found nonce %llu (bits>=%d)\n", (unsigned long long)nonce, bits); }
        nonce++;
    }
    double secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    printf("[abacus] n=%d bits=%d device=%d attempts=%llu found=%llu attempts/s=%.1f\n",
           n, bits, device, (unsigned long long)attempts, (unsigned long long)found, attempts/secs);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
#else
    (void)n; (void)bits; (void)seconds; (void)device;
    fprintf(stderr, "[abacus] built without CUDA\n");
    return 1;
#endif
}
