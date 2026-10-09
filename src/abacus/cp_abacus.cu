// Abacus backend (candidate A) for CPPminer: CUDA nonce search over the verifiable-algebra PoW.
//
// Encoding v2 matches the Abacus prototype exactly (byte-for-byte; Abacus ADR 0010):
//   preheader = "abacus/ph" || chain_id(32) || version(u32 LE) || prev(32) || height(u64 LE)
//               || timestamp(u64 LE) || bits(u32 LE) || nonce(u64 LE)
//   seed      = SHA256("abacus/instance" || preheader)
//   A,B       = expand(seed, 2*n*n)   (SHA256 counter mode, 4 goldilocks elements per hash)
//   C         = A*B over Goldilocks (P = 2^64 - 2^32 + 1)
//   score     = SHA256("abacus/score" || preheader || C_le)   (accept: leading_zero_bits(score) >= bits)
//   dataset   (A') blk[u] = SHA256("abacus/ds" || seed || u || blk[u-1] || blk[ref(u)]),
//               ref(u) = LE64(blk[u-1][0..8]) mod u   (data-dependent)
//
// Modes: a mock/benchmark loop, an A' mock, and a solo/pool client (JOB/SUB) for abacus-node.

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

void build_dataset(const uint8_t epoch_seed[32], uint64_t nblocks, std::vector<uint64_t>& out) {
    out.assign((size_t)nblocks * 4, 0ull);
    uint8_t prev32[32];
    {
        std::vector<uint8_t> m; const char DOM[]="abacus/ds"; m.insert(m.end(),DOM,DOM+strlen(DOM));
        m.insert(m.end(), epoch_seed, epoch_seed+32);
        for (int i=0;i<8;i++) m.push_back(0);
        uint8_t h[32]; sha256(m.data(), m.size(), h);
        memcpy(prev32, h, 32);
        for (int w=0; w<4; ++w) { uint64_t v=0; for (int j=0;j<8;j++) v |= (uint64_t)h[w*8+j] << (8*j); out[w]=v; }
    }
    for (uint64_t u = 1; u < nblocks; ++u) {
        uint64_t rv=0; for (int j=0;j<8;j++) rv |= (uint64_t)prev32[j] << (8*j);
        uint64_t r = rv % u; // data-dependent reference (ADR 0010)
        std::vector<uint8_t> m; const char DOM[]="abacus/ds"; m.insert(m.end(),DOM,DOM+strlen(DOM));
        m.insert(m.end(), epoch_seed, epoch_seed+32);
        for (int i=0;i<8;i++) m.push_back((uint8_t)(u>>(8*i)));
        m.insert(m.end(), prev32, prev32+32);
        const uint8_t* refbytes = (const uint8_t*)&out[(size_t)r*4];
        m.insert(m.end(), refbytes, refbytes+32);
        uint8_t h[32]; sha256(m.data(), m.size(), h);
        memcpy(prev32, h, 32);
        for (int w=0; w<4; ++w) { uint64_t v=0; for (int j=0;j<8;j++) v |= (uint64_t)h[w*8+j] << (8*j); out[(size_t)u*4+w]=v; }
    }
}

// Preheader v2 (bits committed between timestamp and nonce).
void make_preheader(uint8_t* ph, size_t* ph_len, const uint8_t chain_id[32], uint32_t version, const uint8_t prev[32],
                    uint64_t height, uint64_t ts, uint32_t bits, uint64_t nonce) {
    static const char DOM[] = "abacus/ph";
    size_t o = 0; size_t dl = strlen(DOM);
    memcpy(ph+o, DOM, dl); o += dl;
    memcpy(ph+o, chain_id, 32); o += 32;
    for (int i=0;i<4;i++) ph[o++] = (uint8_t)(version >> (8*i));
    memcpy(ph+o, prev, 32); o += 32;
    for (int i=0;i<8;i++) ph[o++] = (uint8_t)(height >> (8*i));
    for (int i=0;i<8;i++) ph[o++] = (uint8_t)(ts >> (8*i));
    for (int i=0;i<4;i++) ph[o++] = (uint8_t)(bits >> (8*i));
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

__constant__ uint32_t c_K[64];

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

__device__ void sha256_dev(const unsigned char* msg, int len, unsigned char out[32]) {
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    const long long bitlen = (long long)len * 8;
    const int padded = ((len + 1 + 8 + 63) / 64) * 64;
    unsigned char block[64];
    for (int off = 0; off < padded; off += 64) {
        for (int k = 0; k < 64; ++k) {
            const int idx = off + k;
            unsigned char b;
            if (idx < len) b = msg[idx];
            else if (idx == len) b = 0x80;
            else if (idx >= padded - 8) b = (unsigned char)((unsigned long long)bitlen >> (8 * (padded - 1 - idx)));
            else b = 0;
            block[k] = b;
        }
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = ((uint32_t)block[4*i]<<24)|((uint32_t)block[4*i+1]<<16)|((uint32_t)block[4*i+2]<<8)|block[4*i+3];
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr32(w[i-15],7)^rotr32(w[i-15],18)^(w[i-15]>>3);
            uint32_t s1 = rotr32(w[i-2],17)^rotr32(w[i-2],19)^(w[i-2]>>10);
            w[i] = w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr32(e,6)^rotr32(e,11)^rotr32(e,25);
            uint32_t ch = (e&f)^((~e)&g);
            uint32_t t1 = hh+S1+ch+c_K[i]+w[i];
            uint32_t S0 = rotr32(a,2)^rotr32(a,13)^rotr32(a,22);
            uint32_t maj = (a&b)^(a&c)^(b&c);
            uint32_t t2 = S0+maj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    for (int i = 0; i < 8; ++i) { out[4*i]=(unsigned char)(h[i]>>24); out[4*i+1]=(unsigned char)(h[i]>>16); out[4*i+2]=(unsigned char)(h[i]>>8); out[4*i+3]=(unsigned char)h[i]; }
}

// seed = SHA256("abacus/instance" || preheader)
__global__ void seed_kernel(const unsigned char* ph, int phlen, unsigned char* seed) {
    unsigned char m[160];
    const char DOM[] = "abacus/instance";
    int dl = 15;
    for (int i = 0; i < dl; ++i) m[i] = DOM[i];
    for (int i = 0; i < phlen; ++i) m[dl + i] = ph[i];
    unsigned char h[32];
    sha256_dev(m, dl + phlen, h);
    for (int i = 0; i < 32; ++i) seed[i] = h[i];
}

// out[4*t + k] = expand elements (SHA256("abacus/expand" || seed || t))
__global__ void expand_kernel(const unsigned char* seed, int count, uint64_t* out) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t * 4 >= count) return;
    unsigned char m[49];
    const char DOM[] = "abacus/expand";
    int dl = 13;
    for (int i = 0; i < dl; ++i) m[i] = DOM[i];
    for (int i = 0; i < 32; ++i) m[dl + i] = seed[i];
    m[45] = (unsigned char)t; m[46] = (unsigned char)(t >> 8); m[47] = (unsigned char)(t >> 16); m[48] = (unsigned char)(t >> 24);
    unsigned char h[32];
    sha256_dev(m, 49, h);
    for (int k = 0; k < 4 && t * 4 + k < count; ++k) {
        uint64_t v = 0;
        for (int j = 0; j < 8; ++j) v |= (uint64_t)h[k * 8 + j] << (8 * j);
        out[t * 4 + k] = v % GOLDI;
    }
}

// Streaming SHA-256 over two device regions: prefix[0..plen) then C[0..count) as little-endian u64.
// Single-thread: adequate for the score (n^2*8 bytes) at modest n; removes the host score.
__device__ void sha256_parts(const unsigned char* prefix, int plen, const uint64_t* c, int count, unsigned char out[32]) {
    long long total = (long long)plen + (long long)count * 8;
    long long bitlen = total * 8;
    long long padded = ((total + 1 + 8 + 63) / 64) * 64;
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    unsigned char block[64];
    for (long long off = 0; off < padded; off += 64) {
        for (int k = 0; k < 64; ++k) {
            long long idx = off + k;
            unsigned char b;
            if (idx < plen) b = prefix[idx];
            else if (idx < total) {
                long long cidx = idx - plen;
                uint64_t word = c[cidx >> 3];
                b = (unsigned char)(word >> (8 * (cidx & 7)));
            } else if (idx == total) b = 0x80;
            else if (idx >= padded - 8) b = (unsigned char)((unsigned long long)bitlen >> (8 * (padded - 1 - idx)));
            else b = 0;
            block[k] = b;
        }
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = ((uint32_t)block[4*i]<<24)|((uint32_t)block[4*i+1]<<16)|((uint32_t)block[4*i+2]<<8)|block[4*i+3];
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr32(w[i-15],7)^rotr32(w[i-15],18)^(w[i-15]>>3);
            uint32_t s1 = rotr32(w[i-2],17)^rotr32(w[i-2],19)^(w[i-2]>>10);
            w[i] = w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=h[0],b=h[1],cc=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr32(e,6)^rotr32(e,11)^rotr32(e,25);
            uint32_t ch = (e&f)^((~e)&g);
            uint32_t t1 = hh+S1+ch+c_K[i]+w[i];
            uint32_t S0 = rotr32(a,2)^rotr32(a,13)^rotr32(a,22);
            uint32_t maj = (a&b)^(a&cc)^(b&cc);
            uint32_t t2 = S0+maj;
            hh=g; g=f; f=e; e=d+t1; d=cc; cc=b; b=a; a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=cc;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    for (int i = 0; i < 8; ++i) { out[4*i]=(unsigned char)(h[i]>>24); out[4*i+1]=(unsigned char)(h[i]>>16); out[4*i+2]=(unsigned char)(h[i]>>8); out[4*i+3]=(unsigned char)h[i]; }
}

// Report the leading-zero bits of the score so the host can decide acceptance (no C copy-back).
__device__ __forceinline__ uint32_t leading_zero_bits_dev(const unsigned char s[32]) {
    uint32_t lead = 0;
    for (int i = 0; i < 32; ++i) {
        if (s[i] == 0) { lead += 8; continue; }
        unsigned char b = s[i]; uint32_t z = 0; while (!(b & 0x80)) { z++; b <<= 1; }
        lead += z; break;
    }
    return lead;
}
__global__ void score_lead_kernel(const unsigned char* ph, int phlen, const uint64_t* C, int count, unsigned int* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned char m[96];
    const char DOM[] = "abacus/score"; int dl = 12;
    for (int i = 0; i < dl; ++i) m[i] = DOM[i];
    for (int i = 0; i < phlen; ++i) m[dl + i] = ph[i];
    unsigned char sc[32];
    sha256_parts(m, dl + phlen, C, count, sc);
    out[0] = leading_zero_bits_dev(sc);
}

// Sequential data-dependent dataset on the device: block u depends on u-1 and on the earlier block
// ref(u) = LE64(blk[u-1][0..8]) mod u. One thread builds it; matches the Rust chain `build_dataset`.
__global__ void dataset_chain_kernel(const unsigned char* epoch_seed, uint64_t nblocks, uint64_t* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned char prev[32];
    unsigned char m[112];
    const char DOMDS[] = "abacus/ds"; int dlds = 9;
    // block 0
    {
        int o = 0; for (int i=0;i<dlds;i++) m[o++]=DOMDS[i];
        for (int i=0;i<32;i++) m[o++]=epoch_seed[i];
        for (int i=0;i<8;i++) m[o++]=0;
        unsigned char h[32]; sha256_dev(m, o, h);
        for (int i=0;i<32;i++) prev[i]=h[i];
        for (int w=0;w<4;w++){ uint64_t v=0; for (int j=0;j<8;j++) v |= (uint64_t)h[w*8+j]<<(8*j); out[w]=v; }
    }
    for (uint64_t u=1; u<nblocks; ++u) {
        uint64_t rv=0; for (int j=0;j<8;j++) rv |= (uint64_t)prev[j]<<(8*j);
        uint64_t r = rv % u;
        { int o=0; for (int i=0;i<dlds;i++) m[o++]=DOMDS[i]; for (int i=0;i<32;i++) m[o++]=epoch_seed[i];
          for (int i=0;i<8;i++) m[o++]=(unsigned char)(u>>(8*i));
          for (int i=0;i<32;i++) m[o++]=prev[i];
          const unsigned char* refbytes = (const unsigned char*)&out[r*4];
          for (int i=0;i<32;i++) m[o++]=refbytes[i];
          unsigned char h[32]; sha256_dev(m,o,h);
          for (int i=0;i<32;i++) prev[i]=h[i];
          for (int w=0;w<4;w++){ uint64_t v=0; for (int j=0;j<8;j++) v |= (uint64_t)h[w*8+j]<<(8*j); out[u*4+w]=v; }
        }
    }
}

// Gathered instance (candidate A'): out[i] = field of dataset block idx[i]%nblocks (matches the Rust
// chain `instance_hard` / `field_from_block`: first u64 of the 32-byte block, LE, mod Goldilocks).
__global__ void gather_kernel(const uint64_t* __restrict__ D, unsigned long long nblocks,
                              const uint64_t* __restrict__ idx, uint64_t* __restrict__ out, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned long long blk = idx[i] % nblocks;
    out[i] = D[blk * 4] % GOLDI;
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
    std::vector<uint64_t> C(nn);
    uint64_t *dAB = nullptr, *dC = nullptr; uint8_t *dc = nullptr, *dPh = nullptr;
    if (cudaMalloc(&dAB, 2*nn*8)!=cudaSuccess || cudaMalloc(&dC, nn*8)!=cudaSuccess ||
        cudaMalloc(&dc, 32)!=cudaSuccess || cudaMalloc(&dPh, 160)!=cudaSuccess) {
        fprintf(stderr, "[abacus] cudaMalloc failed\n"); return 1;
    }
    cudaMemcpyToSymbol(c_K, K256, sizeof(K256));
    const int ethreads = (int)((2*nn + 3) / 4), eblocks = (ethreads + 255) / 256;
    dim3 block(TS, TS), grid(n/TS, n/TS);

    uint8_t chain_id[32]; memset(chain_id, 0xAB, 32);
    uint8_t prev0[32]; memset(prev0, 0, 32);
    std::vector<uint8_t> ph(160);
    uint64_t nonce = 0, attempts = 0, found = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (;;) {
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration<double>(now - t0).count() >= seconds) break;

        size_t phlen = 0;
        make_preheader(ph.data(), &phlen, chain_id, 2, prev0, 0, 0, (uint32_t)bits, nonce);
        cudaMemcpy(dPh, ph.data(), phlen, cudaMemcpyHostToDevice);
        seed_kernel<<<1, 1>>>(dPh, (int)phlen, dc);
        expand_kernel<<<eblocks, 256>>>(dc, (int)(2 * nn), dAB);
        matmul_kernel<<<grid, block>>>(dAB, dAB + nn, dC, n);
        cudaMemcpy(C.data(), dC, nn*8, cudaMemcpyDeviceToHost);

        std::vector<uint8_t> ms; const char DOMS[]="abacus/score"; ms.insert(ms.end(),DOMS,DOMS+strlen(DOMS));
        ms.insert(ms.end(), ph.data(), ph.data()+phlen);
        for (size_t i = 0; i < nn; ++i) { uint64_t x = C[i]; for (int j=0;j<8;j++) ms.push_back((uint8_t)(x >> (8*j))); }
        uint8_t sc[32]; sha256(ms.data(), ms.size(), sc);

        attempts++;
        if ((int)leading_zero_bits(sc) >= bits) { found++; }
        nonce++;
    }
    double secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    printf("[abacus] n=%d bits=%d device=%d attempts=%llu found=%llu attempts/s=%.1f\n",
           n, bits, device, (unsigned long long)attempts, (unsigned long long)found, attempts/secs);

    cudaFree(dAB); cudaFree(dC); cudaFree(dc); cudaFree(dPh);
    return 0;
#else
    (void)n; (void)bits; (void)seconds; (void)device;
    fprintf(stderr, "[abacus] built without CUDA\n");
    return 1;
#endif
}

// Candidate A' memory-hard mock: gather operands from a device dataset instead of expanding.
extern "C" int cp_abacus_cuda_mock_hard(int n, int bits, int seconds, int device, long long nblocks) {
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    if (n <= 0 || (n % TS) != 0) { fprintf(stderr, "[abacus] n must be a positive multiple of %d\n", TS); return 1; }
    if (cudaSetDevice(device) != cudaSuccess) { fprintf(stderr, "[abacus] cudaSetDevice(%d) failed\n", device); return 1; }
    if (nblocks < 2) { fprintf(stderr, "[abacus] --dataset needs >= 2 blocks\n"); return 1; }
    const size_t nn = (size_t)n * n;

    // Build the epoch dataset on the device (matches the Rust chain), as u64 (4 per block).
    uint8_t epoch_seed_host[32]; memset(epoch_seed_host, 7, 32);
    const size_t dbytes = (size_t)nblocks * 32;

    uint64_t *dAB = nullptr, *dC = nullptr, *dIdx = nullptr, *dD = nullptr;
    uint8_t *dc = nullptr, *dPh = nullptr, *dSeed = nullptr;
    if (cudaMalloc(&dAB, 2*nn*8)!=cudaSuccess || cudaMalloc(&dC, nn*8)!=cudaSuccess ||
        cudaMalloc(&dIdx, 2*nn*8)!=cudaSuccess || cudaMalloc(&dD, dbytes)!=cudaSuccess ||
        cudaMalloc(&dc, 32)!=cudaSuccess || cudaMalloc(&dPh, 160)!=cudaSuccess ||
        cudaMalloc(&dSeed, 32)!=cudaSuccess) {
        fprintf(stderr, "[abacus] cudaMalloc failed\n"); return 1;
    }
    cudaMemcpy(dSeed, epoch_seed_host, 32, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_K, K256, sizeof(K256));
    dataset_chain_kernel<<<1, 1>>>(dSeed, (uint64_t)nblocks, dD);
    const int ethreads = (int)((2*nn + 3) / 4), eblocks = (ethreads + 255) / 256;
    const int gblocks = ((int)(2 * nn) + 255) / 256;
    dim3 block(TS, TS), grid(n/TS, n/TS);

    std::vector<uint64_t> C(nn);
    uint8_t chain_id[32]; memset(chain_id, 0xAB, 32);
    uint8_t prev0[32]; memset(prev0, 0, 32);
    std::vector<uint8_t> ph(160);
    uint64_t nonce = 0, attempts = 0, found = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (;;) {
        if (std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() >= seconds) break;
        size_t phlen = 0;
        make_preheader(ph.data(), &phlen, chain_id, 2, prev0, 0, 0, (uint32_t)bits, nonce);
        cudaMemcpy(dPh, ph.data(), phlen, cudaMemcpyHostToDevice);
        seed_kernel<<<1, 1>>>(dPh, (int)phlen, dc);
        expand_kernel<<<eblocks, 256>>>(dc, (int)(2 * nn), dIdx);
        gather_kernel<<<gblocks, 256>>>(dD, (unsigned long long)nblocks, dIdx, dAB, (int)(2 * nn));
        matmul_kernel<<<grid, block>>>(dAB, dAB + nn, dC, n);
        cudaMemcpy(C.data(), dC, nn*8, cudaMemcpyDeviceToHost);
        std::vector<uint8_t> ms; const char DOMS[]="abacus/score"; ms.insert(ms.end(),DOMS,DOMS+strlen(DOMS));
        ms.insert(ms.end(), ph.data(), ph.data()+phlen);
        for (size_t i = 0; i < nn; ++i) { uint64_t x = C[i]; for (int j=0;j<8;j++) ms.push_back((uint8_t)(x >> (8*j))); }
        uint8_t sc[32]; sha256(ms.data(), ms.size(), sc);
        attempts++;
        if ((int)leading_zero_bits(sc) >= bits) { found++; }
        nonce++;
    }
    double secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    double reads = (double)2 * nn * 8 * attempts; // 8 bytes consumed per gathered element (field_from_block)
    printf("[abacus] hard n=%d blocks=%lld attempts=%llu found=%llu attempts/s=%.1f gather_GB/s=%.1f\n",
           n, nblocks, (unsigned long long)attempts, (unsigned long long)found, attempts/secs, reads/secs/1e9);
    cudaFree(dAB); cudaFree(dC); cudaFree(dIdx); cudaFree(dD); cudaFree(dc); cudaFree(dPh); cudaFree(dSeed);
    return 0;
#else
    (void)n;(void)bits;(void)seconds;(void)device;(void)nblocks;
    fprintf(stderr, "[abacus] built without CUDA\n"); return 1;
#endif
}

extern "C" int cp_abacus_cuda_selftest(void) {
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    uint8_t h[32];
    sha256((const uint8_t*)"abc", 3, h);
    printf("sha256(abc)=");
    for (int i = 0; i < 32; ++i) printf("%02x", h[i]);
    printf("\n");
    uint8_t h2[32];
    sha256((const uint8_t*)"", 0, h2);
    printf("sha256(empty)=");
    for (int i = 0; i < 32; ++i) printf("%02x", h2[i]);
    printf("\n");
    uint8_t seed[32];
    memset(seed, 0, 32);
    uint64_t vals[4];
    expand(seed, 32, vals, 4);
    printf("expand0=%llu %llu %llu %llu\n",
           (unsigned long long)vals[0], (unsigned long long)vals[1],
           (unsigned long long)vals[2], (unsigned long long)vals[3]);
    // device expand (same seed) for parity
    cudaMemcpyToSymbol(c_K, K256, sizeof(K256));
    uint8_t* ds = nullptr; uint64_t* dv = nullptr;
    cudaMalloc(&ds, 32); cudaMalloc(&dv, 4 * 8);
    cudaMemcpy(ds, seed, 32, cudaMemcpyHostToDevice);
    expand_kernel<<<1, 256>>>(ds, 4, dv);
    uint64_t dvh[4]; cudaMemcpy(dvh, dv, 32, cudaMemcpyDeviceToHost);
    printf("expand_dev=%llu %llu %llu %llu\n",
           (unsigned long long)dvh[0], (unsigned long long)dvh[1],
           (unsigned long long)dvh[2], (unsigned long long)dvh[3]);
    cudaFree(ds); cudaFree(dv);
    return 0;
#else
    return 1;
#endif
}

// ---------------- solo client (JOB/SUB) ----------------

#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA && !defined(_WIN32)
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <string>

namespace {
std::string to_hex(const uint8_t* p, size_t n) {
    static const char* H = "0123456789abcdef";
    std::string s; s.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) { s.push_back(H[p[i] >> 4]); s.push_back(H[p[i] & 15]); }
    return s;
}
int hexv(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
bool from_hex(const std::string& s, std::vector<uint8_t>& out) {
    if (s.size() % 2) return false;
    out.clear();
    for (size_t i = 0; i < s.size(); i += 2) {
        int hi = hexv(s[i]), lo = hexv(s[i + 1]);
        if (hi < 0 || lo < 0) return false;
        out.push_back((uint8_t)((hi << 4) | lo));
    }
    return true;
}
bool send_line(int fd, const std::string& s) {
    std::string m = s + "\n";
    size_t off = 0;
    while (off < m.size()) { ssize_t r = ::send(fd, m.data() + off, m.size() - off, 0); if (r <= 0) return false; off += (size_t)r; }
    return true;
}
bool recv_line(int fd, std::string& out) {
    out.clear(); char c;
    while (true) { ssize_t r = ::recv(fd, &c, 1, 0); if (r <= 0) return false; if (c == '\n') return true; out.push_back(c); }
}
} // namespace
#endif

int cp_abacus_cuda_solo(const char* host, int port, int n, int seconds, int device, long long nblocks) {
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA && !defined(_WIN32)
    if (n <= 0 || (n % TS) != 0) { fprintf(stderr, "[abacus] n must be a positive multiple of %d\n", TS); return 1; }
    struct addrinfo hints; memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    char portstr[16]; snprintf(portstr, sizeof(portstr), "%d", port);
    struct addrinfo* res = nullptr;
    if (getaddrinfo(host, portstr, &hints, &res) != 0) { fprintf(stderr, "[abacus] resolve %s failed\n", host); return 1; }
    int fd = -1;
    for (struct addrinfo* p = res; p; p = p->ai_next) {
        fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
        close(fd); fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) { fprintf(stderr, "[abacus] connect %s:%d failed\n", host, port); return 1; }
    if (cudaSetDevice(device) != cudaSuccess) { fprintf(stderr, "[abacus] cudaSetDevice(%d) failed\n", device); close(fd); return 1; }

    const size_t nn = (size_t)n * n;
    std::vector<uint64_t> C(nn);
    uint64_t *dAB = nullptr, *dC = nullptr, *dD = nullptr, *dIdx = nullptr; uint8_t *dc = nullptr, *dPh = nullptr;
    cudaMalloc(&dAB, 2*nn*8); cudaMalloc(&dC, nn*8); cudaMalloc(&dc, 32); cudaMalloc(&dPh, 160);
    cudaMemcpyToSymbol(c_K, K256, sizeof(K256));
    const bool hard = nblocks >= 2;
    if (hard) {
        cudaMalloc(&dIdx, 2*nn*8);
        uint8_t epoch_seed[32]; memset(epoch_seed, 7, 32);
        uint8_t* dSeed = nullptr; cudaMalloc(&dSeed, 32); cudaMemcpy(dSeed, epoch_seed, 32, cudaMemcpyHostToDevice);
        cudaMalloc(&dD, (size_t)nblocks * 32);
        dataset_chain_kernel<<<1, 1>>>(dSeed, (uint64_t)nblocks, dD);
        cudaFree(dSeed);
    }
    unsigned int dLead_unused = 0; (void)dLead_unused;
    const int ethreads = (int)((2*nn + 3) / 4), eblocks = (ethreads + 255) / 256;
    const int gblocks = ((int)(2 * nn) + 255) / 256;
    dim3 block(TS, TS), grid(n/TS, n/TS);

    std::vector<uint8_t> ph(160);
    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t found = 0, jobs = 0;

    for (;;) {
        if (std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() >= seconds) break;
        if (!send_line(fd, "JOB")) break;
        std::string line;
        if (!recv_line(fd, line) || line.rfind("JOB ", 0) != 0) break;
        std::vector<uint8_t> jb;
        if (!from_hex(line.substr(4), jb) || jb.size() != 96) break;
        uint8_t chain_id[32]; memcpy(chain_id, jb.data(), 32);
        uint32_t version; memcpy(&version, jb.data()+32, 4);
        uint8_t prev[32]; memcpy(prev, jb.data()+36, 32);
        uint64_t height; memcpy(&height, jb.data()+68, 8);
        uint64_t ts; memcpy(&ts, jb.data()+76, 8);
        uint32_t bits; memcpy(&bits, jb.data()+84, 4);
        uint64_t extranonce; memcpy(&extranonce, jb.data()+88, 8);
        jobs++;

        for (uint64_t ctr = 0;; ctr++) {
            if (std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() >= seconds) break;
            const uint64_t nonce = (extranonce << 32) | (ctr & 0xFFFFFFFFu);
            size_t phlen = 0;
            {
                static const char DOM[] = "abacus/ph"; size_t dl = strlen(DOM); size_t o = 0;
                memcpy(ph.data()+o, DOM, dl); o += dl;
                memcpy(ph.data()+o, chain_id, 32); o += 32;
                for (int i=0;i<4;i++) ph[o++] = (uint8_t)(version >> (8*i));
                memcpy(ph.data()+o, prev, 32); o += 32;
                for (int i=0;i<8;i++) ph[o++] = (uint8_t)(height >> (8*i));
                for (int i=0;i<8;i++) ph[o++] = (uint8_t)(ts >> (8*i));
                for (int i=0;i<4;i++) ph[o++] = (uint8_t)(bits >> (8*i));
                for (int i=0;i<8;i++) ph[o++] = (uint8_t)(nonce >> (8*i));
                phlen = o;
            }
            cudaMemcpy(dPh, ph.data(), phlen, cudaMemcpyHostToDevice);
            seed_kernel<<<1, 1>>>(dPh, (int)phlen, dc);
            if (hard) {
                expand_kernel<<<eblocks, 256>>>(dc, (int)(2 * nn), dIdx);
                gather_kernel<<<gblocks, 256>>>(dD, (unsigned long long)nblocks, dIdx, dAB, (int)(2 * nn));
            } else {
                expand_kernel<<<eblocks, 256>>>(dc, (int)(2 * nn), dAB);
            }
            matmul_kernel<<<grid, block>>>(dAB, dAB + nn, dC, n);
            cudaMemcpy(C.data(), dC, nn*8, cudaMemcpyDeviceToHost);
            std::vector<uint8_t> ms; const char DOMS[]="abacus/score"; ms.insert(ms.end(),DOMS,DOMS+strlen(DOMS));
            ms.insert(ms.end(), ph.data(), ph.data()+phlen);
            for (size_t i = 0; i < nn; ++i) { uint64_t x = C[i]; for (int j=0;j<8;j++) ms.push_back((uint8_t)(x >> (8*j))); }
            uint8_t sc[32]; sha256(ms.data(), ms.size(), sc);
            if ((int)leading_zero_bits(sc) >= bits) {
                // SUB = nonce(u64) || ts(u64) || clen(u32) || C
                std::vector<uint8_t> sb; sb.resize(8+8+4);
                for (int i=0;i<8;i++) sb[i] = (uint8_t)(nonce >> (8*i));
                for (int i=0;i<8;i++) sb[8+i] = (uint8_t)(ts >> (8*i));
                uint32_t cl = (uint32_t)nn; memcpy(&sb[16], &cl, 4);
                for (size_t i = 0; i < nn; ++i) { uint64_t x = C[i]; for (int j=0;j<8;j++) sb.push_back((uint8_t)(x >> (8*j))); }
                std::string reply;
                if (send_line(fd, "SUB " + to_hex(sb.data(), sb.size())) && recv_line(fd, reply) && reply.rfind("OK",0)==0) {
                    found++;
                }
                break;
            }
        }
    }
    double secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    printf("[abacus] solo n=%d node=%s:%d jobs=%llu found=%llu time=%.1fs\n", n, host, port,
           (unsigned long long)jobs, (unsigned long long)found, secs);
    close(fd);
    cudaFree(dAB); cudaFree(dC); cudaFree(dc); cudaFree(dPh);
    if (hard) { cudaFree(dD); cudaFree(dIdx); }
    return 0;
#else
    (void)host; (void)port; (void)n; (void)seconds; (void)device; (void)nblocks;
    fprintf(stderr, "[abacus] solo not supported in this build\n");
    return 1;
#endif
}
