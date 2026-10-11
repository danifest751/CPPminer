// TNet v1 (the Requant coin's work function): what every device backend shares. The work and its tickets, the host side
// of SHA-256 and of the kernels' arguments, CPU references for self-tests, and the interface a backend implements
// (CUDA in cp_tnet.cu, OpenCL in cp_tnet_ocl.cpp). The network and the mining loop are in cp_tnet_net.cpp.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------- SHA-256 on the host

static const uint32_t rqh_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2};

inline uint32_t rqh_rotr(uint32_t x, int s) { return (x >> s) | (x << (32 - s)); }
inline uint32_t rqh_be(const uint8_t* b) {
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | b[3];
}
inline void rqh_compress(uint32_t h[8], const uint32_t w0[16]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) w[i] = w0[i];
    for (int i = 16; i < 64; ++i)
        w[i] = w[i - 16] + (rqh_rotr(w[i - 15], 7) ^ rqh_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)) + w[i - 7] +
               (rqh_rotr(w[i - 2], 17) ^ rqh_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10));
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t t1 = hh + (rqh_rotr(e, 6) ^ rqh_rotr(e, 11) ^ rqh_rotr(e, 25)) + ((e & f) ^ (~e & g)) + rqh_K[i] + w[i];
        const uint32_t t2 = (rqh_rotr(a, 2) ^ rqh_rotr(a, 13) ^ rqh_rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

inline void host_sha256(const std::vector<uint8_t>& m, uint8_t out[32]) {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint32_t w[16];
    std::vector<uint8_t> d = m;
    const uint64_t bits = (uint64_t)m.size() * 8;
    d.push_back(0x80);
    while (d.size() % 64 != 56) d.push_back(0);
    for (int i = 7; i >= 0; --i) d.push_back((uint8_t)(bits >> (8 * i)));
    for (size_t off = 0; off < d.size(); off += 64) {
        for (int q = 0; q < 16; ++q) w[q] = rqh_be(d.data() + off + 4 * q);
        rqh_compress(h, w);
    }
    for (int i = 0; i < 8; ++i) {
        out[4 * i] = (uint8_t)(h[i] >> 24); out[4 * i + 1] = (uint8_t)(h[i] >> 16);
        out[4 * i + 2] = (uint8_t)(h[i] >> 8); out[4 * i + 3] = (uint8_t)h[i];
    }
}

inline std::string to_hex(const uint8_t* b, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; ++i) { s += d[b[i] >> 4]; s += d[b[i] & 15]; }
    return s;
}

// ---------------------------------------------------------------- the kernels' arguments (same layout on every device)

// expansion: out[32 k ..) = SHA256("abacus/expand" || seed || LE32(c0 + k)): the first 44 bytes as big-endian words
struct RqPrefix { uint32_t pw[11]; uint32_t b44; };
inline RqPrefix rq_prefix(const uint8_t seed[32]) {
    uint8_t pre[44];
    memcpy(pre, "abacus/expand", 13);
    memcpy(pre + 13, seed, 31);
    RqPrefix px;
    for (int q = 0; q < 11; ++q) px.pw[q] = rqh_be(pre + 4 * q);
    px.b44 = seed[31];
    return px;
}

struct RqTarget { uint32_t t[8]; };  // big-endian words
inline RqTarget rq_target(const std::vector<uint8_t>& t) {
    RqTarget tg;
    for (int q = 0; q < 8; ++q) tg.t[q] = rqh_be(t.data() + 4 * q);
    return tg;
}

// The last block of a ticket hash, 0x54 || hd || LE64 nonce || LE32 row || LE32 c || padding || length, as big-endian
// words; the same for every ticket of a scan but bytes 41..48 (row and c, in words 10..12), which each thread fills in.
struct RqTail { uint32_t wd[16]; };
inline RqTail rq_tail(const uint8_t hd[32], uint64_t nonce, int w) {
    uint8_t m[64] = {0};
    m[0] = 0x54;
    memcpy(m + 1, hd, 32);
    for (int q = 0; q < 8; ++q) m[33 + q] = (uint8_t)(nonce >> (8 * q));
    m[49] = 0x80;
    const uint64_t len = ((uint64_t)w + 49) * 8;
    for (int q = 0; q < 8; ++q) m[56 + q] = (uint8_t)(len >> (8 * (7 - q)));
    RqTail t;
    for (int q = 0; q < 16; ++q) t.wd[q] = rqh_be(m + 4 * q);
    return t;
}

// seed of the input rows of a nonce, SHA256("abacus/tnet-x0" || header digest || LE64 nonce), and of layer l's weights,
// SHA256("abacus/tnet-w" || epoch seed || LE32 l)
inline void rq_x0_seed(const std::vector<uint8_t>& digest, uint64_t nonce, uint8_t out[32]) {
    std::vector<uint8_t> m((const uint8_t*)"abacus/tnet-x0", (const uint8_t*)"abacus/tnet-x0" + 14);
    m.insert(m.end(), digest.begin(), digest.end());
    for (int q = 0; q < 8; ++q) m.push_back((uint8_t)(nonce >> (8 * q)));
    host_sha256(m, out);
}
inline void rq_w_seed(const std::vector<uint8_t>& epoch_seed, int l, uint8_t out[32]) {
    std::vector<uint8_t> m((const uint8_t*)"abacus/tnet-w", (const uint8_t*)"abacus/tnet-w" + 13);
    m.insert(m.end(), epoch_seed.begin(), epoch_seed.end());
    for (int q = 0; q < 4; ++q) m.push_back((uint8_t)(l >> (8 * q)));
    host_sha256(m, out);
}

// Winning tickets kept per scan (a pool's easy share target can give several in one batch).
#define RQ_MAX_HITS 16

// ---------------------------------------------------------------- CPU references for self-tests

// expansion bytes [32 c0, 32 (c0 + count))
inline std::vector<uint8_t> rq_cpu_expand(const uint8_t seed[32], uint32_t c0, uint32_t count) {
    std::vector<uint8_t> out(32 * (size_t)count);
    for (uint32_t k = 0; k < count; ++k) {
        std::vector<uint8_t> m((const uint8_t*)"abacus/expand", (const uint8_t*)"abacus/expand" + 13);
        m.insert(m.end(), seed, seed + 32);
        for (int q = 0; q < 4; ++q) m.push_back((uint8_t)((c0 + k) >> (8 * q)));
        host_sha256(m, out.data() + 32 * (size_t)k);
    }
    return out;
}

// XO = clamp((X WT^T M + 2^23) >> 24, -128, 127), X rows x n and WT n x n, both row-major
inline std::vector<int8_t> rq_cpu_gemm_rq(const std::vector<int8_t>& x, const std::vector<int8_t>& wt, int rows, int n, int mult) {
    std::vector<int8_t> xo((size_t)rows * n);
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < n; ++j) {
            int32_t s = 0;
            for (int k = 0; k < n; ++k) s += (int32_t)x[(size_t)r * n + k] * wt[(size_t)j * n + k];
            const long long t = ((long long)s * mult + (1LL << 23)) >> 24;
            xo[(size_t)r * n + j] = (int8_t)(t < -128 ? -128 : (t > 127 ? 127 : t));
        }
    return xo;
}

// SHA256(piece || 0x54 || hd || LE64 nonce || LE32 row || LE32 c)
inline void rq_cpu_ticket(const int8_t* piece, int w, const uint8_t hd[32], uint64_t nonce, uint32_t row, uint32_t c,
                          uint8_t out[32]) {
    std::vector<uint8_t> m((const uint8_t*)piece, (const uint8_t*)piece + w);
    m.push_back(0x54);
    m.insert(m.end(), hd, hd + 32);
    for (int q = 0; q < 8; ++q) m.push_back((uint8_t)(nonce >> (8 * q)));
    for (int q = 0; q < 4; ++q) m.push_back((uint8_t)(row >> (8 * q)));
    for (int q = 0; q < 4; ++q) m.push_back((uint8_t)(c >> (8 * q)));
    host_sha256(m, out);
}

// ---------------------------------------------------------------- work, tickets and devices

struct RqWork {
    long long height = 0;
    std::vector<uint8_t> header, digest, seed, target;
    int n = 0, b = 0, L = 0, w = 0, mult = 0;
};

struct RqHit {
    uint64_t nonce = 0;
    uint32_t row = 0, c = 0;
    std::vector<uint8_t> piece;
};

// One GPU running TNet attempts. A backend sets `fused` in init when it is -1 (requantization in the GEMM, or a separate
// pass) and may refuse a mode it does not have.
struct RqDevice {
    int fused = -1;
    // per-phase time in ms, accumulated while `profile` is set
    bool profile = false;
    double ms_expand = 0, ms_layers = 0, ms_hash = 0;
    int scans = 0;

    virtual ~RqDevice() {}
    virtual bool init(const RqWork& wk, int rows) = 0;
    // the epoch's weights (does nothing when they are loaded already)
    virtual void set_epoch(const std::vector<uint8_t>& seed) = 0;
    // rows [r0, r0 + rows) of `nonce`; appends the winning tickets (at most RQ_MAX_HITS) to `hits`
    virtual void scan(const RqWork& wk, uint64_t nonce, uint32_t r0, int rows, std::vector<RqHit>& hits) = 0;
    virtual std::string name() const = 0;
    virtual std::string kernel() const = 0;
    // a device error since the last call
    virtual bool failed() = 0;
};

using RqMake = std::function<std::unique_ptr<RqDevice>()>;

// mining against a node or pool (cp_tnet_net.cpp); `make` builds the device
int rq_solo(const RqMake& make, const char* rpc, const char* payee_hex, const char* worker, int batch, double seconds,
            long long blocks, int device_index);
// offline: ns per ticket at the TNet v1 parameters, requantization fused and separate when the device has both
int rq_bench(const RqMake& make, int batch, double seconds);
