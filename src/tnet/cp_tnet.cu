// TNet v1 miner for the Requant coin (github.com/danifest751/requant, SPEC.md) on CUDA tensor cores: CUTLASS int8 GEMM
// compiled into the miner (cp_tnet_gemm.cu; cuBLAS only as an optional cross-check), mining against a requantd node's or
// pool's JSON-RPC (getwork / submitwork).
//
// Per nonce the input X_0 has B rows (B = 65536 on the test network); rows are independent, so they are
// processed in batches of --batch rows: expand the batch's rows of X_0, run L layers of int8 GEMM + integer
// requantization, hash every w-byte piece of every output row, compare with the 256-bit target. Kernels are
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
#include <deque>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include "cp_tnet_gemm.h"
#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
#include <cublas_v2.h>
#endif

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET sock_t;
#define CLOSESOCK closesocket
#else
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int sock_t;
#define INVALID_SOCKET (-1)
#define CLOSESOCK close
#endif

// ---------------------------------------------------------------- SHA-256 (host and device)

static const uint32_t rq_hK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2};
__constant__ uint32_t rq_dK[64];

__host__ __device__ __forceinline__ uint32_t rq_rotr(uint32_t x, int s) { return (x >> s) | (x << (32 - s)); }

__host__ __device__ __forceinline__ void rq_compress(uint32_t h[8], const uint32_t w0[16]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) w[i] = w0[i];
    for (int i = 16; i < 64; ++i)
        w[i] = w[i - 16] + (rq_rotr(w[i - 15], 7) ^ rq_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)) + w[i - 7] +
               (rq_rotr(w[i - 2], 17) ^ rq_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10));
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
#ifdef __CUDA_ARCH__
        const uint32_t k = rq_dK[i];
#else
        const uint32_t k = rq_hK[i];
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

static void host_sha256(const std::vector<uint8_t>& m, uint8_t out[32]) {
    uint32_t h[8], w[16];
    rq_init(h);
    std::vector<uint8_t> d = m;
    const uint64_t bits = (uint64_t)m.size() * 8;
    d.push_back(0x80);
    while (d.size() % 64 != 56) d.push_back(0);
    for (int i = 7; i >= 0; --i) d.push_back((uint8_t)(bits >> (8 * i)));
    for (size_t off = 0; off < d.size(); off += 64) {
        for (int q = 0; q < 16; ++q) w[q] = rq_be(d.data() + off + 4 * q);
        rq_compress(h, w);
    }
    for (int i = 0; i < 8; ++i) { out[4 * i] = h[i] >> 24; out[4 * i + 1] = h[i] >> 16; out[4 * i + 2] = h[i] >> 8; out[4 * i + 3] = h[i]; }
}

// ---------------------------------------------------------------- kernels

// out[32 k ..) = SHA256("abacus/expand" || seed || LE32(c0 + k)), k < count
struct RqPrefix { uint32_t pw[11]; uint32_t b44; };
static RqPrefix rq_prefix(const uint8_t seed[32]) {
    uint8_t pre[44];
    memcpy(pre, "abacus/expand", 13);
    memcpy(pre + 13, seed, 31);
    RqPrefix px;
    for (int q = 0; q < 11; ++q) px.pw[q] = rq_be(pre + 4 * q);
    px.b44 = seed[31];
    return px;
}
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
    for (int i = 0; i < 8; ++i) {
        uint8_t* o = out + 32 * k + 4 * i;
        o[0] = h[i] >> 24; o[1] = h[i] >> 16; o[2] = h[i] >> 8; o[3] = h[i];
    }
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

struct RqTarget { uint32_t t[8]; };  // big-endian words

// Winning tickets kept per scan (a pool's easy share target can give several in one batch).
#define RQ_MAX_HITS 16

// Ticket (r0 + i, c): SHA256(piece || 0x54 || hd || LE64 nonce || LE32 row || LE32 c) <= target; w % 64 == 0.
__global__ void rq_tickets(const int8_t* __restrict__ X, int n, int rows, int w, uint32_t r0, uint64_t nonce,
                           const uint8_t* __restrict__ hd, RqTarget tg, unsigned int* __restrict__ found,
                           unsigned long long* __restrict__ hits) {
    const int per_row = n / w;
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)rows * per_row) return;
    const int i = (int)(t / per_row), c = (int)(t % per_row);
    const uint32_t row = r0 + (uint32_t)i;
    const uint32_t* piece = reinterpret_cast<const uint32_t*>(X + (size_t)i * n + (size_t)c * w);
    uint32_t h[8], wd[16];
    rq_init(h);
    for (int blk = 0; blk < w / 64; ++blk) {
        for (int q = 0; q < 16; ++q) wd[q] = __byte_perm(piece[blk * 16 + q], 0, 0x0123);
        rq_compress(h, wd);
    }
    uint8_t m[64];
    m[0] = 0x54;
    for (int q = 0; q < 32; ++q) m[1 + q] = hd[q];
    for (int q = 0; q < 8; ++q) m[33 + q] = (uint8_t)(nonce >> (8 * q));
    for (int q = 0; q < 4; ++q) m[41 + q] = (uint8_t)(row >> (8 * q));
    for (int q = 0; q < 4; ++q) m[45 + q] = (uint8_t)((uint32_t)c >> (8 * q));
    m[49] = 0x80;
    for (int q = 50; q < 56; ++q) m[q] = 0;
    const uint64_t len = ((uint64_t)w + 49) * 8;
    for (int q = 0; q < 8; ++q) m[56 + q] = (uint8_t)(len >> (8 * (7 - q)));
    for (int q = 0; q < 16; ++q) wd[q] = rq_be(m + 4 * q);
    rq_compress(h, wd);
    for (int q = 0; q < 8; ++q) {
        if (h[q] < tg.t[q]) break;
        if (h[q] > tg.t[q]) return;
    }
    const unsigned k = atomicAdd(found, 1u);
    if (k < RQ_MAX_HITS) hits[k] = ((unsigned long long)i << 32) | (unsigned)c;
}

// ---------------------------------------------------------------- JSON-RPC client

static std::string to_hex(const uint8_t* b, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; ++i) { s += d[b[i] >> 4]; s += d[b[i] & 15]; }
    return s;
}
static bool from_hex(const std::string& s, std::vector<uint8_t>& out) {
    if (s.size() % 2) return false;
    out.clear();
    for (size_t i = 0; i < s.size(); i += 2) {
        unsigned v;
        if (sscanf(s.c_str() + i, "%2x", &v) != 1) return false;
        out.push_back((uint8_t)v);
    }
    return true;
}

static bool http_post(const std::string& hostport, const std::string& body, std::string& reply) {
#ifdef _WIN32
    static bool wsa = false;
    if (!wsa) { WSADATA d; WSAStartup(MAKEWORD(2, 2), &d); wsa = true; }
#endif
    const size_t colon = hostport.rfind(':');
    if (colon == std::string::npos) return false;
    const std::string host = hostport.substr(0, colon), port = hostport.substr(colon + 1);
    addrinfo hints{}, *res = nullptr;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || !res) return false;
    sock_t s = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (s == INVALID_SOCKET) { freeaddrinfo(res); return false; }
    if (connect(s, res->ai_addr, (int)res->ai_addrlen) != 0) { CLOSESOCK(s); freeaddrinfo(res); return false; }
    freeaddrinfo(res);
    const std::string req = "POST / HTTP/1.1\r\nHost: " + host + "\r\nContent-Type: application/json\r\nContent-Length: " +
                            std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
    size_t sent = 0;
    while (sent < req.size()) {
        const int k = (int)send(s, req.data() + sent, (int)(req.size() - sent), 0);
        if (k <= 0) { CLOSESOCK(s); return false; }
        sent += (size_t)k;
    }
    std::string all;
    char buf[4096];
    for (;;) {
        const int k = (int)recv(s, buf, sizeof buf, 0);
        if (k <= 0) break;
        all.append(buf, (size_t)k);
    }
    CLOSESOCK(s);
    const size_t body_at = all.find("\r\n\r\n");
    if (body_at == std::string::npos) return false;
    reply = all.substr(body_at + 4);
    return true;
}

// Value of "key" in compact JSON: a string (without quotes) or a bare token (number, true, false, null).
static bool json_get(const std::string& s, const char* key, std::string& out) {
    const std::string pat = std::string("\"") + key + "\":";
    const size_t p = s.find(pat);
    if (p == std::string::npos) return false;
    size_t v = p + pat.size();
    if (v < s.size() && s[v] == '"') {
        const size_t e = s.find('"', v + 1);
        if (e == std::string::npos) return false;
        out = s.substr(v + 1, e - v - 1);
        return true;
    }
    size_t e = v;
    while (e < s.size() && s[e] != ',' && s[e] != '}' && s[e] != ']') ++e;
    out = s.substr(v, e - v);
    return true;
}

struct RqWork {
    long long height = 0;
    std::vector<uint8_t> header, digest, seed, target;
    int n = 0, b = 0, L = 0, w = 0, mult = 0;
};

static bool rpc_call(const std::string& ep, const std::string& method, const std::string& params, std::string& reply) {
    if (!http_post(ep, "{\"method\":\"" + method + "\",\"params\":" + params + "}", reply)) return false;
    std::string err;
    if (json_get(reply, "error", err) && err != "null") {
        fprintf(stderr, "[tnet] rpc %s: %s\n", method.c_str(), err.c_str());
        return false;
    }
    return true;
}

static bool get_work(const std::string& ep, const std::string& payee, RqWork& w) {
    std::string r, v;
    if (!rpc_call(ep, "getwork", "[\"" + payee + "\"]", r)) return false;
    auto num = [&](const char* k, int& out) { return json_get(r, k, v) && (out = atoi(v.c_str()), true); };
    if (!json_get(r, "height", v)) return false;
    w.height = atoll(v.c_str());
    if (!json_get(r, "header", v) || !from_hex(v, w.header) || w.header.size() != 116) return false;
    if (!json_get(r, "header_digest", v) || !from_hex(v, w.digest) || w.digest.size() != 32) return false;
    if (!json_get(r, "epoch_seed", v) || !from_hex(v, w.seed) || w.seed.size() != 32) return false;
    if (!json_get(r, "target", v) || !from_hex(v, w.target) || w.target.size() != 32) return false;
    return num("n", w.n) && num("b", w.b) && num("L", w.L) && num("w", w.w) && num("mult", w.mult);
}

static bool get_tip(const std::string& ep, std::string& tip) {
    std::string r;
    return rpc_call(ep, "getinfo", "[]", r) && json_get(r, "tip", tip);
}

struct RqHit {
    uint64_t nonce = 0;
    uint32_t row = 0, c = 0;
    std::vector<uint8_t> piece;
};

// Talks to the node or pool on its own thread (tip polling, submissions), so a slow link never stalls the GPU.
struct RqNet {
    struct Item { std::string params, tip; };
    std::string ep;
    bool pool_mode = false;
    std::mutex mu;
    std::deque<Item> queue;
    std::string tip_seen;              // tip of the work being mined (set by the mining loop)
    std::atomic<bool> restart{false};  // new tip, found block or rejection: fetch new work
    std::atomic<bool> stop{false};
    std::atomic<long long> shares{0}, rejected{0}, blocks{0}, stale{0};
    std::thread th;

    void start() { th = std::thread([this] { run(); }); }
    void submit(const RqWork& wk, const RqHit& h, const std::string& payee, const std::string& worker, const std::string& tip) {
        // the payee (6th parameter) identifies the miner to a pool and the worker (7th) the device; a node ignores both
        Item it;
        it.params = "[\"" + to_hex(wk.digest.data(), 32) + "\"," + std::to_string(h.nonce) + "," + std::to_string(h.row) + "," +
                    std::to_string(h.c) + ",\"" + to_hex(h.piece.data(), h.piece.size()) + "\",\"" + payee + "\",\"" + worker + "\"]";
        it.tip = tip;
        std::lock_guard<std::mutex> g(mu);
        queue.push_back(std::move(it));
    }
    void set_tip(const std::string& tip) { std::lock_guard<std::mutex> g(mu); tip_seen = tip; }
    // Waits (a few seconds at most) for queued submissions, then ends the thread.
    void finish() {
        for (int k = 0; k < 100; ++k) {
            { std::lock_guard<std::mutex> g(mu); if (queue.empty()) break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        stop = true;
        if (th.joinable()) th.join();
    }
    void run() {
        auto last_poll = std::chrono::steady_clock::now() - std::chrono::seconds(10);
        while (!stop) {
            Item it;
            bool have = false;
            std::string tip_now;
            {
                std::lock_guard<std::mutex> g(mu);
                if (!queue.empty()) { it = std::move(queue.front()); queue.pop_front(); have = true; }
                tip_now = tip_seen;
            }
            if (have) {
                // found on a tip the miner has since left: certainly stale, not worth a round trip
                if (it.tip != tip_now) { ++stale; continue; }
                std::string r, acc, blk;
                const bool ok = rpc_call(ep, "submitwork", it.params, r) && json_get(r, "accepted", acc) && acc == "true";
                // a pool answers "block": false for a share that is not a block; a node has no such field
                const bool is_block = !(json_get(r, "block", blk) && blk == "false");
                if (ok && is_block) {
                    std::string h;
                    ++blocks;
                    printf("[tnet] block accepted%s\n", json_get(r, "height", h) ? (" at height " + h).c_str() : "");
                    restart = true;
                } else if (ok) {
                    ++shares;
                } else {
                    ++rejected;
                    printf("[tnet] %s rejected: %s\n", pool_mode ? "share" : "block", r.c_str());
                    restart = true;
                }
                continue;
            }
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_poll).count() >= 1.0) {
                last_poll = now;
                std::string tip;
                if (get_tip(ep, tip) && !tip_now.empty() && tip != tip_now) restart = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
};

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

struct RqEngine {
    int n = 0, L = 0, w = 0, mult = 0, batch = 0;
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

    bool init(const RqWork& wk, int rows, int device) {
        n = wk.n; L = wk.L; w = wk.w; mult = wk.mult; batch = rows;
        if (n % 64 || w % 64 || n % w) { fprintf(stderr, "[tnet] unsupported parameters n=%d w=%d\n", n, w); return false; }
        const size_t nn = (size_t)n * n, bn = (size_t)batch * n;
        gemm = pick_gemm(device);
        if (gemm == CP_TNET_GEMM_NONE) {
            fprintf(stderr, "[tnet] no working int8 GEMM: needs an NVIDIA GPU of compute 7.5 (Turing) or newer\n");
            return false;
        }
#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
        const char* use_cublas = getenv("CP_TNET_CUBLAS");
        if (use_cublas && !strcmp(use_cublas, "1")) cublasCreate(&cb);
#endif
        WT.assign(L, nullptr);
        for (auto& p : WT) if (cudaMalloc(&p, nn) != cudaSuccess) return false;
        return cudaMalloc(&tmp, (nn + 31) / 32 * 32) == cudaSuccess && cudaMalloc(&X, (bn + 31) / 32 * 32 + 32) == cudaSuccess &&
               cudaMalloc(&X2, bn) == cudaSuccess && cudaMalloc(&Y, bn * 4) == cudaSuccess && cudaMalloc(&dhd, 32) == cudaSuccess &&
               cudaMalloc(&dfound, 4) == cudaSuccess && cudaMalloc(&dhits, 8 * RQ_MAX_HITS) == cudaSuccess;
    }

    // W_l = expand(SHA256("abacus/tnet-w" || seed || LE32(l)), n^2), stored transposed.
    void set_epoch(const std::vector<uint8_t>& s) {
        if (s == seed) return;
        const auto t0 = std::chrono::steady_clock::now();
        const size_t nn = (size_t)n * n, hashes = (nn + 31) / 32;
        for (int l = 0; l < L; ++l) {
            std::vector<uint8_t> m((const uint8_t*)"abacus/tnet-w", (const uint8_t*)"abacus/tnet-w" + 13);
            m.insert(m.end(), s.begin(), s.end());
            for (int q = 0; q < 4; ++q) m.push_back((uint8_t)(l >> (8 * q)));
            uint8_t sl[32];
            host_sha256(m, sl);
            rq_expand<<<(unsigned)((hashes + 255) / 256), 256>>>(rq_prefix(sl), 0, hashes, tmp);
            rq_transpose<<<dim3(n / 32, n / 32), dim3(32, 8)>>>((const int8_t*)tmp, WT[l], n);
        }
        cudaDeviceSynchronize();
        seed = s;
        printf("[tnet] epoch weights ready (%.2f s)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    // Rows [r0, r0 + rows) of nonce; appends the winning tickets (at most RQ_MAX_HITS) to `hits`.
    void scan(const RqWork& wk, uint64_t nonce, uint32_t r0, int rows, std::vector<RqHit>& hits) {
        std::vector<uint8_t> m((const uint8_t*)"abacus/tnet-x0", (const uint8_t*)"abacus/tnet-x0" + 14);
        m.insert(m.end(), wk.digest.begin(), wk.digest.end());
        for (int q = 0; q < 8; ++q) m.push_back((uint8_t)(nonce >> (8 * q)));
        uint8_t xs[32];
        host_sha256(m, xs);
        const size_t bn = (size_t)rows * n, hashes = (bn + 31) / 32;
        const uint32_t c0 = (uint32_t)((size_t)r0 * n / 32);
        cudaMemcpy(dhd, wk.digest.data(), 32, cudaMemcpyHostToDevice);
        cudaMemset(dfound, 0, 4);
        rq_expand<<<(unsigned)((hashes + 255) / 256), 256>>>(rq_prefix(xs), c0, hashes, (uint8_t*)X);
        int8_t *cur = X, *nxt = X2;
        for (int l = 0; l < L; ++l) {
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
        rq_tickets<<<(unsigned)((tickets + 127) / 128), 128>>>(cur, n, rows, w, r0, nonce, dhd, tg, dfound, dhits);
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
    cudaMemcpyToSymbol(rq_dK, rq_hK, sizeof(rq_hK));
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
    const int kind = pick_gemm(device);
    const bool mm = kind != CP_TNET_GEMM_NONE;
    printf("[tnet] selftest: sha256 %s, expansion %s, int8 GEMM %s (%s), cuda %s\n", abc ? "ok" : "FAIL", exp ? "ok" : "FAIL",
           mm ? "ok" : "FAIL", cp_tnet_gemm_name(kind), cudaGetErrorString(cudaGetLastError()));
    return abc && exp && mm ? 0 : 1;
}

extern "C" int cp_tnet_cuda_solo(const char* rpc, const char* payee_hex, const char* worker, int device, int batch, double seconds,
                                 long long blocks) {
    if (cudaSetDevice(device) != cudaSuccess) { fprintf(stderr, "[tnet] no CUDA device %d\n", device); return 1; }
    cudaMemcpyToSymbol(rq_dK, rq_hK, sizeof(rq_hK));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    const std::string ep = rpc, payee = payee_hex;
    // device name for a pool's statistics: letters, digits, '.', '_', '-' (the pool applies the same filter)
    std::string worker_name;
    for (const char* q = worker ? worker : ""; *q && worker_name.size() < 32; ++q)
        if (isalnum((unsigned char)*q) || *q == '.' || *q == '_' || *q == '-') worker_name += *q;
    if (worker_name.empty()) worker_name = "default";
    RqWork wk;
    if (!get_work(ep, payee, wk)) { fprintf(stderr, "[tnet] getwork from %s failed\n", rpc); return 1; }
    const int rows = batch < wk.b ? batch : wk.b;
    printf("[tnet] %s | node %s | TNet n=%d B=%d L=%d w=%d | batch %d rows\n", prop.name, rpc, wk.n, wk.b, wk.L, wk.w, rows);
    RqEngine eng;
    if (!eng.init(wk, rows, device)) { fprintf(stderr, "[tnet] GPU set-up failed (try a smaller --batch)\n"); return 1; }
    printf("[tnet] GEMM: %s\n", cp_tnet_gemm_name(eng.gemm));
    std::mt19937_64 rng(std::random_device{}() ^ ((uint64_t)device << 48) ^ (uint64_t)time(nullptr));
    const auto start = std::chrono::steady_clock::now();
    auto last_stat = start, last_work = start;
    unsigned long long tickets = 0, tickets_stat = 0;
    std::string pool_flag;
    RqNet net;
    net.ep = ep;
    net.pool_mode = rpc_call(ep, "getwork", "[\"" + payee + "\"]", pool_flag) && pool_flag.find("\"pool\":true") != std::string::npos;
    if (net.pool_mode) printf("[tnet] pool mode: shares at 2^%s tickets\n", json_get(pool_flag, "share_bits", pool_flag) ? pool_flag.c_str() : "?");
    std::string tip_seen = to_hex(wk.header.data() + 12, 32);
    net.set_tip(tip_seen);
    net.start();
    std::vector<RqHit> hits;
    bool done = false;
    while (!done) {
        eng.set_epoch(wk.seed);
        const uint64_t nonce = rng();
        bool restart = false;
        for (uint32_t r0 = 0; r0 < (uint32_t)wk.b && !restart; r0 += (uint32_t)rows) {
            const int nrows = (int)((uint32_t)wk.b - r0 < (uint32_t)rows ? (uint32_t)wk.b - r0 : (uint32_t)rows);
            hits.clear();
            eng.scan(wk, nonce, r0, nrows, hits);
            tickets += (unsigned long long)nrows * (wk.n / wk.w);
            for (const RqHit& h : hits) net.submit(wk, h, payee, worker_name, tip_seen);
            const auto now = std::chrono::steady_clock::now();
            if (net.restart.exchange(false)) restart = true;
            if (std::chrono::duration<double>(now - last_work).count() > 60.0) restart = true;  // fresh time and transactions
            if (std::chrono::duration<double>(now - last_stat).count() > 10.0) {
                const double dt = std::chrono::duration<double>(now - last_stat).count();
                if (net.pool_mode) {
                    printf("[tnet] height %lld | %.2f M tickets/s | %.1f ns/ticket | shares %lld | rejected %lld | stale %lld | blocks %lld\n",
                           wk.height, (tickets - tickets_stat) / dt / 1e6, dt * 1e9 / (double)(tickets - tickets_stat),
                           net.shares.load(), net.rejected.load(), net.stale.load(), net.blocks.load());
                } else {
                    printf("[tnet] height %lld | %.2f M tickets/s | %.1f ns/ticket | blocks %lld\n", wk.height,
                           (tickets - tickets_stat) / dt / 1e6, dt * 1e9 / (double)(tickets - tickets_stat), net.blocks.load());
                }
                last_stat = now;
                tickets_stat = tickets;
            }
            if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "[tnet] CUDA error\n"); net.finish(); return 1; }
            const double elapsed = std::chrono::duration<double>(now - start).count();
            if ((seconds > 0 && elapsed >= seconds) || (blocks > 0 && net.blocks >= blocks)) { done = true; break; }
        }
        if (done || !restart) continue;
        if (!get_work(ep, payee, wk)) {
            fprintf(stderr, "[tnet] getwork failed; retrying in 2 s\n");
            std::this_thread::sleep_for(std::chrono::seconds(2));
            net.restart = true;
            continue;
        }
        tip_seen = to_hex(wk.header.data() + 12, 32);
        net.set_tip(tip_seen);
        last_work = std::chrono::steady_clock::now();
    }
    net.finish();
    printf("[tnet] done: %lld blocks, %.0f s\n", net.blocks.load(),
           std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    return 0;
}

#endif  // CP_ENABLE_CUBLAS
