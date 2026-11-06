// TNet v1 on the CPU: phones and other machines without a usable GPU. The weights are packed once per epoch, the input
// of each layer once per layer. A GEMM runs in blocks of columns, which the threads take one at a time (this suits a
// phone's fast and slow cores), and in blocks of k inside them, with the accumulators in memory between. Requantization
// and the ticket hash are the plain consensus steps (SPEC.md of Requant); `rq_cpu_selftest` and the vectors in
// tests/cp_tnet_cpu_test.cpp hold them to the references.
#include "cp_tnet_cpu.hpp"

#include "../cpu/gemm/case33_cpu_features.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <random>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__ANDROID__)
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

void rq_cpu_micro_portable(const int8_t* a, const int8_t* b, int kc, int32_t* c) {
    const int groups = kc / 4;
    for (int col = 0; col < 16; ++col) {
        const int8_t* bc = b + static_cast<size_t>(col / 8) * groups * 32 + (col % 8) * 4;
        for (int row = 0; row < 8; ++row) {
            const int8_t* ar = a + row * 4;
            int32_t s = 0;
            for (int g = 0; g < groups; ++g)
                for (int t = 0; t < 4; ++t) s += static_cast<int32_t>(ar[g * 32 + t]) * bc[g * 32 + t];
            c[col * 8 + row] += s;
        }
    }
}

namespace {

constexpr int kMR = 8, kNR = 16;

// Android switches cores off and on as it pleases, and a thread's affinity stays narrowed to the cores that were on when
// it started: on a Snapdragon 860 the miner ended up on three slow cores and a middle one, the fastest never coming back.
// Each thread asks for all the cores again before each pass. Elsewhere the mask is the user's (taskset) and stays.
void rq_reclaim_cores() {
#if defined(__ANDROID__)
#pragma omp parallel
    {
        cpu_set_t all;
        CPU_ZERO(&all);
        const long cores = sysconf(_SC_NPROCESSORS_CONF);
        for (long i = 0; i < cores && i < CPU_SETSIZE; ++i) CPU_SET(i, &all);
        sched_setaffinity(0, sizeof all, &all);
    }
#endif
}

// A buffer on a cache line boundary.
struct RqBuf {
    std::vector<uint8_t> v;
    uint8_t* p = nullptr;
    void resize(size_t bytes) {
        v.resize(bytes + 64);
        p = v.data() + (64 - reinterpret_cast<uintptr_t>(v.data()) % 64) % 64;
    }
};

inline uint32_t rq_bswap(uint32_t x) { return (x >> 24) | ((x >> 8) & 0xff00u) | ((x << 8) & 0xff0000u) | (x << 24); }

inline void rq_init_h(uint32_t h[8]) {
    static const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    memcpy(h, iv, sizeof iv);
}

// out[0 .. bytes) = SHA256("abacus/expand" || seed || LE32(c0 + k)), k = 0, 1, ...
void rq_fill(const uint8_t seed[32], uint32_t c0, uint8_t* out, size_t bytes) {
    const RqPrefix px = rq_prefix(seed);
    const int hashes = static_cast<int>((bytes + 31) / 32);
#pragma omp parallel for schedule(static)
    for (int k = 0; k < hashes; ++k) {
        const uint32_t cc = c0 + static_cast<uint32_t>(k);
        uint32_t w[16], h[8];
        for (int i = 0; i < 11; ++i) w[i] = px.pw[i];
        w[11] = (px.b44 << 24) | ((cc & 0xff) << 16) | (((cc >> 8) & 0xff) << 8) | ((cc >> 16) & 0xff);
        w[12] = ((cc >> 24) << 24) | (0x80u << 16);
        w[13] = 0;
        w[14] = 0;
        w[15] = 49 * 8;
        rq_init_h(h);
        rqh_compress(h, w);
        uint8_t d[32];
        for (int i = 0; i < 8; ++i) {
            d[4 * i] = static_cast<uint8_t>(h[i] >> 24);
            d[4 * i + 1] = static_cast<uint8_t>(h[i] >> 16);
            d[4 * i + 2] = static_cast<uint8_t>(h[i] >> 8);
            d[4 * i + 3] = static_cast<uint8_t>(h[i]);
        }
        const size_t at = static_cast<size_t>(k) * 32;
        memcpy(out + at, d, std::min<size_t>(32, bytes - at));
    }
}

#if defined(__ANDROID__)
// Battery temperature in tenths of a degree; 0 where Android does not let us read it. The thermal zone, as
// /sys/class/power_supply/battery is closed even to adb shell.
int rq_battery_temp() {
    static const std::string path = [] {
        for (int z = 0; z < 256; ++z) {
            const std::string base = "/sys/class/thermal/thermal_zone" + std::to_string(z);
            std::ifstream type(base + "/type");
            std::string name;
            if (type >> name && name == "battery") return base + "/temp";
        }
        return std::string();
    }();
    if (path.empty()) return 0;
    std::ifstream f(path);
    int t = 0;
    f >> t;
    return f ? t / 100 : 0;  // millidegrees
}
#endif

// A phone on a USB port passed 45 C at its battery within 20 minutes, and Android did not stop it. Stop at 45 C, the top of
// the range batteries are generally charged in, and go on at 42 C. CP_TNET_MAX_TEMP sets the limit in degrees, 0 turns it off.
void rq_cool_down() {
#if defined(__ANDROID__)
    static const char* e = getenv("CP_TNET_MAX_TEMP");
    static const int limit = (e ? atoi(e) : 45) * 10;
    static auto last = std::chrono::steady_clock::now() - std::chrono::minutes(1);
    const auto now = std::chrono::steady_clock::now();
    if (limit <= 0 || std::chrono::duration<double>(now - last).count() < 10) return;
    last = now;
    int t = rq_battery_temp();
    if (t < limit) return;
    printf("[tnet] battery at %.1f C: pausing until %.1f C\n", t / 10.0, (limit - 30) / 10.0);
    fflush(stdout);
    while (t >= limit - 30) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        t = rq_battery_temp();
    }
    printf("[tnet] battery at %.1f C: going on\n", t / 10.0);
    fflush(stdout);
#endif
}

int env_block(const char* name, int fallback) {
    const char* e = getenv(name);
    const int v = e ? atoi(e) : 0;
    return v > 0 ? v : fallback;
}

struct RqCpu final : RqDevice {
    int threads, n = 0, L = 0, w = 0, mult = 0, kc_max, nc_tiles;
    RqCpuMicro micro;
    const char* micro_name;
    std::vector<RqBuf> packed;  // the weights, one per layer: per block of k, per tile of 16 columns
    RqBuf x, x2, apack, acc;
    std::vector<uint8_t> seed;

    RqCpu(int threads_, bool portable) : threads(threads_) {
        // shortcut: blocks tuned on a Snapdragon 860 (Cortex-A76 and A55); tune again if another phone's caches disagree.
        kc_max = env_block("CP_TNET_CPU_KC", 1024) & ~3;
        nc_tiles = env_block("CP_TNET_CPU_NC", 128) / kNR;
        if (kc_max < 4) kc_max = 4;
        if (nc_tiles < 1) nc_tiles = 1;
        micro = rq_cpu_micro_portable;
        micro_name = "portable";
#if defined(CASE33_BUILD_AARCH64)
        if (!portable && case33_detect_cpu_features().dotprod) {
            micro = rq_cpu_micro_dotprod;
            micro_name = "AArch64 DotProd";
        }
#else
        (void)portable;
#endif
#ifdef _OPENMP
#if defined(__ANDROID__)
        // Not omp_get_max_threads: it counts the cores that happen to be on right now. A thread with nothing to do sleeps
        // instead of spinning, as there may be more threads than cores on and the heat is better spent on work.
        if (threads <= 0) threads = static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
        kmp_set_blocktime(0);
#endif
        if (threads > 0) omp_set_num_threads(threads);
        else threads = omp_get_max_threads();
#else
        threads = 1;
#endif
    }

    bool init(const RqWork& wk, int rows) override {
        if (fused == 0) {
            fused = 1;  // requantization has no separate form here
            return true;
        }
        fused = 1;
        n = wk.n;
        L = wk.L;
        w = wk.w;
        mult = wk.mult;
        if (n <= 0 || rows <= 0 || n % 32 || w <= 0 || w % 64 || n % w) {
            fprintf(stderr, "[tnet] unsupported parameters n=%d w=%d\n", n, w);
            return false;
        }
#if defined(__ANDROID__)
        // The phone stays usable: everything else on it comes before the miner. Threads started from here inherit this.
        setpriority(PRIO_PROCESS, 0, 10);
#endif
        packed.resize(L);
        for (RqBuf& p : packed) p.resize(static_cast<size_t>(n) * n);
        const size_t bn = static_cast<size_t>(rows) * n, tiles = (static_cast<size_t>(rows) + kMR - 1) / kMR;
        x.resize(bn);
        x2.resize(bn);
        apack.resize(tiles * kMR * n);
        acc.resize(tiles * (n / kNR) * kMR * kNR * sizeof(int32_t));
        seed.clear();
        return true;
    }

    // Weights to the kernel's layout: wt is column j's values, row-major (WT[j][k] = W[k][j]) when `transposed`.
    void pack_weights(const int8_t* src, bool transposed, int8_t* out) const {
        const int blocks = (n + kc_max - 1) / kc_max;
#pragma omp parallel for schedule(dynamic, 1)
        for (int b = 0; b < blocks; ++b) {
            const int k0 = b * kc_max, kc = std::min(kc_max, n - k0), groups = kc / 4;
            int8_t* o = out + static_cast<size_t>(k0) * n;
            for (int tile = 0; tile < n / kNR; ++tile)
                for (int jg = 0; jg < 2; ++jg)
                    for (int g = 0; g < groups; ++g)
                        for (int c = 0; c < 8; ++c)
                            for (int t = 0; t < 4; ++t) {
                                const size_t j = static_cast<size_t>(tile) * kNR + jg * 8 + c, k = static_cast<size_t>(k0) + g * 4 + t;
                                *o++ = transposed ? src[j * n + k] : src[k * n + j];
                            }
        }
    }

    void set_epoch(const std::vector<uint8_t>& s) override {
        if (s == seed) return;
        rq_reclaim_cores();
        const auto t0 = std::chrono::steady_clock::now();
        const size_t nn = static_cast<size_t>(n) * n;
        RqBuf raw;
        raw.resize(nn);
        for (int l = 0; l < L; ++l) {
            uint8_t sl[32];
            rq_w_seed(s, l, sl);
            rq_fill(sl, 0, raw.p, nn);
            pack_weights(reinterpret_cast<const int8_t*>(raw.p), false, reinterpret_cast<int8_t*>(packed[l].p));
        }
        seed = s;
        printf("[tnet] epoch weights ready (%.2f s)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    // xo = clamp((in W_l M + 2^23) >> 24, -128, 127), rows x n.
    void gemm(const int8_t* in, const int8_t* wp, int8_t* xo, int rows) {
        const int rt = (rows + kMR - 1) / kMR, ct = n / kNR;
        int8_t* ap = reinterpret_cast<int8_t*>(apack.p);
        int32_t* c = reinterpret_cast<int32_t*>(acc.p);
        const int tile_ints = kMR * kNR;
#pragma omp parallel
        {
#pragma omp for schedule(static)
            for (int t = 0; t < rt; ++t) {
                int8_t* o = ap + static_cast<size_t>(t) * kMR * n;
                for (int k = 0; k < n; k += 4)
                    for (int r = 0; r < kMR; ++r, o += 4) {
                        const int row = t * kMR + r;
                        if (row < rows) memcpy(o, in + static_cast<size_t>(row) * n + k, 4);
                        else memset(o, 0, 4);  // a batch ends inside a tile: these rows are dropped below
                    }
                memset(c + static_cast<size_t>(t) * ct * tile_ints, 0, static_cast<size_t>(ct) * tile_ints * sizeof(int32_t));
            }
            // A block of columns is one task over all of k, so the threads meet only at the end of the layer: a phone parks
            // hot cores, and a thread stuck on one would otherwise hold up the others at every block of k.
#pragma omp for schedule(dynamic, 1)
            for (int cb = 0; cb < ct; cb += nc_tiles)
                for (int k0 = 0; k0 < n; k0 += kc_max) {
                    const int kc = std::min(kc_max, n - k0);
                    const int8_t* wblock = wp + static_cast<size_t>(k0) * n;
                    for (int j = cb; j < std::min(cb + nc_tiles, ct); ++j) {
                        const int8_t* b = wblock + static_cast<size_t>(j) * kNR * kc;
                        for (int t = 0; t < rt; ++t)
                            micro(ap + static_cast<size_t>(t) * kMR * n + static_cast<size_t>(k0) * kMR, b, kc,
                                  c + (static_cast<size_t>(t) * ct + j) * tile_ints);
                    }
                }
#pragma omp for schedule(static)
            for (int row = 0; row < rows; ++row) {
                const int32_t* ct_row = c + static_cast<size_t>(row / kMR) * ct * tile_ints;
                int8_t* o = xo + static_cast<size_t>(row) * n;
                const int r = row % kMR;
                for (int col = 0; col < n; ++col) {
                    const int64_t s = ct_row[static_cast<size_t>(col / kNR) * tile_ints + (col % kNR) * kMR + r];
                    const int64_t v = (s * mult + (int64_t(1) << 23)) >> 24;
                    o[col] = static_cast<int8_t>(v < -128 ? -128 : (v > 127 ? 127 : v));
                }
            }
        }
    }

    void tickets(const int8_t* in, int rows, uint32_t r0, const RqTail& tail, const RqTarget& tg, uint64_t nonce,
                 std::vector<RqHit>& hits) const {
        const int per_row = n / w;
        std::vector<RqHit> found;
#pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < rows; ++i) {
            const uint32_t row = r0 + static_cast<uint32_t>(i);
            for (int t = 0; t < per_row; ++t) {
                const uint8_t* piece = reinterpret_cast<const uint8_t*>(in) + static_cast<size_t>(i) * n + static_cast<size_t>(t) * w;
                uint32_t h[8], wd[16];
                rq_init_h(h);
                for (int blk = 0; blk < w / 64; ++blk) {
                    for (int q = 0; q < 16; ++q) wd[q] = rqh_be(piece + blk * 64 + 4 * q);
                    rqh_compress(h, wd);
                }
                memcpy(wd, tail.wd, sizeof wd);
                // bytes 40..51: nonce byte 7 | LE32 row | LE32 c | 0x80 0 0
                const uint32_t rb = rq_bswap(row), cb = rq_bswap(static_cast<uint32_t>(t));
                wd[10] = (wd[10] & 0xff000000u) | (rb >> 8);
                wd[11] = (rb << 24) | (cb >> 8);
                wd[12] = (cb << 24) | 0x00800000u;
                rqh_compress(h, wd);
                bool below = true;
                for (int q = 0; q < 8; ++q) {
                    if (h[q] < tg.t[q]) break;
                    if (h[q] > tg.t[q]) { below = false; break; }
                }
                if (!below) continue;
                RqHit hit;
                hit.nonce = nonce;
                hit.row = row;
                hit.c = static_cast<uint32_t>(t);
                hit.piece.assign(piece, piece + w);
#pragma omp critical(rq_cpu_hits)
                found.push_back(std::move(hit));
            }
        }
        std::sort(found.begin(), found.end(), [](const RqHit& a, const RqHit& b) { return a.row != b.row ? a.row < b.row : a.c < b.c; });
        if (found.size() > RQ_MAX_HITS) found.resize(RQ_MAX_HITS);
        for (RqHit& h : found) hits.push_back(std::move(h));
    }

    void scan(const RqWork& wk, uint64_t nonce, uint32_t r0, int rows, std::vector<RqHit>& hits) override {
        rq_cool_down();
        rq_reclaim_cores();
        uint8_t xs[32];
        rq_x0_seed(wk.digest, nonce, xs);
        auto t0 = std::chrono::steady_clock::now();
        auto lap = [&](double& total) {
            if (!profile) return;
            const auto t1 = std::chrono::steady_clock::now();
            total += std::chrono::duration<double, std::milli>(t1 - t0).count();
            t0 = t1;
        };
        rq_fill(xs, static_cast<uint32_t>(static_cast<size_t>(r0) * n / 32), x.p, static_cast<size_t>(rows) * n);
        lap(ms_expand);
        uint8_t* cur = x.p;
        uint8_t* nxt = x2.p;
        for (int l = 0; l < L; ++l) {
            gemm(reinterpret_cast<const int8_t*>(cur), reinterpret_cast<const int8_t*>(packed[l].p), reinterpret_cast<int8_t*>(nxt), rows);
            std::swap(cur, nxt);
        }
        lap(ms_layers);
        tickets(reinterpret_cast<const int8_t*>(cur), rows, r0, rq_tail(wk.digest.data(), nonce, w), rq_target(wk.target), nonce, hits);
        lap(ms_hash);
        if (profile) ++scans;
    }

    std::string name() const override { return "CPU, " + std::to_string(threads) + " threads"; }
    std::string kernel() const override { return std::string("CPU int8 ") + micro_name + ", 8x16 tiles"; }
    bool failed() override { return false; }
};

// One kernel against the references: both clamps, and a batch that ends inside a tile.
bool check_device(RqCpu& d) {
    std::mt19937 rng(11);
    RqWork wk;
    wk.n = 512;
    wk.b = 512;
    wk.L = 1;
    wk.w = 256;
    for (const int mult : {2505, 300000}) {
        wk.mult = mult;
        const int rows = 203;
        if (!d.init(wk, rows)) return false;
        std::vector<int8_t> in(static_cast<size_t>(rows) * wk.n), wt(static_cast<size_t>(wk.n) * wk.n), out(in.size());
        for (auto& v : in) v = static_cast<int8_t>(rng() & 0xff);
        for (auto& v : wt) v = static_cast<int8_t>(rng() & 0xff);
        d.pack_weights(wt.data(), true, reinterpret_cast<int8_t*>(d.packed[0].p));
        d.gemm(in.data(), reinterpret_cast<const int8_t*>(d.packed[0].p), out.data(), rows);
        if (out != rq_cpu_gemm_rq(in, wt, rows, wk.n, mult)) return false;
    }
    return true;
}

}  // namespace

std::unique_ptr<RqDevice> rq_cpu_device(int threads, bool portable) {
    return std::unique_ptr<RqDevice>(new RqCpu(threads, portable));
}

int rq_cpu_selftest(int threads) {
    RqCpu d(threads, false);
    RqCpu plain(threads, true);
    printf("[tnet] selftest on %s (%s)\n", d.name().c_str(), d.kernel().c_str());
    uint8_t seed[32];
    for (int i = 0; i < 32; ++i) seed[i] = static_cast<uint8_t>(i * 5 + 1);
    std::vector<uint8_t> got(32 * 300);
    rq_fill(seed, 1000, got.data(), got.size());
    const bool exp = got == rq_cpu_expand(seed, 1000, 300);

    bool tk = true;
    RqWork wk;
    wk.n = 512;
    wk.b = 4;
    wk.L = 1;
    wk.w = 256;
    wk.mult = 2505;
    wk.digest.resize(32);
    for (int i = 0; i < 32; ++i) wk.digest[i] = static_cast<uint8_t>(i * 7 + 1);
    d.init(wk, 4);
    std::mt19937 rng(5);
    std::vector<int8_t> pieces(4 * 512);
    for (auto& v : pieces) v = static_cast<int8_t>(rng() & 0xff);
    for (int row = 0; row < 4 && tk; ++row)
        for (int c = 0; c < 2 && tk; ++c) {
            uint8_t want[32];
            rq_cpu_ticket(pieces.data() + row * 512 + c * 256, 256, wk.digest.data(), 99, 1000 + row, c, want);
            std::vector<uint8_t> t(want, want + 32);
            std::vector<RqHit> hits;
            d.tickets(pieces.data(), 4, 1000, rq_tail(wk.digest.data(), 99, 256), rq_target(t), 99, hits);
            bool seen = false;
            for (const RqHit& h : hits) seen |= h.row == static_cast<uint32_t>(1000 + row) && h.c == static_cast<uint32_t>(c);
            tk = seen;
        }
    const bool kernel = check_device(d), portable = check_device(plain);
    printf("[tnet] selftest: expansion %s, tickets %s, requantizing GEMM %s, portable GEMM %s\n", exp ? "ok" : "FAIL",
           tk ? "ok" : "FAIL", kernel ? "ok" : "FAIL", portable ? "ok" : "FAIL");
    return exp && tk && kernel && portable ? 0 : 1;
}
