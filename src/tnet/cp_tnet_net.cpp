// TNet: the network (a requantd node's or pool's JSON-RPC, getwork / submitwork, the long poll) and the mining loop,
// shared by the CUDA and OpenCL backends through RqDevice (cp_tnet_common.hpp).

#include "cp_tnet_common.hpp"

#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <mutex>
#include <random>
#include <thread>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET sock_t;
#define CLOSESOCK closesocket
#else
#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
typedef int sock_t;
#define INVALID_SOCKET (-1)
#define CLOSESOCK close
#endif

// ---------------------------------------------------------------- JSON-RPC client

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
    // no answer within 90 s (a long poll waits up to 60 s): give up on this request
#ifdef _WIN32
    const DWORD timeout_ms = 90000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout_ms, sizeof timeout_ms);
#else
    timeval tv{90, 0};
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
#endif
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

// Talks to the node or pool on its own threads (submissions, and watching the tip), so a slow link never stalls
// the GPU. The tip is watched by a long poll (`getwork payee longpollid`, answered when the next block arrives)
// where the node or pool supports it, else by asking for the tip every second.
struct RqNet {
    struct Item { std::string params, tip; };
    std::string ep, payee;
    bool pool_mode = false;
    std::atomic<bool> long_poll_ok{false};  // the long poll works: no need to ask for the tip every second
    std::thread lp;
    std::mutex mu;
    std::deque<Item> queue;
    std::string tip_seen;              // tip of the work being mined (set by the mining loop)
    std::atomic<bool> restart{false};  // new tip, found block or rejection: fetch new work
    std::atomic<bool> stop{false};
    std::atomic<long long> shares{0}, rejected{0}, blocks{0}, stale{0};
    std::thread th;

    void start() {
        th = std::thread([this] { run(); });
        lp = std::thread([this] { long_poll(); });
    }
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
        // the long poll may still wait for its answer (up to a minute): let it end on its own; RqNet outlives it
        if (lp.joinable()) lp.detach();
    }
    void long_poll() {
        int shown = -1;  // the mode last reported in the log (-1: none yet)
        {
            // say at once which mode applies: a plain getwork carries "longpollid" where long polls are answered
            std::string r, id;
            const bool ok = http_post(ep, "{\"method\":\"getwork\",\"params\":[\"" + payee + "\"]}", r) &&
                            json_get(r, "longpollid", id);
            printf(ok ? "[tnet] tip watched by long poll\n" : "[tnet] no long poll here: checking the tip every second\n");
            shown = ok ? 1 : 0;
            long_poll_ok = ok;
        }
        while (!stop) {
            std::string tip_now;
            { std::lock_guard<std::mutex> g(mu); tip_now = tip_seen; }
            if (tip_now.empty()) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); continue; }
            const auto t0 = std::chrono::steady_clock::now();
            std::string r, id;
            const bool ok = http_post(ep, "{\"method\":\"getwork\",\"params\":[\"" + payee + "\",\"" + tip_now + "\"]}", r) &&
                            json_get(r, "longpollid", id);
            if (!ok) {
                // an older node or pool, or no answer: the once-a-second tip check takes over, and this tries again later
                long_poll_ok = false;
                if (shown != 0) printf("[tnet] no long poll here: checking the tip every second\n");
                shown = 0;
                for (int k = 0; k < 100 && !stop; ++k) std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            long_poll_ok = true;
            if (shown != 1) printf("[tnet] tip watched by long poll\n");
            shown = 1;
            if (id != tip_now) {
                std::lock_guard<std::mutex> g(mu);
                if (tip_seen == tip_now) restart = true;  // a new block while mining on tip_now
            } else if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 1.0) {
                std::this_thread::sleep_for(std::chrono::seconds(1));  // answered at once without a new block: no busy loop
            }
        }
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
            if (!long_poll_ok && std::chrono::duration<double>(now - last_poll).count() >= 1.0) {
                last_poll = now;
                std::string tip;
                if (get_tip(ep, tip) && !tip_now.empty() && tip != tip_now) restart = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
};

// ---------------------------------------------------------------- mining and the offline benchmark

int rq_solo(const RqMake& make, const char* rpc, const char* payee_hex, const char* worker, int batch, double seconds,
            long long blocks, int device_index) {
    const std::string ep = rpc, payee = payee_hex;
    // device name for a pool's statistics: letters, digits, '.', '_', '-' (the pool applies the same filter)
    std::string worker_name;
    for (const char* q = worker ? worker : ""; *q && worker_name.size() < 32; ++q)
        if (isalnum((unsigned char)*q) || *q == '.' || *q == '_' || *q == '-') worker_name += *q;
    if (worker_name.empty()) worker_name = "default";
    RqWork wk;
    if (!get_work(ep, payee, wk)) { fprintf(stderr, "[tnet] getwork from %s failed\n", rpc); return 1; }
    const int rows = batch < wk.b ? batch : wk.b;
    std::unique_ptr<RqDevice> dev = make();
    if (!dev || !dev->init(wk, rows)) { fprintf(stderr, "[tnet] GPU set-up failed (try a smaller --batch)\n"); return 1; }
    printf("[tnet] %s | node %s | TNet n=%d B=%d L=%d w=%d | batch %d rows\n", dev->name().c_str(), rpc, wk.n, wk.b, wk.L,
           wk.w, rows);
    printf("[tnet] GEMM: %s, requantization %s\n", dev->kernel().c_str(), dev->fused ? "in the epilogue" : "as a separate pass");
    std::mt19937_64 rng(std::random_device{}() ^ ((uint64_t)device_index << 48) ^ (uint64_t)time(nullptr));
    const auto start = std::chrono::steady_clock::now();
    auto last_stat = start, last_work = start;
    unsigned long long tickets = 0, tickets_stat = 0;
    std::string pool_flag;
    // on the heap and never freed: the long-poll thread may still wait for an answer after mining ends
    RqNet& net = *new RqNet;
    net.ep = ep;
    net.payee = payee;
    net.pool_mode = rpc_call(ep, "getwork", "[\"" + payee + "\"]", pool_flag) && pool_flag.find("\"pool\":true") != std::string::npos;
    if (net.pool_mode) printf("[tnet] pool mode: shares at 2^%s tickets\n", json_get(pool_flag, "share_bits", pool_flag) ? pool_flag.c_str() : "?");
    std::string tip_seen = to_hex(wk.header.data() + 12, 32);
    net.set_tip(tip_seen);
    net.start();
    std::vector<RqHit> hits;
    bool done = false;
    while (!done) {
        dev->set_epoch(wk.seed);
        const uint64_t nonce = rng();
        bool restart = false;
        for (uint32_t r0 = 0; r0 < (uint32_t)wk.b && !restart; r0 += (uint32_t)rows) {
            const int nrows = (int)((uint32_t)wk.b - r0 < (uint32_t)rows ? (uint32_t)wk.b - r0 : (uint32_t)rows);
            hits.clear();
            dev->scan(wk, nonce, r0, nrows, hits);
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
                fflush(stdout);
                last_stat = now;
                tickets_stat = tickets;
            }
            if (dev->failed()) { fprintf(stderr, "[tnet] GPU error\n"); net.finish(); return 1; }
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

// Offline: the attempt at the TNet v1 parameters (n = 8192, L = 8, w = 256, B = 65536) on random weights and input, with
// a target nothing meets; first with the requantization in the GEMM, then as a separate pass (when the device has it).
int rq_bench(const RqMake& make, int batch, double seconds) {
    RqWork wk;
    wk.n = 8192;
    wk.L = 8;
    wk.w = 256;
    wk.b = 65536;
    wk.mult = (int)std::lround(16777216.0 / (74.0 * std::sqrt(8192.0)));
    std::mt19937_64 rng(7);
    wk.digest.resize(32);
    wk.seed.resize(32);
    for (auto& v : wk.digest) v = (uint8_t)rng();
    for (auto& v : wk.seed) v = (uint8_t)rng();
    wk.target.assign(32, 0);
    const int rows = batch < wk.b ? batch : wk.b;
    const double macs_per_ticket = (double)wk.L * wk.n * wk.w;
    for (int mode = 1; mode >= 0; --mode) {
        std::unique_ptr<RqDevice> dev = make();
        if (!dev) return 1;
        dev->fused = mode;
        if (!dev->init(wk, rows)) { fprintf(stderr, "[tnet] GPU set-up failed (try a smaller --batch)\n"); return 1; }
        if (mode == 1)
            printf("[tnet] bench on %s (%s): n=%d L=%d w=%d M=%d, batch %d rows, %.0f s per mode\n", dev->name().c_str(),
                   dev->kernel().c_str(), wk.n, wk.L, wk.w, wk.mult, rows, seconds);
        if (dev->fused != mode) continue;  // the device does not have this mode, or its check failed (it said why)
        dev->set_epoch(wk.seed);
        std::vector<RqHit> hits;
        dev->scan(wk, 1, 0, rows, hits);  // warm-up
        dev->profile = true;
        unsigned long long tickets = 0;
        uint64_t nonce = 2;
        const auto t0 = std::chrono::steady_clock::now();
        double dt = 0;
        while (dt < seconds) {
            for (uint32_t r0 = 0; r0 < (uint32_t)wk.b && dt < seconds; r0 += (uint32_t)rows) {
                dev->scan(wk, nonce, r0, rows, hits);
                tickets += (unsigned long long)rows * (wk.n / wk.w);
                dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            }
            ++nonce;
        }
        if (dev->failed()) { fprintf(stderr, "[tnet] GPU error\n"); return 1; }
        const double ns = dt * 1e9 / (double)tickets, s = dev->scans;
        printf("[tnet] %s: %.1f ns/ticket, %.2f M tickets/s, %.1f TMAC/s; per batch: expand %.2f ms, layers %.2f ms, hash %.2f ms"
               " (%d batches)\n",
               mode ? "requant in the GEMM epilogue" : "requant as a separate pass", ns, tickets / dt / 1e6,
               macs_per_ticket / ns / 1e3, dev->ms_expand / s, dev->ms_layers / s, dev->ms_hash / s, dev->scans);
        fflush(stdout);
    }
    return 0;
}
