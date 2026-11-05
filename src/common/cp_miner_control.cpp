#include "cp_miner_control.h"
#include "cp_platform.h"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <cstdio>
#include <cstdint>
#include <thread>
#ifndef _WIN32
#include <sys/select.h>
#endif
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

namespace {
CpMinerControl control;
bool enabled = false;
std::string owner, instance, credential;
int device_id = -1;

bool hex_id(const std::string& s) {
    return s.size() == 64 && std::all_of(s.begin(), s.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}
bool equal_secret(const std::string& a, const std::string& b) {
    unsigned diff = (unsigned)(a.size() ^ b.size());
    for (size_t i = 0; i < b.size(); ++i) diff |= (i < a.size() ? (unsigned char)a[i] : 0) ^ (unsigned char)b[i];
    return diff == 0;
}
std::string status_json() {
    std::string state, lease;
    control.snapshot(state, lease);
    return "{\"protocol\":\"requant-miner-control-v1\",\"instance\":\"" + instance +
           "\",\"owner\":\"" + owner + "\",\"device\":" + std::to_string(device_id) +
           ",\"state\":\"" + state + "\",\"lease\":\"" + lease +
           "\",\"gpu_quiescent\":" + (state == "quiescent" ? "true}" : "false}");
}
// Strict bounded header-only protocol: no request bodies, redirects or CORS.
int dispatch(const std::string& req) {
    const auto end = req.find("\r\n\r\n"), line_end = req.find("\r\n");
    if (end == std::string::npos || end + 4 != req.size() || line_end == std::string::npos) return 400;
    const std::string first = req.substr(0, line_end);
    const auto sp = first.find(' '), sp2 = first.find(' ', sp == std::string::npos ? 0 : sp + 1);
    if (sp == std::string::npos || sp2 == std::string::npos || first.substr(sp2 + 1) != "HTTP/1.1") return 400;
    const std::string method = first.substr(0, sp), path = first.substr(sp + 1, sp2 - sp - 1);
    std::map<std::string, std::string> headers;
    size_t off = line_end + 2;
    while (off < end) {
        const size_t e = req.find("\r\n", off), colon = req.find(':', off);
        if (e == std::string::npos || colon == std::string::npos || colon >= e || colon == off) return 400;
        std::string name = req.substr(off, colon - off);
        for (char& c : name) {
            if (!std::isalnum((unsigned char)c) && c != '-') return 400;
            c = (char)std::tolower((unsigned char)c);
        }
        size_t start = colon + 1, stop = e;
        while (start < stop && req[start] == ' ') ++start;
        while (stop > start && req[stop - 1] == ' ') --stop;
        const std::string value = req.substr(start, stop - start);
        for (unsigned char c : value) if (c < 32 || c == 127) return 400;
        if (!headers.emplace(name, value).second) return 400;
        off = e + 2;
    }
    if (headers.count("transfer-encoding") || (headers.count("content-length") && headers["content-length"] != "0")) return 400;
    if (!equal_secret(headers["authorization"], "Bearer " + credential)) return 401;
    if (headers["x-requant-miner-owner"] != owner) return 409;
    // Only an authenticated owner may discover the new process instance.
    const bool discovery = path == "/v1/control/status" && method == "GET" && headers["x-requant-miner-instance"].empty();
    if (!discovery && headers["x-requant-miner-instance"] != instance) return 409;
    if (path == "/v1/control/status") return method == "GET" ? 200 : 405;
    const std::string lease = headers["x-requant-miner-lease"];
    if (!hex_id(lease)) return 400;
    if (method != "POST") return 405;
    if (path == "/v1/control/quiesce") return control.command("quiesce", lease);
    if (path == "/v1/control/resume") return control.command("resume", lease);
    return 404;
}
void handle(cp_sock_t socket) {
#ifdef SO_NOSIGPIPE
    int suppress_sigpipe = 1;
    setsockopt(socket, SOL_SOCKET, SO_NOSIGPIPE, (const char*)&suppress_sigpipe, sizeof(suppress_sigpipe));
#endif
    std::string req;
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (req.size() < 4096 && std::chrono::steady_clock::now() < until) {
        fd_set reads; FD_ZERO(&reads); FD_SET(socket, &reads);
        timeval tv{0, 100000};
        if (select((int)socket + 1, &reads, nullptr, nullptr, &tv) <= 0) continue;
        char buf[1024];
        const int n = (int)recv(socket, buf, (int)std::min(sizeof(buf), 4096 - req.size()), 0);
        if (n <= 0) break;
        req.append(buf, (size_t)n);
        if (req.find("\r\n\r\n") != std::string::npos) break;
    }
    const int code = dispatch(req);
    const std::string body = code == 200 ? status_json() : "{\"error\":\"miner_control_rejected\"}";
    std::string res = "HTTP/1.1 " + std::to_string(code) + (code == 200 ? " OK" : " Error") +
        "\r\nContent-Type: application/json\r\nCache-Control: no-store\r\nConnection: close\r\nContent-Length: " +
        std::to_string(body.size()) + "\r\n\r\n" + body;
#ifdef _WIN32
    DWORD send_timeout = 2000;
#else
    timeval send_timeout{2, 0};
#endif
    setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, (const char*)&send_timeout, sizeof(send_timeout));
    size_t sent = 0;
    while (sent < res.size()) {
        const int n = (int)send(socket, res.data() + sent, (int)(res.size() - sent), MSG_NOSIGNAL);
        if (n <= 0) break;
        sent += (size_t)n;
    }
    CP_SOCK_CLOSE(socket);
}
}

int cp_miner_control_start(int port, const char* owner_arg, const char* key_env, int device) {
    const char* key = key_env ? std::getenv(key_env) : nullptr;
    if (enabled || port < 1 || port > 65535 || device < 0 || !owner_arg ||
        !hex_id(owner_arg) || !key || std::strlen(key) < 32 || std::strlen(key) > 256) return -1;
    for (const char* p = key; *p; ++p) if (*p < 33 || *p > 126) return -1;
    std::string fresh_instance;
    try {
        std::random_device entropy;
        if (entropy.entropy() <= 0) return -1;
        static const char* digits = "0123456789abcdef";
        for (int i = 0; i < 8; ++i) {
            const uint32_t word = entropy();
            for (int shift = 0; shift < 32; shift += 4) fresh_instance += digits[(word >> shift) & 15];
        }
    } catch (...) { return -1; }
    if (cp_net_init() != 0) return -1;
    const cp_sock_t listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listener == CP_INVALID_SOCK) return -1;
#ifndef _WIN32
    int one = 1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof(one));
#endif
    sockaddr_in sa{};
    sa.sin_family = AF_INET; sa.sin_port = htons((unsigned short)port);
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(listener, (sockaddr*)&sa, sizeof(sa)) != 0 || listen(listener, 8) != 0) {
        CP_SOCK_CLOSE(listener); return -1;
    }
    owner = owner_arg; instance = fresh_instance; credential = key; device_id = device; enabled = true;
    control.bootstrap(instance);
    std::thread([listener] {
        for (;;) {
            sockaddr_in peer{}; socklen_t len = sizeof(peer);
            const cp_sock_t client = accept(listener, (sockaddr*)&peer, &len);
            if (client == CP_INVALID_SOCK) { cp_sleep(1); continue; }
            if ((ntohl(peer.sin_addr.s_addr) >> 24) != 127) { CP_SOCK_CLOSE(client); continue; }
            handle(client);
        }
    }).detach();
    std::fprintf(stderr, "[control] instance=%s bootstrap_lease=%s awaiting owner runtime-idle reconciliation\n", instance.c_str(), instance.c_str());
    return control.wait_start();
}
int cp_miner_control_checkpoint(const std::function<bool()>& idle) {
    return enabled ? control.checkpoint(idle) : 0;
}
