// Abacus backend entry point: argument handling and dispatch to the CUDA mock / solo loop.
//
// Usage (subset parsed here; the rest of CPPminer's args are ignored for abacus):
//   cppminer --algo abacus --backend cuda [-d DEVICE] --mock [--n 64] [--bits 4] [--seconds 5]
//   cppminer --algo abacus --backend cuda [-d DEVICE] --node HOST:PORT [--n 64] [--seconds 30]
//   add --dataset NBLOCKS (candidate A': 32-byte blocks, e.g. 33554432 = 1 GiB) to either mode.
//
// Solo mode connects to an Abacus node (abacus-node), requests jobs and submits blocks.

#include "cp_abacus.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static const char* getval(int argc, char** argv, const char* name) {
    for (int i = 1; i + 1 < argc; ++i)
        if (!strcmp(argv[i], name)) return argv[i + 1];
    return nullptr;
}

static bool hasflag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], name)) return true;
    return false;
}

extern "C" int cp_abacus_main(int argc, char** argv) {
    if (hasflag(argc, argv, "--selftest")) {
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
        return cp_abacus_cuda_selftest();
#else
        fprintf(stderr, "[abacus] no CUDA\n");
        return 1;
#endif
    }
    const bool mock = hasflag(argc, argv, "--mock");
    const char* node = getval(argc, argv, "--node");
    const char* n_s = getval(argc, argv, "--n");
    const char* b_s = getval(argc, argv, "--bits");
    const char* s_s = getval(argc, argv, "--seconds");
    const char* d_s = getval(argc, argv, "-d");
    if (!d_s) d_s = getval(argc, argv, "--devices");
    const char* ds_s = getval(argc, argv, "--dataset");
    if (getval(argc, argv, "--seg"))
        fprintf(stderr, "[abacus] --seg is not supported (the gather reads one 8-byte word per entry); ignored\n");

    const int n = n_s ? atoi(n_s) : 64;
    const int bits = b_s ? atoi(b_s) : 4;
    const int seconds = s_s ? atoi(s_s) : (node ? 30 : 5);
    const int device = d_s ? atoi(d_s) : 0;
    const long long nblocks = ds_s ? atoll(ds_s) : 0; // dataset size in 32-byte blocks, not MiB

    printf("[mode] algo=abacus backend=cuda%s%s%s\n", mock ? " (mock)" : "", node ? " (solo)" : "", nblocks ? " (hard)" : "");

    if (node) {
        std::string s(node);
        const auto pos = s.rfind(':');
        if (pos == std::string::npos) {
            fprintf(stderr, "[abacus] --node expects HOST:PORT\n");
            return 1;
        }
        const std::string host = s.substr(0, pos);
        const int port = atoi(s.substr(pos + 1).c_str());
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
        return cp_abacus_cuda_solo(host.c_str(), port, n, seconds, device, nblocks);
#else
        fprintf(stderr, "[abacus] this build has no CUDA backend\n");
        return 1;
#endif
    }

    if (!mock) {
        fprintf(stderr, "[abacus] pass --mock (benchmark) or --node HOST:PORT (solo)\n");
        return 1;
    }

#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    if (nblocks > 0)
        return cp_abacus_cuda_mock_hard(n, bits, seconds, device, nblocks);
    return cp_abacus_cuda_mock(n, bits, seconds, device);
#else
    fprintf(stderr, "[abacus] this build has no CUDA backend\n");
    return 1;
#endif
}
