// Abacus backend entry point: argument handling and dispatch to the CUDA mock loop.
//
// Usage (subset parsed here; the rest of CPPminer's args are ignored for abacus):
//   cppminer --algo abacus --backend cuda [-d DEVICE] --mock [--n 64] [--bits 4] [--seconds 5]
//
// Network (solo/pool) mining is not implemented yet: the Abacus chain prototype does not expose a
// job/submit protocol. Only --mock is accepted for now.

#include "cp_abacus.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

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
    const bool mock = hasflag(argc, argv, "--mock");
    const char* n_s = getval(argc, argv, "--n");
    const char* b_s = getval(argc, argv, "--bits");
    const char* s_s = getval(argc, argv, "--seconds");
    const char* d_s = getval(argc, argv, "-d");
    if (!d_s) d_s = getval(argc, argv, "--devices");

    const int n = n_s ? atoi(n_s) : 64;
    const int bits = b_s ? atoi(b_s) : 4;
    const int seconds = s_s ? atoi(s_s) : 5;
    const int device = d_s ? atoi(d_s) : 0;

    printf("[mode] algo=abacus backend=cuda%s\n", mock ? " (mock)" : "");

    if (!mock) {
        fprintf(stderr, "[abacus] network solo mining is not implemented yet; use --mock\n");
        return 1;
    }

#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    return cp_abacus_cuda_mock(n, bits, seconds, device);
#else
    fprintf(stderr, "[abacus] this build has no CUDA backend\n");
    return 1;
#endif
}
