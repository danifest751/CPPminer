// TNet v1 on the CPU (phones and other machines without a usable GPU). cp_tnet_cpu.cpp.
#pragma once

#include "cp_tnet_common.hpp"

// One tile of 8 output rows x 16 output columns over kc values of k (a multiple of 4), added to its accumulators.
//   a: 8 rows x kc, 32 bytes per 4 k, row r at +4r
//   b: two groups of 8 columns, each 32 bytes per 4 k, column c at +4c; the second group follows the first
//   c: 16 columns of 8 int32, column-major
using RqCpuMicro = void (*)(const int8_t* a, const int8_t* b, int kc, int32_t* c);

void rq_cpu_micro_portable(const int8_t* a, const int8_t* b, int kc, int32_t* c);
#if defined(CASE33_BUILD_AARCH64)
void rq_cpu_micro_dotprod(const int8_t* a, const int8_t* b, int kc, int32_t* c);
#endif

// threads 0: all the CPUs OpenMP sees. portable: the plain C++ kernel whatever the CPU has.
std::unique_ptr<RqDevice> rq_cpu_device(int threads, bool portable);
// The kernels, expansion and tickets against the references in cp_tnet_common.hpp; 0 when all agree.
int rq_cpu_selftest(int threads);
