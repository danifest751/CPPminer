// TNet v1 on the CPU (phones and other machines without a usable GPU). cp_tnet_cpu.cpp.
#pragma once

#include "cp_tnet_common.hpp"

// One tile of 8 output rows x 16 output columns over kc values of k, added to its accumulators. A kernel takes k in units
// of u values (4, or 8 for I8MM), and kc is a multiple of u.
//   a: 8 rows x kc, 8u bytes per u k, row r at +ur
//   b: two groups of 8 columns, each 8u bytes per u k, column c at +uc; the second group follows the first
//   c: 16 columns of 8 int32, column-major
using RqCpuMicro = void (*)(const int8_t* a, const int8_t* b, int kc, int32_t* c);

void rq_cpu_micro_portable(const int8_t* a, const int8_t* b, int kc, int32_t* c);
#if defined(CASE33_BUILD_AARCH64)
void rq_cpu_micro_dotprod(const int8_t* a, const int8_t* b, int kc, int32_t* c);
#if !defined(_MSC_VER)  // MSVC has no I8MM intrinsics
void rq_cpu_micro_i8mm(const int8_t* a, const int8_t* b, int kc, int32_t* c);
#endif
#endif

// threads 0: all the CPUs OpenMP sees. portable: the plain C++ kernel whatever the CPU has.
std::unique_ptr<RqDevice> rq_cpu_device(int threads, bool portable);
// The kernels, expansion and tickets against the references in cp_tnet_common.hpp; 0 when all agree.
int rq_cpu_selftest(int threads);
