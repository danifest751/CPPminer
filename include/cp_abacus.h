#ifndef CP_ABACUS_H
#define CP_ABACUS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Abacus (verifiable GPU-algebra PoW) entry point, called from main when --algo abacus.
 * Handles its own argument subset and returns a process exit code. */
int cp_abacus_main(int argc, char** argv);

#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
/* CUDA mock mine loop: search `seconds` for nonces with `bits` leading zero score bits.
 * Returns 0 on success, non-zero on error. Defined in cp_abacus.cu. */
int cp_abacus_cuda_mock(int n, int bits, int seconds, int device);
#endif

#ifdef __cplusplus
}
#endif

#endif /* CP_ABACUS_H */
