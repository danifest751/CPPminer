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

/* Solo client: connect to an Abacus node, request jobs, mine and submit blocks.
 * Defined in cp_abacus.cu (POSIX only). */
int cp_abacus_cuda_solo(const char* host, int port, int n, int seconds, int device, long long nblocks);

/* Print SHA-256 known-vector self-test; returns 0 on success. */
int cp_abacus_cuda_selftest(void);

/* Candidate A' mock: gather operands from a device dataset of `nblocks` 32-byte blocks. */
int cp_abacus_cuda_mock_hard(int n, int bits, int seconds, int device, long long nblocks);
#endif

#ifdef __cplusplus
}
#endif

#endif /* CP_ABACUS_H */
