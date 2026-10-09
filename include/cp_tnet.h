#ifndef CP_TNET_H
#define CP_TNET_H

#ifdef __cplusplus
extern "C" {
#endif

/* TNet v1 (the Requant coin's work function) entry point, called from main for --algo tnet (alias requant).
 * Handles its own arguments and returns a process exit code. */
int cp_tnet_main(int argc, char** argv);

#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
/* Solo mining against a requantd node's JSON-RPC (getwork/submitwork). Defined in cp_tnet.cu. */
int cp_tnet_cuda_solo(const char* rpc, const char* payee_hex, const char* worker, int device, int batch, double seconds,
                      long long blocks);
/* Known-answer self-test of the device SHA-256 and expansion. */
int cp_tnet_cuda_selftest(int device);
#endif

#ifdef __cplusplus
}
#endif

#endif /* CP_TNET_H */
