// --algo tnet (alias requant): argument handling. The backends are CUDA (cp_tnet.cu, NVIDIA tensor cores), OpenCL
// (cp_tnet_ocl.cpp, AMD, Intel and other GPUs) and the CPU (cp_tnet_cpu.cpp, phones); the network and the mining loop are
// shared (cp_tnet_net.cpp).
#include "cp_tnet.h"
#include "cp_miner_control.h"
#include "cp_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
#define TNET_HAS_CUDA 1
#else
#define TNET_HAS_CUDA 0
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
#define TNET_HAS_OPENCL 1
#else
#define TNET_HAS_OPENCL 0
#endif
#if defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
#define TNET_HAS_CPU 1
#include "cp_tnet_cpu.hpp"
#else
#define TNET_HAS_CPU 0
#endif

static void tnet_usage(void)
{
    fprintf(stderr,
            "usage: cppminer --algo tnet --rpc HOST:PORT --payee KEY_HASH_HEX [--worker NAME] [--backend cuda|opencl|cpu]\n"
            "                [--device N] [--threads N] [--batch ROWS] [--seconds S] [--blocks K]\n"
            "                [--control-port N --control-owner HEX64 --control-key-env ENV] [--api-port N]\n"
            "       cppminer --algo tnet --selftest [--backend cuda|opencl|cpu] [--device N]\n"
            "       cppminer --algo tnet --bench [--backend cuda|opencl|cpu] [--device N] [--threads N] [--batch ROWS] [--seconds S]\n"
            "  --rpc     requantd JSON-RPC or pool address (default 127.0.0.1:19445, regtest)\n"
            "  --payee   32-byte key hash to pay (requant-wallet address KEYFILE prints it)\n"
            "  --worker  name of this device in a pool's statistics (rewards still go to the payee)\n"
            "  --backend cuda (NVIDIA tensor cores; the default where built), opencl (AMD, Intel and other GPUs) or cpu\n"
            "            (the default on Android)\n"
            "  --device  CUDA device id, or the OpenCL device's flat index (cppminer --backend opencl --list-devices)\n"
            "  --threads CPU threads (default: all)\n"
            "  --batch   rows per pass (default 8192 on a GPU, lower it on GPUs with little memory; 256 on the CPU, 64 on a\n"
            "            phone's GPU)\n"
            "  --bench   offline: ns per ticket with requantization in the GEMM and as a separate pass\n"
            "            (default 20 s each); CP_TNET_FUSED=0 mines with the separate pass (CUDA)\n");
}

int cp_tnet_main(int argc, char** argv)
{
    const char* rpc = "127.0.0.1:19445";
    const char* payee = NULL;
    const char* worker = "default";
#if defined(__ANDROID__) && TNET_HAS_CPU
    const char* backend = "cpu";  // long GPU passes can freeze a phone
#else
    const char* backend = TNET_HAS_CUDA ? "cuda" : (TNET_HAS_OPENCL ? "opencl" : "cpu");
#endif
    int device = 0, batch = 0, threads = 0, selftest = 0, bench = 0;
    int control_port = 0, api_port = 0;
    const char *control_owner = NULL, *control_key_env = NULL;
    double seconds = 0;
    long long blocks = 0;
    for(int i = 1; i < argc; ++i){
        const char* a = argv[i];
        const char* v = i + 1 < argc ? argv[i + 1] : NULL;
        if(!strcmp(a, "--algo") && v){ ++i; }
        else if(!strcmp(a, "--rpc") && v){ rpc = v; ++i; }
        else if(!strcmp(a, "--payee") && v){ payee = v; ++i; }
        else if(!strcmp(a, "--worker") && v){ worker = v; ++i; }
        else if(!strcmp(a, "--backend") && v){ backend = v; ++i; }
        else if((!strcmp(a, "--device") || !strcmp(a, "--devices")) && v){ device = atoi(v); ++i; }
        else if(!strcmp(a, "--batch") && v){ batch = atoi(v); if(batch <= 0){ tnet_usage(); return 2; } ++i; }
        else if(!strcmp(a, "--threads") && v){ threads = atoi(v); if(threads <= 0){ tnet_usage(); return 2; } ++i; }
        else if(!strcmp(a, "--seconds") && v){ seconds = atof(v); ++i; }
        else if(!strcmp(a, "--blocks") && v){ blocks = atoll(v); ++i; }
        else if(!strcmp(a, "--api-port") && v){
            char* end = NULL;
            const long port = strtol(v, &end, 10);
            if (!*v || *end || port < 1 || port > 65535) { tnet_usage(); return 2; }
            api_port = (int)port; ++i;
        }
        else if(!strcmp(a, "--control-port") && v){
            char* end = NULL;
            const long port = strtol(v, &end, 10);
            if (!*v || *end || port < 1 || port > 65535) { tnet_usage(); return 2; }
            control_port = (int)port; ++i;
        }
        else if(!strcmp(a, "--control-owner") && v){ control_owner = v; ++i; }
        else if(!strcmp(a, "--control-key-env") && v){ control_key_env = v; ++i; }
        else if(!strcmp(a, "--selftest")){ selftest = 1; }
        else if(!strcmp(a, "--bench")){ bench = 1; }
        else { tnet_usage(); return 2; }
    }
    const int cuda = !strcmp(backend, "cuda"), opencl = !strcmp(backend, "opencl"), cpu = !strcmp(backend, "cpu");
    if(!cuda && !opencl && !cpu){
        fprintf(stderr, "[tnet] --backend: cuda, opencl or cpu\n");
        return 2;
    }
    if((cuda && !TNET_HAS_CUDA) || (opencl && !TNET_HAS_OPENCL) || (cpu && !TNET_HAS_CPU)){
        fprintf(stderr, "[tnet] this build has no %s backend (build with -DCP_ENABLE_%s=ON)\n", backend, cuda ? "CUDA" : (opencl ? "OPENCL" : "CPU"));
        return 1;
    }
#if defined(__ANDROID__)
    if(!batch) batch = cpu ? 256 : 64;  // a pass of a few seconds at most, or new blocks go unnoticed
#else
    if(!batch) batch = cpu ? 256 : 8192;
#endif
    if(batch < 4 || batch % 4){
        fprintf(stderr, "[tnet] --batch must be a positive multiple of 4\n");
        return 2;
    }
    if(!selftest && !bench && (!payee || strlen(payee) != 64)){
        fprintf(stderr, "[tnet] --payee must be a 64-digit hex key hash\n");
        tnet_usage();
        return 2;
    }
    if ((control_port || control_owner || control_key_env) &&
        (!control_port || !control_owner || !control_key_env || !cuda || selftest || bench)) {
        fprintf(stderr, "[tnet] control requires all ownership options and CUDA solo/pool mode\n");
        return 2;
    }
    if (api_port) {
        if (selftest || bench) { fprintf(stderr, "[tnet] stats API requires mining mode\n"); return 2; }
        cp_api_set_algo("tnet", backend);
        if (cp_api_start("127.0.0.1", api_port) != 0) { fprintf(stderr, "[tnet] stats API unavailable\n"); return 1; }
    }
    (void)rpc; (void)payee; (void)worker; (void)device; (void)seconds; (void)blocks;
#if TNET_HAS_CUDA
    if(cuda){
        if(selftest) return cp_tnet_cuda_selftest(device);
        if(bench) return cp_tnet_cuda_bench(device, batch, seconds > 0 ? seconds : 20);
        if (control_port && cp_miner_control_start(control_port, control_owner, control_key_env, device) != 0) {
            fprintf(stderr, "[tnet] cannot start explicit loopback control; refusing this opted-in run\n");
            return 1;
        }
        return cp_tnet_cuda_solo(rpc, payee, worker, device, batch, seconds, blocks);
    }
#endif
#if TNET_HAS_CPU
    if(cpu){
        const RqMake make = [threads] { return rq_cpu_device(threads, false); };
        if(selftest) return rq_cpu_selftest(threads);
        if(bench) return rq_bench(make, batch, seconds > 0 ? seconds : 20);
        return rq_solo(make, rpc, payee, worker, batch, seconds, blocks, device);
    }
#endif
#if TNET_HAS_OPENCL
    if(opencl){
        if(selftest) return cp_tnet_ocl_selftest(device);
        if(bench) return cp_tnet_ocl_bench(device, batch, seconds > 0 ? seconds : 20);
        return cp_tnet_ocl_solo(rpc, payee, worker, device, batch, seconds, blocks);
    }
#endif
    return 1;
}
