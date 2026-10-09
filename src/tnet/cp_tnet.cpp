// --algo tnet (alias requant): argument handling. The miner (TNet v1 on CUDA + cuBLAS) is in cp_tnet.cu.
#include "cp_tnet.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void tnet_usage(void)
{
    fprintf(stderr,
            "usage: cppminer --algo tnet --rpc HOST:PORT --payee KEY_HASH_HEX [--device N] [--batch ROWS]\n"
            "                [--seconds S] [--blocks K]\n"
            "       cppminer --algo tnet --selftest [--device N]\n"
            "  --rpc     requantd JSON-RPC address (default 127.0.0.1:19445, regtest)\n"
            "  --payee   32-byte key hash to pay (requant-wallet address KEYFILE prints it)\n"
            "  --batch   rows per GPU pass (default 8192; lower it on GPUs with little memory)\n");
}

int cp_tnet_main(int argc, char** argv)
{
    const char* rpc = "127.0.0.1:19445";
    const char* payee = NULL;
    int device = 0, batch = 8192, selftest = 0;
    double seconds = 0;
    long long blocks = 0;
    for(int i = 1; i < argc; ++i){
        const char* a = argv[i];
        const char* v = i + 1 < argc ? argv[i + 1] : NULL;
        if(!strcmp(a, "--algo") && v){ ++i; }
        else if(!strcmp(a, "--rpc") && v){ rpc = v; ++i; }
        else if(!strcmp(a, "--payee") && v){ payee = v; ++i; }
        else if(!strcmp(a, "--device") && v){ device = atoi(v); ++i; }
        else if(!strcmp(a, "--batch") && v){ batch = atoi(v); ++i; }
        else if(!strcmp(a, "--seconds") && v){ seconds = atof(v); ++i; }
        else if(!strcmp(a, "--blocks") && v){ blocks = atoll(v); ++i; }
        else if(!strcmp(a, "--selftest")){ selftest = 1; }
        else { tnet_usage(); return 2; }
    }
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA && defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
    if(selftest) return cp_tnet_cuda_selftest(device);
    if(!payee || strlen(payee) != 64){
        fprintf(stderr, "[tnet] --payee must be a 64-digit hex key hash\n");
        tnet_usage();
        return 2;
    }
    if(batch < 4 || batch % 4){
        fprintf(stderr, "[tnet] --batch must be a positive multiple of 4\n");
        return 2;
    }
    return cp_tnet_cuda_solo(rpc, payee, device, batch, seconds, blocks);
#else
    (void)rpc; (void)payee; (void)device; (void)batch; (void)seconds; (void)blocks; (void)selftest;
    fprintf(stderr, "[tnet] this build has no CUDA + cuBLAS; rebuild with -DCP_ENABLE_CUDA=ON -DCP_ENABLE_CUBLAS=ON\n");
    return 1;
#endif
}
