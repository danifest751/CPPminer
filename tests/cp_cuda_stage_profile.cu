// Standalone ablation harness. Generated headers are used ONLY for diagnosis.
// Removing hashes/milestones produces invalid mining work; this never connects to a pool.
#include "cp_cutlass_gemm_types.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call) do { cudaError_t e = (call); if(e != cudaSuccess){ \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); std::exit(1); } } while(0)

#ifndef CP_PROFILE_VARIANT
#define CP_PROFILE_VARIANT "full"
#endif

__global__ void fill_input(int8_t* data, size_t bytes, unsigned salt)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(i < bytes){
        unsigned x = (unsigned)i ^ salt;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        data[i] = (int8_t)((int)(x & 127) - 64);
    }
}

static int host_input(size_t i, unsigned salt)
{
    unsigned x = (unsigned)i ^ salt;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return (int)(x & 127) - 64;
}

static uint32_t reference_diagnostic(size_t index, int N)
{
    const size_t cta = index / 256;
    int row0, col0;
    MmaLaneTile128x128::thread_cell_global((int)(cta / (N / 128)) * 128,
        (int)(cta % (N / 128)) * 128, (int)(index % 256), row0, col0);
    const int rows[8] = {0,1,2,3,16,17,18,19};
    const int columns[8] = {0,1,2,3,32,33,34,35};
    int32_t accum[8][8] = {};
    uint32_t words[16] = {};
    for(int step=0; step<K_DIM/R_RANK; ++step){
        uint32_t folded = 0;
        for(int u=0;u<8;++u) for(int v=0;v<8;++v){
            for(int l=step*R_RANK;l<(step+1)*R_RANK;++l)
                accum[u][v] += host_input((size_t)(row0+rows[u])*K_DIM+l,123) *
                               host_input((size_t)(col0+columns[v])*K_DIM+l,456);
            folded ^= (uint32_t)accum[u][v];
        }
        if(!std::strcmp(CP_PROFILE_VARIANT,"final-milestone-only") && step != K_DIM/R_RANK-1) continue;
        uint32_t& word = words[step%16];
        word = (word << 13 | word >> 19) ^ folded;
    }
    uint32_t diagnostic = 0;
    for(auto word : words) diagnostic ^= word;
    return diagnostic;
}

template<typename T> static void measure(int M, int N, int repeats, const char* tile)
{
    const int K = K_DIM;
    const size_t count = (size_t)M * N / 64;
    int8_t *a, *b;
    uint32_t* key;
    int *found, *output, *cols;
    CHECK(cudaMalloc(&a, (size_t)M * K)); CHECK(cudaMalloc(&b, (size_t)N * K));
    CHECK(cudaMalloc(&key, 32)); CHECK(cudaMemset(key, 37, 32));
    CHECK(cudaMalloc(&found, sizeof(int))); CHECK(cudaMemset(found, 0, sizeof(int)));
    CHECK(cudaMalloc(&output, count * sizeof(int))); CHECK(cudaMemset(output, 0, count * sizeof(int)));
    CHECK(cudaMalloc(&cols, sizeof(int))); CHECK(cudaMemset(cols, 0, sizeof(int)));
    fill_input<<<((size_t)M * K + 255) / 256, 256>>>(a, (size_t)M * K, 123);
    fill_input<<<((size_t)N * K + 255) / 256, 256>>>(b, (size_t)N * K, 456);
    CHECK(cudaDeviceSynchronize());
    CpCutlassJackpotLaunch jackpot{};
    jackpot.d_a_key8 = key; jackpot.d_found = found;
    jackpot.d_out_t_rows = output; jackpot.d_out_t_cols = cols;
    cp_cutlass::FusedMilestoneGemmOp<T> op;
    if(op.initialize(M, N, K, M, N, a, b, nullptr, N / 128, count, &jackpot) != cutlass::Status::kSuccess) std::exit(2);
    for(int i = 0; i < 5; ++i){ if(op() != cutlass::Status::kSuccess) std::exit(2); }
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&end));
    std::vector<float> times;
    for(int i = 0; i < repeats; ++i){
        CHECK(cudaEventRecord(start));
        if(op() != cutlass::Status::kSuccess) std::exit(2);
        CHECK(cudaEventRecord(end)); CHECK(cudaEventSynchronize(end));
        float ms; CHECK(cudaEventElapsedTime(&ms, start, end)); times.push_back(ms);
    }
    cudaFuncAttributes attr;
    CHECK(cudaFuncGetAttributes(&attr, cp_cutlass::FusedKernelEntry<T>));
    int blocks;
    const int smem = sizeof(typename T::GemmKernel::SharedStorage);
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, cp_cutlass::FusedKernelEntry<T>, T::GemmKernel::kThreadCount, smem));
    std::vector<int> host(count);
    CHECK(cudaMemcpy(host.data(), output, count * sizeof(int), cudaMemcpyDeviceToHost));
    if(std::strcmp(CP_PROFILE_VARIANT,"full")){
        for(size_t index : {size_t(0), count/2+113, count-1}){
            if((uint32_t)host[index] != reference_diagnostic(index,N)){
                std::fprintf(stderr,"Ablation CPU diagnostic mismatch at %zu\n",index); std::exit(4);
            }
        }
    }
    uint64_t checksum = 0;
    for(auto value : host) checksum = checksum * 1315423911u + (uint32_t)value;
    int hits; CHECK(cudaMemcpy(&hits, found, sizeof(int), cudaMemcpyDeviceToHost));
    if(hits) std::exit(3);
    std::printf("{\"variant\":\"%s\",\"tile\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"registers\":%d,\"local_bytes\":%zu,\"shared_bytes\":%d,\"active_blocks_per_sm\":%d,\"checksum\":%llu,\"milliseconds\":[", CP_PROFILE_VARIANT, tile, M, N, K, attr.numRegs, attr.localSizeBytes, smem, blocks, (unsigned long long)checksum);
    for(size_t i=0;i<times.size();++i) std::printf("%s%.6f", i ? "," : "", times[i]);
    std::puts("]}");
    CHECK(cudaFree(a)); CHECK(cudaFree(b)); CHECK(cudaFree(key)); CHECK(cudaFree(found)); CHECK(cudaFree(output)); CHECK(cudaFree(cols));
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(end));
}

int main(int argc, char** argv)
{
    const int repeats = argc > 1 ? std::atoi(argv[1]) : 30;
    if(repeats < 1 || repeats > 1000) return 1;
    const bool large = argc > 2;
    if(large) measure<cp_cutlass::Gemm256x128TensorOp>(4096, 131072, repeats, "256x128");
    else measure<cp_cutlass::Gemm128x128TensorOp>(4096, 131072, repeats, "128x128");
    return 0;
}
