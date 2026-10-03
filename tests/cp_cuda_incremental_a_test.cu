// Standalone GPU test; see docs/cuda_incremental_a_experiment.md for build/run.
#include "../src/cuda/cp_incremental_a.cuh"
#include "blake3.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call) do { cudaError_t e = (call); if(e != cudaSuccess){ \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); std::exit(1); } } while(0)

static void run_case(int leaves, bool quick)
{
    const size_t bytes = (size_t)leaves * D_B3_CHUNK;
    // Small cases exercise repeated updates to the same leaf/parent.
    const int cols = 1024, rows = (int)(bytes / cols);
    const int count = (leaves + CP_MT_THREADS - 1) / CP_MT_THREADS;
    int8_t* signal;
    uint8_t *tree, *key_gpu, *subroots, *root, *reference_subroots;
    unsigned* dirty;
    uint8_t key[32];
    for(int i = 0; i < 32; ++i) key[i] = (uint8_t)(i * 7 + leaves);
    CHECK(cudaMalloc(&signal, bytes));
    CHECK(cudaMalloc(&tree, (size_t)2 * leaves * D_B3_OUT));
    CHECK(cudaMalloc(&dirty, (size_t)2 * leaves * sizeof(unsigned)));
    CHECK(cudaMalloc(&key_gpu, 32));
    CHECK(cudaMalloc(&subroots, (size_t)count * 32));
    CHECK(cudaMalloc(&reference_subroots, (size_t)count * 32));
    CHECK(cudaMalloc(&root, 32));
    CHECK(cudaMemset(signal, 0, bytes));
    CHECK(cudaMemset(dirty, 1, (size_t)2 * leaves * sizeof(unsigned)));
    CHECK(cudaMemcpy(key_gpu, key, 32, cudaMemcpyHostToDevice));
    std::vector<uint8_t> host(bytes), actual_subroots((size_t)count * 32), expected_subroots((size_t)count * 32);
    const int rounds = quick ? 8 : (leaves == 524288 ? 4 : 24);
    for(int round = 0; round < rounds; ++round){
        // Re-key the persistent signal halfway through without losing its bytes.
        if(round == rounds / 2){
            key[0] ^= 0xa5;
            CHECK(cudaMemcpy(key_gpu, key, 32, cudaMemcpyHostToDevice));
            CHECK(cudaMemset(dirty, 1, (size_t)2 * leaves * sizeof(unsigned)));
        }
        // Round zero builds the all-zero tree; later rounds mutate/mark paths.
        if(round)
            cp_sparse_a_update_kernel<<<(cols + 255) / 256, 256>>>(
                signal, rows, cols, (uint64_t)round * 0x9e3779b97f4a7c15ULL, dirty, leaves);
        cp_incremental_leaves_kernel<<<(leaves + 255) / 256, 256>>>(
            (const uint8_t*)signal, bytes, key_gpu, tree, dirty, leaves);
        for(int level = leaves / 2; level; level /= 2)
            cp_incremental_parents_kernel<<<(level + 255) / 256, 256>>>(
                key_gpu, tree, dirty, level, level);
        cp_incremental_publish_kernel<<<(count + 255) / 256, 256>>>(
            key_gpu, tree, leaves, subroots, root);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(host.data(), signal, bytes, cudaMemcpyDeviceToHost));
        uint8_t expected[32], actual[32];
        blake3_hasher hasher;
        blake3_hasher_init_keyed(&hasher, key);
        blake3_hasher_update(&hasher, host.data(), bytes);
        blake3_hasher_finalize(&hasher, expected, 32);
        CHECK(cudaMemcpy(actual, root, 32, cudaMemcpyDeviceToHost));
        if(std::memcmp(actual, expected, 32)){
            std::fprintf(stderr, "CPU root mismatch: leaves=%d round=%d\n", leaves, round);
            std::exit(1);
        }
        cp_keyed_chunk_roots_kernel<<<count, CP_MT_THREADS, CP_MT_SMEM_BYTES>>>(
            (const uint8_t*)signal, bytes, bytes, key_gpu, reference_subroots, leaves);
        CHECK(cudaMemcpy(actual_subroots.data(), subroots, (size_t)count * 32, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(expected_subroots.data(), reference_subroots, (size_t)count * 32, cudaMemcpyDeviceToHost));
        if(actual_subroots != expected_subroots){
            std::fprintf(stderr, "Full GPU subroot mismatch: leaves=%d round=%d\n", leaves, round);
            std::exit(1);
        }
        for(auto byte : host){
            const int value = (int)(int8_t)byte;
            if(value < -64 || value > 63) std::exit(1);
        }
        std::vector<unsigned> flags((size_t)2 * leaves);
        CHECK(cudaMemcpy(flags.data(), dirty, flags.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
        for(size_t node = 1; node < flags.size(); ++node)
            if(flags[node]){ std::fprintf(stderr, "Dirty node left after hashing\n"); std::exit(1); }
    }
    CHECK(cudaFree(signal)); CHECK(cudaFree(tree)); CHECK(cudaFree(dirty));
    CHECK(cudaFree(key_gpu)); CHECK(cudaFree(subroots)); CHECK(cudaFree(root));
    CHECK(cudaFree(reference_subroots));
    std::printf("PASS leaves=%d rounds=%d: CPU roots, full GPU subroots, re-key, range, dirty flags\n", leaves, rounds);
    std::fflush(stdout);
}

int main(int argc, char** argv)
{
    const bool quick = argc == 2 && !std::strcmp(argv[1], "--quick");
    if(argc > 1 && !quick) return 1;
    for(int leaves : {2, 4, 128, 256, 512, 4096, 32768, 524288})
        if(!quick || leaves <= 4096) run_case(leaves, quick);
    return 0;
}
