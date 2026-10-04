// Independent CPU recurrence and reference BLAKE3 for every GPU prefix/state.
// Build through scripts/build_jackpot_register_profile.py so each generated
// fold implementation is checked separately, with no production source edits.
#include "cp_cutlass_jackpot.cuh"
#include "blake3.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call) do { cudaError_t e = (call); if(e != cudaSuccess){ \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); std::exit(1); } } while(0)

__host__ __device__ static uint32_t sample(unsigned stream, unsigned step)
{
    if(stream % 8 == 0) return 0;
    if(stream % 8 == 1) return 0xffffffffu;
    uint32_t x = stream * 0x9e3779b9u + step * 0x85ebca6bu;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

__global__ static void prefixes(uint32_t* states, uint32_t* digests, int streams, int steps)
{
    const int stream = blockIdx.x * blockDim.x + threadIdx.x;
    if(stream >= streams) return;
    uint32_t words[16], key[8];
    #pragma unroll
    for(int word=0; word<16; ++word) words[word] = stream%2 ? sample(stream,word+1000) : 0;
    #pragma unroll
    for(int word=0; word<8; ++word) key[word] = sample(stream+2,word+2000);
    for(int step=0; step<steps; ++step){
        cp_cutlass_jackpot_fold_step(words,step,sample(stream,step));
        const size_t index = (size_t)stream*steps + step;
        #pragma unroll
        for(int word=0; word<16; ++word) states[index*16+word] = words[word];
        uint32_t digest[8];
        b3_compress64(key,words,digest);
        #pragma unroll
        for(int word=0; word<8; ++word) digests[index*8+word] = digest[word];
    }
}

int main()
{
    const int streams=1024, steps=64;
    const size_t count=(size_t)streams*steps;
    uint32_t *gpu_states, *gpu_digests;
    CHECK(cudaMalloc(&gpu_states,count*16*sizeof(uint32_t)));
    CHECK(cudaMalloc(&gpu_digests,count*8*sizeof(uint32_t)));
    prefixes<<<(streams+127)/128,128>>>(gpu_states,gpu_digests,streams,steps);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> states(count*16),digests(count*8);
    CHECK(cudaMemcpy(states.data(),gpu_states,states.size()*4,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(digests.data(),gpu_digests,digests.size()*4,cudaMemcpyDeviceToHost));
    for(int stream=0; stream<streams; ++stream){
        uint32_t words[16], key[8];
        for(int word=0; word<16; ++word) words[word] = stream%2 ? sample(stream,word+1000) : 0;
        for(int word=0; word<8; ++word) key[word] = sample(stream+2,word+2000);
        for(int step=0; step<steps; ++step){
            const int word=step&15;
            const uint32_t prior=words[word];
            words[word]=((prior*8192u) | (prior/524288u)) ^ sample(stream,step);
            const size_t index=(size_t)stream*steps+step;
            if(std::memcmp(words,states.data()+index*16,sizeof(words))){
                std::fprintf(stderr,"State mismatch stream=%d step=%d\n",stream,step); return 2;
            }
            uint8_t expected[32];
            blake3_hasher hasher;
            blake3_hasher_init_keyed(&hasher,(const uint8_t*)key);
            blake3_hasher_update(&hasher,words,sizeof(words));
            blake3_hasher_finalize(&hasher,expected,sizeof(expected));
            if(std::memcmp(expected,digests.data()+index*8,sizeof(expected))){
                std::fprintf(stderr,"BLAKE3 mismatch stream=%d step=%d\n",stream,step); return 3;
            }
        }
    }
    CHECK(cudaFree(gpu_states)); CHECK(cudaFree(gpu_digests));
    std::printf("{\"streams\":%d,\"prefixes_per_stream\":%d,\"state_words_checked\":%zu,\"digests_checked\":%zu,\"passed\":true}\n",streams,steps,count*16,count);
    return 0;
}
