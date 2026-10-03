#ifndef CP_INCREMENTAL_A_CUH
#define CP_INCREMENTAL_A_CUH

#include "cp_gpu_gen.cuh"
#include "cp_merkle_tree.cuh"

/* Experimental signal-A cache. The complete binary tree stores non-root
 * BLAKE3 CVs; ROOT is applied separately when publishing the matrix digest. */
struct CpIncrementalA {
    int8_t* signal = nullptr;
    uint8_t* tree = nullptr;
    unsigned* dirty = nullptr;
    int leaves = 0;
    uint8_t job_key[32] = {};
    bool ready = false;
};

/* Different columns always address different bytes, even if rows coincide.
 * Dirty paths are marked before any hashing starts on the same stream. */
__global__ void cp_sparse_a_update_kernel(int8_t* signal, int rows, int cols,
                                         uint64_t seed, unsigned* dirty, int leaves)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col >= cols) return;
    const uint64_t random = cp_splitmix64(seed ^ (uint64_t)col * 0x9E3779B97F4A7C15ULL);
    const int row = (int)((uint32_t)random % (unsigned)rows);
    const size_t pos = (size_t)row * cols + col;
    signal[pos] = (int8_t)((int)((random >> 32) & 127) - 64);
    if(dirty){
        unsigned node = (unsigned)leaves + (unsigned)(pos / D_B3_CHUNK);
        while(node){
            // Another updater that marked this node will mark its ancestors.
            if(atomicExch(dirty + node, 1u)) break;
            node >>= 1;
        }
    }
}

__global__ void cp_incremental_leaves_kernel(const uint8_t* signal, size_t bytes,
                                            const uint8_t* key, uint8_t* tree,
                                            unsigned* dirty, int leaves)
{
    const int leaf = blockIdx.x * blockDim.x + threadIdx.x;
    if(leaf >= leaves || !dirty[leaves + leaf]) return;
    d_b3_keyed_chunk_cv_glob(key, (uint64_t)leaf, signal,
                            (size_t)leaf * D_B3_CHUNK, bytes, D_B3_CHUNK,
                            tree + (size_t)(leaves + leaf) * D_B3_OUT);
    dirty[leaves + leaf] = 0;
}

__global__ void cp_incremental_parents_kernel(const uint8_t* key_bytes,
                                             uint8_t* tree, unsigned* dirty,
                                             int first, int count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= count) return;
    const int node = first + index;
    if(!dirty[node]) return;
    uint32_t key[8], left[8], right[8], result[8];
    d_mt_load_key(key_bytes, key);
    for(int i = 0; i < 8; ++i){
        left[i] = d_b3_load32(tree + (size_t)(2 * node) * D_B3_OUT + 4 * i);
        right[i] = d_b3_load32(tree + (size_t)(2 * node + 1) * D_B3_OUT + 4 * i);
    }
    d_mt_parent_cv(key, left, right, false, result);
    for(int i = 0; i < 8; ++i)
        d_b3_store32(tree + (size_t)node * D_B3_OUT + 4 * i, result[i]);
    dirty[node] = 0;
}

__global__ void cp_incremental_publish_kernel(const uint8_t* key_bytes,
                                             const uint8_t* tree, int leaves,
                                             uint8_t* subroots, uint8_t* root)
{
    const int sub = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = (leaves + CP_MT_THREADS - 1) / CP_MT_THREADS;
    const int first = leaves < CP_MT_THREADS ? 1 : leaves / CP_MT_THREADS;
    if(sub < count){
        for(int i = 0; i < D_B3_OUT; ++i)
            subroots[(size_t)sub * D_B3_OUT + i] = tree[(size_t)(first + sub) * D_B3_OUT + i];
    }
    if(sub == 0){
        uint32_t key[8], left[8], right[8], result[8];
        d_mt_load_key(key_bytes, key);
        for(int i = 0; i < 8; ++i){
            left[i] = d_b3_load32(tree + 2 * D_B3_OUT + 4 * i);
            right[i] = d_b3_load32(tree + 3 * D_B3_OUT + 4 * i);
        }
        d_mt_parent_cv(key, left, right, true, result);
        for(int i = 0; i < 8; ++i) d_b3_store32(root + 4 * i, result[i]);
    }
}

#endif
