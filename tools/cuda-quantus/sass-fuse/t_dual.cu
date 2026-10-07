// t_dual.cu: observe the two carry-out predicates of a 3-input IADD3 on sm_75.
// ptxas emits IADD3 Rd, P0, P1, a, b, c ; IADD3.X Rc, RZ, RZ, RZ, P0, P1 for (u64)a+b+c >> 32.
// The SASS is patched afterwards so the second instruction reads only P0 or only P1.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
typedef uint32_t u32; typedef uint64_t u64;
__global__ void k(const u32* a, const u32* b, const u32* c, u32* lo, u32* cnt, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 s = (u64)a[i] + b[i] + c[i];
    lo[i] = (u32)s;
    cnt[i] = (u32)(s >> 32);
}
int main()
{
    const int n = 12;
    u32 A[n] = {0x80000000u, 0x80000000u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x7FFFFFFFu, 0x40000000u, 0x80000000u, 0x00000001u, 0xC0000000u, 0xFFFFFFFFu, 0x00000000u, 0x80000001u};
    u32 B[n] = {0x80000000u, 0x00000000u, 0xFFFFFFFFu, 0x00000001u, 0x7FFFFFFFu, 0x40000000u, 0x7FFFFFFFu, 0xFFFFFFFFu, 0xC0000000u, 0x00000000u, 0xFFFFFFFFu, 0x80000001u};
    u32 C[n] = {0x00000000u, 0x80000000u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000002u, 0x40000000u, 0x00000001u, 0x00000000u, 0xC0000000u, 0xFFFFFFFFu, 0x00000001u, 0x80000001u};
    u32 *da, *db, *dc, *dlo, *dcnt;
    cudaMalloc(&da, n * 4); cudaMalloc(&db, n * 4); cudaMalloc(&dc, n * 4); cudaMalloc(&dlo, n * 4); cudaMalloc(&dcnt, n * 4);
    cudaMemcpy(da, A, n * 4, cudaMemcpyHostToDevice); cudaMemcpy(db, B, n * 4, cudaMemcpyHostToDevice); cudaMemcpy(dc, C, n * 4, cudaMemcpyHostToDevice);
    k<<<1, 32>>>(da, db, dc, dlo, dcnt, n);
    u32 lo[n], cnt[n];
    cudaMemcpy(lo, dlo, n * 4, cudaMemcpyDeviceToHost); cudaMemcpy(cnt, dcnt, n * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        u64 ab = (u64)A[i] + B[i];
        u32 cab = (u32)(ab >> 32), cabc = (u32)(((ab & 0xFFFFFFFFu) + C[i]) >> 32);
        u32 maj = ((A[i] >> 31) + (B[i] >> 31) + (C[i] >> 31)) >= 2;
        u64 tot = (u64)A[i] + B[i] + C[i];
        printf("a=%08x b=%08x c=%08x lo=%08x out=%u  total=%llu chain(c_ab=%u c_abc=%u) maj31=%u\n", A[i], B[i], C[i], lo[i], cnt[i],
               (unsigned long long)(tot >> 32), cab, cabc, maj);
    }
    return 0;
}
