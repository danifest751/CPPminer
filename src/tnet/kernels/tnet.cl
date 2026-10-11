/* TNet v1 (the Requant coin's work function) on OpenCL: expansion (SHA-256), weight transpose, the int8 GEMM with the
 * requantization in its epilogue, and the ticket hash. Byte-compatible with the CUDA kernels (cp_tnet.cu) and the Rust
 * reference; the host is cp_tnet_ocl.cpp.
 *
 * Build option RQ_DOT, how four int8 products are summed: 1 the integer dot product of OpenCL 3.0
 * (cl_khr_integer_dot_product), 2 AMD's v_dot4_i32_iu8 (RDNA3/4, both operands signed), 3 AMD's v_dot4_i32_i8 (GCN5,
 * RDNA2), 0 written out. The host builds them in turn and keeps the first that agrees with the CPU. -DRQ_DPAS=1 adds
 * rq_gemm_dpas, the GEMM on Intel XMX. */

typedef struct { uint pw[11]; uint b44; } RqPrefix;
typedef struct { uint t[8]; } RqTarget;
typedef struct { uint wd[16]; } RqTail;

__constant uint RQ_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2};

#define ROTR(x, s) rotate((uint)(x), (uint)(32 - (s)))
#define BSWAP(x) as_uint(as_uchar4((uint)(x)).wzyx)

static void rq_compress(uint h[8], const uint w0[16]) {
    uint w[64];
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = w0[i];
#pragma unroll
    for (int i = 16; i < 64; ++i)
        w[i] = w[i - 16] + (ROTR(w[i - 15], 7) ^ ROTR(w[i - 15], 18) ^ (w[i - 15] >> 3)) + w[i - 7] +
               (ROTR(w[i - 2], 17) ^ ROTR(w[i - 2], 19) ^ (w[i - 2] >> 10));
    uint a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
#pragma unroll
    for (int i = 0; i < 64; ++i) {
        const uint t1 = hh + (ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25)) + ((e & f) ^ (~e & g)) + RQ_K[i] + w[i];
        const uint t2 = (ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

static void rq_init(uint h[8]) {
    h[0] = 0x6a09e667; h[1] = 0xbb67ae85; h[2] = 0x3c6ef372; h[3] = 0xa54ff53a;
    h[4] = 0x510e527f; h[5] = 0x9b05688c; h[6] = 0x1f83d9ab; h[7] = 0x5be0cd19;
}

/* out[32 k ..) = SHA256("abacus/expand" || seed || LE32(c0 + k)), k < count; out is 32-byte aligned */
__kernel void rq_expand(RqPrefix px, uint c0, ulong count, __global uint* restrict out, int pack_n) {
    const ulong k = get_global_id(0);
    if (k >= count) return;
    const uint cc = c0 + (uint)k;
    uint w[16], h[8];
    for (int i = 0; i < 11; ++i) w[i] = px.pw[i];
    w[11] = (px.b44 << 24) | ((cc & 0xff) << 16) | (((cc >> 8) & 0xff) << 8) | ((cc >> 16) & 0xff);
    w[12] = ((cc >> 24) << 24) | (0x80u << 16);
    w[13] = 0; w[14] = 0; w[15] = 49 * 8;
    rq_init(h);
    rq_compress(h, w);
    /* pack_n: the batch is A packed for the XMX GEMM (rows of pack_n bytes in 8 x 32-byte chunks), else row-major */
    size_t at = 8 * k;
    if (pack_n) {
        const size_t byte = 32 * k, row = byte / (size_t)pack_n, col = byte % (size_t)pack_n;
        at = (((row / 8) * (size_t)(pack_n / 32) + col / 32) * 256 + (row % 8) * 32) / 4;
    }
    __global uint* o = out + at;
    vstore4((uint4)(BSWAP(h[0]), BSWAP(h[1]), BSWAP(h[2]), BSWAP(h[3])), 0, o);
    vstore4((uint4)(BSWAP(h[4]), BSWAP(h[5]), BSWAP(h[6]), BSWAP(h[7])), 1, o);
}

/* WT[j n + k] = W[k n + j]; work-groups of 32 x 8 over 32 x 32 tiles */
__kernel void rq_transpose(__global const char* restrict W, __global char* restrict WT, int n) {
    __local char tile[32][33];
    const int bx = get_group_id(0) * 32, by = get_group_id(1) * 32;
    const int tx = get_local_id(0), ty = get_local_id(1);
    for (int y = ty; y < 32; y += 8) tile[y][tx] = W[(size_t)(by + y) * n + bx + tx];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int y = ty; y < 32; y += 8) WT[(size_t)(bx + y) * n + by + tx] = tile[tx][y];
}

#if RQ_DOT == 1
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#define DOT4(acc, a, b) ((acc) += dot_4x8packed_ss_int((a), (b)))
#elif RQ_DOT == 2
#define DOT4(acc, a, b) ((acc) = __builtin_amdgcn_sudot4(true, as_int(a), true, as_int(b), (acc), false))
#elif RQ_DOT == 3
#define DOT4(acc, a, b) ((acc) = __builtin_amdgcn_sdot4(as_int(a), as_int(b), (acc), false))
#else
#define DOT4(acc, a, b)                                                                                                    \
    do {                                                                                                                   \
        const char4 x_ = as_char4(a), y_ = as_char4(b);                                                                    \
        (acc) += (int)x_.x * y_.x + (int)x_.y * y_.y + (int)x_.z * y_.z + (int)x_.w * y_.w;                                \
    } while (0)
#endif

/* XO (rows x n) = clamp((X WT^T M + 2^23) >> 24, -128, 127): X rows x n and WT n x n, both row-major int8, n a multiple
 * of 64. Work-groups of 16 x 16 cover 64 x 64 outputs, each work-item 4 x 4 (rows ty + 16 i, columns tx + 16 j); k in
 * steps of 64 through local memory. */
#define TILE 64
#define KS 16 /* ints (4 int8 each) per k step */
__kernel __attribute__((reqd_work_group_size(16, 16, 1))) void rq_gemm_rq(__global const uint* restrict X,
                                                                        __global const uint* restrict WT,
                                                                        __global char* restrict XO, int rows, int n,
                                                                        int mult) {
    __local uint As[TILE][KS + 1];
    __local uint Bs[TILE][KS + 1];
    const int tx = get_local_id(0), ty = get_local_id(1), t = ty * 16 + tx;
    const int row0 = get_group_id(1) * TILE, col0 = get_group_id(0) * TILE;
    const int nw = n / 4; /* ints per row */
    int acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) acc[i][j] = 0;
    for (int k0 = 0; k0 < nw; k0 += KS) {
        /* 64 x 16 ints of each operand, four per work-item */
        for (int q = 0; q < 4; ++q) {
            const int idx = t + 256 * q, r = idx / KS, kk = idx % KS;
            const int gr = row0 + r;
            As[r][kk] = gr < rows ? X[(size_t)gr * nw + k0 + kk] : 0u;
            Bs[r][kk] = WT[(size_t)(col0 + r) * nw + k0 + kk];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            uint a[4], b[4];
            for (int i = 0; i < 4; ++i) a[i] = As[ty + 16 * i][kk];
            for (int j = 0; j < 4; ++j) b[j] = Bs[tx + 16 * j][kk];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) DOT4(acc[i][j], a[i], b[j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i = 0; i < 4; ++i) {
        const int gr = row0 + ty + 16 * i;
        if (gr >= rows) continue;
        for (int j = 0; j < 4; ++j) {
            const long v = ((long)acc[i][j] * mult + (1L << 23)) >> 24;
            XO[(size_t)gr * n + col0 + tx + 16 * j] = (char)(v < -128 ? -128 : (v > 127 ? 127 : v));
        }
    }
}

#if RQ_DPAS
/* The same product on Intel XMX (cl_intel_subgroup_matrix_multiply_accumulate, Xe-HPG: sub-group size 8). One DPAS is
 * M 8 x N 8 x K 32: a.s<m> = row m, k bytes 4 lane .. 4 lane + 3; b = column `lane`, b.s<i> = k bytes 4 i .. 4 i + 3;
 * result.s<m> = C[row m][column lane] (the layout the Pearl DPAS path of this miner checked on an Arc A380).
 *
 * Both operands are kept packed so that each DPAS operand is one 256-byte block read (intel_sub_group_block_read8:
 * lane l gets dwords l, l + 8, ..., l + 56):
 *   A (activations), chunk (row block rb of 8 rows, k block kb of 32): at ((rb (n/32) + kb) 256), dword m 8 + lane =
 *     row 8 rb + m, k bytes 32 kb + 4 lane .. + 3 (the chunk is the 8 x 32 bytes, row after row);
 *   B (weights), chunk (column block cb of 8, k block kb): at ((cb (n/32) + kb) 256), dword i 8 + lane = column 8 cb + lane,
 *     k bytes 32 kb + 4 i .. + 3.
 * The expansion writes A packed, every layer but the last writes its output packed, the last row-major for the ticket
 * hash. Rows a multiple of 8, n a multiple of 128.
 *
 * A sub-group owns 32 rows x 16 columns (4 x 2 DPAS blocks); a work-group of 8 sub-groups 32 rows x 128 columns;
 * work-groups next to each other take the next rows of the same columns, so the weights they share stay in the cache. */
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

/* B packed from W row-major (W[k n + j], as the expansion writes it): one dword per work-item */
__kernel void rq_pack_w(__global const uchar* restrict W, __global uint* restrict P, int n) {
    const size_t d = get_global_id(0);
    if (d >= (size_t)n * n / 4) return;
    const int kbs = n / 32;
    const size_t chunk = d / 64;
    const int within = (int)(d % 64), i = within / 8, lane = within % 8;
    const int j = (int)(chunk / kbs) * 8 + lane, k = (int)(chunk % kbs) * 32 + 4 * i;
    P[d] = (uint)W[(size_t)k * n + j] | ((uint)W[(size_t)(k + 1) * n + j] << 8) | ((uint)W[(size_t)(k + 2) * n + j] << 16) |
           ((uint)W[(size_t)(k + 3) * n + j] << 24);
}

__kernel __attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(64, 1, 1)))
void rq_gemm_dpas(__global const uint* restrict A, __global const uint* restrict B, __global char* restrict XO, int rows,
                  int n, int mult, int out_packed) {
    const int lane = get_sub_group_local_id(), sg = get_sub_group_id();
    const int r0 = get_group_id(0) * 32, c0 = get_group_id(1) * 128 + sg * 16;
    const int kbs = n / 32;
    int8 acc[4][2];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) acc[i][j] = (int8)(0);
    __global const uint* ap[4];
    __global const uint* bp[2];
    for (int i = 0; i < 4; ++i) ap[i] = A + (size_t)((r0 / 8 + i) * kbs) * 64;
    for (int j = 0; j < 2; ++j) bp[j] = B + (size_t)((c0 / 8 + j) * kbs) * 64;
    const int blocks = min(4, (rows - r0 + 7) / 8); /* row blocks of this tile that exist */
    for (int kb = 0; kb < kbs; ++kb) {
        int8 a[4], b[2];
#pragma unroll
        for (int i = 0; i < 4; ++i) a[i] = i < blocks ? as_int8(intel_sub_group_block_read8(ap[i] + kb * 64)) : (int8)(0);
#pragma unroll
        for (int j = 0; j < 2; ++j) b[j] = as_int8(intel_sub_group_block_read8(bp[j] + kb * 64));
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j) acc[i][j] = intel_sub_group_i8_i8_matrix_mad_k32(a[i], b[j], acc[i][j]);
    }
    for (int i = 0; i < blocks; ++i)
        for (int j = 0; j < 2; ++j) {
            const int col = c0 + 8 * j + lane;
            int v8[8];
            vstore8(acc[i][j], 0, v8);
            for (int m = 0; m < 8; ++m) {
                const int row = r0 + 8 * i + m;
                const long v = ((long)v8[m] * mult + (1L << 23)) >> 24;
                const char q = (char)(v < -128 ? -128 : (v > 127 ? 127 : v));
                if (out_packed)
                    XO[((size_t)(row / 8) * kbs + col / 32) * 256 + (row % 8) * 32 + col % 32] = q;
                else
                    XO[(size_t)row * n + col] = q;
            }
        }
}
#endif

/* Ticket (r0 + i, c): SHA256(piece || 0x54 || hd || LE64 nonce || LE32 row || LE32 c) <= target; w % 64 == 0. The fixed
 * last block comes in `tail`; each work-item fills in row and c (bytes 41..48). */
__kernel void rq_tickets(__global const uint* restrict X, int n, int rows, int w, uint r0, RqTail tail, RqTarget tg,
                         volatile __global uint* found, __global ulong* hits) {
    const int per_row = n / w;
    const size_t t = get_global_id(0);
    if (t >= (size_t)rows * per_row) return;
    const int i = (int)(t / per_row), c = (int)(t % per_row);
    const uint row = r0 + (uint)i;
    __global const uint* piece = X + ((size_t)i * n + (size_t)c * w) / 4;
    uint h[8], wd[16];
    rq_init(h);
    for (int blk = 0; blk < w / 64; ++blk) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint4 v = vload4(blk * 4 + q, piece);
            wd[4 * q] = BSWAP(v.x);
            wd[4 * q + 1] = BSWAP(v.y);
            wd[4 * q + 2] = BSWAP(v.z);
            wd[4 * q + 3] = BSWAP(v.w);
        }
        rq_compress(h, wd);
    }
#pragma unroll
    for (int q = 0; q < 16; ++q) wd[q] = tail.wd[q];
    /* bytes 40..51: nonce byte 7 | LE32 row | LE32 c | 0x80 0 0 */
    const uint rb = BSWAP(row), cb = BSWAP((uint)c);
    wd[10] = (wd[10] & 0xff000000u) | (rb >> 8);
    wd[11] = (rb << 24) | (cb >> 8);
    wd[12] = (cb << 24) | 0x00800000u;
    rq_compress(h, wd);
    for (int q = 0; q < 8; ++q) {
        if (h[q] < tg.t[q]) break;
        if (h[q] > tg.t[q]) return;
    }
    const uint k = atomic_inc(found);
    if (k < 16) hits[k] = ((ulong)i << 32) | (uint)c;
}
