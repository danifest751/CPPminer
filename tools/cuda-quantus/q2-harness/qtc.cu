// Standalone CUDA Quantus (Poseidon2 / Goldilocks) mining kernel: verify + bench.
//   qtc test            field-op edge tests + hash cross-check against the CPU reference
//   qtc bench SEC [TPB] [NPT]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>
#include "qpow/poseidon2.hpp"

typedef uint32_t u32;
typedef uint64_t u64;
#define DEV __device__ __forceinline__

#ifndef QTC_NOEXT
#define QTC_NOEXT 0
#endif
#ifndef QTC_SYNC
#define QTC_SYNC 0
#endif
#ifndef QTC_TPB
#define QTC_TPB 256
#endif

// ---------------------------------------------------------------- field ops
// Elements are any 64-bit value (not necessarily < p); every op is exact mod p.

DEV u64 mk64(u32 lo, u32 hi) { u64 r; asm("mov.b64 %0, {%1,%2};" : "=l"(r) : "r"(lo), "r"(hi)); return r; }
DEV void sp64(u64 x, u32& lo, u32& hi) { asm("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(x)); }

// (r3:r2:r1:r0) -> 64-bit value congruent mod p (not necessarily < p).
// V = lo + r2*2^64 + r3*2^96 = lo + r2*EPS - r3 (mod p). t = lo + r2*EPS with carry c; if c,
// 2^64 = EPS turns -r3 into +EPS - r3 = c*2^32 - (r3 + c), which cannot overflow. Exact except
// when r2 == 0 and lo < r3 with no carry (probability ~2^-64 for hash data); a miss there only
// changes the candidate filter, and every candidate is re-hashed on the host.
// Precondition r3 <= 2^32 - 2: always true for a product; for a*b + c it holds when
// b < 2^64 - 2^32 (the MDS diagonal is below 0xF3FB...).
// EPS = 0xFFFFFFFF from constant memory: as a literal, ptxas turns hi(r2*EPS) into IMAD.HI,
// which is slow on Turing; through a constant bank operand it stays one IMAD.WIDE.
__constant__ u32 c_eps;

#ifndef QTC_RED_ALU
#define QTC_RED_ALU 0
#endif
DEV u64 red128(u32 r0, u32 r1, u32 r2, u32 r3)
{
    u32 z0, z1;
#if QTC_RED_ALU
    // r2*EPS = (r2 << 32) - r2 from adds: keeps the reduction on the ALU pipe, which Turing
    // has spare, instead of a wide multiply on the busier FMA pipe.
    asm("{.reg .u32 u, h, t0, t1, rc, tc;\n\t"
        "sub.cc.u32     u, 0, %2;\n\t"
        "subc.u32       h, %2, 0;\n\t"
        "add.cc.u32     t0, %3, u;\n\t"
        "addc.cc.u32    t1, %4, h;\n\t"
        "addc.u32       rc, %5, 0;\n\t"
        "addc.u32       tc, t1, 0;\n\t"
        "sub.cc.u32     %0, t0, rc;\n\t"
        "subc.u32       %1, tc, 0;}"
        : "=r"(z0), "=r"(z1) : "r"(r2), "r"(r0), "r"(r1), "r"(r3));
#else
    asm("{.reg .u32 t0, t1, rc, tc;\n\t"
        "mad.lo.cc.u32  t0, %2, %6, %3;\n\t"
        "madc.hi.cc.u32 t1, %2, %6, %4;\n\t"
        "addc.u32       rc, %5, 0;\n\t"
        "addc.u32       tc, t1, 0;\n\t"
        "sub.cc.u32     %0, t0, rc;\n\t"
        "subc.u32       %1, tc, 0;}"
        : "=r"(z0), "=r"(z1) : "r"(r2), "r"(r0), "r"(r1), "r"(r3), "r"(c_eps));
#endif
    return mk64(z0, z1);
}

// a*b (+ c0 + c1*2^32 when FMA) as 128 bits: a0*b0, a1*b1, a0*b1 as wide products, a1*b0
// accumulated onto a0*b1 with one carry, then three adds.
template <bool FMA>
DEV void mulk(u64 a, u64 b, u32 c0, u32 c1, u32& r0, u32& r1, u32& r2, u32& r3)
{
    u32 a0, a1, b0, b1; sp64(a, a0, a1); sp64(b, b0, b1);
    u64 P, Q, T;
    if (FMA) {
        asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(P) : "r"(a0), "r"(b0), "l"((u64)c0));
        asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(T) : "r"(a0), "r"(b1), "l"((u64)c1));
    } else {
        asm("mul.wide.u32 %0, %1, %2;" : "=l"(P) : "r"(a0), "r"(b0));
        asm("mul.wide.u32 %0, %1, %2;" : "=l"(T) : "r"(a0), "r"(b1));
    }
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(Q) : "r"(a1), "r"(b1));
    u32 p0, p1, q0, q1, t0, t1; sp64(P, p0, p1); sp64(Q, q0, q1); sp64(T, t0, t1);
    asm("{.reg .u32 m0, m1, mc;\n\t"
        "mad.lo.cc.u32  m0, %3, %4, %5;\n\t"
        "madc.hi.cc.u32 m1, %3, %4, %6;\n\t"
        "addc.u32       mc, 0, 0;\n\t"
        "add.cc.u32     %0, %7, m0;\n\t"
        "addc.cc.u32    %1, %8, m1;\n\t"
        "addc.u32       %2, %9, mc;}"
        : "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(a1), "r"(b0), "r"(t0), "r"(t1), "r"(p1), "r"(q0), "r"(q1));
    r0 = p0;
}

DEV void mul128(u64 a, u64 b, u32& r0, u32& r1, u32& r2, u32& r3) { mulk<false>(a, b, 0, 0, r0, r1, r2, r3); }
#ifndef QTC_SQR3
#define QTC_SQR3 1
#endif
// a^2 from three wide products: a0^2 + 2*a0*a1*2^32 + a1^2*2^64.
DEV void sqr128(u64 a, u32& r0, u32& r1, u32& r2, u32& r3)
{
#if QTC_SQR3 == 2
    /* 2*a0*a1 by shifting the 64-bit product left once (65 bits: d2:d1:d0), then one 3-word add */
    u32 a0, a1; sp64(a, a0, a1);
    u64 P, Q, T;
    asm("mul.wide.u32 %0, %1, %1;" : "=l"(P) : "r"(a0));
    asm("mul.wide.u32 %0, %1, %1;" : "=l"(Q) : "r"(a1));
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(T) : "r"(a0), "r"(a1));
    u32 p0, p1, q0, q1, t0, t1; sp64(P, p0, p1); sp64(Q, q0, q1); sp64(T, t0, t1);
    u32 d0, d1, d2;
    asm("shl.b32 %0, %1, 1;" : "=r"(d0) : "r"(t0));
    asm("shf.l.wrap.b32 %0, %1, %2, 1;" : "=r"(d1) : "r"(t0), "r"(t1));
    asm("shr.b32 %0, %1, 31;" : "=r"(d2) : "r"(t1));
    asm("add.cc.u32  %0, %3, %6;\n\t"
        "addc.cc.u32 %1, %4, %7;\n\t"
        "addc.u32    %2, %5, %8;"
        : "=r"(r1), "=r"(r2), "=r"(r3) : "r"(p1), "r"(q0), "r"(q1), "r"(d0), "r"(d1), "r"(d2));
    r0 = p0;
#elif QTC_SQR3
    u32 a0, a1; sp64(a, a0, a1);
    u64 P, Q, T;
    asm("mul.wide.u32 %0, %1, %1;" : "=l"(P) : "r"(a0));
    asm("mul.wide.u32 %0, %1, %1;" : "=l"(Q) : "r"(a1));
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(T) : "r"(a0), "r"(a1));
    u32 p0, p1, q0, q1, t0, t1; sp64(P, p0, p1); sp64(Q, q0, q1); sp64(T, t0, t1);
    asm("add.cc.u32  %0, %3, %6;\n\t"
        "addc.cc.u32 %1, %4, %7;\n\t"
        "addc.u32    %2, %5, 0;\n\t"
        "add.cc.u32  %0, %0, %6;\n\t"
        "addc.cc.u32 %1, %1, %7;\n\t"
        "addc.u32    %2, %2, 0;"
        : "=r"(r1), "=r"(r2), "=r"(r3) : "r"(p1), "r"(q0), "r"(q1), "r"(t0), "r"(t1));
    r0 = p0;
#else
    mulk<false>(a, a, 0, 0, r0, r1, r2, r3);
#endif
}
// a*b + c as 128 bits. c rides as the 64-bit addend of a0*b0 (one wide multiply with carry k),
// a0*b1 + a1*b0 is the middle term with carry mc, and k joins the r2 column.
DEV void fma128(u64 a, u64 b, u64 c, u32& r0, u32& r1, u32& r2, u32& r3)
{
    u32 a0, a1, b0, b1, c0, c1; sp64(a, a0, a1); sp64(b, b0, b1); sp64(c, c0, c1);
    u64 T, Q;
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(T) : "r"(a0), "r"(b1));
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(Q) : "r"(a1), "r"(b1));
    u32 t0, t1, q0, q1; sp64(T, t0, t1); sp64(Q, q0, q1);
    asm("{.reg .u32 k, m0, m1, mc;\n\t"
        "mad.lo.cc.u32  %0, %4, %6, %8;\n\t"
        "madc.hi.cc.u32 %1, %4, %6, %9;\n\t"
        "addc.u32       k, 0, 0;\n\t"
        "mad.lo.cc.u32  m0, %5, %6, %10;\n\t"
        "madc.hi.cc.u32 m1, %5, %6, %11;\n\t"
        "addc.u32       mc, 0, 0;\n\t"
        "add.cc.u32     %1, %1, m0;\n\t"
        "addc.cc.u32    %2, %12, m1;\n\t"
        "addc.u32       %3, %13, mc;\n\t"
        "add.cc.u32     %2, %2, k;\n\t"
        "addc.u32       %3, %3, 0;}"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(t0), "r"(t1), "r"(q0), "r"(q1));
}

DEV u64 gmul(u64 a, u64 b) { u32 r0, r1, r2, r3; mul128(a, b, r0, r1, r2, r3); return red128(r0, r1, r2, r3); }
DEV u64 gsqr(u64 a) { u32 r0, r1, r2, r3; sqr128(a, r0, r1, r2, r3); return red128(r0, r1, r2, r3); }
DEV u64 gfma(u64 a, u64 b, u64 c) { u32 r0, r1, r2, r3; fma128(a, b, c, r0, r1, r2, r3); return red128(r0, r1, r2, r3); }

// Lazy sum: value = (hi:lo) + t*2^64.
struct W { u32 lo, hi, t; };

DEV W w2(u64 a, u64 b)
{
    W r; u32 a0, a1, b0, b1; sp64(a, a0, a1); sp64(b, b0, b1);
    asm("add.cc.u32 %0, %3, %5;\n\t addc.cc.u32 %1, %4, %6;\n\t addc.u32 %2, 0, 0;"
        : "=r"(r.lo), "=r"(r.hi), "=r"(r.t) : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    return r;
}
DEV W wadd(W a, W b)
{
    W r;
    asm("add.cc.u32 %0, %3, %6;\n\t addc.cc.u32 %1, %4, %7;\n\t addc.u32 %2, %5, %8;"
        : "=r"(r.lo), "=r"(r.hi), "=r"(r.t)
        : "r"(a.lo), "r"(a.hi), "r"(a.t), "r"(b.lo), "r"(b.hi), "r"(b.t));
    return r;
}
DEV W wadd64(W a, u64 b)
{
    W r; u32 b0, b1; sp64(b, b0, b1);
    asm("add.cc.u32 %0, %3, %6;\n\t addc.cc.u32 %1, %4, %7;\n\t addc.u32 %2, %5, 0;"
        : "=r"(r.lo), "=r"(r.hi), "=r"(r.t) : "r"(a.lo), "r"(a.hi), "r"(a.t), "r"(b0), "r"(b1));
    return r;
}
// (hi:lo) + t*2^64, t < 2^31 -> 64-bit. w = t*EPS + lo < 2^63; adding hi*2^32 carries c at most
// once, and then the high word is < 2^31, so +EPS cannot carry again.
DEV u64 wred(W a)
{
    u32 w0, w1, c;
    asm("mad.lo.cc.u32  %0, %3, %6, %4;\n\t"
        "madc.hi.u32    %1, %3, %6, 0;\n\t"
        "add.cc.u32     %1, %1, %5;\n\t"
        "addc.u32       %2, 0, 0;\n\t"
        : "=r"(w0), "=r"(w1), "=r"(c) : "r"(a.t), "r"(a.lo), "r"(a.hi), "r"(c_eps));
    u32 n = 0u - c;
    asm("add.cc.u32 %0, %0, %2;\n\t addc.u32 %1, %1, 0;" : "+r"(w0), "+r"(w1) : "r"(n));
    return mk64(w0, w1);
}

DEV u64 gadd(u64 a, u64 b) { return wred(w2(a, b)); }

DEV u64 canon(u64 a) { return a >= 0xFFFFFFFF00000001ull ? a - 0xFFFFFFFF00000001ull : a; }

/* r2*EPS = (r2 << 32) - r2 on the ALU pipe instead of an IMAD.WIDE */
DEV u64 red128a(u32 r0, u32 r1, u32 r2, u32 r3)
{
    u32 z0, z1;
    asm("{.reg .u32 u, h, t0, t1, rc, tc;\n\t"
        "sub.cc.u32     u, 0, %2;\n\t"
        "subc.u32       h, %2, 0;\n\t"
        "add.cc.u32     t0, %3, u;\n\t"
        "addc.cc.u32    t1, %4, h;\n\t"
        "addc.u32       rc, %5, 0;\n\t"
        "addc.u32       tc, t1, 0;\n\t"
        "sub.cc.u32     %0, t0, rc;\n\t"
        "subc.u32       %1, tc, 0;}"
        : "=r"(z0), "=r"(z1) : "r"(r2), "r"(r0), "r"(r1), "r"(r3));
    return mk64(z0, z1);
}
#ifndef QTC_SBRED
#define QTC_SBRED 0  /* bit k: S-box reduction k (x2, x3, x4, x7) on the ALU */
#endif
template <int K> DEV u64 redk(u32 r0, u32 r1, u32 r2, u32 r3)
{
    return ((QTC_SBRED >> K) & 1) ? red128a(r0, r1, r2, r3) : red128(r0, r1, r2, r3);
}
DEV u64 sbox(u64 x)
{
    u32 r0, r1, r2, r3;
    sqr128(x, r0, r1, r2, r3); const u64 x2 = redk<0>(r0, r1, r2, r3);
    mul128(x2, x, r0, r1, r2, r3); const u64 x3 = redk<1>(r0, r1, r2, r3);
    sqr128(x2, r0, r1, r2, r3); const u64 x4 = redk<2>(r0, r1, r2, r3);
    mul128(x3, x4, r0, r1, r2, r3); return redk<3>(r0, r1, r2, r3);
}

// ---------------------------------------------------------------- constants
// Row g (0..14): what is added after the linear layer that follows full round g; row 15 is
// RC_INITIAL[0] for the second permutation's first linear layer.
__constant__ u64 c_post[16][12];
__constant__ u64 c_rci[23];   // RC_INTERNAL[1..21], 0, 0
__constant__ u64 c_rct0[12];  // RC_TERMINAL[0]

struct Params {
    u64 pre[12];  // ext(mid + nonce words, s7 without the counter) + RC_INITIAL[0]
    u64 t0;       // first 8 target bytes, big-endian
    u64* dump;    // optional canonical out0 per nonce
    u32* out;     // [0] = candidate count, [1..15] = nonce index
    u32 w0;       // big-endian counter word at index 0 (QTC_SUB: first t)
    u32 count;
    u64 K[12];    // QTC_SUB: lanes after the first full round with f = g = 0
    u64* cyc;     // optional: SM cycles per block
};
#ifndef QTC_SUB
#define QTC_SUB 0
#endif

__constant__ u64 c_diag[12] = {
    0xc3b6c08e23ba9300ull, 0xd84b5de94a324fb6ull, 0x0d0c371c5b35b84full, 0x7964f570e7188037ull,
    0x5daf18bbd996604bull, 0x6743bc47b9595257ull, 0x5528b9362c59bb70ull, 0xac45e25b7127b68bull,
    0xa2077d7dfbb606b5ull, 0xf3faac6faee378aeull, 0x0c6388b51545e883ull, 0xd27dbb6944917b60ull};

DEV void mat4(u64 x0, u64 x1, u64 x2, u64 x3, W& y0, W& y1, W& y2, W& y3)
{
    W t01 = w2(x0, x1), t23 = w2(x2, x3);
    W t0123 = wadd(t01, t23);
    W t01123 = wadd64(t0123, x1);
    W t01233 = wadd64(t0123, x3);
    y3 = wadd(t01233, w2(x0, x0));
    y1 = wadd(t01123, w2(x2, x2));
    y0 = wadd(t01123, t01);
    y2 = wadd(t01233, t23);
}

// s = M_ext * s + add
DEV void ext_add(u64 s[12], const u64* add)
{
#if QTC_NOEXT  /* timing ablation: only the constants, no mixing (wrong hashes) */
#pragma unroll
    for (int i = 0; i < 12; i++) s[i] = gadd(s[i], add[i]);
    return;
#endif
    W y[12];
#pragma unroll
    for (int k = 0; k < 3; k++) mat4(s[4 * k], s[4 * k + 1], s[4 * k + 2], s[4 * k + 3], y[4 * k], y[4 * k + 1], y[4 * k + 2], y[4 * k + 3]);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        W sum = wadd(wadd(y[j], y[4 + j]), y[8 + j]);
#pragma unroll
        for (int k = 0; k < 3; k++) s[4 * k + j] = wred(wadd64(wadd(y[4 * k + j], sum), add[4 * k + j]));
    }
}

#ifndef QTC_INT_W
#define QTC_INT_W 1
#endif
// a*b + w with w a lazy 96-bit sum (lo, hi, t): w enters the product chain unreduced.
// a*b + w < 2^128 - 2^96 when b < 0xF3FB... (MDS diagonal), so r3 <= 2^32 - 2 holds.
DEV u64 red128_alu(u32 r0, u32 r1, u32 r2, u32 r3)
{
    u32 z0, z1;
    asm("{.reg .u32 u, h, t0, t1, rc, tc;\n\t"
        "sub.cc.u32     u, 0, %2;\n\t"
        "subc.u32       h, %2, 0;\n\t"
        "add.cc.u32     t0, %3, u;\n\t"
        "addc.cc.u32    t1, %4, h;\n\t"
        "addc.u32       rc, %5, 0;\n\t"
        "addc.u32       tc, t1, 0;\n\t"
        "sub.cc.u32     %0, t0, rc;\n\t"
        "subc.u32       %1, tc, 0;}"
        : "=r"(z0), "=r"(z1) : "r"(r2), "r"(r0), "r"(r1), "r"(r3));
    return mk64(z0, z1);
}
#ifndef QTC_ALU_MASK
#define QTC_ALU_MASK 0
#endif
template <bool ALU>
DEV u64 gfma_w(u64 a, u64 b, W w)
{
    u32 a0, a1, b0, b1; sp64(a, a0, a1); sp64(b, b0, b1);
    u32 r0, r1, r2, r3;
    asm("mad.lo.cc.u32  %0, %4, %6, %8;\n\t"
        "madc.hi.cc.u32 %1, %4, %6, %9;\n\t"
        "madc.lo.cc.u32 %2, %5, %7, %10;\n\t"
        "madc.hi.u32    %3, %5, %7, 0;\n\t"
        "mad.lo.cc.u32  %1, %4, %7, %1;\n\t"
        "madc.hi.cc.u32 %2, %4, %7, %2;\n\t"
        "addc.u32       %3, %3, 0;\n\t"
        "mad.lo.cc.u32  %1, %5, %6, %1;\n\t"
        "madc.hi.cc.u32 %2, %5, %6, %2;\n\t"
        "addc.u32       %3, %3, 0;"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(w.lo), "r"(w.hi), "r"(w.t));
    return ALU ? red128_alu(r0, r1, r2, r3) : red128(r0, r1, r2, r3);
}

#ifndef QTC_GFMA2
#define QTC_GFMA2 0
#endif
/* gECC-style a*b + w: three wide products take one zero-extended word of w each as their
 * 64-bit addend (no overflow: (2^32-1)^2 + 2^32-1 < 2^64; the pairs are shared by all lanes),
 * the fourth takes the whole a0*b1 product with carry-out, then one 3-word merge. */
struct WX { u64 lo, hi, t; };
DEV WX wx(W w)
{
    WX r;
    r.lo = mk64(w.lo, 0); r.hi = mk64(w.hi, 0); r.t = mk64(w.t, 0);
    return r;
}
DEV u64 gfma_x(u64 a, u64 b, const WX& w)
{
    u32 a0, a1, b0, b1; sp64(a, a0, a1); sp64(b, b0, b1);
    u64 X, P, Q;
    asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(X) : "r"(a0), "r"(b1), "l"(w.hi));
    asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(P) : "r"(a0), "r"(b0), "l"(w.lo));
    asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(Q) : "r"(a1), "r"(b1), "l"(w.t));
    u32 x0, x1, p0, p1, q0, q1; sp64(X, x0, x1); sp64(P, p0, p1); sp64(Q, q0, q1);
    u32 m0, m1, c, r1, r2, r3;
    asm("mad.lo.cc.u32  %0, %3, %4, %5;\n\t"
        "madc.hi.cc.u32 %1, %3, %4, %6;\n\t"
        "addc.u32       %2, 0, 0;"
        : "=r"(m0), "=r"(m1), "=r"(c) : "r"(a1), "r"(b0), "r"(x0), "r"(x1));
    asm("add.cc.u32  %0, %3, %4;\n\t"
        "addc.cc.u32 %1, %5, %6;\n\t"
        "addc.u32    %2, %7, %8;"
        : "=r"(r1), "=r"(r2), "=r"(r3) : "r"(p1), "r"(m0), "r"(m1), "r"(q0), "r"(q1), "r"(c));
    return red128(p0, r1, r2, r3);
}

/* Even chain a0*b0 + w.lo/w.hi, a1*b1 + w.t; odd chain a0*b1 + a1*b0 on its own natural
 * register pair (no pair assembly), merged at word 1. */
DEV u64 gfma_y(u64 a, u64 b, W w)
{
    u32 a0, a1, b0, b1; sp64(a, a0, a1); sp64(b, b0, b1);
    u32 r0, r1, r2, r3, x0, x1, x2;
    asm("mul.lo.u32     %0, %3, %5;\n\t"
        "mul.hi.u32     %1, %3, %5;\n\t"
        "mad.lo.cc.u32  %0, %4, %6, %0;\n\t"
        "madc.hi.cc.u32 %1, %4, %6, %1;\n\t"
        "addc.u32       %2, 0, 0;"
        : "=r"(x0), "=r"(x1), "=r"(x2) : "r"(a0), "r"(a1), "r"(b1), "r"(b0));
    asm("mad.lo.cc.u32  %0, %4, %6, %8;\n\t"
        "madc.hi.cc.u32 %1, %4, %6, %9;\n\t"
        "madc.lo.cc.u32 %2, %5, %7, %10;\n\t"
        "madc.hi.u32    %3, %5, %7, 0;\n\t"
        "add.cc.u32     %1, %1, %11;\n\t"
        "addc.cc.u32    %2, %2, %12;\n\t"
        "addc.u32       %3, %3, %13;"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(w.lo), "r"(w.hi), "r"(w.t), "r"(x0), "r"(x1), "r"(x2));
    return red128(r0, r1, r2, r3);
}

DEV void internal22(u64 s[12], u64 rc0)
{
    // s[0] arrives with RC_INTERNAL[0] already added.
#ifndef QTC_NINT
#define QTC_NINT 22
#endif
#ifndef QTC_IUNR
#define QTC_IUNR 1
#endif
#define QTC_PRAGMA(x) _Pragma(#x)
#define QTC_UNROLL(n) QTC_PRAGMA(unroll n)
    QTC_UNROLL(QTC_IUNR)
    for (int r = 0; r < QTC_NINT; r++) {
#if QTC_SYNC >= 2
        __syncthreads();
#endif
#if QTC_INT_W
        // lanes 1..11 do not depend on the S-box: sum them while it runs
        W rest = w2(s[1], s[2]);
#pragma unroll
        for (int i = 3; i < 12; i++) rest = wadd64(rest, s[i]);
        s[0] = sbox(s[0]);
        W sg = wadd64(rest, s[0]);
        W sg0 = wadd64(sg, c_rci[r]);
#if QTC_GFMA2 == 2
        s[0] = gfma_y(s[0], c_diag[0], sg0);
#pragma unroll
        for (int i = 1; i < 12; i++) s[i] = gfma_y(s[i], c_diag[i], sg);
#elif QTC_GFMA2
        s[0] = gfma_x(s[0], c_diag[0], wx(sg0));
        const WX sgx = wx(sg);
#pragma unroll
        for (int i = 1; i < 12; i++) s[i] = gfma_x(s[i], c_diag[i], sgx);
#else
        s[0] = gfma_w<false>(s[0], c_diag[0], sg0);
#pragma unroll
        for (int i = 1; i < 12; i++)
            s[i] = ((QTC_ALU_MASK >> i) & 1) ? gfma_w<true>(s[i], c_diag[i], sg) : gfma_w<false>(s[i], c_diag[i], sg);
#endif
#else
        s[0] = sbox(s[0]);
        W acc = w2(s[0], s[1]);
#pragma unroll
        for (int i = 2; i < 12; i++) acc = wadd64(acc, s[i]);
        u64 sg = wred(acc);
        u64 sg0 = gadd(sg, c_rci[r]);
        s[0] = gfma(s[0], c_diag[0], sg0);
#pragma unroll
        for (int i = 1; i < 12; i++) s[i] = gfma(s[i], c_diag[i], sg);
#endif
    }
    (void)rc0;
#pragma unroll
    for (int i = 0; i < 12; i++) s[i] = gadd(s[i], c_rct0[i]);
}

DEV u64 qtc_hash_out0(const Params& p, u32 x)
{
    u64 s[12];
#if QTC_SUB
    /* Nonce words 0..3 = B + t*u, 4..7 = B - t*u with u = adj(M4) column 0: after the first
     * linear layer only lane 0 (-35t) and lane 4 (+35t) depend on t. The host gives the other
     * ten S-box outputs folded through the next linear layer as K; column 0 and column 4 of
     * M_ext are m0 * (2,1,1 | 1,2,1 per block), m0 = M4 column 0 = (2,1,1,3). */
    {
        const u64 d = 35ull * x;
        const u64 f = sbox(gadd(p.pre[0], 0xFFFFFFFF00000001ull - d));
        const u64 g = sbox(gadd(p.pre[4], d));
        const W c2 = w2(f, g);
        const W c[3] = {wadd64(c2, f), wadd64(c2, g), c2};
        const int m0[4] = {2, 1, 1, 3};
#pragma unroll
        for (int k = 0; k < 3; k++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                W v = c[k];
                if (m0[j] >= 2) v = wadd(v, c[k]);
                if (m0[j] == 3) v = wadd(v, c[k]);
                s[4 * k + j] = wred(wadd64(v, p.K[4 * k + j]));
            }
    }
    const int g0 = 1;
#else
    // ext(mid + x*e7) = ext(mid) + x * M[:,7]
    const u32 col[12] = {1, 1, 3, 2, 2, 2, 6, 4, 1, 1, 3, 2};
#pragma unroll
    for (int i = 0; i < 12; i++) {
        u32 a0, a1; sp64(p.pre[i], a0, a1);
        u64 t; asm("mul.wide.u32 %0, %1, %2;" : "=l"(t) : "r"(x), "r"(col[i]));
        s[i] = gadd(p.pre[i], t);
    }
    const int g0 = 0;
#endif
#pragma unroll 1
    for (int g = g0;; g++) {
#if QTC_SYNC
        __syncthreads();
#endif
#pragma unroll
        for (int i = 0; i < 12; i++) s[i] = sbox(s[i]);
        if (g == 15) break;
#pragma unroll 1
        for (int e = 0; e < 1 + (g == 7); e++) ext_add(s, c_post[e ? 15 : g]);
#ifndef QTC_NFULL_SKIP
        if ((g & 7) == 3) internal22(s, 0);
#else
        if (g == 0) internal22(s, 0);
#endif
    }
    // out0 = (M_ext s)[0] = 2*y00 + y10 + y20 with y_k0 = 2x0 + 3x1 + x2 + x3 of chunk k.
    W acc = w2(s[0], s[0]);
#pragma unroll
    for (int k = 0; k < 3; k++) {
        const int m = k == 0 ? 2 : 1;
        for (int rep = 0; rep < m; rep++) {
            if (!(k == 0 && rep == 0)) { acc = wadd64(acc, s[4 * k]); acc = wadd64(acc, s[4 * k]); }
            acc = wadd64(acc, s[4 * k + 1]); acc = wadd64(acc, s[4 * k + 1]); acc = wadd64(acc, s[4 * k + 1]);
            acc = wadd64(acc, s[4 * k + 2]); acc = wadd64(acc, s[4 * k + 3]);
        }
    }
    return canon(wred(acc));
}

#ifndef QTC_MINB
#define QTC_MINB 1
#endif
__global__ void __launch_bounds__(QTC_TPB, QTC_MINB) qtc_scan(const Params p)
{
    const long long t0 = clock64();
    const u32 stride = gridDim.x * blockDim.x;
    for (u32 base = blockIdx.x * blockDim.x; base < p.count; base += stride) {
        const u32 idx = base + threadIdx.x;
#if QTC_SUB
        u32 x = p.w0 + idx;  /* t */
#else
        u32 x = __byte_perm(p.w0 + idx, 0, 0x0123);
#endif
        u64 o = qtc_hash_out0(p, x);
        u64 key; { u32 l, h; sp64(o, l, h); key = mk64(__byte_perm(h, 0, 0x0123), __byte_perm(l, 0, 0x0123)); }
        if (idx >= p.count) continue;
        if (p.dump) p.dump[idx] = o;
        if (key <= p.t0) {
            u32 slot = atomicAdd(p.out, 1u);
            if (slot < 15) p.out[1 + slot] = idx;
        }
    }
    if (p.cyc) { __syncthreads(); if (threadIdx.x == 0) p.cyc[blockIdx.x] = (u64)(clock64() - t0); }
}

// field-op test kernel: op 0 mul, 1 sqr, 2 fma, 3 add, 4 wred of (a + b + c) lazily
__global__ void field_test(const u64* a, const u64* b, const u64* c, u64* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[5 * i + 0] = canon(gmul(a[i], b[i]));
    out[5 * i + 1] = canon(gsqr(a[i]));
    out[5 * i + 2] = canon(gfma(a[i], b[i], c[i]));
    out[5 * i + 3] = canon(gadd(a[i], b[i]));
    W w = wadd(w2(a[i], b[i]), w2(c[i], a[i]));
    out[5 * i + 4] = canon(wred(w));
}

// ---------------------------------------------------------------- host
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static u64 h_canon(u64 a) { return qpow::gf_canon(a); }

static void host_ext(u64 s[12]) { qpow::ext_layer(s); }

static void setup_constants()
{
    { u32 e = 0xFFFFFFFFu; CK(cudaMemcpyToSymbol(c_eps, &e, 4)); }
    u64 post[16][12] = {};
    for (int g = 0; g < 15; g++) {
        int gg = g & 7;
        if (gg <= 2) memcpy(post[g], qpow::RC_INITIAL[gg + 1], 96);
        else if (gg == 3) post[g][0] = qpow::RC_INTERNAL[0];
        else if (gg <= 6) memcpy(post[g], qpow::RC_TERMINAL[gg - 3], 96);
        else { post[g][0] = 1; post[g][1] = 1; }
    }
    memcpy(post[15], qpow::RC_INITIAL[0], 96);
    CK(cudaMemcpyToSymbol(c_post, post, sizeof(post)));
    u64 rci[23] = {};
    for (int r = 0; r < 21; r++) rci[r] = qpow::RC_INTERNAL[r + 1];
    CK(cudaMemcpyToSymbol(c_rci, rci, sizeof(rci)));
    CK(cudaMemcpyToSymbol(c_rct0, qpow::RC_TERMINAL[0], 96));
}

static void le32_words(const uint8_t b[32], u32 w[8]) { for (int i = 0; i < 8; i++) memcpy(&w[i], b + 4 * i, 4); }

// Kernel params for nonce = start + idx (counter in the last 4 BE bytes, no wrap inside a launch).
static void make_params(const uint8_t header[32], const uint8_t nonce[64], Params& p)
{
    u64 mid[12];
    qpow::mining_midstate(header, nonce, mid);
    u32 w[8]; le32_words(nonce + 32, w);
    u64 base[12];
    memcpy(base, mid, sizeof(mid));
    for (int i = 0; i < 7; i++) base[i] = qpow::gf_add(base[i], (u64)w[i]);
    host_ext(base);
#if QTC_SUB
    memcpy(base, mid, sizeof(mid));
    for (int i = 0; i < 8; i++) base[i] = qpow::gf_add(base[i], (u64)w[i]);
    host_ext(base);
    for (int i = 0; i < 12; i++) p.pre[i] = h_canon(qpow::gf_add(base[i], qpow::RC_INITIAL[0][i]));
    /* first full round with lanes 0 and 4 left out: K = M_ext * sbox(pre except 0, 4) + RC_INITIAL[1] */
    u64 k[12];
    for (int i = 0; i < 12; i++) k[i] = (i == 0 || i == 4) ? 0 : qpow::gf_sbox(p.pre[i]);
    host_ext(k);
    for (int i = 0; i < 12; i++) p.K[i] = h_canon(qpow::gf_add(k[i], qpow::RC_INITIAL[1][i]));
    p.w0 = 0;
#else
    for (int i = 0; i < 12; i++) p.pre[i] = h_canon(qpow::gf_add(base[i], qpow::RC_INITIAL[0][i]));
    p.w0 = ((u32)nonce[60] << 24) | ((u32)nonce[61] << 16) | ((u32)nonce[62] << 8) | nonce[63];
#endif
}

/* QTC_SUB nonce family: words 0..3 = B + t*u, words 4..7 = B - t*u, u = (4, -17, 11, -3). */
static const int k_sub_u[8] = {4, -17, 11, -3, -4, 17, -11, 3};
static void sub_fix(uint8_t n[64])
{
#if QTC_SUB
    for (int i = 0; i < 8; i++) {
        u32 w; memcpy(&w, n + 32 + 4 * i, 4);
        w = 0x40000000u | (w & 0x3FFFFFFFu);  /* room for |17 t| < 2^29 either way */
        memcpy(n + 32 + 4 * i, &w, 4);
    }
#else
    (void)n;
#endif
}
static void nonce_add(uint8_t n[64], u32 v);
static void nonce_at(const uint8_t base[64], u32 t, uint8_t out[64])
{
    memcpy(out, base, 64);
#if QTC_SUB
    for (int i = 0; i < 8; i++) {
        u32 w; memcpy(&w, out + 32 + 4 * i, 4);
        w += (u32)((int64_t)k_sub_u[i] * t);
        memcpy(out + 32 + 4 * i, &w, 4);
    }
#else
    nonce_add(out, t);
#endif
}

static u64 cpu_out0(const uint8_t header[32], const uint8_t nonce[64])
{
    u64 mid[12];
    qpow::mining_midstate(header, nonce, mid);
    u64 s[12]; memcpy(s, mid, sizeof(s));
    qpow::absorb32(s, nonce + 32);
    qpow::permute(s);
    s[0] = qpow::gf_add(s[0], 1); s[1] = qpow::gf_add(s[1], 1);
    qpow::permute(s);
    return h_canon(s[0]);
}

static void nonce_add(uint8_t n[64], u32 v)
{
    u64 c = v;
    for (int i = 63; i >= 0 && c; i--) { c += n[i]; n[i] = (uint8_t)c; c >>= 8; }
}

static int test_field()
{
    std::mt19937_64 rng(1);
    const u64 P = 0xFFFFFFFF00000001ull;
    std::vector<u64> edge = {0, 1, 2, 0xFFFFFFFFull, 0x100000000ull, P - 1, P, P + 1, ~0ull, ~0ull - 1,
                             0xFFFFFFFF00000000ull, 0x00000000FFFFFFFFull, 0x8000000000000000ull, 0xFFFFFFFEFFFFFFFFull};
    std::vector<u64> a, b, c;
    for (u64 x : edge) for (u64 y : edge) for (u64 z : edge) { a.push_back(x); b.push_back(y); c.push_back(z); }
    for (int i = 0; i < 1 << 20; i++) { a.push_back(rng()); b.push_back(rng()); c.push_back(rng()); }
    int n = (int)a.size();
    u64 *da, *db, *dc, *dout;
    CK(cudaMalloc(&da, 8 * n)); CK(cudaMalloc(&db, 8 * n)); CK(cudaMalloc(&dc, 8 * n)); CK(cudaMalloc(&dout, 40 * n));
    CK(cudaMemcpy(da, a.data(), 8 * n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, b.data(), 8 * n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dc, c.data(), 8 * n, cudaMemcpyHostToDevice));
    field_test<<<(n + 255) / 256, 256>>>(da, db, dc, dout, n);
    std::vector<u64> out(5 * n);
    CK(cudaMemcpy(out.data(), dout, 40 * n, cudaMemcpyDeviceToHost));
    int bad = 0, approx = 0, outside = 0;
    for (int i = 0; i < n; i++) {
        unsigned __int128 A = a[i] % P, B = b[i] % P, C = c[i] % P;
        u64 ref[5] = {(u64)(A * B % P), (u64)(A * A % P), (u64)((A * B + C) % P), (u64)((A + B) % P),
                      (u64)((2 * A + B + C) % P)};
        for (int k = 0; k < 5; k++) {
            if (out[5 * i + k] == ref[k]) continue;
            if (k <= 2) {  // the documented red128 gap: r2 == 0 && lo < r3 without carry
                unsigned __int128 v = (unsigned __int128)a[i] * (k == 1 ? a[i] : b[i]) + (k == 2 ? c[i] : 0);
                u64 lo = (u64)v; u32 r2 = (u32)(v >> 64), r3 = (u32)(v >> 96);
                if (r2 == 0 && lo < r3) { approx++; continue; }
                if (r3 == 0xFFFFFFFFu) { outside++; continue; }  // fma precondition, see red128
            }
            if (bad++ < 10)
                printf("field op %d mismatch a=%016llx b=%016llx c=%016llx got %016llx ref %016llx\n", k,
                       (unsigned long long)a[i], (unsigned long long)b[i], (unsigned long long)c[i],
                       (unsigned long long)out[5 * i + k], (unsigned long long)ref[k]);
        }
    }
    printf("field ops: %d values, %d mismatches, %d in the documented 2^-64 gap (edge inputs)\n", n, bad, approx);
    cudaFree(da); cudaFree(db); cudaFree(dc); cudaFree(dout);
    return bad;
}

static int test_hash()
{
    std::mt19937_64 rng(7);
    int bad = 0;
    for (int trial = 0; trial < 4; trial++) {
        uint8_t header[32], nonce[64];
        for (auto& v : header) v = (uint8_t)rng();
        for (auto& v : nonce) v = (uint8_t)rng();
        sub_fix(nonce);
        nonce[60] = 0x7f;  // room for the batch inside the 32-bit counter word
        Params p{};
        make_params(header, nonce, p);
        const u32 n = 1u << 16;
        p.count = n;
        p.t0 = 0;
        u64* dd; u32* dout;
        CK(cudaMalloc(&dd, 8ull * n)); CK(cudaMalloc(&dout, 64));
        CK(cudaMemset(dout, 0, 64));
        p.dump = dd; p.out = dout;
        qtc_scan<<<(n + QTC_TPB - 1) / QTC_TPB, QTC_TPB>>>(p);
        CK(cudaGetLastError());
        std::vector<u64> got(n);
        CK(cudaMemcpy(got.data(), dd, 8ull * n, cudaMemcpyDeviceToHost));
        // CPU reference on a sample (CPU permute is slow)
        int checked = 0;
        for (u32 i = 0; i < n; i += (i < 64 ? 1 : 97)) {
            uint8_t nn[64]; nonce_at(nonce, i, nn);
            u64 ref = cpu_out0(header, nn);
            checked++;
            if (got[i] != ref && bad++ < 10)
                printf("hash mismatch trial %d idx %u: got %016llx ref %016llx\n", trial, i,
                       (unsigned long long)got[i], (unsigned long long)ref);
        }
        printf("hash trial %d: %d nonces checked\n", trial, checked);
        cudaFree(dd); cudaFree(dout);
    }
    // candidate path: target from a real-looking difficulty, check every reported nonce on CPU
    {
        uint8_t header[32], nonce[64];
        for (auto& v : header) v = (uint8_t)rng();
        for (auto& v : nonce) v = (uint8_t)rng();
        sub_fix(nonce);
        nonce[60] = 0;
        Params p{};
        make_params(header, nonce, p);
        p.count = 1u << 24;
        p.t0 = 0x0000004000000000ull;  // ~2^-26 per nonce -> ~0.25 expected... use a looser one
        p.t0 = 0x0000040000000000ull;  // 2^-22 -> ~4 candidates
        u32* dout; CK(cudaMalloc(&dout, 64)); CK(cudaMemset(dout, 0, 64));
        p.out = dout; p.dump = nullptr;
        qtc_scan<<<56 * 8, QTC_TPB>>>(p);
        u32 res[16]; CK(cudaMemcpy(res, dout, 64, cudaMemcpyDeviceToHost));
        printf("candidates: %u\n", res[0]);
        for (u32 k = 0; k < res[0] && k < 15; k++) {
            uint8_t nn[64]; nonce_at(nonce, res[1 + k], nn);
            u64 ref = cpu_out0(header, nn);
            u64 key = __builtin_bswap64(ref);
            printf("  idx %u out0 %016llx key %016llx %s\n", res[1 + k], (unsigned long long)ref,
                   (unsigned long long)key, key <= p.t0 ? "ok" : "BAD");
            if (key > p.t0) bad++;
        }
        cudaFree(dout);
    }
    return bad;
}

static void bench(double sec, int blocks_per_sm)
{
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    uint8_t header[32] = {1, 2, 3}, nonce[64] = {};
    sub_fix(nonce);
    Params p{};
    make_params(header, nonce, p);
    p.t0 = 0;
    u32* dout; CK(cudaMalloc(&dout, 64)); CK(cudaMemset(dout, 0, 64));
    p.out = dout;
    int grid = prop.multiProcessorCount * blocks_per_sm;
    p.count = 1u << 24;
    qtc_scan<<<grid, QTC_TPB>>>(p);
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    qtc_scan<<<grid, QTC_TPB>>>(p);
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    int reps = (int)(sec * 1000 / ms) + 1;
    u64 total = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) {
        p.w0 = (u32)(i & 0xff) << 24;
        qtc_scan<<<grid, QTC_TPB>>>(p);
        total += p.count;
    }
    CK(cudaDeviceSynchronize());
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, qtc_scan);
    printf("{\"mhs\": %.2f, \"regs\": %d, \"tpb\": %d, \"grid\": %d, \"launch_ms\": %.2f}\n",
           total / s / 1e6, fa.numRegs, QTC_TPB, grid, s * 1000 / reps);
}

int main(int argc, char** argv)
{
    setup_constants();
    if (argc > 1 && !strcmp(argv[1], "test")) {
        int bad = test_field();
        bad += test_hash();
        printf(bad ? "FAIL\n" : "PASS\n");
        return bad != 0;
    }
    if (argc > 1 && !strcmp(argv[1], "cyc")) {
        /* SM cycles per hash: one resident wave of blocks, each looping over many hashes */
        cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
        int per_sm = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, qtc_scan, QTC_TPB, 0));
        const int grid = prop.multiProcessorCount * per_sm;
        uint8_t header[32] = {1, 2, 3}, nonce[64] = {};
        sub_fix(nonce);
        Params p{};
        make_params(header, nonce, p);
        p.t0 = 0;
        u32* dout; CK(cudaMalloc(&dout, 64)); CK(cudaMemset(dout, 0, 64)); p.out = dout;
        u64* cyc; CK(cudaMalloc(&cyc, 8 * grid)); p.cyc = cyc;
        p.count = 1u << 22; qtc_scan<<<grid, QTC_TPB>>>(p); CK(cudaDeviceSynchronize());
        p.count = (u32)grid * QTC_TPB * (argc > 2 ? atoi(argv[2]) : 64);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0); qtc_scan<<<grid, QTC_TPB>>>(p); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        std::vector<u64> c(grid); CK(cudaMemcpy(c.data(), cyc, 8 * grid, cudaMemcpyDeviceToHost));
        double m = 0; for (u64 v : c) m += (double)v; m /= grid;
        cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, qtc_scan);
        printf("{\"smclk_per_hash\": %.1f, \"mhs_per_ghz\": %.1f, \"mhs\": %.1f, \"eff_mhz\": %.0f, \"blocks_per_sm\": %d, \"regs\": %d}\n",
               m * prop.multiProcessorCount / p.count, 1e3 * p.count / m, p.count / (ms * 1e3), m / (ms * 1e3),
               per_sm, fa.numRegs);
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "bench")) {
        bench(argc > 2 ? atof(argv[2]) : 10, argc > 3 ? atoi(argv[3]) : 8);
        return 0;
    }
    fprintf(stderr, "usage: qtc test | bench SEC [blocks_per_sm]\n");
    return 2;
}
