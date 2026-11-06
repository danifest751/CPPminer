// The TNet CPU tile on AArch64 DotProd; this file alone is built with +dotprod and runs only where the CPU reports it.
#include "cp_tnet_cpu.hpp"

#if (defined(__aarch64__) || defined(_M_ARM64)) && \
        (defined(__ARM_FEATURE_DOTPROD) || (defined(_MSC_VER) && defined(__ARM_ARCH) && __ARM_ARCH >= 802))
#include <arm_neon.h>

// A column group's 16 accumulators stay in registers for the whole block: one sdot gives 16 products.
void rq_cpu_micro_dotprod(const int8_t* a, const int8_t* b, int kc, int32_t* c) {
    const int groups = kc / 4;
    for (int jg = 0; jg < 2; ++jg) {
        int32_t* cj = c + jg * 64;
        int32x4_t lo[8], hi[8];
        for (int x = 0; x < 8; ++x) {
            lo[x] = vld1q_s32(cj + x * 8);
            hi[x] = vld1q_s32(cj + x * 8 + 4);
        }
        const int8_t* ag = a;
        const int8_t* bg = b + static_cast<size_t>(jg) * groups * 32;
        for (int g = 0; g < groups; ++g, ag += 32, bg += 32) {
            const int8x16_t a_lo = vld1q_s8(ag), a_hi = vld1q_s8(ag + 16);
            const int8x16_t b0 = vld1q_s8(bg), b1 = vld1q_s8(bg + 16);
            lo[0] = vdotq_laneq_s32(lo[0], a_lo, b0, 0);
            hi[0] = vdotq_laneq_s32(hi[0], a_hi, b0, 0);
            lo[1] = vdotq_laneq_s32(lo[1], a_lo, b0, 1);
            hi[1] = vdotq_laneq_s32(hi[1], a_hi, b0, 1);
            lo[2] = vdotq_laneq_s32(lo[2], a_lo, b0, 2);
            hi[2] = vdotq_laneq_s32(hi[2], a_hi, b0, 2);
            lo[3] = vdotq_laneq_s32(lo[3], a_lo, b0, 3);
            hi[3] = vdotq_laneq_s32(hi[3], a_hi, b0, 3);
            lo[4] = vdotq_laneq_s32(lo[4], a_lo, b1, 0);
            hi[4] = vdotq_laneq_s32(hi[4], a_hi, b1, 0);
            lo[5] = vdotq_laneq_s32(lo[5], a_lo, b1, 1);
            hi[5] = vdotq_laneq_s32(hi[5], a_hi, b1, 1);
            lo[6] = vdotq_laneq_s32(lo[6], a_lo, b1, 2);
            hi[6] = vdotq_laneq_s32(hi[6], a_hi, b1, 2);
            lo[7] = vdotq_laneq_s32(lo[7], a_lo, b1, 3);
            hi[7] = vdotq_laneq_s32(hi[7], a_hi, b1, 3);
        }
        for (int x = 0; x < 8; ++x) {
            vst1q_s32(cj + x * 8, lo[x]);
            vst1q_s32(cj + x * 8 + 4, hi[x]);
        }
    }
}
#elif defined(CASE33_BUILD_AARCH64)
// A compiler without DotProd: the dispatcher only calls this if the CPU has it, so keep the answer right anyway.
void rq_cpu_micro_dotprod(const int8_t* a, const int8_t* b, int kc, int32_t* c) { rq_cpu_micro_portable(a, b, kc, c); }
#endif
