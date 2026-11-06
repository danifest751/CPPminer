// The TNet CPU tile on AArch64 I8MM; this file alone is built with +i8mm and runs only where the CPU reports it.
#include "cp_tnet_cpu.hpp"

#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
#include <arm_neon.h>

// One smmla multiplies 2 rows by 2 columns over 8 k, 32 products to sdot's 16. Each accumulator holds a 2x2 square, so a
// block starts and ends by turning the caller's columns into squares and back.
void rq_cpu_micro_i8mm(const int8_t* a, const int8_t* b, int kc, int32_t* c) {
    const int groups = kc / 8;
    for (int jg = 0; jg < 2; ++jg) {
        int32_t* cj = c + jg * 64;
        int32x4_t s[4][4];  // [pair of rows][pair of columns]
        for (int p = 0; p < 4; ++p) {
            const int32_t* cp = cj + p * 16;
            const int32x4_t c0 = vld1q_s32(cp), c1 = vld1q_s32(cp + 4), d0 = vld1q_s32(cp + 8), d1 = vld1q_s32(cp + 12);
            s[0][p] = vzip1q_s32(c0, d0);
            s[1][p] = vzip2q_s32(c0, d0);
            s[2][p] = vzip1q_s32(c1, d1);
            s[3][p] = vzip2q_s32(c1, d1);
        }
        const int8_t* ag = a;
        const int8_t* bg = b + static_cast<size_t>(jg) * groups * 64;
        for (int g = 0; g < groups; ++g, ag += 64, bg += 64) {
            const int8x16_t a0 = vld1q_s8(ag), a1 = vld1q_s8(ag + 16), a2 = vld1q_s8(ag + 32), a3 = vld1q_s8(ag + 48);
            const int8x16_t b0 = vld1q_s8(bg), b1 = vld1q_s8(bg + 16), b2 = vld1q_s8(bg + 32), b3 = vld1q_s8(bg + 48);
            s[0][0] = vmmlaq_s32(s[0][0], a0, b0);
            s[1][0] = vmmlaq_s32(s[1][0], a1, b0);
            s[2][0] = vmmlaq_s32(s[2][0], a2, b0);
            s[3][0] = vmmlaq_s32(s[3][0], a3, b0);
            s[0][1] = vmmlaq_s32(s[0][1], a0, b1);
            s[1][1] = vmmlaq_s32(s[1][1], a1, b1);
            s[2][1] = vmmlaq_s32(s[2][1], a2, b1);
            s[3][1] = vmmlaq_s32(s[3][1], a3, b1);
            s[0][2] = vmmlaq_s32(s[0][2], a0, b2);
            s[1][2] = vmmlaq_s32(s[1][2], a1, b2);
            s[2][2] = vmmlaq_s32(s[2][2], a2, b2);
            s[3][2] = vmmlaq_s32(s[3][2], a3, b2);
            s[0][3] = vmmlaq_s32(s[0][3], a0, b3);
            s[1][3] = vmmlaq_s32(s[1][3], a1, b3);
            s[2][3] = vmmlaq_s32(s[2][3], a2, b3);
            s[3][3] = vmmlaq_s32(s[3][3], a3, b3);
        }
        for (int p = 0; p < 4; ++p) {
            int32_t* cp = cj + p * 16;
            vst1q_s32(cp, vuzp1q_s32(s[0][p], s[1][p]));
            vst1q_s32(cp + 4, vuzp1q_s32(s[2][p], s[3][p]));
            vst1q_s32(cp + 8, vuzp2q_s32(s[0][p], s[1][p]));
            vst1q_s32(cp + 12, vuzp2q_s32(s[2][p], s[3][p]));
        }
    }
}
#endif
