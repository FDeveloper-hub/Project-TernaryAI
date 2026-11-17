#include "ternary_neon.h"
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) && defined(__GNUC__)
#include <arm_neon.h>
#if !defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_NEON
#define HWCAP_NEON (1 << 12)
#endif
#endif

int ternary_neon_enabled(void) {
    static int v = -1;
    if (v < 0) {
        const char *off = getenv("TERNARYAI_NO_NEON");
        int hw = 1;
#if !defined(__aarch64__) && defined(__linux__)
        hw = (getauxval(AT_HWCAP) & HWCAP_NEON) != 0;
#endif
        v = hw && !(off && off[0] && off[0] != '0');
    }
    return v;
}

/* Bit j (0..7) de cada byte de un par de bytes de mascara -> 0xFF en los 16 bytes que corresponden a esos 16 bits. */
static inline uint8x16_t mask16_to_bytes(uint32_t m16) {
    static const uint8_t sel[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t b = vcombine_u8(vdup_n_u8((uint8_t)(m16 & 0xFF)), vdup_n_u8((uint8_t)((m16 >> 8) & 0xFF)));
    return vtstq_u8(b, vld1q_u8(sel));
}

/* Bit j (0..3) de un nibble -> 0xFFFFFFFF en el carril j. */
static inline uint32x4_t mask4_to_lanes(uint32_t nib) {
    static const uint32_t sel[4] = {1, 2, 4, 8};
    return vtstq_u32(vdupq_n_u32(nib), vld1q_u32(sel));
}

int32_t ternary_neon_dot_u8(const uint64_t *wm, const uint64_t *ws, size_t n, const uint8_t *q, int32_t zp) {
    const size_t nw = (n + 63) / 64;
    uint16x8_t ap = vdupq_n_u16(0), an = vdupq_n_u16(0);    /* sumas parciales por carril: 4*510 por palabra, se vuelcan cada 32 */
    uint16x4_t cp = vdup_n_u16(0), cn = vdup_n_u16(0);      /* cuentas de pesos +1 y -1 */
    uint32x4_t tp = vdupq_n_u32(0), tn = vdupq_n_u32(0);
    uint32_t sp = 0, sn = 0;
    int32_t npos = 0, nneg = 0;
    uint8_t tmp[64];
    for (size_t k = 0; k < nw; k++) {
        const uint64_t pos = wm[k] & ~ws[k], neg = wm[k] & ws[k];
        const size_t rem = n - 64 * k;
        const uint8_t *src = q + 64 * k;
        if (rem < 64) { memset(tmp, 0, sizeof(tmp)); memcpy(tmp, src, rem); src = tmp; }   /* nunca se lee mas alla de n */
        cp = vpadal_u8(cp, vcnt_u8(vcreate_u8(pos)));
        cn = vpadal_u8(cn, vcnt_u8(vcreate_u8(neg)));
        for (int j = 0; j < 4; j++) {
            const uint8x16_t v = vld1q_u8(src + 16 * j);
            ap = vpadalq_u8(ap, vandq_u8(v, mask16_to_bytes((uint32_t)(pos >> (16 * j)) & 0xFFFF)));
            an = vpadalq_u8(an, vandq_u8(v, mask16_to_bytes((uint32_t)(neg >> (16 * j)) & 0xFFFF)));
        }
        if ((k & 31) == 31 || k + 1 == nw) {
            tp = vpadalq_u16(tp, ap); tn = vpadalq_u16(tn, an);
            ap = vdupq_n_u16(0); an = vdupq_n_u16(0);
            npos += (int32_t)(vget_lane_u16(cp, 0) + vget_lane_u16(cp, 1) + vget_lane_u16(cp, 2) + vget_lane_u16(cp, 3));
            nneg += (int32_t)(vget_lane_u16(cn, 0) + vget_lane_u16(cn, 1) + vget_lane_u16(cn, 2) + vget_lane_u16(cn, 3));
            cp = vdup_n_u16(0); cn = vdup_n_u16(0);
        }
    }
    sp = vgetq_lane_u32(tp, 0) + vgetq_lane_u32(tp, 1) + vgetq_lane_u32(tp, 2) + vgetq_lane_u32(tp, 3);
    sn = vgetq_lane_u32(tn, 0) + vgetq_lane_u32(tn, 1) + vgetq_lane_u32(tn, 2) + vgetq_lane_u32(tn, 3);
    return (int32_t)sp - (int32_t)sn - zp * (npos - nneg);
}

void ternary_neon_u8_to_f32(const uint8_t *q, size_t n, float sx, int32_t zp, float *out) {
    const float32x4_t vs = vdupq_n_f32(sx);
    const int32x4_t vz = vdupq_n_s32(zp);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const uint8x16_t b = vld1q_u8(q + i);
        const uint16x8_t lo = vmovl_u8(vget_low_u8(b)), hi = vmovl_u8(vget_high_u8(b));
        const int32x4_t a0 = vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo))), vz);
        const int32x4_t a1 = vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(lo))), vz);
        const int32x4_t a2 = vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi))), vz);
        const int32x4_t a3 = vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(hi))), vz);
        vst1q_f32(out + i,      vmulq_f32(vs, vcvtq_f32_s32(a0)));
        vst1q_f32(out + i + 4,  vmulq_f32(vs, vcvtq_f32_s32(a1)));
        vst1q_f32(out + i + 8,  vmulq_f32(vs, vcvtq_f32_s32(a2)));
        vst1q_f32(out + i + 12, vmulq_f32(vs, vcvtq_f32_s32(a3)));
    }
    for (; i < n; i++) out[i] = sx * (float)((int32_t)q[i] - zp);
}

void ternary_neon_outer_acc(float *gw, size_t n, float g, const float *xf) {
    const float32x4_t vg = vdupq_n_f32(g);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t p = vmulq_f32(vg, vld1q_f32(xf + i));
        vst1q_f32(gw + i, vaddq_f32(vld1q_f32(gw + i), p));
    }
    for (; i < n; i++) gw[i] += g * xf[i];
}

void ternary_neon_gradx_row(float *gx, size_t n, const uint64_t *wm, const uint64_t *ws, float gs) {
    const float32x4_t vg = vdupq_n_f32(gs);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const size_t w = i >> 6, sh = i & 63;
        const uint32_t m = (uint32_t)(wm[w] >> sh) & 0xF, sg = (uint32_t)(ws[w] >> sh) & 0xF;
        const float32x4_t a = vld1q_f32(gx + i);
        float32x4_t r = vbslq_f32(mask4_to_lanes(m & ~sg), vaddq_f32(a, vg), a);
        r = vbslq_f32(mask4_to_lanes(m & sg), vsubq_f32(a, vg), r);
        vst1q_f32(gx + i, r);
    }
    for (; i < n; i++) {
        const uint64_t bit = (uint64_t)1 << (i & 63);
        const size_t k = i >> 6;
        if (wm[k] & bit) gx[i] += (ws[k] & bit) ? -gs : gs;
    }
}

/* ---- busqueda de ids en un rango [lo, hi) (marcadores especiales de ternary_targets) ---------------------------------- */
size_t ternary_neon_find_range(const int32_t *ids, size_t from, size_t n, int32_t lo, int32_t hi) {
    if (lo >= hi || from >= n) return n;
    const uint32_t span = (uint32_t)hi - (uint32_t)lo;
    const uint32x4_t vlo = vdupq_n_u32((uint32_t)lo), vspan = vdupq_n_u32(span);
    size_t i = from;
    for (; i + 16 <= n; i += 16) {                       /* 4 vectores de 4 ids por vuelta: una decision por 16 ids */
        const uint32x4_t m0 = vcltq_u32(vsubq_u32(vreinterpretq_u32_s32(vld1q_s32(ids + i)), vlo), vspan);
        const uint32x4_t m1 = vcltq_u32(vsubq_u32(vreinterpretq_u32_s32(vld1q_s32(ids + i + 4)), vlo), vspan);
        const uint32x4_t m2 = vcltq_u32(vsubq_u32(vreinterpretq_u32_s32(vld1q_s32(ids + i + 8)), vlo), vspan);
        const uint32x4_t m3 = vcltq_u32(vsubq_u32(vreinterpretq_u32_s32(vld1q_s32(ids + i + 12)), vlo), vspan);
        const uint32x4_t any = vorrq_u32(vorrq_u32(m0, m1), vorrq_u32(m2, m3));
        const uint32x2_t r = vorr_u32(vget_low_u32(any), vget_high_u32(any));      /* sin vmaxvq: tambien vale en ARMv7 */
        if ((vget_lane_u32(r, 0) | vget_lane_u32(r, 1)) == 0) continue;
        for (size_t k = i; k < i + 16; k++)                 /* hay uno en este bloque: el primero */
            if ((uint32_t)ids[k] - (uint32_t)lo < span) return k;
    }
    for (; i < n; i++)
        if ((uint32_t)ids[i] - (uint32_t)lo < span) return i;
    return n;
}

#endif /* __ARM_NEON: los esqueletos para las demas compilaciones estan en ternary_neon_stubs.c */
