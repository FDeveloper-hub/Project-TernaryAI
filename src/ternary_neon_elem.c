/* Nucleos NEON de 128 bits: elemento a elemento, smear y R2: u8_quantize_row, add_f32, acc_add2, scale_f32, shift_round_i32, sign_f32, scale_i32_f32, scatter_i32, smear_k, smear_grad_rows, elemnorm_row, r2_row.
 * Interfaz y reglas: include/ternary_neon.h. Los esqueletos para compilaciones sin NEON estan en ternary_neon_stubs.c.
 *
 * Mismos bits que el escalar de la libreria (un mul y un add separados, sin vmla/vfma). Todas las funciones son reentrantes.
 * Diferencias aceptadas: denormales a cero (NEON). Aqui no hay division ni raiz vectoriales (ARMv7): donde el escalar divide,
 * se evita la division con una reciprocidad aproximada + comprobacion exacta de los casos dudosos (elemnorm) o se deja
 * una division escalar por elemento (VFP). */
#include "ternary_neon.h"
#include "ternary_fexp.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) && defined(__GNUC__)
#include <arm_neon.h>

#define NINL static inline __attribute__((always_inline))
NINL uint32x4_t u32(uint32_t v) { return vdupq_n_u32(v); }
NINL float32x4_t f32(float v) { return vdupq_n_f32(v); }
NINL float32x4_t bitsf(uint32x4_t v) { return vreinterpretq_f32_u32(v); }
NINL uint32x4_t fbits(float32x4_t v) { return vreinterpretq_u32_f32(v); }
/* Bit j (0..3) de un nibble -> 0xFFFFFFFF en el carril j. */
NINL uint32x4_t mask4(uint32_t nib) {
    static const uint32_t sel[4] = {1, 2, 4, 8};
    return vtstq_u32(vdupq_n_u32(nib & 15u), vld1q_u32(sel));
}
/* (float)sign(z): +1, -1 o 0. */
NINL float32x4_t sign4(int32x4_t z) {
    const uint32x4_t pos = vcgtq_s32(z, vdupq_n_s32(0)), neg = vcltq_s32(z, vdupq_n_s32(0));
    return bitsf(vorrq_u32(vandq_u32(pos, u32(0x3F800000u)), vandq_u32(neg, u32(0xBF800000u))));
}
/* Suma horizontal de un uint32x4 (los carriles tienen bits disjuntos: suma = OR). */
NINL uint32_t hsum_u32(uint32x4_t v) {
    uint32x2_t t = vpadd_u32(vget_low_u32(v), vget_high_u32(v));
    t = vpadd_u32(t, t);
    return vget_lane_u32(t, 0);
}

/* ---------------------------------------------------------------------------------------------------------------- */
void ternary_neon_add_f32(float *out, const float *a, const float *b, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const float32x4_t x0 = vld1q_f32(a + i), x1 = vld1q_f32(a + i + 4), y0 = vld1q_f32(b + i), y1 = vld1q_f32(b + i + 4);
        vst1q_f32(out + i, vaddq_f32(x0, y0)); vst1q_f32(out + i + 4, vaddq_f32(x1, y1));
    }
    for (; i + 4 <= n; i += 4) vst1q_f32(out + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < n; i++) out[i] = a[i] + b[i];
}

void ternary_neon_acc_add2(float *dst, const float *a, const float *b, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const float32x4_t x0 = vld1q_f32(a + i), x1 = vld1q_f32(a + i + 4), y0 = vld1q_f32(b + i), y1 = vld1q_f32(b + i + 4);
        const float32x4_t d0 = vld1q_f32(dst + i), d1 = vld1q_f32(dst + i + 4);
        vst1q_f32(dst + i, vaddq_f32(d0, vaddq_f32(x0, y0))); vst1q_f32(dst + i + 4, vaddq_f32(d1, vaddq_f32(x1, y1)));
    }
    for (; i + 4 <= n; i += 4) vst1q_f32(dst + i, vaddq_f32(vld1q_f32(dst + i), vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i))));
    for (; i < n; i++) dst[i] += (a[i] + b[i]);
}

void ternary_neon_scale_f32(float *x, size_t n, float s) {
    const float32x4_t vs = f32(s);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const float32x4_t a = vld1q_f32(x + i), b = vld1q_f32(x + i + 4);
        vst1q_f32(x + i, vmulq_f32(a, vs)); vst1q_f32(x + i + 4, vmulq_f32(b, vs));
    }
    for (; i + 4 <= n; i += 4) vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), vs));
    for (; i < n; i++) x[i] *= s;
}

void ternary_neon_shift_round_i32(const int32_t *z, size_t n, int shift, int32_t *out) {
    const int32x4_t vr = vdupq_n_s32((int32_t)(1u << (shift - 1))), vsh = vdupq_n_s32(-shift);     /* desplazamiento aritmetico a la derecha */
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const int32x4_t a = vld1q_s32(z + i), b = vld1q_s32(z + i + 4);
        vst1q_s32(out + i, vshlq_s32(vaddq_s32(a, vr), vsh)); vst1q_s32(out + i + 4, vshlq_s32(vaddq_s32(b, vr), vsh));
    }
    for (; i + 4 <= n; i += 4) vst1q_s32(out + i, vshlq_s32(vaddq_s32(vld1q_s32(z + i), vr), vsh));
    for (; i < n; i++) out[i] = (int32_t)((uint32_t)z[i] + (1u << (shift - 1))) >> shift;
}

/* Escribe 4 carriles en out[(i+k)*stride] (sin scatter en NEON: carriles con indice constante). */
NINL void store4_strided(float *out, size_t i, size_t stride, float32x4_t v) {
    vst1q_lane_f32(out + (i + 0) * stride, v, 0); vst1q_lane_f32(out + (i + 1) * stride, v, 1);
    vst1q_lane_f32(out + (i + 2) * stride, v, 2); vst1q_lane_f32(out + (i + 3) * stride, v, 3);
}

void ternary_neon_sign_f32(const int32_t *z, size_t n, float *out, size_t stride) {
    size_t i = 0;
    if (stride == 1) {
        for (; i + 4 <= n; i += 4) vst1q_f32(out + i, sign4(vld1q_s32(z + i)));
    } else {
        for (; i + 4 <= n; i += 4) store4_strided(out, i, stride, sign4(vld1q_s32(z + i)));
    }
    for (; i < n; i++) out[i * stride] = (float)((z[i] > 0) - (z[i] < 0));
}

void ternary_neon_scale_i32_f32(const int32_t *z, size_t n, float k, float *out, size_t stride) {
    const float32x4_t vk = f32(k);
    size_t i = 0;
    if (stride == 1) {
        for (; i + 8 <= n; i += 8) {
            const float32x4_t a = vcvtq_f32_s32(vld1q_s32(z + i)), b = vcvtq_f32_s32(vld1q_s32(z + i + 4));
            vst1q_f32(out + i, vmulq_f32(vk, a)); vst1q_f32(out + i + 4, vmulq_f32(vk, b));
        }
        for (; i + 4 <= n; i += 4) vst1q_f32(out + i, vmulq_f32(vk, vcvtq_f32_s32(vld1q_s32(z + i))));
    } else {
        for (; i + 4 <= n; i += 4) store4_strided(out, i, stride, vmulq_f32(vk, vcvtq_f32_s32(vld1q_s32(z + i))));
    }
    for (; i < n; i++) out[i * stride] = k * (float)z[i];
}

void ternary_neon_scatter_i32(const int32_t *src, size_t n, int32_t *dst, size_t stride) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const int32x4_t v = vld1q_s32(src + i);
        vst1q_lane_s32(dst + (i + 0) * stride, v, 0); vst1q_lane_s32(dst + (i + 1) * stride, v, 1);
        vst1q_lane_s32(dst + (i + 2) * stride, v, 2); vst1q_lane_s32(dst + (i + 3) * stride, v, 3);
    }
    for (; i < n; i++) dst[i * stride] = src[i];
}

void ternary_neon_smear_k(const int32_t *in, int32_t *out, int32_t *prev, size_t n, int first) {
    size_t i = 0;
    if (first) {
        for (; i + 4 <= n; i += 4) { const int32x4_t v = vld1q_s32(in + i); vst1q_s32(out + i, v); vst1q_s32(prev + i, v); }
        for (; i < n; i++) { const int32_t a = in[i]; out[i] = a; prev[i] = a; }
        return;
    }
    for (; i + 8 <= n; i += 8) {
        const int32x4_t v0 = vld1q_s32(in + i), v1 = vld1q_s32(in + i + 4), p0 = vld1q_s32(prev + i), p1 = vld1q_s32(prev + i + 4);
        vst1q_s32(out + i, vshrq_n_s32(vaddq_s32(v0, p0), 1)); vst1q_s32(out + i + 4, vshrq_n_s32(vaddq_s32(v1, p1), 1));
        vst1q_s32(prev + i, v0); vst1q_s32(prev + i + 4, v1);
    }
    for (; i + 4 <= n; i += 4) {
        const int32x4_t v = vld1q_s32(in + i), p = vld1q_s32(prev + i);
        vst1q_s32(out + i, vshrq_n_s32(vaddq_s32(v, p), 1)); vst1q_s32(prev + i, v);
    }
    for (; i < n; i++) { const int32_t a = in[i]; out[i] = (int32_t)((uint32_t)a + (uint32_t)prev[i]) >> 1; prev[i] = a; }
}

void ternary_neon_smear_grad_rows(float *g, size_t p0, size_t p1, size_t T, size_t w, const float *next_of_last) {
    const float32x4_t half = f32(0.5f), zero = f32(0.0f);
    for (size_t p = p0; p < p1; p++) {
        float *row = g + p * w;
        const float *nrow = (p + 1 == p1 && next_of_last) ? next_of_last : row + w;
        const int has_next = p + 1 < T;
        size_t d = 0;
        for (; d + 4 <= w; d += 4) {
            const float32x4_t r = vld1q_f32(row + d);
            const float32x4_t own = p == 0 ? r : vmulq_f32(half, r);
            const float32x4_t nxt = has_next ? vmulq_f32(half, vld1q_f32(nrow + d)) : zero;
            vst1q_f32(row + d, vaddq_f32(own, nxt));
        }
        for (; d < w; d++) {
            const float own = p == 0 ? row[d] : 0.5f * row[d];
            const float nxt = has_next ? 0.5f * nrow[d] : 0.0f;
            row[d] = own + nxt;
        }
    }
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Cuantizacion uint8 de una fila. rintf(y) = (y + 1.5*2^23) - 1.5*2^23 (al par mas cercano) vale para |y| < 2^22 y aqui |y| <= ~255. */
#define RINT_M 12582912.0f
void ternary_neon_u8_quantize_row(const float *x, size_t n, uint8_t *q, float *scale_out, int32_t *zp_out) {
    float lo = 0.0f, hi = 0.0f;                         /* el rango incluye el 0 */
    size_t i = 0;
    {
        float32x4_t l0 = vdupq_n_f32(0.0f), l1 = l0, h0 = l0, h1 = l0;
        for (; i + 8 <= n; i += 8) {
            const float32x4_t a = vld1q_f32(x + i), b = vld1q_f32(x + i + 4);
            l0 = vminq_f32(l0, a); l1 = vminq_f32(l1, b); h0 = vmaxq_f32(h0, a); h1 = vmaxq_f32(h1, b);
        }
        for (; i + 4 <= n; i += 4) { const float32x4_t a = vld1q_f32(x + i); l0 = vminq_f32(l0, a); h0 = vmaxq_f32(h0, a); }
        const float32x4_t l = vminq_f32(l0, l1), h = vmaxq_f32(h0, h1);
        const float32x2_t l2 = vmin_f32(vget_low_f32(l), vget_high_f32(l)), h2 = vmax_f32(vget_low_f32(h), vget_high_f32(h));
        lo = vget_lane_f32(vmin_f32(l2, vdup_lane_f32(l2, 1)), 0); hi = vget_lane_f32(vmax_f32(h2, vdup_lane_f32(h2, 1)), 0);
        for (; i < n; i++) { const float v = x[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
        if (lo != lo || hi != hi) {                     /* algun NaN: vmin/vmax lo propagan; el escalar lo ignora */
            lo = 0.0f; hi = 0.0f;
            for (size_t k = 0; k < n; k++) { const float v = x[k]; if (v < lo) lo = v; if (v > hi) hi = v; }
        }
    }
    const float sc = (hi - lo) / 255.0f;
    if (!(sc > 0.0f) || !isfinite(sc)) { *scale_out = 1.0f; *zp_out = 0; memset(q, 0, n); return; }
    int zp = (int)rintf(-lo / sc);
    if (zp < 0) zp = 0;
    if (zp > 255) zp = 255;
    *scale_out = sc; *zp_out = zp;
    const float inv = 1.0f / sc;
    const float32x4_t vinv = f32(inv), vzp = f32((float)zp), vm = f32(RINT_M), z0 = f32(0.0f), c255 = f32(255.0f);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t a = vmulq_f32(vld1q_f32(x + i), vinv), b = vmulq_f32(vld1q_f32(x + i + 4), vinv);
        a = vaddq_f32(vsubq_f32(vaddq_f32(a, vm), vm), vzp); b = vaddq_f32(vsubq_f32(vaddq_f32(b, vm), vm), vzp);
        a = vminq_f32(vmaxq_f32(a, z0), c255); b = vminq_f32(vmaxq_f32(b, z0), c255);
        const uint16x8_t w = vcombine_u16(vmovn_u32(vcvtq_u32_f32(a)), vmovn_u32(vcvtq_u32_f32(b)));
        vst1_u8(q + i, vmovn_u16(w));
    }
    for (; i < n; i++) {
        float qf = rintf(x[i] * inv) + (float)zp;
        if (qf < 0.0f) qf = 0.0f;
        if (qf > 255.0f) qf = 255.0f;
        q[i] = (uint8_t)qf;
    }
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Regla R2 (ver ternary_dense.c: ternary_r2_row_scalar). 16 pesos por tile en 4 vectores de 4 carriles. */
NINL uint16_t bits16(const uint64_t *w, size_t bit0) {
    const size_t i = bit0 >> 6, b = bit0 & 63;
    uint64_t v = w[i] >> b;
    if (b > 48) v |= w[i + 1] << (64 - b);
    return (uint16_t)v;
}
NINL void apply_bits(uint64_t *mask, uint64_t *sign, size_t bit0, uint16_t P, uint16_t N) {
    if (!(P | N)) return;
    const size_t w = bit0 >> 6, b = bit0 & 63;
    for (int part = 0; part < 2; part++) {
        uint64_t Pw, Nw;
        if (part == 0) { Pw = (uint64_t)P << b; Nw = (uint64_t)N << b; }
        else { if (b <= 48) break; Pw = (uint64_t)P >> (64 - b); Nw = (uint64_t)N >> (64 - b); }
        uint64_t *mp = mask + w + part, *sp = sign + w + part;
        const uint64_t m = *mp, s = *sp, any = Pw | Nw;
        const uint64_t nm = ~(m & ~s);
        *mp = (m & ~any) | (Pw & ~(m & s)) | (Nw & nm);
        *sp = (s & ~any) | (Nw & nm);
    }
}

void ternary_neon_r2_row(uint64_t *mask, uint64_t *sign, int8_t *cnt, size_t base, const float *g, size_t n,
                         float beta, float inc, int ti, int tin, int tout, int ri, int pre) {
    float inv = 1.0f;
    if (!pre) { const double ms = ternary_canon_sumsq(g, n) / (double)n; inv = 1.0f / ((float)sqrt(ms) + 1e-12f); }
    const float cb = beta, cg = inc * 127.0f;
    const float32x4_t vinv = f32(inv), vcb = f32(cb), vcg = f32(cg), c3 = f32(3.0f), cm3 = f32(-3.0f), vm = f32(RINT_M);
    const int32x4_t v127 = vdupq_n_s32(127), vm127 = vdupq_n_s32(-127), vti = vdupq_n_s32(ti), vtin = vdupq_n_s32(tin), vtout = vdupq_n_s32(tout);
    const int32x4_t vrp = vdupq_n_s32(-ri), vrn = vdupq_n_s32(ri), vz = vdupq_n_s32(0);
    static const uint32_t wt0[4] = {1, 2, 4, 8}, wt1[4] = {16, 32, 64, 128}, wt2[4] = {256, 512, 1024, 2048}, wt3[4] = {4096, 8192, 16384, 32768};
    const uint32x4_t W0 = vld1q_u32(wt0), W1 = vld1q_u32(wt1), W2 = vld1q_u32(wt2), W3 = vld1q_u32(wt3);
    for (size_t t = 0; t * 16 < n; t++) {
        const size_t rem = n - 16 * t, cnt_rem = rem < 16 ? rem : 16;
        float gt[16]; int8_t ct[16];
        const float *gp = g + 16 * t; const int8_t *cp = cnt + 16 * t;
        if (cnt_rem < 16) { memset(gt, 0, sizeof gt); memset(ct, 0, sizeof ct); memcpy(gt, gp, cnt_rem * 4); memcpy(ct, cp, cnt_rem); gp = gt; cp = ct; }
        const uint16_t m16 = bits16(mask, base + 16 * t), s16 = bits16(sign, base + 16 * t);
        const int8x16_t c8 = vld1q_s8(cp);
        const int16x8_t cl = vmovl_s8(vget_low_s8(c8)), ch = vmovl_s8(vget_high_s8(c8));
        const int32x4_t ci[4] = {vmovl_s16(vget_low_s16(cl)), vmovl_s16(vget_high_s16(cl)), vmovl_s16(vget_low_s16(ch)), vmovl_s16(vget_high_s16(ch))};
        int32x4_t vv[4]; uint32x4_t Pm = vdupq_n_u32(0), Nm = vdupq_n_u32(0);
        const uint32x4_t wts[4] = {W0, W1, W2, W3};
        for (int j = 0; j < 4; j++) {
            float32x4_t gh = vmulq_f32(vld1q_f32(gp + 4 * j), vinv);
            gh = vminq_f32(vmaxq_f32(gh, cm3), c3);                                  /* NaN se propaga: luego x = 0 */
            float32x4_t x = vsubq_f32(vmulq_f32(vcb, vcvtq_f32_s32(ci[j])), vmulq_f32(vcg, gh));
            x = bitsf(vandq_u32(fbits(x), vceqq_f32(x, x)));                         /* NaN -> 0 */
            x = vsubq_f32(vaddq_f32(x, vm), vm);                                     /* rintf */
            int32x4_t v = vminq_s32(vmaxq_s32(vcvtq_s32_f32(x), vm127), v127);
            const uint32x4_t m4 = mask4((uint32_t)(m16 >> (4 * j))), s4 = mask4((uint32_t)(s16 >> (4 * j)));
            const uint32x4_t zero = vmvnq_u32(m4), pos = vbicq_u32(m4, s4), neg = vandq_u32(m4, s4);
            const uint32x4_t dpos = vcgtq_s32(v, vz), dneg = vmvnq_u32(dpos);
            const uint32x4_t tow0 = vorrq_u32(vandq_u32(dpos, neg), vandq_u32(dneg, pos));
            int32x4_t th = vbslq_s32(tow0, vtout, vti);
            th = vbslq_s32(zero, vtin, th);
            const uint32x4_t F = vorrq_u32(vcgeq_s32(v, th), vcleq_s32(v, vnegq_s32(th)));
            const uint32x4_t P = vandq_u32(F, dpos), N = vandq_u32(F, dneg);
            v = vbslq_s32(P, vrp, v); v = vbslq_s32(N, vrn, v);
            vv[j] = v;
            Pm = vorrq_u32(Pm, vandq_u32(P, wts[j])); Nm = vorrq_u32(Nm, vandq_u32(N, wts[j]));
        }
        const int8x16_t out8 = vcombine_s8(vmovn_s16(vcombine_s16(vmovn_s32(vv[0]), vmovn_s32(vv[1]))), vmovn_s16(vcombine_s16(vmovn_s32(vv[2]), vmovn_s32(vv[3]))));
        uint32_t P16 = hsum_u32(Pm), N16 = hsum_u32(Nm);
        if (cnt_rem < 16) {
            int8_t o[16]; vst1q_s8(o, out8); memcpy(cnt + 16 * t, o, cnt_rem);
            const uint32_t lim = (1u << cnt_rem) - 1u; P16 &= lim; N16 &= lim;
        } else vst1q_s8(cnt + 16 * t, out8);
        apply_bits(mask, sign, base + 16 * t, (uint16_t)P16, (uint16_t)N16);
    }
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Una fila de la normalizacion por elemento. Mismos bits que ternary_elemnorm_apply_scalar (ternary_dense.c).
 * Codigo logaritmico sin division por elemento: q' = v * (1/mx) difiere de q = v/mx en <= 2 ulp; el codigo solo cambia al cruzar
 * m = 2^-0.5 (fraccion 0x3504F3), asi que si la fraccion de q' esta a menos de 16 de ese valor se rehace ese grupo con la division
 * exacta. Quedan una division escalar por elemento (g / den[c]) y 16 raices por fila (tabla den[c]). */
#define EN_MT 0.70710678f
#define EN_FRAC_MT 0x3504F3u
static inline int en_code_s(float x) {
    if (!(x >= 0x1p-16f)) return 15;
    int e; const float m = frexpf(x, &e);
    int r = -e + (m < EN_MT ? 1 : 0);
    return r < 0 ? 0 : (r > 15 ? 15 : r);
}
void ternary_neon_elemnorm_row(uint8_t *cr, float *rmax, float *g, float *v, size_t W, float beta2, float omb, float bc, float eps) {
    const size_t st = (W + 1) / 2;
    const float rm = *rmax;
    const float32x4_t vb = f32(beta2), vo = f32(omb), vrm = f32(rm);
    float32x4_t mxv = vdupq_n_f32(0.0f);
    /* paso 1: v = beta2*vh + omb*g^2, con vh = rm * 2^-c (c = nibble del codigo guardado) */
    for (size_t t = 0; t * 16 < W; t++) {
        const size_t rem = W - 16 * t, cnt = rem < 16 ? rem : 16, nb = (cnt + 1) / 2;
        uint8_t bt[8]; memset(bt, 0, 8); memcpy(bt, cr + 8 * t, nb);
        const uint8x8_t b8 = vld1_u8(bt);
        const uint8x8x2_t zz = vzip_u8(vand_u8(b8, vdup_n_u8(15)), vshr_n_u8(b8, 4));           /* 16 codigos en orden */
        const uint8x16_t c16 = vcombine_u8(zz.val[0], zz.val[1]);
        const uint16x8_t cl = vmovl_u8(vget_low_u8(c16)), ch = vmovl_u8(vget_high_u8(c16));
        const uint32x4_t cq[4] = {vmovl_u16(vget_low_u16(cl)), vmovl_u16(vget_high_u16(cl)), vmovl_u16(vget_low_u16(ch)), vmovl_u16(vget_high_u16(ch))};
        for (int j = 0; j < 4; j++) {
            if ((size_t)(4 * j) >= cnt) continue;                     /* v[W..] no se escribe ni se lee */
            const size_t r4 = cnt - 4 * j;
            float32x4_t x;
            if (r4 >= 4) x = vld1q_f32(g + 16 * t + 4 * j);
            else { float tg[4] = {0, 0, 0, 0}; for (size_t i = 0; i < r4; i++) tg[i] = g[16 * t + 4 * j + i]; x = vld1q_f32(tg); }
            const float32x4_t p2 = bitsf(vshlq_n_u32(vsubq_u32(u32(127), cq[j]), 23));            /* 2^-c */
            const float32x4_t vh = vmulq_f32(vrm, p2);
            float32x4_t vn = vaddq_f32(vmulq_f32(vb, vh), vmulq_f32(vo, vmulq_f32(x, x)));
            if (r4 < 4) { float tv[4]; vst1q_f32(tv, vn); for (size_t i = r4; i < 4; i++) tv[i] = 0.0f; vn = vld1q_f32(tv); }
            if (r4 >= 4) vst1q_f32(v + 16 * t + 4 * j, vn);
            else { float tv[4]; vst1q_f32(tv, vn); for (size_t i = 0; i < r4; i++) v[16 * t + 4 * j + i] = tv[i]; }
            mxv = vmaxq_f32(mxv, vn);
        }
    }
    float mx;
    { const float32x2_t m2 = vmax_f32(vget_low_f32(mxv), vget_high_f32(mxv)); mx = vget_lane_f32(vmax_f32(m2, vdup_lane_f32(m2, 1)), 0); }
    if (mx != mx) {                                          /* algun NaN: el escalar lo ignora al tomar el maximo */
        mx = 0.0f;
        for (size_t k = 0; k < W; k++) if (v[k] > mx) mx = v[k];
    }
    *rmax = mx;
    float den[16];
    for (int c = 0; c < 16; c++) den[c] = sqrtf(ldexpf(mx, -c) / bc) + eps;
    const uint8_t keep_hi = (W & 1) ? (uint8_t)(cr[st - 1] & 0xF0) : 0;
    const int fast = mx >= 1e-30f && mx <= 1e30f;
    const float inv = fast ? 1.0f / mx : 0.0f;
    const float32x4_t vinv = f32(inv), vlo = f32(0x1p-16f);
    /* paso 2: codigos, g / den[c] */
    for (size_t t = 0; t * 16 < W; t++) {
        const size_t rem = W - 16 * t, cnt = rem < 16 ? rem : 16, nb = (cnt + 1) / 2;
        int32_t rr[16];
        if (mx > 0.0f) {
            float vt[16];
            if (cnt == 16) memcpy(vt, v + 16 * t, 64); else { memset(vt, 0, 64); memcpy(vt, v + 16 * t, cnt * 4); }
            for (int j = 0; j < 4; j++) {
                int exact = !fast;
                if (fast) {
                    const float32x4_t qv = vmulq_f32(vld1q_f32(vt + 4 * j), vinv);
                    const uint32x4_t qb = fbits(qv), ok = vcgeq_f32(qv, vlo);
                    const int32x4_t E = vreinterpretq_s32_u32(vandq_u32(vshrq_n_u32(qb, 23), u32(0xFF)));
                    const uint32x4_t fr = vandq_u32(qb, u32(0x7FFFFF));
                    const uint32x4_t lt = vcltq_u32(fr, u32(EN_FRAC_MT));
                    int32x4_t r = vsubq_s32(vdupq_n_s32(126), E);
                    r = vsubq_s32(r, vreinterpretq_s32_u32(lt));                                  /* lt = -1 -> +1 */
                    r = vminq_s32(vmaxq_s32(r, vdupq_n_s32(0)), vdupq_n_s32(15));
                    r = vbslq_s32(ok, r, vdupq_n_s32(15));
                    vst1q_s32(rr + 4 * j, r);
                    /* dudoso: fraccion a menos de 16 de la frontera (|fr - T| < 16) y valor normal >= 2^-16 */
                    const uint32x4_t d = vabdq_u32(fr, u32(EN_FRAC_MT));
                    const uint32x4_t amb = vandq_u32(vcltq_u32(d, u32(16)), ok);
                    const uint32x2_t a2 = vorr_u32(vget_low_u32(amb), vget_high_u32(amb));
                    exact = (vget_lane_u32(a2, 0) | vget_lane_u32(a2, 1)) != 0;
                    /* NaN en q': vcgeq da falso -> 15, igual que el escalar (!(x >= lo)) */
                }
                if (exact)
                    for (int i = 0; i < 4; i++) rr[4 * j + i] = en_code_s(vt[4 * j + i] / mx);
            }
        } else for (int i = 0; i < 16; i++) rr[i] = 15;
        for (size_t i = 0; i < cnt; i++) g[16 * t + i] = g[16 * t + i] / den[rr[i]];
        uint8_t ob[8];
        for (int i = 0; i < 8; i++) ob[i] = (uint8_t)(rr[2 * i] | (rr[2 * i + 1] << 4));
        memcpy(cr + 8 * t, ob, nb);
    }
    if (W & 1) cr[st - 1] = (uint8_t)((cr[st - 1] & 0x0F) | keep_hi);
}

#endif
