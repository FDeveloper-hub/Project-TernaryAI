/* Nucleos NEON de 128 bits: atencion, softmax y RoPE: scores_i32, qk_grad, outer_add, attn_scores, attn_pv, attn_bwd, attn_pv2, attn_bwd2, softmax_fwd, softmax_bwd, rope_rotate_int, rope_unrotate_grad.
 * Interfaz y reglas: include/ternary_neon.h. Los esqueletos para compilaciones sin NEON estan en ternary_neon_stubs.c. */
#include "ternary_neon.h"
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) && defined(__GNUC__)
#include <arm_neon.h>

/* Todas las funciones son reentrantes (solo pila y las salidas recibidas). Productos y sumas float van en instrucciones
 * separadas (vmulq + vaddq, nunca vmla/vfma). Nunca se leen ni escriben elementos de salida mas alla de n; las lecturas
 * de K/V traspuestas pueden llegar al relleno de la ultima tanda de 16 columnas (legible por contrato). */

static inline float32x4_t gather4(const float *tab, const uint32_t *k) {
    float32x4_t e = vld1q_dup_f32(tab + k[0]);
    e = vld1q_lane_f32(tab + k[1], e, 1);
    e = vld1q_lane_f32(tab + k[2], e, 2);
    return vld1q_lane_f32(tab + k[3], e, 3);
}

/* Suma de 16 parciales (carril l = parte p[l/4][l%4]) en el arbol fijo de ternary_sum16: (l,l+8), (l,l+4), (0+2, 1+3), a+b. */
static inline float sum16_v(float32x4_t p0, float32x4_t p1, float32x4_t p2, float32x4_t p3) {
    const float32x4_t t4 = vaddq_f32(vaddq_f32(p0, p2), vaddq_f32(p1, p3));
    const float a = vgetq_lane_f32(t4, 0) + vgetq_lane_f32(t4, 2), b = vgetq_lane_f32(t4, 1) + vgetq_lane_f32(t4, 3);
    return a + b;
}

/* e_j = tab[min(m - sc[j], K0)] para una tanda de 16 claves (r = claves validas, 1..16): los carriles >= r valen 0 y no se lee sc[r..]. */
static inline void exp_tile16(const int32_t *sc, size_t r, int32_t m, const float *tab, uint32_t K0, float32x4_t ev[4]) {
    const int32x4_t vm = vdupq_n_s32(m);
    const uint32x4_t vk0 = vdupq_n_u32(K0);
    uint32_t kk[16];
    if (r >= 16) {
        for (int v = 0; v < 4; v++) vst1q_u32(kk + 4 * v, vminq_u32(vreinterpretq_u32_s32(vsubq_s32(vm, vld1q_s32(sc + 4 * v))), vk0));
        for (int v = 0; v < 4; v++) ev[v] = gather4(tab, kk + 4 * v);
    } else {
        int32_t ts[16]; float te[16];
        for (size_t i = 0; i < 16; i++) ts[i] = i < r ? sc[i] : m;
        for (int v = 0; v < 4; v++) vst1q_u32(kk + 4 * v, vminq_u32(vreinterpretq_u32_s32(vsubq_s32(vm, vld1q_s32(ts + 4 * v))), vk0));
        for (int v = 0; v < 4; v++) vst1q_f32(te + 4 * v, gather4(tab, kk + 4 * v));
        for (size_t i = r; i < 16; i++) te[i] = 0.0f;
        for (int v = 0; v < 4; v++) ev[v] = vld1q_f32(te + 4 * v);
    }
}

/* Guarda los r primeros de 16 floats (4 vectores). */
static inline void store_tile16(float *dst, size_t r, const float32x4_t v[4]) {
    if (r >= 16) { for (int i = 0; i < 4; i++) vst1q_f32(dst + 4 * i, v[i]); return; }
    float t[16];
    for (int i = 0; i < 4; i++) vst1q_f32(t + 4 * i, v[i]);
    memcpy(dst, t, r * sizeof(float));
}

/* ---------------------------------------------------------------------------------------------------------------- */
void ternary_neon_scores_i32(const int32_t *a, const int32_t *kb, size_t kstride, size_t H, size_t count, int shift,
                             int32_t lim, int32_t *out) {
    const size_t H4 = H & ~(size_t)3;
    size_t j = 0;
    for (; j + 2 <= count; j += 2) {
        const int32_t *b0 = kb + j * kstride, *b1 = b0 + kstride;
        int64x2_t a0l = vdupq_n_s64(0), a0h = a0l, a1l = a0l, a1h = a0l;
        for (size_t d = 0; d < H4; d += 4) {
            const int32x4_t av = vld1q_s32(a + d), k0 = vld1q_s32(b0 + d), k1 = vld1q_s32(b1 + d);
            a0l = vmlal_s32(a0l, vget_low_s32(av), vget_low_s32(k0));
            a1l = vmlal_s32(a1l, vget_low_s32(av), vget_low_s32(k1));
            a0h = vmlal_s32(a0h, vget_high_s32(av), vget_high_s32(k0));
            a1h = vmlal_s32(a1h, vget_high_s32(av), vget_high_s32(k1));
        }
        const int64x2_t s0 = vaddq_s64(a0l, a0h), s1 = vaddq_s64(a1l, a1h);
        int64_t d0 = vgetq_lane_s64(s0, 0) + vgetq_lane_s64(s0, 1), d1 = vgetq_lane_s64(s1, 0) + vgetq_lane_s64(s1, 1);
        for (size_t d = H4; d < H; d++) { d0 += (int64_t)a[d] * (int64_t)b0[d]; d1 += (int64_t)a[d] * (int64_t)b1[d]; }
        d0 >>= shift; d1 >>= shift;
        if (d0 > lim) d0 = lim; else if (d0 < -lim) d0 = -lim;
        if (d1 > lim) d1 = lim; else if (d1 < -lim) d1 = -lim;
        out[j] = (int32_t)d0; out[j + 1] = (int32_t)d1;
    }
    for (; j < count; j++) {
        const int32_t *b0 = kb + j * kstride;
        int64x2_t al = vdupq_n_s64(0), ah = al;
        for (size_t d = 0; d < H4; d += 4) {
            const int32x4_t av = vld1q_s32(a + d), k0 = vld1q_s32(b0 + d);
            al = vmlal_s32(al, vget_low_s32(av), vget_low_s32(k0));
            ah = vmlal_s32(ah, vget_high_s32(av), vget_high_s32(k0));
        }
        const int64x2_t s0 = vaddq_s64(al, ah);
        int64_t d0 = vgetq_lane_s64(s0, 0) + vgetq_lane_s64(s0, 1);
        for (size_t d = H4; d < H; d++) d0 += (int64_t)a[d] * (int64_t)b0[d];
        d0 >>= shift;
        if (d0 > lim) d0 = lim; else if (d0 < -lim) d0 = -lim;
        out[j] = (int32_t)d0;
    }
}

/* gq[d] += ds[j]*(float)kb[j][d]; gk[j][d] += ds[j]*(float)qv[d]: por tramo de W vectores, gq vive en registros durante los j. */
static inline __attribute__((always_inline)) void qkg_chunk(const int W, float *gq, const int32_t *qv, const int32_t *kb, float *gk,
                                                            size_t kstride, size_t count, const float *ds) {
    float32x4_t aq[4], qf[4];
    for (int w = 0; w < W; w++) { aq[w] = vld1q_f32(gq + 4 * w); qf[w] = vcvtq_f32_s32(vld1q_s32(qv + 4 * w)); }
    for (size_t j = 0; j < count; j++) {
        const float32x4_t dj = vld1q_dup_f32(ds + j);
        const int32_t *kr = kb + j * kstride; float *gkj = gk + j * kstride;
        for (int w = 0; w < W; w++) {
            aq[w] = vaddq_f32(aq[w], vmulq_f32(dj, vcvtq_f32_s32(vld1q_s32(kr + 4 * w))));
            vst1q_f32(gkj + 4 * w, vaddq_f32(vld1q_f32(gkj + 4 * w), vmulq_f32(dj, qf[w])));
        }
    }
    for (int w = 0; w < W; w++) vst1q_f32(gq + 4 * w, aq[w]);
}

void ternary_neon_qk_grad(float *gq, const int32_t *qv, const int32_t *kb, float *gk, size_t kstride, size_t H,
                          size_t count, const float *ds) {
    const size_t H4 = H & ~(size_t)3;
    size_t c = 0;
    for (; c + 16 <= H4; c += 16) qkg_chunk(4, gq + c, qv + c, kb + c, gk + c, kstride, count, ds);
    switch ((H4 - c) / 4) {
        case 3: qkg_chunk(3, gq + c, qv + c, kb + c, gk + c, kstride, count, ds); break;
        case 2: qkg_chunk(2, gq + c, qv + c, kb + c, gk + c, kstride, count, ds); break;
        case 1: qkg_chunk(1, gq + c, qv + c, kb + c, gk + c, kstride, count, ds); break;
        default: break;
    }
    for (size_t d = H4; d < H; d++) {
        const float qd = (float)qv[d];
        for (size_t j = 0; j < count; j++) {
            gq[d] += ds[j] * (float)kb[j * kstride + d];
            gk[j * kstride + d] += ds[j] * qd;
        }
    }
}

void ternary_neon_outer_add(float *dst, size_t stride, size_t H, size_t count, const float *p, const float *ga) {
    const size_t H4 = H & ~(size_t)3;
    for (size_t j = 0; j < count; j++) {
        const float32x4_t vp = vld1q_dup_f32(p + j);
        float *d = dst + j * stride;
        size_t i = 0;
        for (; i + 8 <= H4; i += 8) {
            const float32x4_t g0 = vld1q_f32(ga + i), g1 = vld1q_f32(ga + i + 4), d0 = vld1q_f32(d + i), d1 = vld1q_f32(d + i + 4);
            vst1q_f32(d + i, vaddq_f32(d0, vmulq_f32(vp, g0)));
            vst1q_f32(d + i + 4, vaddq_f32(d1, vmulq_f32(vp, g1)));
        }
        for (; i < H4; i += 4) vst1q_f32(d + i, vaddq_f32(vld1q_f32(d + i), vmulq_f32(vp, vld1q_f32(ga + i))));
        for (; i < H; i++) d[i] += p[j] * ga[i];
    }
}

/* ---- Atencion: puntajes ------------------------------------------------------------------------------------------
 * Tandas de 8 claves x 2 cabezas: 8 acumuladores int64x2; cada carga de K (2 vectores por d) alimenta las dos cabezas.
 * El recorte a [-lim, lim] se hace con vqmovn_s64 (satura a int32) + vmin/vmax: equivalente porque lim cabe en int32. */
static inline __attribute__((always_inline)) void scores_block(const int NB, const int32_t *const *q, const int32_t *kT, size_t ld, size_t H,
                                                               size_t n, int shift, int32_t lim, int32_t *sc, size_t scs, int32_t *m_out) {
    const int64x2_t vsh = vdupq_n_s64(-(int64_t)shift);
    const int32x4_t vlim = vdupq_n_s32(lim), vnlim = vdupq_n_s32(-lim);
    int32x4_t mx[2]; int32_t ms[2];
    for (int b = 0; b < NB; b++) { mx[b] = vdupq_n_s32(INT32_MIN); ms[b] = INT32_MIN; }
    for (size_t j0 = 0; j0 < n; j0 += 8) {
        int64x2_t acc[2][4];
        for (int b = 0; b < NB; b++) for (int i = 0; i < 4; i++) acc[b][i] = vdupq_n_s64(0);
        size_t d = 0;
        for (; d + 2 <= H; d += 2) {
            const int32x4_t k0 = vld1q_s32(kT + d * ld + j0), k1 = vld1q_s32(kT + d * ld + j0 + 4);
            const int32x4_t k2 = vld1q_s32(kT + (d + 1) * ld + j0), k3 = vld1q_s32(kT + (d + 1) * ld + j0 + 4);
            for (int b = 0; b < NB; b++) {
                const int32x2_t qq = vld1_s32(q[b] + d);
                acc[b][0] = vmlal_lane_s32(acc[b][0], vget_low_s32(k0), qq, 0);
                acc[b][1] = vmlal_lane_s32(acc[b][1], vget_high_s32(k0), qq, 0);
                acc[b][2] = vmlal_lane_s32(acc[b][2], vget_low_s32(k1), qq, 0);
                acc[b][3] = vmlal_lane_s32(acc[b][3], vget_high_s32(k1), qq, 0);
                acc[b][0] = vmlal_lane_s32(acc[b][0], vget_low_s32(k2), qq, 1);
                acc[b][1] = vmlal_lane_s32(acc[b][1], vget_high_s32(k2), qq, 1);
                acc[b][2] = vmlal_lane_s32(acc[b][2], vget_low_s32(k3), qq, 1);
                acc[b][3] = vmlal_lane_s32(acc[b][3], vget_high_s32(k3), qq, 1);
            }
        }
        if (d < H) {
            const int32x4_t k0 = vld1q_s32(kT + d * ld + j0), k1 = vld1q_s32(kT + d * ld + j0 + 4);
            for (int b = 0; b < NB; b++) {
                const int32x2_t qq = vld1_dup_s32(q[b] + d);
                acc[b][0] = vmlal_lane_s32(acc[b][0], vget_low_s32(k0), qq, 0);
                acc[b][1] = vmlal_lane_s32(acc[b][1], vget_high_s32(k0), qq, 0);
                acc[b][2] = vmlal_lane_s32(acc[b][2], vget_low_s32(k1), qq, 0);
                acc[b][3] = vmlal_lane_s32(acc[b][3], vget_high_s32(k1), qq, 0);
            }
        }
        const size_t r = n - j0;
        for (int b = 0; b < NB; b++) {
            int32x4_t s0 = vcombine_s32(vqmovn_s64(vshlq_s64(acc[b][0], vsh)), vqmovn_s64(vshlq_s64(acc[b][1], vsh)));
            int32x4_t s1 = vcombine_s32(vqmovn_s64(vshlq_s64(acc[b][2], vsh)), vqmovn_s64(vshlq_s64(acc[b][3], vsh)));
            s0 = vminq_s32(vmaxq_s32(s0, vnlim), vlim);
            s1 = vminq_s32(vmaxq_s32(s1, vnlim), vlim);
            int32_t *dst = sc + (size_t)b * scs + j0;
            if (r >= 8) {
                vst1q_s32(dst, s0); vst1q_s32(dst + 4, s1);
                mx[b] = vmaxq_s32(mx[b], vmaxq_s32(s0, s1));
            } else {
                int32_t t[8];
                vst1q_s32(t, s0); vst1q_s32(t + 4, s1);
                for (size_t i = 0; i < r; i++) { dst[i] = t[i]; if (t[i] > ms[b]) ms[b] = t[i]; }
            }
        }
    }
    for (int b = 0; b < NB; b++) {
        const int32x2_t m2 = vmax_s32(vget_low_s32(mx[b]), vget_high_s32(mx[b]));
        int32_t m = vget_lane_s32(m2, 0) > vget_lane_s32(m2, 1) ? vget_lane_s32(m2, 0) : vget_lane_s32(m2, 1);
        m_out[b] = m > ms[b] ? m : ms[b];
    }
}

void ternary_neon_attn_scores(const int32_t *const *q, size_t G, const int32_t *kT, size_t ld, size_t H, size_t n,
                              int shift, int32_t lim, int32_t *sc, size_t scs, int32_t *m_out) {
    size_t b = 0;
    for (; b + 2 <= G; b += 2) scores_block(2, q + b, kT, ld, H, n, shift, lim, sc + b * scs, scs, m_out + b);
    if (b < G) scores_block(1, q + b, kT, ld, H, n, shift, lim, sc + b * scs, scs, m_out + b);
}

/* ---- Atencion: forward (exponenciales + PV) ---------------------------------------------------------------------- */
/* Exponenciales y 1/suma de una cabeza sobre uno o dos tramos (A en [0,nA), B desde spA). e se guarda solo para j < n del tramo. */
static inline void exp_head(const int32_t *sc, int32_t m, const float *tab, uint32_t K0, size_t nA, size_t nB, size_t spA,
                            float *e, float *inv_out) {
    float32x4_t part[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    for (int seg = 0; seg < 2; seg++) {
        const size_t n = seg ? nB : nA, so = seg ? spA : 0;
        for (size_t t = 0; t < n; t += 16) {
            const size_t r = n - t;
            float32x4_t ev[4];
            exp_tile16(sc + so + t, r, m, tab, K0, ev);
            store_tile16(e + so + t, r, ev);
            for (int v = 0; v < 4; v++) part[v] = vaddq_f32(part[v], ev[v]);
        }
    }
    *inv_out = 1.0f / sum16_v(part[0], part[1], part[2], part[3]);
}

/* acc[b][w] = sum_j e[b][j] * v[j][c0 + 4w..] (j ascendente, tramo A y luego B), por GB cabezas x W vectores. */
static inline __attribute__((always_inline)) void pv_chunk(const int GB, const int W, size_t nA, size_t nB, size_t spA, const float *vA,
                                                           const float *vB, size_t vs, const float *e, size_t es, const float *inv,
                                                           float *out, size_t H, size_t c0) {
    float32x4_t acc[2][4];
    for (int b = 0; b < GB; b++) for (int w = 0; w < W; w++) acc[b][w] = vdupq_n_f32(0.0f);
    for (int seg = 0; seg < 2; seg++) {
        const size_t n = seg ? nB : nA;
        const float *v = (seg ? vB : vA) + c0, *eo = e + (seg ? spA : 0);
        for (size_t j = 0; j < n; j++) {
            float32x4_t vv[4];
            for (int w = 0; w < W; w++) vv[w] = vld1q_f32(v + j * vs + 4 * w);
            for (int b = 0; b < GB; b++) {
                const float32x4_t eb = vld1q_dup_f32(eo + b * es + j);
                for (int w = 0; w < W; w++) acc[b][w] = vaddq_f32(acc[b][w], vmulq_f32(eb, vv[w]));
            }
        }
    }
    for (int b = 0; b < GB; b++) {
        const float32x4_t iv = vdupq_n_f32(inv[b]);
        for (int w = 0; w < W; w++) vst1q_f32(out + b * H + c0 + 4 * w, vmulq_f32(acc[b][w], iv));
    }
}

static inline __attribute__((always_inline)) void pv_heads(const int GB, size_t nA, size_t nB, size_t spA, const float *vA, const float *vB,
                                                           size_t vs, size_t H, const float *e, size_t es, const float *inv, float *out) {
    const size_t H4 = H & ~(size_t)3;
    size_t c = 0;
    for (; c + 16 <= H4; c += 16) pv_chunk(GB, 4, nA, nB, spA, vA, vB, vs, e, es, inv, out, H, c);
    switch ((H4 - c) / 4) {
        case 3: pv_chunk(GB, 3, nA, nB, spA, vA, vB, vs, e, es, inv, out, H, c); break;
        case 2: pv_chunk(GB, 2, nA, nB, spA, vA, vB, vs, e, es, inv, out, H, c); break;
        case 1: pv_chunk(GB, 1, nA, nB, spA, vA, vB, vs, e, es, inv, out, H, c); break;
        default: break;
    }
    for (size_t d = H4; d < H; d++)
        for (int b = 0; b < GB; b++) {
            float acc = 0.0f;
            for (int seg = 0; seg < 2; seg++) {
                const size_t n = seg ? nB : nA; const float *v = seg ? vB : vA, *eo = e + b * es + (seg ? spA : 0);
                for (size_t j = 0; j < n; j++) acc += eo[j] * v[j * vs + d];
            }
            out[b * H + d] = acc * inv[b];
        }
}

void ternary_neon_attn_pv2(size_t G, const int32_t *sc, size_t scs, const int32_t *m, const float *tab, uint32_t K0,
                           size_t nA, size_t nB, size_t spA, const float *vA, const float *vB, size_t vs, size_t H,
                           float *e, size_t es, float *out, float *inv_out) {
    for (size_t b = 0; b < G; b++) exp_head(sc + b * scs, m[b], tab, K0, nA, nB, spA, e + b * es, inv_out + b);
    size_t b = 0;
    for (; b + 2 <= G; b += 2) pv_heads(2, nA, nB, spA, vA, vB, vs, H, e + b * es, es, inv_out + b, out + b * H);
    if (b < G) pv_heads(1, nA, nB, spA, vA, vB, vs, H, e + b * es, es, inv_out + b, out + b * H);
}

void ternary_neon_attn_pv(size_t G, const int32_t *sc, size_t scs, const int32_t *m, const float *tab, uint32_t K0,
                          size_t n, const float *v, size_t vs, size_t H, float *e, size_t es, float *out, float *inv_out) {
    ternary_neon_attn_pv2(G, sc, scs, m, tab, K0, n, 0, (n + 15) / 16 * 16, v, v, vs, H, e, es, out, inv_out);
}

/* ---- Atencion: backward ------------------------------------------------------------------------------------------ */
/* Fase 1 (NBH cabezas): D, dprob por tanda de 16 claves (una carga de V traspuesta alimenta las NBH cabezas), p y ds. */
static inline __attribute__((always_inline)) void bwd_probs(const int NBH, size_t nA, size_t nB, size_t spA, size_t H, const int32_t *sc, size_t scs,
                                                           const int32_t *m, const float *inv, const float *tab, uint32_t K0, const float *ga,
                                                           const float *o, const float *vTA, const float *vTB, size_t ld, float it, float gam,
                                                           float *p, float *ds, size_t es) {
    float D[2];
    for (int b = 0; b < NBH; b++) {
        float d0 = 0.0f;
        for (size_t d = 0; d < H; d++) d0 += ga[b * H + d] * o[b * H + d];   /* D = dO . O, d ascendente */
        D[b] = d0;
    }
    const float32x4_t vit = vdupq_n_f32(it), vgam = vdupq_n_f32(gam);
    for (int seg = 0; seg < 2; seg++) {
        const size_t n = seg ? nB : nA, so = seg ? spA : 0; const float *vT = seg ? vTB : vTA;
        for (size_t t = 0; t < n; t += 16) {
            float32x4_t dp[2][4];
            for (int b = 0; b < NBH; b++) for (int v = 0; v < 4; v++) dp[b][v] = vdupq_n_f32(0.0f);
            for (size_t d = 0; d < H; d++) {
                const float *vr = vT + d * ld + t;
                const float32x4_t v0 = vld1q_f32(vr), v1 = vld1q_f32(vr + 4), v2 = vld1q_f32(vr + 8), v3 = vld1q_f32(vr + 12);
                for (int b = 0; b < NBH; b++) {
                    const float32x4_t g = vld1q_dup_f32(ga + b * H + d);
                    dp[b][0] = vaddq_f32(dp[b][0], vmulq_f32(g, v0));
                    dp[b][1] = vaddq_f32(dp[b][1], vmulq_f32(g, v1));
                    dp[b][2] = vaddq_f32(dp[b][2], vmulq_f32(g, v2));
                    dp[b][3] = vaddq_f32(dp[b][3], vmulq_f32(g, v3));
                }
            }
            const size_t r = n - t;
            for (int b = 0; b < NBH; b++) {
                float32x4_t ev[4], pv[4], dsv[4];
                exp_tile16(sc + b * scs + so + t, r, m[b], tab, K0, ev);
                const float32x4_t viv = vdupq_n_f32(inv[b]), vD = vdupq_n_f32(D[b]);
                for (int v = 0; v < 4; v++) {
                    pv[v] = vmulq_f32(ev[v], viv);
                    dsv[v] = vmulq_f32(vmulq_f32(vmulq_f32(vit, pv[v]), vsubq_f32(dp[b][v], vD)), vgam);
                }
                store_tile16(p + b * es + so + t, r, pv);
                store_tile16(ds + b * es + so + t, r, dsv);
            }
        }
    }
}

/* Fase 2: gv[j] += p*ga, gq += ds*kf, gk[j] += ds*qf; j ascendente por tramo, cabezas del bloque en orden. */
static inline __attribute__((always_inline)) void bwd_acc(const int GB, const int W, size_t nA, size_t nB, size_t spA, size_t H, const float *ga,
                                                          const float *qf, const float *p, const float *ds, size_t es, const int32_t *krA,
                                                          const int32_t *krB, size_t ks, float *gq, float *gkA, float *gkB, float *gvA,
                                                          float *gvB, size_t gs, size_t c0) {
    float32x4_t aq[2][2];
    for (int b = 0; b < GB; b++) for (int w = 0; w < W; w++) aq[b][w] = vld1q_f32(gq + b * H + c0 + 4 * w);
    for (int seg = 0; seg < 2; seg++) {
        const size_t n = seg ? nB : nA, po = seg ? spA : 0;
        const int32_t *kr = (seg ? krB : krA) + c0; float *gk = (seg ? gkB : gkA) + c0, *gv = (seg ? gvB : gvA) + c0;
        for (size_t j = 0; j < n; j++) {
            float32x4_t kf[2], gvv[2], gkk[2];
            for (int w = 0; w < W; w++) {
                kf[w] = vcvtq_f32_s32(vld1q_s32(kr + j * ks + 4 * w));
                gvv[w] = vld1q_f32(gv + j * gs + 4 * w);
                gkk[w] = vld1q_f32(gk + j * gs + 4 * w);
            }
            for (int b = 0; b < GB; b++) {
                const float32x4_t pj = vld1q_dup_f32(p + b * es + po + j), dj = vld1q_dup_f32(ds + b * es + po + j);
                for (int w = 0; w < W; w++) {
                    gvv[w] = vaddq_f32(gvv[w], vmulq_f32(pj, vld1q_f32(ga + b * H + c0 + 4 * w)));
                    aq[b][w] = vaddq_f32(aq[b][w], vmulq_f32(dj, kf[w]));
                    gkk[w] = vaddq_f32(gkk[w], vmulq_f32(dj, vld1q_f32(qf + b * H + c0 + 4 * w)));
                }
            }
            for (int w = 0; w < W; w++) { vst1q_f32(gv + j * gs + 4 * w, gvv[w]); vst1q_f32(gk + j * gs + 4 * w, gkk[w]); }
        }
    }
    for (int b = 0; b < GB; b++) for (int w = 0; w < W; w++) vst1q_f32(gq + b * H + c0 + 4 * w, aq[b][w]);
}

static inline __attribute__((always_inline)) void bwd_acc_heads(const int GB, size_t nA, size_t nB, size_t spA, size_t H, const float *ga,
                                                                const float *qf, const float *p, const float *ds, size_t es, const int32_t *krA,
                                                                const int32_t *krB, size_t ks, float *gq, float *gkA, float *gkB, float *gvA,
                                                                float *gvB, size_t gs) {
    const size_t H4 = H & ~(size_t)3;
    size_t c = 0;
    for (; c + 8 <= H4; c += 8) bwd_acc(GB, 2, nA, nB, spA, H, ga, qf, p, ds, es, krA, krB, ks, gq, gkA, gkB, gvA, gvB, gs, c);
    if (c < H4) bwd_acc(GB, 1, nA, nB, spA, H, ga, qf, p, ds, es, krA, krB, ks, gq, gkA, gkB, gvA, gvB, gs, c);
    for (size_t d = H4; d < H; d++)                               /* columnas sobrantes (H % 4): escalar, mismo orden por elemento */
        for (int b = 0; b < GB; b++)
            for (int seg = 0; seg < 2; seg++) {
                const size_t n = seg ? nB : nA, po = seg ? spA : 0;
                const int32_t *kr = seg ? krB : krA; float *gk = seg ? gkB : gkA, *gv = seg ? gvB : gvA;
                for (size_t j = 0; j < n; j++) {
                    const float pj = p[b * es + po + j], dj = ds[b * es + po + j];
                    gv[j * gs + d] += pj * ga[b * H + d];
                    gq[b * H + d] += dj * (float)kr[j * ks + d];
                    gk[j * gs + d] += dj * qf[b * H + d];
                }
            }
}

void ternary_neon_attn_bwd2(size_t G, size_t nA, size_t nB, size_t spA, size_t H, const int32_t *sc, size_t scs,
                            const int32_t *m, const float *inv, const float *tab, uint32_t K0, const float *ga,
                            const float *o, const float *vTA, const float *vTB, size_t ld, float it, float gam,
                            const int32_t *qr, const int32_t *krA, const int32_t *krB, size_t ks, float *gq,
                            float *gkA, float *gkB, float *gvA, float *gvB, size_t gs, float *p, float *ds,
                            size_t es, float *qf) {
    const size_t H4 = H & ~(size_t)3;
    size_t b = 0;
    while (b < G) {
        const int nb = (G - b >= 2) ? 2 : 1;
        if (nb == 2) bwd_probs(2, nA, nB, spA, H, sc + b * scs, scs, m + b, inv + b, tab, K0, ga + b * H, o + b * H, vTA, vTB, ld, it, gam, p + b * es, ds + b * es, es);
        else         bwd_probs(1, nA, nB, spA, H, sc + b * scs, scs, m + b, inv + b, tab, K0, ga + b * H, o + b * H, vTA, vTB, ld, it, gam, p + b * es, ds + b * es, es);
        for (int i = 0; i < nb; i++) {
            const int32_t *q0 = qr + (b + i) * H; float *f0 = qf + (b + i) * H;
            for (size_t d = 0; d < H4; d += 4) vst1q_f32(f0 + d, vcvtq_f32_s32(vld1q_s32(q0 + d)));
            for (size_t d = H4; d < H; d++) f0[d] = (float)q0[d];
        }
        if (nb == 2) bwd_acc_heads(2, nA, nB, spA, H, ga + b * H, qf + b * H, p + b * es, ds + b * es, es, krA, krB, ks, gq + b * H, gkA, gkB, gvA, gvB, gs);
        else         bwd_acc_heads(1, nA, nB, spA, H, ga + b * H, qf + b * H, p + b * es, ds + b * es, es, krA, krB, ks, gq + b * H, gkA, gkB, gvA, gvB, gs);
        b += (size_t)nb;
    }
}

void ternary_neon_attn_bwd(size_t G, size_t n, size_t H, const int32_t *sc, size_t scs, const int32_t *m,
                           const float *inv, const float *tab, uint32_t K0, const float *ga, const float *o,
                           const float *vT, size_t ld, float it, float gam, const int32_t *qr, const int32_t *kr,
                           size_t ks, float *gq, float *gk, float *gv, size_t gs, float *p, float *ds, size_t es,
                           float *qf) {
    ternary_neon_attn_bwd2(G, n, 0, (n + 15) / 16 * 16, H, sc, scs, m, inv, tab, K0, ga, o, vT, vT, ld, it, gam, qr, kr, kr, ks,
                           gq, gk, gk, gv, gv, gs, p, ds, es, qf);
}

/* ---- Softmax entero ---------------------------------------------------------------------------------------------- */
void ternary_neon_softmax_fwd(const int32_t *scores, size_t n_valid, const float *table, size_t table_len, float *out) {
    int32x4_t mv = vdupq_n_s32(INT32_MIN);
    int32_t ms = INT32_MIN;
    size_t i = 0;
    for (; i + 4 <= n_valid; i += 4) mv = vmaxq_s32(mv, vld1q_s32(scores + i));
    for (; i < n_valid; i++) if (scores[i] > ms) ms = scores[i];
    const int32x2_t m2 = vmax_s32(vget_low_s32(mv), vget_high_s32(mv));
    int32_t m = vget_lane_s32(m2, 0) > vget_lane_s32(m2, 1) ? vget_lane_s32(m2, 0) : vget_lane_s32(m2, 1);
    if (ms > m) m = ms;
    const int32x4_t vm = vdupq_n_s32(m), vcap = vdupq_n_s32((int32_t)table_len - 1);
    float32x4_t part[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    size_t t = 0;
    for (; t + 16 <= n_valid; t += 16) {                         /* k = m - s >= 0 (saturado: igual que la cuenta en int64 + tope) */
        uint32_t kk[16];
        for (int v = 0; v < 4; v++) vst1q_u32(kk + 4 * v, vreinterpretq_u32_s32(vminq_s32(vqsubq_s32(vm, vld1q_s32(scores + t + 4 * v)), vcap)));
        for (int v = 0; v < 4; v++) {
            const float32x4_t ev = gather4(table, kk + 4 * v);
            vst1q_f32(out + t + 4 * v, ev);
            part[v] = vaddq_f32(part[v], ev);
        }
    }
    if (t < n_valid) {
        const size_t r = n_valid - t;
        int32_t ts[16]; uint32_t kk[16]; float te[16];
        for (size_t q = 0; q < 16; q++) ts[q] = q < r ? scores[t + q] : m;
        for (int v = 0; v < 4; v++) vst1q_u32(kk + 4 * v, vreinterpretq_u32_s32(vminq_s32(vqsubq_s32(vm, vld1q_s32(ts + 4 * v)), vcap)));
        for (int v = 0; v < 4; v++) vst1q_f32(te + 4 * v, gather4(table, kk + 4 * v));
        for (size_t q = r; q < 16; q++) te[q] = 0.0f;
        for (int v = 0; v < 4; v++) part[v] = vaddq_f32(part[v], vld1q_f32(te + 4 * v));
        memcpy(out + t, te, r * sizeof(float));
    }
    const float inv = 1.0f / sum16_v(part[0], part[1], part[2], part[3]);
    const float32x4_t vinv = vdupq_n_f32(inv);
    for (i = 0; i + 4 <= n_valid; i += 4) vst1q_f32(out + i, vmulq_f32(vld1q_f32(out + i), vinv));
    for (; i < n_valid; i++) out[i] *= inv;
}

void ternary_neon_softmax_bwd(const float *y, const float *g, size_t n, float inv_temp, float *out_ds) {
    float32x4_t part[4] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    size_t t = 0;
    for (; t + 16 <= n; t += 16)
        for (int v = 0; v < 4; v++) part[v] = vaddq_f32(part[v], vmulq_f32(vld1q_f32(g + t + 4 * v), vld1q_f32(y + t + 4 * v)));
    if (t < n) {
        const size_t r = n - t;
        float tg[16], ty[16];
        memset(tg, 0, sizeof tg); memset(ty, 0, sizeof ty);
        memcpy(tg, g + t, r * sizeof(float)); memcpy(ty, y + t, r * sizeof(float));
        for (int v = 0; v < 4; v++) part[v] = vaddq_f32(part[v], vmulq_f32(vld1q_f32(tg + 4 * v), vld1q_f32(ty + 4 * v)));
    }
    const float dot = sum16_v(part[0], part[1], part[2], part[3]);
    const float32x4_t vdot = vdupq_n_f32(dot), vt = vdupq_n_f32(inv_temp);
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(out_ds + i, vmulq_f32(vmulq_f32(vt, vld1q_f32(y + i)), vsubq_f32(vld1q_f32(g + i), vdot)));
    for (; i < n; i++) out_ds[i] = inv_temp * y[i] * (g[i] - dot);
}

/* ---- RoPE entera ------------------------------------------------------------------------------------------------- */
void ternary_neon_rope_rotate_int(const int32_t *z, size_t heads, size_t H, size_t P, const int16_t *c, const int16_t *s, int32_t *out) {
    const size_t P4 = P & ~(size_t)3;
    for (size_t h = 0; h < heads; h++) {
        const int32_t *zp = z + h * H; int32_t *op = out + h * H;
        size_t i = 0;
        for (; i < P4; i += 4) {
            const int32x4x2_t ab = vld2q_s32(zp + 2 * i);
            const int32x4_t cv = vmovl_s16(vld1_s16(c + i)), sv = vmovl_s16(vld1_s16(s + i));
            int32x4x2_t o;
            o.val[0] = vsubq_s32(vmulq_s32(ab.val[0], cv), vmulq_s32(ab.val[1], sv));
            o.val[1] = vaddq_s32(vmulq_s32(ab.val[0], sv), vmulq_s32(ab.val[1], cv));
            vst2q_s32(op + 2 * i, o);
        }
        for (; i < P; i++) {
            const int32_t a = zp[2 * i], b = zp[2 * i + 1];
            op[2 * i] = (int32_t)((uint32_t)a * (uint32_t)(int32_t)c[i] - (uint32_t)b * (uint32_t)(int32_t)s[i]);
            op[2 * i + 1] = (int32_t)((uint32_t)a * (uint32_t)(int32_t)s[i] + (uint32_t)b * (uint32_t)(int32_t)c[i]);
        }
    }
}

void ternary_neon_rope_unrotate_grad(float *grad, size_t heads, size_t H, size_t P, const int16_t *c, const int16_t *s, float inv) {
    const size_t P4 = P & ~(size_t)3;
    const float32x4_t vinv = vdupq_n_f32(inv);
    for (size_t h = 0; h < heads; h++) {
        float *gp = grad + h * H;
        size_t i = 0;
        for (; i < P4; i += 4) {
            const float32x4x2_t ab = vld2q_f32(gp + 2 * i);
            const float32x4_t cf = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vld1_s16(c + i))), vinv);
            const float32x4_t sf = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vld1_s16(s + i))), vinv);
            float32x4x2_t o;
            o.val[0] = vaddq_f32(vmulq_f32(ab.val[0], cf), vmulq_f32(ab.val[1], sf));
            o.val[1] = vaddq_f32(vmulq_f32(vnegq_f32(ab.val[0]), sf), vmulq_f32(ab.val[1], cf));
            vst2q_f32(gp + 2 * i, o);
        }
        for (; i < P; i++) {
            const float ga = gp[2 * i], gb = gp[2 * i + 1];
            const float cf = c[i] * inv, sf = s[i] * inv;
            gp[2 * i] = ga * cf + gb * sf;
            gp[2 * i + 1] = -ga * sf + gb * cf;
        }
    }
}

#endif
