#include "ternary_mem.h"
#include "ternary_avx512.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void ternary_avx512_split_planes(uint16_t *plus, uint16_t *minus, size_t n, const uint64_t *mask, const uint64_t *sign) {
    const size_t nt = (n + 15) / 16;
    const unsigned char *mb = (const unsigned char *)mask, *sb = (const unsigned char *)sign;
    for (size_t t = 0; t < nt; t++) {
        uint16_t m = (uint16_t)(mb[2 * t] | (mb[2 * t + 1] << 8)), s = (uint16_t)(sb[2 * t] | (sb[2 * t + 1] << 8));
        if (t + 1 == nt && (n & 15)) { const uint16_t lim = (uint16_t)((1u << (n & 15)) - 1u); m &= lim; s &= lim; }
        plus[t] = (uint16_t)(m & (uint16_t)~s);
        minus[t] = (uint16_t)(m & s);
    }
}

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>

int ternary_avx512_enabled(void) {
    static int v = -1;
    if (v < 0) {
        __builtin_cpu_init();
        const char *off = getenv("TERNARYAI_NO_AVX512");
        v = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
            __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl") &&
            !(off && off[0] && off[0] != '0');
    }
    return v;
}

/* VNNI (vpdpbusd): ademas de AVX-512, y se puede apagar con TERNARYAI_NO_VNNI=1. Es aritmetica entera exacta: el resultado
 * no depende de que se use o no. */
int ternary_avx512_vnni_enabled(void) {
    static int v = -1;
    if (v < 0) {
        const char *off = getenv("TERNARYAI_NO_VNNI");
        v = ternary_avx512_enabled() && __builtin_cpu_supports("avx512vnni") && !(off && off[0] && off[0] != '0');
    }
    return v;
}

#define AVX512_FN __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl")))
#define AVX512_INL __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl"), always_inline)) static inline

/* ---- grad_w ------------------------------------------------------------------------------------------------
 * T tiles de 16 columnas completos a la vez: cada acumulador es una cadena independiente, asi que T=8 oculta la
 * latencia de la suma (la cadena de una columna es secuencial en s: es el orden de siempre). */
AVX512_INL void gw_tiles(float *rowp, size_t S, const float *g, const uint16_t *xp, const uint16_t *xn, size_t nt16, const int T) {
    __m512 acc[8];
    for (int t = 0; t < T; t++) acc[t] = _mm512_loadu_ps(rowp + 16 * t);
    for (size_t s = 0; s < S; s++) {
        const __m512 gv = _mm512_set1_ps(g[s]);
        const uint16_t *p = xp + s * nt16, *q = xn + s * nt16;
        for (int t = 0; t < T; t++) {
            acc[t] = _mm512_mask_add_ps(acc[t], (__mmask16)p[t], acc[t], gv);
            acc[t] = _mm512_mask_sub_ps(acc[t], (__mmask16)q[t], acc[t], gv);
        }
    }
    for (int t = 0; t < T; t++) _mm512_storeu_ps(rowp + 16 * t, acc[t]);
}

AVX512_FN void ternary_avx512_gradw(float *base, size_t ld, size_t outd, size_t n, size_t S, const float *gT,
                                    const uint16_t *xp, const uint16_t *xn) {
    const size_t nt16 = (n + 15) / 16, full = n / 16;
    for (size_t o = 0; o < outd; o++) {
        float *row = base + o * ld;
        const float *g = gT + o * S;
        size_t t = 0;
        for (; t + 8 <= full; t += 8) gw_tiles(row + 16 * t, S, g, xp + t, xn + t, nt16, 8);
        if (t + 4 <= full) { gw_tiles(row + 16 * t, S, g, xp + t, xn + t, nt16, 4); t += 4; }
        for (; t < full; t++) gw_tiles(row + 16 * t, S, g, xp + t, xn + t, nt16, 1);
        if (full < nt16) {   /* ultima tile parcial: carga/guarda con mascara */
            const __mmask16 lim = (__mmask16)((1u << (n & 15)) - 1u);
            __m512 acc = _mm512_maskz_loadu_ps(lim, row + 16 * full);
            for (size_t s = 0; s < S; s++) {
                const __m512 gv = _mm512_set1_ps(g[s]);
                acc = _mm512_mask_add_ps(acc, (__mmask16)xp[s * nt16 + full], acc, gv);
                acc = _mm512_mask_sub_ps(acc, (__mmask16)xn[s * nt16 + full], acc, gv);
            }
            _mm512_mask_storeu_ps(row + 16 * full, lim, acc);
        }
    }
}

/* ---- grad_x ------------------------------------------------------------------------------------------------
 * SB muestras x T tiles: las mascaras de W (una lectura por fila y tile) las comparten las SB muestras. */
/* Posiciones por bloque (acumuladores independientes: 4 x 8 tiles = 32 serian demasiados, con T = 8 se usan 4 x 8 = 32 zmm solo
 * cuando T = 8: el compilador reparte; medido: 4 da el mejor aprovechamiento de los dos puertos FP) y mascaras cargadas directo a
 * los registros k (kmovw desde memoria: no pasa por el puerto 5, que comparte con las sumas). */
#ifndef GX_SB
#define GX_SB 4
#endif
#define GX_MASK(ptr) _load_mask16((__mmask16 *)(ptr))
AVX512_INL void gx_block(float *const *gx, size_t s0, const int SB, size_t t0, const int T, size_t outd, const float *gs,
                         const uint16_t *wp, const uint16_t *wn, size_t nt16, const float scale, const int hs) {
    __m512 acc[4][8];
    for (int b = 0; b < SB; b++) for (int t = 0; t < T; t++) acc[b][t] = _mm512_loadu_ps(gx[s0 + b] + 16 * (t0 + t));
    for (size_t o = 0; o < outd; o++) {
        const uint16_t *p = wp + o * nt16 + t0, *q = wn + o * nt16 + t0;
        __m512 gv[4];
        for (int b = 0; b < SB; b++) gv[b] = _mm512_set1_ps(hs ? gs[(s0 + b) * outd + o] * scale : gs[(s0 + b) * outd + o]);
        for (int t = 0; t < T; t++) {
            const __mmask16 kp = GX_MASK(p + t), kq = GX_MASK(q + t);
            for (int b = 0; b < SB; b++) {
                acc[b][t] = _mm512_mask_add_ps(acc[b][t], kp, acc[b][t], gv[b]);
                acc[b][t] = _mm512_mask_sub_ps(acc[b][t], kq, acc[b][t], gv[b]);
            }
        }
    }
    for (int b = 0; b < SB; b++) for (int t = 0; t < T; t++) _mm512_storeu_ps(gx[s0 + b] + 16 * (t0 + t), acc[b][t]);
}

AVX512_INL void gradx_impl(float *const *gx, size_t S, size_t outd, size_t n, const float *gs,
                           const uint16_t *wp, const uint16_t *wn, const float scale, const int hs) {
    const size_t nt16 = (n + 15) / 16, full = n / 16;
    size_t t = 0;
    #define GX_COLS(TV) { \
        size_t s = 0; \
        for (; s + GX_SB <= S; s += GX_SB) gx_block(gx, s, GX_SB, t, TV, outd, gs, wp, wn, nt16, scale, hs); \
        switch (S - s) { \
            case 3: gx_block(gx, s, 3, t, TV, outd, gs, wp, wn, nt16, scale, hs); break; \
            case 2: gx_block(gx, s, 2, t, TV, outd, gs, wp, wn, nt16, scale, hs); break; \
            case 1: gx_block(gx, s, 1, t, TV, outd, gs, wp, wn, nt16, scale, hs); break; \
            default: break; } }
    for (; t + 8 <= full; t += 8) GX_COLS(8)
    if (t + 4 <= full) { GX_COLS(4) t += 4; }
    for (; t < full; t++) GX_COLS(1)
    #undef GX_COLS
    if (full < nt16) {
        const __mmask16 lim = (__mmask16)((1u << (n & 15)) - 1u);
        for (size_t s = 0; s < S; s++) {
            __m512 acc = _mm512_maskz_loadu_ps(lim, gx[s] + 16 * full);
            for (size_t o = 0; o < outd; o++) {
                const __m512 gv = _mm512_set1_ps(hs ? gs[s * outd + o] * scale : gs[s * outd + o]);
                acc = _mm512_mask_add_ps(acc, (__mmask16)wp[o * nt16 + full], acc, gv);
                acc = _mm512_mask_sub_ps(acc, (__mmask16)wn[o * nt16 + full], acc, gv);
            }
            _mm512_mask_storeu_ps(gx[s] + 16 * full, lim, acc);
        }
    }
}

AVX512_FN void ternary_avx512_gradx(float *const *gx, size_t S, size_t outd, size_t n, const float *gs,
                                    const uint16_t *wp, const uint16_t *wn) {
    gradx_impl(gx, S, outd, n, gs, wp, wn, 1.0f, 0);
}

/* Igual que ternary_avx512_gradx pero con g = gs[...] * scale calculado dentro (el mismo producto float que
 * ternary_dense_u8_gradx hace antes de sumar): no hace falta un buffer con los gradientes ya escalados. */
AVX512_FN void ternary_avx512_gradx_scaled(float *const *gx, size_t S, size_t outd, size_t n, const float *g,
                                           float scale, const uint16_t *wp, const uint16_t *wn) {
    gradx_impl(gx, S, outd, n, g, wp, wn, scale, 1);
}

/* Operaciones SIN contraccion (GCC fusionaria a*b+c en FMA con los operadores genericos; las versiones _round son
 * builtins y se mantienen separadas). */
#define RN (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
#define MULPS(a, b) _mm512_mul_round_ps((a), (b), RN)
#define ADDPS(a, b) _mm512_add_round_ps((a), (b), RN)
#define SUBPS(a, b) _mm512_sub_round_ps((a), (b), RN)

static inline __mmask16 tail_lim(size_t n, size_t t) {
    const size_t rem = n - 16 * t;
    return (rem >= 16) ? (__mmask16)0xFFFFu : (__mmask16)((1u << rem) - 1u);
}

AVX512_FN void ternary_avx512_pack_thr(const float *x, size_t n, float thr, uint64_t *mask, uint64_t *sign) {
    unsigned char *mb = (unsigned char *)mask, *sb = (unsigned char *)sign;
    const __m512 vt = _mm512_set1_ps(thr), vnt = _mm512_set1_ps(-thr);
    const size_t nt = (n + 15) / 16;
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(n, t);
        const __m512 v = _mm512_maskz_loadu_ps(lim, x + 16 * t);
        const uint16_t p = (uint16_t)(_mm512_cmp_ps_mask(v, vt, _CMP_GT_OQ) & lim);
        const uint16_t q = (uint16_t)(_mm512_cmp_ps_mask(v, vnt, _CMP_LT_OQ) & lim);
        uint16_t m = (uint16_t)(p | q), s = q;
        if (lim != 0xFFFFu) {
            uint16_t om, os;
            memcpy(&om, mb + 2 * t, 2); memcpy(&os, sb + 2 * t, 2);
            m = (uint16_t)((om & ~lim) | m); s = (uint16_t)((os & ~lim) | s);
        }
        memcpy(mb + 2 * t, &m, 2); memcpy(sb + 2 * t, &s, 2);
    }
}

/* Suma de los 16 carriles en el mismo arbol que ternary_sum16: (l, l+8), (l, l+4), (0+2, 1+3), luego (a + b). */
AVX512_INL float sum16_ps(__m512 p) {
    const __m256 t8 = _mm256_add_ps(_mm512_castps512_ps256(p), _mm512_extractf32x8_ps(p, 1));
    const __m128 t4 = _mm_add_ps(_mm256_castps256_ps128(t8), _mm256_extractf128_ps(t8, 1));
    const __m128 s2 = _mm_add_ps(t4, _mm_movehl_ps(t4, t4));
    return _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)));
}
/* Igual que ternary_canon_reduce_d16 (ternary_fexp.h): (a0+a1), luego (l, l+4), (l, l+2), (0+1). */
AVX512_INL double reduce_d16(__m512d a0, __m512d a1) {
    const __m512d v = _mm512_add_pd(a0, a1);
    const __m256d t4 = _mm256_add_pd(_mm512_castpd512_pd256(v), _mm512_extractf64x4_pd(v, 1));
    const __m128d t2 = _mm_add_pd(_mm256_castpd256_pd128(t4), _mm256_extractf128_pd(t4, 1));
    return _mm_cvtsd_f64(t2) + _mm_cvtsd_f64(_mm_unpackhi_pd(t2, t2));
}

AVX512_FN float ternary_avx512_sum_abs(const float *x, size_t n) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    const size_t nt = (n + 15) / 16;
    size_t t = 0;
    for (; t + 2 <= nt && 16 * (t + 2) <= n; t += 2) {
        a0 = ADDPS(a0, _mm512_abs_ps(_mm512_loadu_ps(x + 16 * t)));
        a1 = ADDPS(a1, _mm512_abs_ps(_mm512_loadu_ps(x + 16 * t + 16)));
    }
    for (; t < nt; t++)
        a0 = ADDPS(a0, _mm512_abs_ps(_mm512_maskz_loadu_ps(tail_lim(n, t), x + 16 * t)));
    return sum16_ps(ADDPS(a0, a1));
}

AVX512_INL double ss_row(const float *x, size_t D) {
    __m512d a0 = _mm512_setzero_pd(), a1 = _mm512_setzero_pd();
    const size_t nt = (D + 15) / 16;
    for (size_t t = 0; t < nt; t++) {
        const __m512 v = _mm512_maskz_loadu_ps(tail_lim(D, t), x + 16 * t);
        const __m512d lo = _mm512_cvtps_pd(_mm512_castps512_ps256(v));
        const __m512d hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v, 1));
        a0 = _mm512_add_pd(a0, _mm512_mul_pd(lo, lo));   /* x*x en double es exacto: FMA daria lo mismo */
        a1 = _mm512_add_pd(a1, _mm512_mul_pd(hi, hi));
    }
    return reduce_d16(a0, a1);
}

static inline float inv_rms_from_ss(double ss, size_t D, float eps) {
    return (float)(1.0 / sqrt(ss / (double)D + (double)eps));
}

AVX512_FN void ternary_avx512_rmsnorm_forward(size_t rows, size_t D, float eps, float scale, const uint16_t *gp,
                                              const uint16_t *gm, const float *x, float *out) {
    const size_t nt = (D + 15) / 16;
    const __m512 sgn = _mm512_set1_ps(-0.0f);
    for (size_t i = 0; i < rows; i++) {
        const float *xi = x + i * D;
        float *oi = out + i * D;
        const float f = scale * inv_rms_from_ss(ss_row(xi, D), D, eps);
        const __m512 vf = _mm512_set1_ps(f);
        for (size_t t = 0; t < nt; t++) {
            const __mmask16 lim = tail_lim(D, t);
            const __m512 v = MULPS(vf, _mm512_maskz_loadu_ps(lim, xi + 16 * t));
            __m512 r = _mm512_maskz_mov_ps(gp[t], v);
            r = _mm512_mask_xor_ps(r, gm[t], v, sgn);
            _mm512_mask_storeu_ps(oi + 16 * t, lim, r);
        }
    }
}

AVX512_FN void ternary_avx512_rmsnorm_backward(size_t rows, size_t D, float eps, float scale, const uint16_t *gp,
                                               const uint16_t *gm, const float *x, const float *go,
                                               float *grad_gain_out, float *grad_x_out, double *gscale) {
    const size_t nt = (D + 15) / 16;
    const __m512 one = _mm512_set1_ps(1.0f), mone = _mm512_set1_ps(-1.0f), vs = _mm512_set1_ps(scale);
    for (size_t i = 0; i < rows; i++) {
        const float *xi = x + i * D, *gi = go + i * D;
        const float r = inv_rms_from_ss(ss_row(xi, D), D, eps);
        const __m512 vr = _mm512_set1_ps(r);
        __m512d d0 = _mm512_setzero_pd(), d1 = _mm512_setzero_pd();
        for (size_t t = 0; t < nt; t++) {
            const __mmask16 lim = tail_lim(D, t);
            const __m512 xv = _mm512_maskz_loadu_ps(lim, xi + 16 * t), gv = _mm512_maskz_loadu_ps(lim, gi + 16 * t);
            /* grad_gain[j] += ((go*s)*x)*r  (mismo orden de productos que el escalar) */
            const __m512 gg = MULPS(MULPS(MULPS(gv, vs), xv), vr);
            const __m512 acc = _mm512_maskz_loadu_ps(lim, grad_gain_out + 16 * t);
            _mm512_mask_storeu_ps(grad_gain_out + 16 * t, lim, ADDPS(acc, gg));
            /* dot += g*go*x en double: g*go es +-go o 0 y el producto con x es exacto en double */
            const __m512 sg = _mm512_mask_mov_ps(_mm512_maskz_mov_ps(gp[t], gv), gm[t], _mm512_xor_ps(gv, _mm512_set1_ps(-0.0f)));
            const __m256 sgx_lo = _mm512_castps512_ps256(sg), sgx_hi = _mm512_extractf32x8_ps(sg, 1);
            d0 = _mm512_add_pd(d0, _mm512_mul_pd(_mm512_cvtps_pd(sgx_lo), _mm512_cvtps_pd(_mm512_castps512_ps256(xv))));
            d1 = _mm512_add_pd(d1, _mm512_mul_pd(_mm512_cvtps_pd(sgx_hi), _mm512_cvtps_pd(_mm512_extractf32x8_ps(xv, 1))));
        }
        const double dot = reduce_d16(d0, d1);
        *gscale += dot * (double)r;
        if (grad_x_out) {
            float *gx = grad_x_out + i * D;
            const float c = (float)(dot * (double)r * (double)r / (double)D);
            const float sr = scale * r;
            const __m512 vc = _mm512_set1_ps(c), vsr = _mm512_set1_ps(sr);
            for (size_t t = 0; t < nt; t++) {
                const __mmask16 lim = tail_lim(D, t);
                const __m512 xv = _mm512_maskz_loadu_ps(lim, xi + 16 * t), gv = _mm512_maskz_loadu_ps(lim, gi + 16 * t);
                const __m512 g = _mm512_mask_mov_ps(_mm512_maskz_mov_ps(gp[t], one), gm[t], mone);
                const __m512 inner = SUBPS(MULPS(g, gv), MULPS(xv, vc));
                const __m512 acc = _mm512_maskz_loadu_ps(lim, gx + 16 * t);
                _mm512_mask_storeu_ps(gx + 16 * t, lim, ADDPS(acc, MULPS(vsr, inner)));
            }
        }
    }
}

AVX512_INL __m512 expand_pm(__mmask16 p, __mmask16 m) {
    return _mm512_mask_mov_ps(_mm512_maskz_mov_ps(p, _mm512_set1_ps(1.0f)), m, _mm512_set1_ps(-1.0f));
}

AVX512_FN void ternary_avx512_bwd1_row(size_t in, const uint16_t *xp, const uint16_t *xn, const uint16_t *wp,
                                       const uint16_t *wn, float g, float gs, float *grad_row, float *gx,
                                       int accumulate) {
    const size_t nt = (in + 15) / 16;
    const __m512 vg = _mm512_set1_ps(g), vgs = _mm512_set1_ps(gs);
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(in, t);
        const __m512 xv = expand_pm(xp[t], xn[t]), wv = expand_pm(wp[t], wn[t]);
        __m512 gr = MULPS(vg, xv);
        if (accumulate) gr = ADDPS(_mm512_maskz_loadu_ps(lim, grad_row + 16 * t), gr);
        _mm512_mask_storeu_ps(grad_row + 16 * t, lim, gr);
        _mm512_mask_storeu_ps(gx + 16 * t, lim, ADDPS(_mm512_maskz_loadu_ps(lim, gx + 16 * t), MULPS(vgs, wv)));
    }
}

static inline uint16_t get16(const uint64_t *pl, size_t nbits_total, size_t pos) {
    const unsigned char *b = (const unsigned char *)pl;
    const size_t nbytes = (nbits_total + 7) / 8, by = pos >> 3;
    uint32_t v = 0;
    for (int i = 0; i < 3; i++) if (by + i < nbytes) v |= (uint32_t)b[by + i] << (8 * i);
    return (uint16_t)(v >> (pos & 7));
}

AVX512_FN void ternary_avx512_unpack_row(const uint64_t *mask, const uint64_t *sign, size_t nbits_total, size_t bit0,
                                         size_t n, float scale, float *out, int32_t *raw) {
    const size_t nt = (n + 15) / 16;
    const __m512 vs = _mm512_set1_ps(scale);
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(n, t);
        const uint16_t m = (uint16_t)(get16(mask, nbits_total, bit0 + 16 * t) & lim), sg = (uint16_t)(get16(sign, nbits_total, bit0 + 16 * t) & lim);
        const uint16_t p = (uint16_t)(m & ~sg), q = (uint16_t)(m & sg);
        _mm512_mask_storeu_ps(out + 16 * t, lim, MULPS(vs, expand_pm(p, q)));
        if (raw) _mm512_mask_storeu_epi32(raw + 16 * t, lim, _mm512_mask_mov_epi32(_mm512_maskz_mov_epi32(p, _mm512_set1_epi32(1)), q, _mm512_set1_epi32(-1)));
    }
}

AVX512_FN void ternary_avx512_scores_i32(const int32_t *a, const int32_t *kb, size_t kstride, size_t H, size_t count,
                                         int shift, int32_t lim, int32_t *out) {
    const size_t nt = (H + 15) / 16;
    for (size_t j = 0; j < count; j++) {
        const int32_t *b = kb + j * kstride;
        __m512i acc = _mm512_setzero_si512();
        for (size_t t = 0; t < nt; t++) {
            const __mmask16 lm = tail_lim(H, t);
            const __m512i av = _mm512_maskz_loadu_epi32(lm, a + 16 * t), bv = _mm512_maskz_loadu_epi32(lm, b + 16 * t);
            acc = _mm512_add_epi64(acc, _mm512_mul_epi32(av, bv));
            acc = _mm512_add_epi64(acc, _mm512_mul_epi32(_mm512_srli_epi64(av, 32), _mm512_srli_epi64(bv, 32)));
        }
        int64_t dot = _mm512_reduce_add_epi64(acc);
        dot >>= shift;
        if (dot > lim) dot = lim; else if (dot < -lim) dot = -lim;
        out[j] = (int32_t)dot;
    }
}

AVX512_FN void ternary_avx512_qk_grad(float *gq, const int32_t *qv, const int32_t *kb, float *gk, size_t kstride,
                                      size_t H, size_t count, const float *ds) {
    const size_t nt = (H + 15) / 16;
    for (size_t j = 0; j < count; j++) {
        const __m512 vd = _mm512_set1_ps(ds[j]);
        for (size_t t = 0; t < nt; t++) {
            const __mmask16 lm = tail_lim(H, t);
            const __m512 kf = _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(lm, kb + j * kstride + 16 * t));
            const __m512 qf = _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(lm, qv + 16 * t));
            _mm512_mask_storeu_ps(gq + 16 * t, lm, ADDPS(_mm512_maskz_loadu_ps(lm, gq + 16 * t), MULPS(vd, kf)));
            _mm512_mask_storeu_ps(gk + j * kstride + 16 * t, lm, ADDPS(_mm512_maskz_loadu_ps(lm, gk + j * kstride + 16 * t), MULPS(vd, qf)));
        }
    }
}

AVX512_FN void ternary_avx512_outer_add(float *dst, size_t stride, size_t H, size_t count, const float *p, const float *ga) {
    const size_t nt = (H + 15) / 16;
    for (size_t j = 0; j < count; j++) {
        const __m512 vp = _mm512_set1_ps(p[j]);
        for (size_t t = 0; t < nt; t++) {
            const __mmask16 lm = tail_lim(H, t);
            float *d = dst + j * stride + 16 * t;
            _mm512_mask_storeu_ps(d, lm, ADDPS(_mm512_maskz_loadu_ps(lm, d), MULPS(vp, _mm512_maskz_loadu_ps(lm, ga + 16 * t))));
        }
    }
}


/* ---- Atencion flash ternaria --------------------------------------------------------------------------------- */
#define ATT_GB 4
static inline __mmask16 key_mask(size_t n, size_t t) { const size_t r = n - 16 * t; return (__mmask16)(r >= 16 ? 0xFFFFu : ((1u << r) - 1u)); }

AVX512_FN void ternary_avx512_attn_scores(const int32_t *const *q, size_t G, const int32_t *kT, size_t ld, size_t H, size_t n,
                                          int shift, int32_t lim, int32_t *sc, size_t scs, int32_t *m_out) {
    const size_t nt = (n + 15) / 16;
    const __m512i vlim = _mm512_set1_epi64(lim), vnlim = _mm512_set1_epi64(-(int64_t)lim);
    const __m512i perm = _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
    const __m128i vsh = _mm_cvtsi32_si128(shift);
    for (size_t g0 = 0; g0 < G; g0 += ATT_GB) {
        const size_t gb = (G - g0 < ATT_GB) ? (G - g0) : ATT_GB;
        __m512i mx[ATT_GB];
        for (size_t b = 0; b < ATT_GB; b++) mx[b] = _mm512_set1_epi32(INT32_MIN);
        for (size_t t = 0; t < nt; t++) {
            __m512i ae[ATT_GB], ao[ATT_GB];
            for (size_t b = 0; b < ATT_GB; b++) { ae[b] = _mm512_setzero_si512(); ao[b] = _mm512_setzero_si512(); }
            const int32_t *kp = kT + 16 * t;
            if (gb == ATT_GB) {
                for (size_t d = 0; d < H; d++) {
                    const __m512i kv = _mm512_loadu_si512((const void *)(kp + d * ld)), ko = _mm512_srli_epi64(kv, 32);
                    for (int b = 0; b < ATT_GB; b++) {
                        const __m512i qb = _mm512_set1_epi32(q[g0 + b][d]);
                        ae[b] = _mm512_add_epi64(ae[b], _mm512_mul_epi32(qb, kv));
                        ao[b] = _mm512_add_epi64(ao[b], _mm512_mul_epi32(qb, ko));
                    }
                }
            } else {
                for (size_t d = 0; d < H; d++) {
                    const __m512i kv = _mm512_loadu_si512((const void *)(kp + d * ld)), ko = _mm512_srli_epi64(kv, 32);
                    for (size_t b = 0; b < gb; b++) {
                        const __m512i qb = _mm512_set1_epi32(q[g0 + b][d]);
                        ae[b] = _mm512_add_epi64(ae[b], _mm512_mul_epi32(qb, kv));
                        ao[b] = _mm512_add_epi64(ao[b], _mm512_mul_epi32(qb, ko));
                    }
                }
            }
            const __mmask16 vm = key_mask(n, t);
            for (size_t b = 0; b < gb; b++) {
                __m512i e64 = _mm512_sra_epi64(ae[b], vsh), o64 = _mm512_sra_epi64(ao[b], vsh);
                e64 = _mm512_min_epi64(_mm512_max_epi64(e64, vnlim), vlim);
                o64 = _mm512_min_epi64(_mm512_max_epi64(o64, vnlim), vlim);
                __m512i s = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi64_epi32(e64)), _mm512_cvtepi64_epi32(o64), 1);
                s = _mm512_permutexvar_epi32(perm, s);
                _mm512_storeu_si512((void *)(sc + (g0 + b) * scs + 16 * t), s);
                mx[b] = _mm512_mask_max_epi32(mx[b], vm, mx[b], s);
            }
        }
        for (size_t b = 0; b < gb; b++) m_out[g0 + b] = _mm512_reduce_max_epi32(mx[b]);
    }
}

/* e_j para un bloque de claves (gather enmascarado de la tabla compacta; los carriles >= n dan 0). */
AVX512_INL __m512 exp_tile(const int32_t *sc, __m512i vm, __m512i vk0, const float *tab, __mmask16 mk) {
    const __m512i k = _mm512_sub_epi32(vm, _mm512_loadu_si512((const void *)sc));
    return _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mk, _mm512_min_epu32(k, vk0), tab, 4);
}

AVX512_INL void pv_block(const int GB, const int NT, size_t n, const float *v, size_t vs, const __mmask16 *tm,
                         const float *e, size_t es, const float *inv, float *out, size_t H) {
    __m512 acc[4][4];
    for (int b = 0; b < GB; b++) for (int t = 0; t < NT; t++) acc[b][t] = _mm512_setzero_ps();
    for (size_t j = 0; j < n; j++) {
        __m512 vv[4];
        for (int t = 0; t < NT; t++) vv[t] = _mm512_maskz_loadu_ps(tm[t], v + j * vs + 16 * t);
        for (int b = 0; b < GB; b++) {
            const __m512 eb = _mm512_set1_ps(e[b * es + j]);
            for (int t = 0; t < NT; t++) acc[b][t] = ADDPS(acc[b][t], MULPS(eb, vv[t]));
        }
    }
    for (int b = 0; b < GB; b++) {
        const __m512 iv = _mm512_set1_ps(inv[b]);
        for (int t = 0; t < NT; t++) _mm512_mask_storeu_ps(out + b * H + 16 * t, tm[t], MULPS(acc[b][t], iv));
    }
}

AVX512_FN void ternary_avx512_attn_pv(size_t G, const int32_t *sc, size_t scs, const int32_t *m, const float *tab, uint32_t K0,
                                      size_t n, const float *v, size_t vs, size_t H, float *e, size_t es, float *out, float *inv_out) {
    const size_t nt = (n + 15) / 16, NT = (H + 15) / 16;
    const __m512i vk0 = _mm512_set1_epi32((int)K0);
    for (size_t b = 0; b < G; b++) {                                  /* exponenciales y suma de cada cabeza */
        const __m512i vm = _mm512_set1_epi32(m[b]);
        __m512 part = _mm512_setzero_ps();
        for (size_t t = 0; t < nt; t++) {
            const __m512 ev = exp_tile(sc + b * scs + 16 * t, vm, vk0, tab, key_mask(n, t));
            _mm512_storeu_ps(e + b * es + 16 * t, ev);
            part = ADDPS(part, ev);
        }
        inv_out[b] = 1.0f / sum16_ps(part);
    }
    __mmask16 tm[8];
    for (size_t t = 0; t < NT && t < 8; t++) tm[t] = tail_lim(H, t);
    size_t b = 0;
    while (b < G) {                                                   /* PV: V se lee una vez por bloque de cabezas */
        const size_t left = G - b;
        if (NT == 1)      { if (left >= 4) { pv_block(4, 1, n, v, vs, tm, e + b * es, es, inv_out + b, out + b * H, H); b += 4; } else if (left >= 2) { pv_block(2, 1, n, v, vs, tm, e + b * es, es, inv_out + b, out + b * H, H); b += 2; } else { pv_block(1, 1, n, v, vs, tm, e + b * es, es, inv_out + b, out + b * H, H); b += 1; } }
        else if (NT == 2) { if (left >= 4) { pv_block(4, 2, n, v, vs, tm, e + b * es, es, inv_out + b, out + b * H, H); b += 4; } else if (left >= 2) { pv_block(2, 2, n, v, vs, tm, e + b * es, es, inv_out + b, out + b * H, H); b += 2; } else { pv_block(1, 2, n, v, vs, tm, e + b * es, es, inv_out + b, out + b * H, H); b += 1; } }
        else if (NT == 3) { if (left >= 4) { pv_block(4, 3, n, v, vs, tm, e + b * es, es, inv_out + b, out + b * H, H); b += 4; } else if (left >= 2) { pv_block(2, 3, n, v, vs, tm, e + b * es, es, inv_out + b, out + b * H, H); b += 2; } else { pv_block(1, 3, n, v, vs, tm, e + b * es, es, inv_out + b, out + b * H, H); b += 1; } }
        else if (NT == 4) { if (left >= 2) { pv_block(2, 4, n, v, vs, tm, e + b * es, es, inv_out + b, out + b * H, H); b += 2; } else { pv_block(1, 4, n, v, vs, tm, e + b * es, es, inv_out + b, out + b * H, H); b += 1; } }
        else {                                                        /* cabezas anchas (> 64): una cabeza, d en trozos de 4 vectores */
            for (size_t t0 = 0; t0 < NT; t0 += 4) {
                const size_t w = (NT - t0 < 4) ? (NT - t0) : 4;
                __mmask16 tl[4]; for (size_t t = 0; t < w; t++) tl[t] = tail_lim(H, t0 + t);
                const float *vp = v + 16 * t0;
                if (w == 4) pv_block(1, 4, n, vp, vs, tl, e + b * es, es, inv_out + b, out + b * H + 16 * t0, H);
                else if (w == 3) pv_block(1, 3, n, vp, vs, tl, e + b * es, es, inv_out + b, out + b * H + 16 * t0, H);
                else if (w == 2) pv_block(1, 2, n, vp, vs, tl, e + b * es, es, inv_out + b, out + b * H + 16 * t0, H);
                else pv_block(1, 1, n, vp, vs, tl, e + b * es, es, inv_out + b, out + b * H + 16 * t0, H);
            }
            b += 1;
        }
    }
}

AVX512_INL void bwd_acc_block(const int GB, const int NT, size_t n, size_t H, const __mmask16 *tm, const float *ga, const float *qf,
                              const float *p, const float *ds, size_t es, const int32_t *kr, size_t ks, float *gq, float *gk,
                              float *gv, size_t gs) {
    __m512 aq[4][4];
    for (int b = 0; b < GB; b++) for (int t = 0; t < NT; t++) aq[b][t] = _mm512_maskz_loadu_ps(tm[t], gq + b * H + 16 * t);
    for (size_t j = 0; j < n; j++) {
        __m512 kf[4], gvv[4], gkk[4];
        for (int t = 0; t < NT; t++) {
            kf[t] = _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(tm[t], kr + j * ks + 16 * t));
            gvv[t] = _mm512_maskz_loadu_ps(tm[t], gv + j * gs + 16 * t);
            gkk[t] = _mm512_maskz_loadu_ps(tm[t], gk + j * gs + 16 * t);
        }
        for (int b = 0; b < GB; b++) {
            const __m512 pj = _mm512_set1_ps(p[b * es + j]), dj = _mm512_set1_ps(ds[b * es + j]);
            for (int t = 0; t < NT; t++) {
                gvv[t] = ADDPS(gvv[t], MULPS(pj, _mm512_maskz_loadu_ps(tm[t], ga + b * H + 16 * t)));
                aq[b][t] = ADDPS(aq[b][t], MULPS(dj, kf[t]));
                gkk[t] = ADDPS(gkk[t], MULPS(dj, _mm512_maskz_loadu_ps(tm[t], qf + b * H + 16 * t)));
            }
        }
        for (int t = 0; t < NT; t++) {
            _mm512_mask_storeu_ps(gv + j * gs + 16 * t, tm[t], gvv[t]);
            _mm512_mask_storeu_ps(gk + j * gs + 16 * t, tm[t], gkk[t]);
        }
    }
    for (int b = 0; b < GB; b++) for (int t = 0; t < NT; t++) _mm512_mask_storeu_ps(gq + b * H + 16 * t, tm[t], aq[b][t]);
}

/* Fase 1 del backward de una fila: p, ds (G x es) y qf (G x H) a partir de las puntuaciones. */
static AVX512_FN void bwd_pds(size_t G, size_t n, size_t H, const int32_t *sc, size_t scs, const int32_t *m,
                       const float *inv, const float *tab, uint32_t K0, const float *ga, const float *o,
                       const float *vT, size_t ld, float it, float gam, const int32_t *qr,
                       float *p, float *ds, size_t es, float *qf) {
    const size_t nt = (n + 15) / 16, NT = (H + 15) / 16;
    const __m512i vk0 = _mm512_set1_epi32((int)K0);
    const __m512 vit = _mm512_set1_ps(it), vgam = _mm512_set1_ps(gam);
    for (size_t b = 0; b < G; b++) {
        const float *gab = ga + b * H, *ob = o + b * H;
        float D = 0.0f;
        for (size_t d = 0; d < H; d++) D += gab[d] * ob[d];            /* D = dO . O (escalar, mismo orden que la referencia) */
        const __m512 vD = _mm512_set1_ps(D), viv = _mm512_set1_ps(inv[b]);
        const __m512i vm = _mm512_set1_epi32(m[b]);
        for (size_t t = 0; t < nt; t++) {
            __m512 dp = _mm512_setzero_ps();                           /* dprob_j = sum_d ga[d]*v[j][d], d ascendente */
            for (size_t d = 0; d < H; d++)
                dp = ADDPS(dp, MULPS(_mm512_set1_ps(gab[d]), _mm512_loadu_ps(vT + d * ld + 16 * t)));
            const __m512 e = exp_tile(sc + b * scs + 16 * t, vm, vk0, tab, key_mask(n, t));
            const __m512 pv = MULPS(e, viv);
            const __m512 dsv = MULPS(MULPS(MULPS(vit, pv), SUBPS(dp, vD)), vgam);
            _mm512_storeu_ps(p + b * es + 16 * t, pv);
            _mm512_storeu_ps(ds + b * es + 16 * t, dsv);
        }
        for (size_t t = 0; t < NT; t++)
            _mm512_mask_storeu_ps(qf + b * H + 16 * t, tail_lim(H, t), _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(tail_lim(H, t), qr + b * H + 16 * t)));
    }
}

AVX512_FN void ternary_avx512_attn_bwd_pds(size_t G, size_t n, size_t H, const int32_t *sc, size_t scs, const int32_t *m,
                                           const float *inv, const float *tab, uint32_t K0, const float *ga, const float *o,
                                           const float *vT, size_t ld, float it, float gam, const int32_t *qr,
                                           float *p, float *ds, size_t es, float *qf) {
    bwd_pds(G, n, H, sc, scs, m, inv, tab, K0, ga, o, vT, ld, it, gam, qr, p, ds, es, qf);
}

AVX512_FN void ternary_avx512_attn_bwd(size_t G, size_t n, size_t H, const int32_t *sc, size_t scs, const int32_t *m,
                                       const float *inv, const float *tab, uint32_t K0, const float *ga, const float *o,
                                       const float *vT, size_t ld, float it, float gam, const int32_t *qr, const int32_t *kr,
                                       size_t ks, float *gq, float *gk, float *gv, size_t gs, float *p, float *ds, size_t es,
                                       float *qf) {
    const size_t NT = (H + 15) / 16;
    bwd_pds(G, n, H, sc, scs, m, inv, tab, K0, ga, o, vT, ld, it, gam, qr, p, ds, es, qf);
    __mmask16 tm[8];
    for (size_t t = 0; t < NT && t < 8; t++) tm[t] = tail_lim(H, t);
    size_t b = 0;
    while (b < G) {
        const size_t left = G - b;
        if (NT == 1)      { if (left >= 4) { bwd_acc_block(4, 1, n, H, tm, ga + b * H, qf + b * H, p + b * es, ds + b * es, es, kr, ks, gq + b * H, gk, gv, gs); b += 4; } else { bwd_acc_block(1, 1, n, H, tm, ga + b * H, qf + b * H, p + b * es, ds + b * es, es, kr, ks, gq + b * H, gk, gv, gs); b += 1; } }
        else if (NT == 2) { if (left >= 4) { bwd_acc_block(4, 2, n, H, tm, ga + b * H, qf + b * H, p + b * es, ds + b * es, es, kr, ks, gq + b * H, gk, gv, gs); b += 4; } else { bwd_acc_block(1, 2, n, H, tm, ga + b * H, qf + b * H, p + b * es, ds + b * es, es, kr, ks, gq + b * H, gk, gv, gs); b += 1; } }
        else if (NT == 3) { if (left >= 2) { bwd_acc_block(2, 3, n, H, tm, ga + b * H, qf + b * H, p + b * es, ds + b * es, es, kr, ks, gq + b * H, gk, gv, gs); b += 2; } else { bwd_acc_block(1, 3, n, H, tm, ga + b * H, qf + b * H, p + b * es, ds + b * es, es, kr, ks, gq + b * H, gk, gv, gs); b += 1; } }
        else if (NT == 4) { if (left >= 2) { bwd_acc_block(2, 4, n, H, tm, ga + b * H, qf + b * H, p + b * es, ds + b * es, es, kr, ks, gq + b * H, gk, gv, gs); b += 2; } else { bwd_acc_block(1, 4, n, H, tm, ga + b * H, qf + b * H, p + b * es, ds + b * es, es, kr, ks, gq + b * H, gk, gv, gs); b += 1; } }
        else {                                                        /* cabezas anchas: una cabeza, d en trozos de 4 vectores */
            for (size_t t0 = 0; t0 < NT; t0 += 4) {
                const size_t w = (NT - t0 < 4) ? (NT - t0) : 4;
                __mmask16 tl[4]; for (size_t t = 0; t < w; t++) tl[t] = tail_lim(H, t0 + t);
                const size_t o0 = 16 * t0;
                if (w == 4) bwd_acc_block(1, 4, n, H, tl, ga + b * H + o0, qf + b * H + o0, p + b * es, ds + b * es, es, kr + o0, ks, gq + b * H + o0, gk + o0, gv + o0, gs);
                else if (w == 3) bwd_acc_block(1, 3, n, H, tl, ga + b * H + o0, qf + b * H + o0, p + b * es, ds + b * es, es, kr + o0, ks, gq + b * H + o0, gk + o0, gv + o0, gs);
                else if (w == 2) bwd_acc_block(1, 2, n, H, tl, ga + b * H + o0, qf + b * H + o0, p + b * es, ds + b * es, es, kr + o0, ks, gq + b * H + o0, gk + o0, gv + o0, gs);
                else bwd_acc_block(1, 1, n, H, tl, ga + b * H + o0, qf + b * H + o0, p + b * es, ds + b * es, es, kr + o0, ks, gq + b * H + o0, gk + o0, gv + o0, gs);
            }
            b += 1;
        }
    }
}

/* ---- Backward de atencion por bloques de R filas de consulta consecutivas ---------------------------------------------
 * Misma cuenta que R llamadas seguidas a attn_bwd (fase 2), pero cada fila j de gk/gv se lee y se escribe UNA vez por bloque y los
 * acumuladores de gq de las R filas viven en registros. El orden de cada suma no cambia: gk/gv[j] reciben las filas en orden
 * ascendente y, dentro de cada fila, las cabezas en orden ascendente; gq[r] recibe las claves ascendentes. Fila r usa nr[r] claves
 * (nr no decreciente). Los acumuladores de gq pasan por memoria solo en las claves de la diagonal (j >= nr[0]). */
#define BR_MAXRN 16
AVX512_INL void bwd_rows_main(const int R, const int GB, const int NT, const size_t *nr, size_t H, const __mmask16 *tm,
                              const float *ga, size_t gas, const float *qf, const float *p, const float *ds, size_t es, size_t rs,
                              const int32_t *kr, size_t ks, float *gq, size_t gqs, float *gk, float *gv, size_t gs, int doA) {
    __m512 aq[4][8][2];
    for (int r = 0; r < R; r++) for (int b = 0; b < GB; b++) for (int t = 0; t < NT; t++)
        aq[r][b][t] = _mm512_maskz_loadu_ps(tm[t], gq + r * gqs + b * H + 16 * t);
    const size_t n0 = nr[0];
    for (size_t j = 0; j < n0; j++) {
        __m512 kf[2], gvv[2] = {_mm512_setzero_ps(), _mm512_setzero_ps()}, gkk[2] = {_mm512_setzero_ps(), _mm512_setzero_ps()};
        for (int t = 0; t < NT; t++) {
            kf[t] = _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(tm[t], kr + j * ks + 16 * t));
            if (doA) { gvv[t] = _mm512_maskz_loadu_ps(tm[t], gv + j * gs + 16 * t); gkk[t] = _mm512_maskz_loadu_ps(tm[t], gk + j * gs + 16 * t); }
        }
        for (int r = 0; r < R; r++)
            for (int b = 0; b < GB; b++) {
                const __m512 pj = _mm512_set1_ps(p[r * rs + b * es + j]), dj = _mm512_set1_ps(ds[r * rs + b * es + j]);
                for (int t = 0; t < NT; t++) {
                    if (doA) {
                        gvv[t] = ADDPS(gvv[t], MULPS(pj, _mm512_maskz_loadu_ps(tm[t], ga + r * gas + b * H + 16 * t)));
                        gkk[t] = ADDPS(gkk[t], MULPS(dj, _mm512_maskz_loadu_ps(tm[t], qf + (r * GB + b) * H + 16 * t)));
                    }
                    aq[r][b][t] = ADDPS(aq[r][b][t], MULPS(dj, kf[t]));
                }
            }
        if (doA)
            for (int t = 0; t < NT; t++) {
                _mm512_mask_storeu_ps(gv + j * gs + 16 * t, tm[t], gvv[t]);
                _mm512_mask_storeu_ps(gk + j * gs + 16 * t, tm[t], gkk[t]);
            }
    }
    for (int r = 0; r < R; r++) for (int b = 0; b < GB; b++) for (int t = 0; t < NT; t++)
        _mm512_mask_storeu_ps(gq + r * gqs + b * H + 16 * t, tm[t], aq[r][b][t]);
    /* diagonal: claves j >= nr[0]; la fila r participa si j < nr[r] (las acumuladoras de gq ya estan en memoria) */
    for (size_t j = n0; j < nr[R - 1]; j++) {
        int r0 = 0; while (nr[r0] <= j) r0++;
        __m512 kf[2], gvv[2] = {_mm512_setzero_ps(), _mm512_setzero_ps()}, gkk[2] = {_mm512_setzero_ps(), _mm512_setzero_ps()};
        for (int t = 0; t < NT; t++) {
            kf[t] = _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(tm[t], kr + j * ks + 16 * t));
            if (doA) { gvv[t] = _mm512_maskz_loadu_ps(tm[t], gv + j * gs + 16 * t); gkk[t] = _mm512_maskz_loadu_ps(tm[t], gk + j * gs + 16 * t); }
        }
        for (int r = r0; r < R; r++)
            for (int b = 0; b < GB; b++) {
                const __m512 pj = _mm512_set1_ps(p[r * rs + b * es + j]), dj = _mm512_set1_ps(ds[r * rs + b * es + j]);
                for (int t = 0; t < NT; t++) {
                    if (doA) {
                        gvv[t] = ADDPS(gvv[t], MULPS(pj, _mm512_maskz_loadu_ps(tm[t], ga + r * gas + b * H + 16 * t)));
                        gkk[t] = ADDPS(gkk[t], MULPS(dj, _mm512_maskz_loadu_ps(tm[t], qf + (r * GB + b) * H + 16 * t)));
                    }
                    float *q = gq + r * gqs + b * H + 16 * t;
                    _mm512_mask_storeu_ps(q, tm[t], ADDPS(_mm512_maskz_loadu_ps(tm[t], q), MULPS(dj, kf[t])));
                }
            }
        if (doA)
            for (int t = 0; t < NT; t++) {
                _mm512_mask_storeu_ps(gv + j * gs + 16 * t, tm[t], gvv[t]);
                _mm512_mask_storeu_ps(gk + j * gs + 16 * t, tm[t], gkk[t]);
            }
    }
}

size_t ternary_avx512_attn_bwd_rows_cap(size_t G, size_t H) {
    const size_t NT = (H + 15) / 16;
    if (G < 1 || NT < 1 || NT > 2) return 0;
    const size_t gn = G * NT;
    if (gn <= 4) return 4;
    if (gn <= 8) return 2;
    return 0;
}

AVX512_FN void ternary_avx512_attn_bwd_rows(size_t R, size_t G, size_t H, const size_t *nr, const float *ga, size_t gas,
                                            const float *qf, const float *p, const float *ds, size_t es, size_t rs,
                                            const int32_t *kr, size_t ks, float *gq, size_t gqs, float *gk, float *gv,
                                            size_t gs, int doA) {
    const size_t NT = (H + 15) / 16;
    __mmask16 tm[2];
    for (size_t t = 0; t < NT; t++) tm[t] = tail_lim(H, t);
#define BRC(R_, G_, NT_) if (R == R_ && G == G_ && NT == NT_) { bwd_rows_main(R_, G_, NT_, nr, H, tm, ga, gas, qf, p, ds, es, rs, kr, ks, gq, gqs, gk, gv, gs, doA); return; }
    BRC(4,1,1) BRC(4,2,1) BRC(4,3,1) BRC(4,4,1) BRC(4,1,2) BRC(4,2,2)
    BRC(2,1,1) BRC(2,2,1) BRC(2,3,1) BRC(2,4,1) BRC(2,1,2) BRC(2,2,2) BRC(2,3,2) BRC(2,4,2) BRC(2,5,1) BRC(2,6,1) BRC(2,7,1) BRC(2,8,1)
#undef BRC
    abort();
}


/* ---- Atencion con ventana: dos tramos de claves (sinks A + ventana B) ---------------------------------------------
 * Indices de trabajo: el tramo A (nA claves) ocupa [0, spA) y el B (nB claves) [spA, spA + nB), con spA = nA redondeado a
 * 16: los carriles de la suma por fila son (indice & 15), asi que cada tramo empieza en el carril 0 (igual que la referencia
 * escalar: carril = (t < nA ? t : t - nA) & 15). Las claves del tramo A se leen de columna colA, las del B de colB. */
AVX512_INL void pv_block2(const int GB, const int NT, size_t nA, const float *vA, size_t nB, const float *vB, size_t vs,
                          const __mmask16 *tm, const float *e, size_t es, size_t spA, const float *inv, float *out, size_t H) {
    __m512 acc[4][4];
    for (int b = 0; b < GB; b++) for (int t = 0; t < NT; t++) acc[b][t] = _mm512_setzero_ps();
    for (int seg = 0; seg < 2; seg++) {
        const size_t n = seg ? nB : nA; const float *v = seg ? vB : vA; const float *eo = e + (seg ? spA : 0);
        for (size_t j = 0; j < n; j++) {
            __m512 vv[4];
            for (int t = 0; t < NT; t++) vv[t] = _mm512_maskz_loadu_ps(tm[t], v + j * vs + 16 * t);
            for (int b = 0; b < GB; b++) {
                const __m512 eb = _mm512_set1_ps(eo[b * es + j]);
                for (int t = 0; t < NT; t++) acc[b][t] = ADDPS(acc[b][t], MULPS(eb, vv[t]));
            }
        }
    }
    for (int b = 0; b < GB; b++) {
        const __m512 iv = _mm512_set1_ps(inv[b]);
        for (int t = 0; t < NT; t++) _mm512_mask_storeu_ps(out + b * H + 16 * t, tm[t], MULPS(acc[b][t], iv));
    }
}

AVX512_FN void ternary_avx512_attn_pv2(size_t G, const int32_t *sc, size_t scs, const int32_t *m, const float *tab, uint32_t K0,
                                       size_t nA, size_t nB, size_t spA, const float *vA, const float *vB, size_t vs, size_t H,
                                       float *e, size_t es, float *out, float *inv_out) {
    const size_t NT = (H + 15) / 16;
    const __m512i vk0 = _mm512_set1_epi32((int)K0);
    for (size_t b = 0; b < G; b++) {
        const __m512i vm = _mm512_set1_epi32(m[b]);
        __m512 part = _mm512_setzero_ps();
        for (size_t t = 0; t < (nA + 15) / 16; t++) {
            const __m512 ev = exp_tile(sc + b * scs + 16 * t, vm, vk0, tab, key_mask(nA, t));
            _mm512_storeu_ps(e + b * es + 16 * t, ev);
            part = ADDPS(part, ev);
        }
        for (size_t t = 0; t < (nB + 15) / 16; t++) {
            const __m512 ev = exp_tile(sc + b * scs + spA + 16 * t, vm, vk0, tab, key_mask(nB, t));
            _mm512_storeu_ps(e + b * es + spA + 16 * t, ev);
            part = ADDPS(part, ev);
        }
        inv_out[b] = 1.0f / sum16_ps(part);
    }
    __mmask16 tm[8];
    for (size_t t = 0; t < NT && t < 8; t++) tm[t] = tail_lim(H, t);
#define PVB2(GB_, NT_) pv_block2(GB_, NT_, nA, vA, nB, vB, vs, tm, e + b * es, es, spA, inv_out + b, out + b * H, H)
    size_t b = 0;
    while (b < G) {
        const size_t left = G - b;
        if (NT == 1)      { if (left >= 4) { PVB2(4, 1); b += 4; } else if (left >= 2) { PVB2(2, 1); b += 2; } else { PVB2(1, 1); b += 1; } }
        else if (NT == 2) { if (left >= 4) { PVB2(4, 2); b += 4; } else if (left >= 2) { PVB2(2, 2); b += 2; } else { PVB2(1, 2); b += 1; } }
        else if (NT == 3) { if (left >= 4) { PVB2(4, 3); b += 4; } else if (left >= 2) { PVB2(2, 3); b += 2; } else { PVB2(1, 3); b += 1; } }
        else if (NT == 4) { if (left >= 2) { PVB2(2, 4); b += 2; } else { PVB2(1, 4); b += 1; } }
        else {
            for (size_t t0 = 0; t0 < NT; t0 += 4) {
                const size_t w = (NT - t0 < 4) ? (NT - t0) : 4;
                __mmask16 tl[4]; for (size_t t = 0; t < w; t++) tl[t] = tail_lim(H, t0 + t);
                const float *va = vA + 16 * t0, *vb = vB + 16 * t0;
#define PVW(W_) pv_block2(1, W_, nA, va, nB, vb, vs, tl, e + b * es, es, spA, inv_out + b, out + b * H + 16 * t0, H)
                if (w == 4) PVW(4); else if (w == 3) PVW(3); else if (w == 2) PVW(2); else PVW(1);
#undef PVW
            }
            b += 1;
        }
    }
#undef PVB2
}

AVX512_INL void bwd_acc_block2(const int GB, const int NT, size_t nA, size_t nB, size_t spA, size_t H, const __mmask16 *tm,
                               const float *ga, const float *qf, const float *p, const float *ds, size_t es,
                               const int32_t *krA, const int32_t *krB, size_t ks, float *gq, float *gkA, float *gkB,
                               float *gvA, float *gvB, size_t gs) {
    __m512 aq[4][4];
    for (int b = 0; b < GB; b++) for (int t = 0; t < NT; t++) aq[b][t] = _mm512_maskz_loadu_ps(tm[t], gq + b * H + 16 * t);
    for (int seg = 0; seg < 2; seg++) {
        const size_t n = seg ? nB : nA, po = seg ? spA : 0;
        const int32_t *kr = seg ? krB : krA; float *gk = seg ? gkB : gkA, *gv = seg ? gvB : gvA;
        for (size_t j = 0; j < n; j++) {
            __m512 kf[4], gvv[4], gkk[4];
            for (int t = 0; t < NT; t++) {
                kf[t] = _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(tm[t], kr + j * ks + 16 * t));
                gvv[t] = _mm512_maskz_loadu_ps(tm[t], gv + j * gs + 16 * t);
                gkk[t] = _mm512_maskz_loadu_ps(tm[t], gk + j * gs + 16 * t);
            }
            for (int b = 0; b < GB; b++) {
                const __m512 pj = _mm512_set1_ps(p[b * es + po + j]), dj = _mm512_set1_ps(ds[b * es + po + j]);
                for (int t = 0; t < NT; t++) {
                    gvv[t] = ADDPS(gvv[t], MULPS(pj, _mm512_maskz_loadu_ps(tm[t], ga + b * H + 16 * t)));
                    aq[b][t] = ADDPS(aq[b][t], MULPS(dj, kf[t]));
                    gkk[t] = ADDPS(gkk[t], MULPS(dj, _mm512_maskz_loadu_ps(tm[t], qf + b * H + 16 * t)));
                }
            }
            for (int t = 0; t < NT; t++) {
                _mm512_mask_storeu_ps(gv + j * gs + 16 * t, tm[t], gvv[t]);
                _mm512_mask_storeu_ps(gk + j * gs + 16 * t, tm[t], gkk[t]);
            }
        }
    }
    for (int b = 0; b < GB; b++) for (int t = 0; t < NT; t++) _mm512_mask_storeu_ps(gq + b * H + 16 * t, tm[t], aq[b][t]);
}

AVX512_FN void ternary_avx512_attn_bwd2(size_t G, size_t nA, size_t nB, size_t spA, size_t H, const int32_t *sc, size_t scs,
                                        const int32_t *m, const float *inv, const float *tab, uint32_t K0, const float *ga,
                                        const float *o, const float *vTA, const float *vTB, size_t ld, float it, float gam,
                                        const int32_t *qr, const int32_t *krA, const int32_t *krB, size_t ks, float *gq,
                                        float *gkA, float *gkB, float *gvA, float *gvB, size_t gs, float *p, float *ds,
                                        size_t es, float *qf) {
    const size_t NT = (H + 15) / 16;
    const __m512i vk0 = _mm512_set1_epi32((int)K0);
    const __m512 vit = _mm512_set1_ps(it), vgam = _mm512_set1_ps(gam);
    for (size_t b = 0; b < G; b++) {
        const float *gab = ga + b * H, *ob = o + b * H;
        float D = 0.0f;
        for (size_t d = 0; d < H; d++) D += gab[d] * ob[d];
        const __m512 vD = _mm512_set1_ps(D), viv = _mm512_set1_ps(inv[b]);
        const __m512i vm = _mm512_set1_epi32(m[b]);
        for (int seg = 0; seg < 2; seg++) {
            const size_t n = seg ? nB : nA, so = seg ? spA : 0; const float *vT = seg ? vTB : vTA;
            for (size_t t = 0; t < (n + 15) / 16; t++) {
                __m512 dp = _mm512_setzero_ps();
                for (size_t d = 0; d < H; d++)
                    dp = ADDPS(dp, MULPS(_mm512_set1_ps(gab[d]), _mm512_loadu_ps(vT + d * ld + 16 * t)));
                const __m512 e = exp_tile(sc + b * scs + so + 16 * t, vm, vk0, tab, key_mask(n, t));
                const __m512 pv = MULPS(e, viv);
                const __m512 dsv = MULPS(MULPS(MULPS(vit, pv), SUBPS(dp, vD)), vgam);
                _mm512_storeu_ps(p + b * es + so + 16 * t, pv);
                _mm512_storeu_ps(ds + b * es + so + 16 * t, dsv);
            }
        }
        for (size_t t = 0; t < NT; t++)
            _mm512_mask_storeu_ps(qf + b * H + 16 * t, tail_lim(H, t), _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(tail_lim(H, t), qr + b * H + 16 * t)));
    }
    __mmask16 tm[8];
    for (size_t t = 0; t < NT && t < 8; t++) tm[t] = tail_lim(H, t);
#define BWB2(GB_, NT_) bwd_acc_block2(GB_, NT_, nA, nB, spA, H, tm, ga + b * H, qf + b * H, p + b * es, ds + b * es, es, krA, krB, ks, gq + b * H, gkA, gkB, gvA, gvB, gs)
    size_t b = 0;
    while (b < G) {
        const size_t left = G - b;
        if (NT == 1)      { if (left >= 4) { BWB2(4, 1); b += 4; } else { BWB2(1, 1); b += 1; } }
        else if (NT == 2) { if (left >= 4) { BWB2(4, 2); b += 4; } else { BWB2(1, 2); b += 1; } }
        else if (NT == 3) { if (left >= 2) { BWB2(2, 3); b += 2; } else { BWB2(1, 3); b += 1; } }
        else if (NT == 4) { if (left >= 2) { BWB2(2, 4); b += 2; } else { BWB2(1, 4); b += 1; } }
        else {
            for (size_t t0 = 0; t0 < NT; t0 += 4) {
                const size_t w = (NT - t0 < 4) ? (NT - t0) : 4;
                __mmask16 tl[4]; for (size_t t = 0; t < w; t++) tl[t] = tail_lim(H, t0 + t);
                const size_t o0 = 16 * t0;
#define BWW(W_) bwd_acc_block2(1, W_, nA, nB, spA, H, tl, ga + b * H + o0, qf + b * H + o0, p + b * es, ds + b * es, es, krA + o0, krB + o0, ks, gq + b * H + o0, gkA + o0, gkB + o0, gvA + o0, gvB + o0, gs)
                if (w == 4) BWW(4); else if (w == 3) BWW(3); else if (w == 2) BWW(2); else BWW(1);
#undef BWW
            }
            b += 1;
        }
    }
#undef BWB2
}


/* ======================================================================================================
 * Entrada uint8 (ver ternary_u8.h). Todo entero o elemento a elemento: mismos bits que el escalar.
 * ====================================================================================================== */
AVX512_FN int32_t ternary_avx512_dot_u8(const uint64_t *wm, const uint64_t *ws, size_t n, const uint8_t *q, int32_t zp) {
    const size_t nw = (n + 63) / 64;
    const __m512i z = _mm512_setzero_si512();
    __m512i ap = z, an = z;
    int32_t npos = 0, nneg = 0;
    for (size_t k = 0; k < nw; k++) {
        const uint64_t pos = wm[k] & ~ws[k], neg = wm[k] & ws[k];
        npos += __builtin_popcountll(pos);
        nneg += __builtin_popcountll(neg);
        const size_t rem = n - 64 * k;
        const __mmask64 lm = (rem >= 64) ? ~(__mmask64)0 : (((__mmask64)1 << rem) - 1);
        const __m512i v = _mm512_maskz_loadu_epi8(lm, q + 64 * k);
        ap = _mm512_add_epi64(ap, _mm512_sad_epu8(_mm512_maskz_mov_epi8((__mmask64)pos, v), z));
        an = _mm512_add_epi64(an, _mm512_sad_epu8(_mm512_maskz_mov_epi8((__mmask64)neg, v), z));
    }
    const int32_t sp = (int32_t)_mm512_reduce_add_epi64(ap), sn = (int32_t)_mm512_reduce_add_epi64(an);
    return sp - sn - zp * (npos - nneg);
}

AVX512_FN void ternary_avx512_u8_to_f32(const uint8_t *q, size_t n, float sx, int32_t zp, float *out) {
    const __m512 vs = _mm512_set1_ps(sx);
    const __m512i vz = _mm512_set1_epi32(zp);
    const size_t nt = (n + 15) / 16;
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(n, t);
        const __m512i b = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(lim, q + 16 * t));
        _mm512_mask_storeu_ps(out + 16 * t, lim, MULPS(vs, _mm512_cvtepi32_ps(_mm512_sub_epi32(b, vz))));
    }
}

/* gw[o][:] += g[p*gs + o] * x[p*xs + :] para o en [o0, o1) y p ascendente (g == 0 se salta). Los acumuladores de una
 * fila viven en registros durante todo el recorrido de p (el bucle por posicion y por fila de antes leia y escribia
 * gw en cada paso): las mismas sumas y el mismo orden por elemento (producto y suma separados, sin FMA), mismos bits. */
AVX512_FN void ternary_avx512_scale_terms(float *terms, size_t ts, const float *g, size_t gs, const float *sx, const int32_t *z,
                                          size_t zs, size_t P, size_t o0, size_t o1) {
    const size_t n = o1 - o0;
    for (size_t p = 0; p < P; p++) {
        const __m512 vs = _mm512_set1_ps(sx[p]);
        for (size_t t = 0; t < (n + 15) / 16; t++) {
            const size_t o = o0 + 16 * t; const __mmask16 lim = tail_lim(n, t);
            const __m512 gv = _mm512_maskz_loadu_ps(lim, g + p * gs + o);
            const __m512 zf = _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(lim, z + p * zs + o));
            const __mmask16 nz = _mm512_mask_cmpneq_ps_mask(lim, gv, _mm512_setzero_ps());   /* g == 0 -> 0 (como el bucle escalar) */
            _mm512_mask_storeu_ps(terms + p * ts + o, lim, _mm512_maskz_mul_ps(nz, gv, MULPS(vs, zf)));
        }
    }
}

AVX512_FN void ternary_avx512_outer_acc_rows(float *gw, size_t n, size_t o0, size_t o1, size_t P, const float *g, size_t gs,
                                             const float *x, size_t xs) {
    const size_t full = n / 16, nt = (n + 15) / 16;
    size_t o = o0;
    /* de dos en dos filas: 16 acumuladores independientes (la cadena de sumas de cada fila es de latencia 4 y con 8 solo se
     * mantiene ocupado un puerto) y cada trozo de x se carga una vez para las dos filas */
    for (; o + 2 <= o1; o += 2) {
        float *r0 = gw + o * n, *r1 = r0 + n;
        size_t t = 0;
        for (; t + 8 <= full; t += 8) {
            __m512 a[8], b[8];
            for (int k = 0; k < 8; k++) { a[k] = _mm512_loadu_ps(r0 + 16 * (t + k)); b[k] = _mm512_loadu_ps(r1 + 16 * (t + k)); }
            for (size_t p = 0; p < P; p++) {
                const float g0 = g[p * gs + o], g1 = g[p * gs + o + 1];
                const float *xp = x + p * xs + 16 * t;
                if (g0 != 0.0f && g1 != 0.0f) {
                    const __m512 v0 = _mm512_set1_ps(g0), v1 = _mm512_set1_ps(g1);
                    for (int k = 0; k < 8; k++) {
                        const __m512 xv = _mm512_loadu_ps(xp + 16 * k);
                        a[k] = _mm512_fmadd_ps(v0, xv, a[k]);
                        b[k] = _mm512_fmadd_ps(v1, xv, b[k]);
                    }
                } else if (g0 != 0.0f) {
                    const __m512 v0 = _mm512_set1_ps(g0);
                    for (int k = 0; k < 8; k++) a[k] = _mm512_fmadd_ps(v0, _mm512_loadu_ps(xp + 16 * k), a[k]);
                } else if (g1 != 0.0f) {
                    const __m512 v1 = _mm512_set1_ps(g1);
                    for (int k = 0; k < 8; k++) b[k] = _mm512_fmadd_ps(v1, _mm512_loadu_ps(xp + 16 * k), b[k]);
                }
            }
            for (int k = 0; k < 8; k++) { _mm512_storeu_ps(r0 + 16 * (t + k), a[k]); _mm512_storeu_ps(r1 + 16 * (t + k), b[k]); }
        }
        for (size_t oo = o; oo < o + 2; oo++) {   /* trozos restantes de las dos filas, de a uno */
            float *row = gw + oo * n;
            for (size_t tt = t; tt < nt; tt++) {
                const __mmask16 lim = tail_lim(n, tt);
                __m512 a = _mm512_maskz_loadu_ps(lim, row + 16 * tt);
                for (size_t p = 0; p < P; p++) {
                    const float gv = g[p * gs + oo];
                    if (gv == 0.0f) continue;
                    a = _mm512_fmadd_ps(_mm512_set1_ps(gv), _mm512_maskz_loadu_ps(lim, x + p * xs + 16 * tt), a);
                }
                _mm512_mask_storeu_ps(row + 16 * tt, lim, a);
            }
        }
    }
    for (; o < o1; o++) {
        float *row = gw + o * n;
        size_t t = 0;
        for (; t + 8 <= full; t += 8) {
            __m512 a[8];
            for (int k = 0; k < 8; k++) a[k] = _mm512_loadu_ps(row + 16 * (t + k));
            for (size_t p = 0; p < P; p++) {
                const float gv = g[p * gs + o];
                if (gv == 0.0f) continue;
                const __m512 vg = _mm512_set1_ps(gv);
                const float *xp = x + p * xs + 16 * t;
                for (int k = 0; k < 8; k++) a[k] = _mm512_fmadd_ps(vg, _mm512_loadu_ps(xp + 16 * k), a[k]);
            }
            for (int k = 0; k < 8; k++) _mm512_storeu_ps(row + 16 * (t + k), a[k]);
        }
        for (; t < nt; t++) {
            const __mmask16 lim = tail_lim(n, t);
            __m512 a = _mm512_maskz_loadu_ps(lim, row + 16 * t);
            for (size_t p = 0; p < P; p++) {
                const float gv = g[p * gs + o];
                if (gv == 0.0f) continue;
                a = _mm512_fmadd_ps(_mm512_set1_ps(gv), _mm512_maskz_loadu_ps(lim, x + p * xs + 16 * t), a);
            }
            _mm512_mask_storeu_ps(row + 16 * t, lim, a);
        }
    }
}

AVX512_FN void ternary_avx512_outer_acc(float *gw, size_t n, float g, const float *xf) {
    const __m512 vg = _mm512_set1_ps(g);
    const size_t nt = (n + 15) / 16;
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(n, t);
        const __m512 a = _mm512_maskz_loadu_ps(lim, gw + 16 * t), x = _mm512_maskz_loadu_ps(lim, xf + 16 * t);
        _mm512_mask_storeu_ps(gw + 16 * t, lim, ADDPS(a, MULPS(vg, x)));
    }
}

AVX512_FN void ternary_avx512_gradx_row(float *gx, size_t n, const uint64_t *wm, const uint64_t *ws, float gs) {
    const __m512 vg = _mm512_set1_ps(gs);
    const size_t nt = (n + 15) / 16;
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(n, t);
        const size_t w = (16 * t) >> 6, sh = (16 * t) & 63;
        const uint64_t m = wm[w] >> sh, sg = ws[w] >> sh;
        const __mmask16 pos = (__mmask16)(m & ~sg) & lim, neg = (__mmask16)(m & sg) & lim;
        __m512 a = _mm512_maskz_loadu_ps(lim, gx + 16 * t);
        a = _mm512_mask_add_round_ps(a, pos, a, vg, RN);
        a = _mm512_mask_sub_round_ps(a, neg, a, vg, RN);
        _mm512_mask_storeu_ps(gx + 16 * t, lim, a);
    }
}


/* ======================================================================================================
 * Forward entero con VNNI. Pesos en {-1,0,+1} como int8 en bloques de 4 entradas: wt[(k*Opad + o)*4 + j] = w[o][4k+j]
 * (Opad = O redondeado a 16, relleno con ceros). vpdpbusd suma 4 productos u8 x int8 en cada lane int32, asi que un zmm
 * produce 16 salidas a la vez (64 MAC) y no hace falta ninguna reduccion horizontal. z = sum(q*w) - zp*sum(w) + bias es el
 * mismo entero exacto que da ternary_avx512_dot_u8 (sin redondeos: el orden de las sumas no importa).
 * ====================================================================================================== */
#ifndef VNNI_NB
#define VNNI_NB 2
#endif
#ifndef VNNI_PB
#define VNNI_PB 8
#endif
#define VNNI_FN __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni")))
#define VNNI_INL __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni"), always_inline)) static inline

/* n pesos (mascara/signo) -> n int8 en {-1,0,1}; escribe exactamente n bytes. */
AVX512_FN void ternary_avx512_expand_i8(const uint64_t *mask, const uint64_t *sign, size_t n, int8_t *out) {
    const __m512i one = _mm512_set1_epi8(1), mone = _mm512_set1_epi8(-1);
    for (size_t k = 0; 64 * k < n; k++) {
        const size_t rem = n - 64 * k;
        const __mmask64 lm = (rem >= 64) ? ~(__mmask64)0 : (((__mmask64)1 << rem) - 1);
        const __mmask64 pos = (__mmask64)(mask[k] & ~sign[k]) & lm, neg = (__mmask64)(mask[k] & sign[k]) & lm;
        _mm512_mask_storeu_epi8(out + 64 * k, lm, _mm512_mask_mov_epi8(_mm512_maskz_mov_epi8(pos, one), neg, mone));
    }
}

VNNI_INL void vnni_tile(const int8_t *wt, size_t Opad, size_t in, size_t ob, const int NB, const int PB, const uint8_t *q, size_t qs,
                        size_t p0, const float *sx, const int32_t *zp, const int32_t *wsum, const int32_t *bias, size_t O, float scale,
                        float *out, size_t os, int32_t *raw, size_t rs) {
    __m512i acc[8][4];
    for (int p = 0; p < PB; p++) for (int b = 0; b < NB; b++) acc[p][b] = _mm512_setzero_si512();
    const size_t k4 = in / 4;
    const int8_t *wb = wt + ob * 4;
    const uint8_t *qp[8];
    for (int p = 0; p < PB; p++) qp[p] = q + (p0 + p) * qs;
    for (size_t k = 0; k < k4; k++) {
        __m512i w[4];
        for (int b = 0; b < NB; b++) w[b] = _mm512_loadu_si512((const void *)(wb + k * Opad * 4 + (size_t)b * 64));
        for (int p = 0; p < PB; p++) {
            int32_t d; memcpy(&d, qp[p] + 4 * k, 4);
            const __m512i qv = _mm512_set1_epi32(d);
            for (int b = 0; b < NB; b++) acc[p][b] = _mm512_dpbusd_epi32(acc[p][b], qv, w[b]);
        }
    }
    if (in & 3) {   /* ultimo bloque parcial: los pesos de relleno son 0 */
        __m512i w[4];
        for (int b = 0; b < NB; b++) w[b] = _mm512_loadu_si512((const void *)(wb + k4 * Opad * 4 + (size_t)b * 64));
        for (int p = 0; p < PB; p++) {
            int32_t d = 0; memcpy(&d, qp[p] + 4 * k4, in & 3);
            const __m512i qv = _mm512_set1_epi32(d);
            for (int b = 0; b < NB; b++) acc[p][b] = _mm512_dpbusd_epi32(acc[p][b], qv, w[b]);
        }
    }
    for (int p = 0; p < PB; p++) {
        const __m512i vzp = _mm512_set1_epi32(zp[p0 + p]);
        const __m512 vk = _mm512_set1_ps(scale * sx[p0 + p]);
        for (int b = 0; b < NB; b++) {
            const size_t o0 = ob + 16 * (size_t)b;
            const size_t rem = O - o0;
            const __mmask16 lim = rem >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << rem) - 1u);
            __m512i z = _mm512_sub_epi32(acc[p][b], _mm512_mullo_epi32(vzp, _mm512_loadu_si512((const void *)(wsum + o0))));
            if (bias) z = _mm512_add_epi32(z, _mm512_loadu_si512((const void *)(bias + o0)));
            if (raw) _mm512_mask_storeu_epi32(raw + (p0 + p) * rs + o0, lim, z);
            if (out) _mm512_mask_storeu_ps(out + (p0 + p) * os + o0, lim, MULPS(vk, _mm512_cvtepi32_ps(z)));
        }
    }
}

VNNI_FN void ternary_vnni_forward(const int8_t *wt, const int32_t *wsum, const int32_t *bias, size_t Opad, size_t O, size_t in, size_t P,
                                  const uint8_t *q, size_t qs, const float *sx, const int32_t *zp, float scale,
                                  float *out, size_t os, int32_t *raw, size_t rs) {
    size_t ob = 0;
    #define VNNI_TILE(NBV, PBV) vnni_tile(wt, Opad, in, ob, NBV, PBV, q, qs, p, sx, zp, wsum, bias, O, scale, out, os, raw, rs)
    #define VNNI_GROUP(NBV) \
        for (; ob + 16 * (NBV) <= Opad; ob += 16 * (NBV)) { \
            size_t p = 0; \
            for (; p + VNNI_PB <= P; p += VNNI_PB) VNNI_TILE(NBV, VNNI_PB); \
            switch (P - p) { \
                case 7: VNNI_TILE(NBV, 7); break; case 6: VNNI_TILE(NBV, 6); break; case 5: VNNI_TILE(NBV, 5); break; \
                case 4: VNNI_TILE(NBV, 4); break; case 3: VNNI_TILE(NBV, 3); break; case 2: VNNI_TILE(NBV, 2); break; \
                case 1: VNNI_TILE(NBV, 1); break; default: break; \
            } \
        }
    VNNI_GROUP(VNNI_NB) VNNI_GROUP(2) VNNI_GROUP(1)
    #undef VNNI_GROUP
    #undef VNNI_TILE
}

/* ======================================================================================================
 * SwiGLU: producto y compuerta elemento a elemento (silu / sign / ReLU2). Mismos bits que el escalar.
 * ====================================================================================================== */
/* 2^y, y en [CUTOFF, 0]: las mismas operaciones y el mismo orden que ternary_fexp2_core (ternary_fexp.h). */
AVX512_INL __m512 fexp2_v(__m512 y) {
    const __m512 fl = _mm512_roundscale_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m512 f = SUBPS(y, fl);
    __m512 p = _mm512_set1_ps(1.535336188319500e-4f);
    p = MULPS(p, f); p = ADDPS(p, _mm512_set1_ps(1.339887440266574e-3f));
    p = MULPS(p, f); p = ADDPS(p, _mm512_set1_ps(9.618437357674640e-3f));
    p = MULPS(p, f); p = ADDPS(p, _mm512_set1_ps(5.550332471162809e-2f));
    p = MULPS(p, f); p = ADDPS(p, _mm512_set1_ps(2.402264791363012e-1f));
    p = MULPS(p, f); p = ADDPS(p, _mm512_set1_ps(6.931472028550421e-1f));
    __m512 r = MULPS(f, p); r = ADDPS(r, _mm512_set1_ps(1.0f));
    const __m512i e = _mm512_slli_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(fl), _mm512_set1_epi32(127)), 23);
    return MULPS(r, _mm512_castsi512_ps(e));
}

/* silu(a) y su derivada (igual que silu_f de ternary_swiglu.c). */
AVX512_INL void silu_v(__m512 a, __m512 *sv, __m512 *dsv) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 y = MULPS(_mm512_sub_ps(_mm512_setzero_ps(), _mm512_abs_ps(a)), _mm512_set1_ps(1.44269504088896341f));
    const __mmask16 ok = _mm512_cmp_ps_mask(y, _mm512_set1_ps(-66.0f), _CMP_NLT_UQ);   /* !(y < CUTOFF) */
    const __m512 e = _mm512_maskz_mov_ps(ok, fexp2_v(_mm512_mask_mov_ps(_mm512_set1_ps(-1.0f), ok, y)));
    const __m512 den = ADDPS(one, e);
    const __m512 sg = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(a, _mm512_setzero_ps(), _CMP_GE_OQ),
                                           _mm512_div_round_ps(e, den, RN), _mm512_div_round_ps(one, den, RN));
    *sv = MULPS(a, sg);
    *dsv = MULPS(sg, ADDPS(one, MULPS(a, SUBPS(one, sg))));
}

/* (float)sign(z) como vector de floats: +1, -1 o 0. */
AVX512_INL __m512 sign_f(__m512i z) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512i zero = _mm512_setzero_si512();
    __m512 r = _mm512_maskz_mov_ps(_mm512_cmpgt_epi32_mask(z, zero), one);
    return _mm512_mask_mov_ps(r, _mm512_cmplt_epi32_mask(z, zero), _mm512_set1_ps(-1.0f));
}

/* h de una posicion desde los Z enteros (product_from_z): modo 0 silu por tabla, 1 sign, 2 ReLU2. */
AVX512_FN void ternary_avx512_swiglu_prod_z(int mode, size_t H, const int32_t *zg, const int32_t *zu, float su,
                                            const float *silu_tab, int32_t D, float *h) {
    const __m512 vsu = _mm512_set1_ps(su);
    const size_t nt = (H + 15) / 16;
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(H, t);
        const __m512i u = _mm512_maskz_loadu_epi32(lim, zu + 16 * t);
        __m512 r;
        if (mode == 0) {
            __m512i k = _mm512_add_epi32(_mm512_maskz_loadu_epi32(lim, zg + 16 * t), _mm512_set1_epi32(D));
            k = _mm512_min_epi32(_mm512_max_epi32(k, _mm512_setzero_si512()), _mm512_set1_epi32(2 * D));
            r = MULPS(_mm512_i32gather_ps(k, silu_tab, 4), MULPS(vsu, _mm512_cvtepi32_ps(u)));
        } else if (mode == 2) {
            const __m512 a = MULPS(vsu, _mm512_cvtepi32_ps(u));
            r = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(a, _mm512_setzero_ps(), _CMP_GT_OQ), MULPS(a, a));
        } else {
            const __m512i g = _mm512_maskz_loadu_epi32(lim, zg + 16 * t);
            const __m512i z = _mm512_setzero_si512();
            __m512i v = _mm512_maskz_mov_epi32(_mm512_cmpneq_epi32_mask(g, z), u);
            v = _mm512_mask_sub_epi32(v, _mm512_cmplt_epi32_mask(g, z), z, v);
            r = _mm512_cvtepi32_ps(v);
        }
        _mm512_mask_storeu_ps(h + 16 * t, lim, r);
    }
}

/* Backward de producto y compuerta de una posicion (ruta con tablas): gu y ga (ga solo si mode != 2). */
AVX512_FN void ternary_avx512_swiglu_bwd_z(int mode, size_t H, const int32_t *zg, const int32_t *zu, const float *gt, float su,
                                           const float *silu_tab, const float *dsilu_tab, int32_t D, float *gu, float *ga) {
    const __m512 vsu = _mm512_set1_ps(su);
    const size_t nt = (H + 15) / 16;
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(H, t);
        const __m512 g = _mm512_maskz_loadu_ps(lim, gt + 16 * t);
        const __m512 u = MULPS(vsu, _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(lim, zu + 16 * t)));
        if (mode == 2) {
            const __m512 r = MULPS(MULPS(g, _mm512_set1_ps(2.0f)), u);
            _mm512_mask_storeu_ps(gu + 16 * t, lim, _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(u, _mm512_setzero_ps(), _CMP_GT_OQ), r));
        } else if (mode == 0) {
            __m512i k = _mm512_add_epi32(_mm512_maskz_loadu_epi32(lim, zg + 16 * t), _mm512_set1_epi32(D));
            k = _mm512_min_epi32(_mm512_max_epi32(k, _mm512_setzero_si512()), _mm512_set1_epi32(2 * D));
            _mm512_mask_storeu_ps(gu + 16 * t, lim, MULPS(g, _mm512_i32gather_ps(k, silu_tab, 4)));
            _mm512_mask_storeu_ps(ga + 16 * t, lim, MULPS(MULPS(g, u), _mm512_i32gather_ps(k, dsilu_tab, 4)));
        } else {
            _mm512_mask_storeu_ps(gu + 16 * t, lim, MULPS(g, sign_f(_mm512_maskz_loadu_epi32(lim, zg + 16 * t))));
            _mm512_mask_storeu_ps(ga + 16 * t, lim, MULPS(g, u));
        }
    }
}

/* Ruta uint8: h = compuerta(fa) * fu con silu exacto (exponencial compartida). zg solo en modo sign. */
AVX512_FN void ternary_avx512_swiglu_fwd_u8(int mode, size_t H, const float *fa, const float *fu, const int32_t *zg, float *h) {
    const size_t nt = (H + 15) / 16;
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(H, t);
        __m512 r;
        if (mode == 2) {
            const __m512 a = _mm512_maskz_loadu_ps(lim, fu + 16 * t);
            r = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(a, _mm512_setzero_ps(), _CMP_GT_OQ), MULPS(a, a));
        } else if (mode == 0) {
            __m512 sv, dsv; silu_v(_mm512_maskz_loadu_ps(lim, fa + 16 * t), &sv, &dsv);
            r = MULPS(sv, _mm512_maskz_loadu_ps(lim, fu + 16 * t));
        } else {
            r = MULPS(sign_f(_mm512_maskz_loadu_epi32(lim, zg + 16 * t)), _mm512_maskz_loadu_ps(lim, fu + 16 * t));
        }
        _mm512_mask_storeu_ps(h + 16 * t, lim, r);
    }
}

/* Backward uint8 de una posicion: a = kg*zg, u = ku*zu (kg = escala_gate*sx, ku = escala_up*sx). */
AVX512_FN void ternary_avx512_swiglu_bwd_u8(int mode, size_t H, const int32_t *zg, const int32_t *zu, const float *gt,
                                            float kg, float ku, float *gu, float *ga) {
    const __m512 vkg = _mm512_set1_ps(kg), vku = _mm512_set1_ps(ku);
    const size_t nt = (H + 15) / 16;
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(H, t);
        const __m512 g = _mm512_maskz_loadu_ps(lim, gt + 16 * t);
        const __m512 u = MULPS(vku, _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(lim, zu + 16 * t)));
        if (mode == 2) {
            const __m512 r = MULPS(MULPS(g, _mm512_set1_ps(2.0f)), u);
            _mm512_mask_storeu_ps(gu + 16 * t, lim, _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(u, _mm512_setzero_ps(), _CMP_GT_OQ), r));
        } else if (mode == 0) {
            const __m512 a = MULPS(vkg, _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(lim, zg + 16 * t)));
            __m512 sv, dsv; silu_v(a, &sv, &dsv);
            _mm512_mask_storeu_ps(gu + 16 * t, lim, MULPS(g, sv));
            _mm512_mask_storeu_ps(ga + 16 * t, lim, MULPS(MULPS(g, u), dsv));
        } else {
            _mm512_mask_storeu_ps(gu + 16 * t, lim, MULPS(g, sign_f(_mm512_maskz_loadu_epi32(lim, zg + 16 * t))));
            _mm512_mask_storeu_ps(ga + 16 * t, lim, MULPS(g, u));
        }
    }
}


/* ======================================================================================================
 * Entropia cruzada (una fila) y sumas del gradiente de la escala de la cabeza.
 * ====================================================================================================== */
AVX512_INL __m512 expf_nonpos_v(__m512 x) {      /* igual que ternary_expf_nonpos (ternary_fexp.h) */
    const __m512 y = MULPS(x, _mm512_set1_ps(1.4426950408889634f));
    const __mmask16 ok = _mm512_cmp_ps_mask(y, _mm512_set1_ps(-66.0f), _CMP_NLT_UQ);
    return _mm512_maskz_mov_ps(ok, fexp2_v(_mm512_mask_mov_ps(_mm512_set1_ps(-1.0f), ok, y)));
}

AVX512_FN void ternary_avx512_xent_row(const float *l, size_t V, size_t t, size_t n, float *g, int *ok_out, double *loss_out) {
    const size_t nt = (V + 15) / 16;
    __m512 mv = _mm512_set1_ps(-INFINITY);
    for (size_t k = 0; k < nt; k++) mv = _mm512_max_ps(mv, _mm512_mask_loadu_ps(_mm512_set1_ps(-INFINITY), tail_lim(V, k), l + 16 * k));
    const float m = _mm512_reduce_max_ps(mv);
    size_t arg = 0;
    for (size_t k = 0; k < nt; k++) {
        const __mmask16 eq = _mm512_cmp_ps_mask(_mm512_maskz_loadu_ps(tail_lim(V, k), l + 16 * k), _mm512_set1_ps(m), _CMP_EQ_OQ) & tail_lim(V, k);
        if (eq) { arg = 16 * k + (size_t)__builtin_ctz((unsigned)eq); break; }
    }
    *ok_out = (arg == t);
    const __m512 vm = _mm512_set1_ps(m);
    __m512d a0 = _mm512_setzero_pd(), a1 = _mm512_setzero_pd();
    for (size_t k = 0; k < nt; k++) {
        const __mmask16 lim = tail_lim(V, k);
        const __m512 e = _mm512_maskz_mov_ps(lim, expf_nonpos_v(SUBPS(_mm512_maskz_loadu_ps(lim, l + 16 * k), vm)));
        a0 = _mm512_add_pd(a0, _mm512_cvtps_pd(_mm512_castps512_ps256(e)));
        a1 = _mm512_add_pd(a1, _mm512_cvtps_pd(_mm512_extractf32x8_ps(e, 1)));
    }
    const double sum = reduce_d16(a0, a1);
    *loss_out = log(sum) - (double)(l[t] - m);
    if (g) {
        const __m512d vinv = _mm512_set1_pd(1.0 / (sum * (double)n));
        for (size_t k = 0; k < nt; k++) {
            const __mmask16 lim = tail_lim(V, k);
            const __m512 e = _mm512_maskz_mov_ps(lim, expf_nonpos_v(SUBPS(_mm512_maskz_loadu_ps(lim, l + 16 * k), vm)));
            const __m256 lo = _mm512_cvtpd_ps(_mm512_mul_round_pd(_mm512_cvtps_pd(_mm512_castps512_ps256(e)), vinv, RN));
            const __m256 hi = _mm512_cvtpd_ps(_mm512_mul_round_pd(_mm512_cvtps_pd(_mm512_extractf32x8_ps(e, 1)), vinv, RN));
            _mm512_mask_storeu_ps(g + 16 * k, lim, _mm512_insertf32x8(_mm512_castps256_ps512(lo), hi, 1));
        }
        g[t] -= (float)(1.0 / (double)n);
    }
}

/* sum (double)g*(double)z, y con factor: ((double)g*sx)*(double)z; 16 carriles, orden de ternary_canon_dot_gz[sx]. */
AVX512_FN double ternary_avx512_dot_gz(const float *g, const int32_t *z, size_t n, int use_sx, double sx) {
    const size_t nt = (n + 15) / 16;
    __m512d a0 = _mm512_setzero_pd(), a1 = _mm512_setzero_pd();
    const __m512d vs = _mm512_set1_pd(sx);
    for (size_t k = 0; k < nt; k++) {
        const __mmask16 lim = tail_lim(n, k);
        const __m512 gv = _mm512_maskz_loadu_ps(lim, g + 16 * k);
        const __m512i zv = _mm512_maskz_loadu_epi32(lim, z + 16 * k);
        __m512d g0 = _mm512_cvtps_pd(_mm512_castps512_ps256(gv)), g1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(gv, 1));
        if (use_sx) { g0 = _mm512_mul_round_pd(g0, vs, RN); g1 = _mm512_mul_round_pd(g1, vs, RN); }
        const __m512d z0 = _mm512_cvtepi32_pd(_mm512_castsi512_si256(zv)), z1 = _mm512_cvtepi32_pd(_mm512_extracti32x8_epi32(zv, 1));
        a0 = _mm512_add_round_pd(a0, _mm512_mul_round_pd(g0, z0, RN), RN);
        a1 = _mm512_add_round_pd(a1, _mm512_mul_round_pd(g1, z1, RN), RN);
    }
    return reduce_d16(a0, a1);
}

AVX512_FN void ternary_avx512_scale_f32(float *x, size_t n, float s) {
    const __m512 vs = _mm512_set1_ps(s);
    const size_t nt = (n + 15) / 16;
    for (size_t k = 0; k < nt; k++) {
        const __mmask16 lim = tail_lim(n, k);
        _mm512_mask_storeu_ps(x + 16 * k, lim, MULPS(_mm512_maskz_loadu_ps(lim, x + 16 * k), vs));
    }
}


/* ======================================================================================================
 * Regla R2: una fila. Mismas operaciones y orden que ternary_r2_row_scalar (ternary_dense.c).
 * ====================================================================================================== */
AVX512_INL void apply_bits(uint64_t *mask, uint64_t *sign, size_t bit0, uint16_t P, uint16_t N) {
    if (!(P | N)) return;
    const size_t w = bit0 >> 6, b = bit0 & 63;
    for (int part = 0; part < 2; part++) {
        uint64_t Pw, Nw;
        if (part == 0) { Pw = (uint64_t)P << b; Nw = (uint64_t)N << b; }
        else { if (b <= 48) break; Pw = (uint64_t)P >> (64 - b); Nw = (uint64_t)N >> (64 - b); }
        uint64_t *mp = mask + w + part, *sp = sign + w + part;
        const uint64_t m = *mp, s = *sp, any = Pw | Nw;
        const uint64_t nm = ~(m & ~s);                       /* mascara destino con paso -1: ~(w == +1) */
        *mp = (m & ~any) | (Pw & ~(m & s)) | (Nw & nm);
        *sp = (s & ~any) | (Nw & nm);
    }
}

/* Regla R2: 16 pesos por tile; mismos pasos y orden que ternary_r2_update_row (ternary_dense.c). Los umbrales (ti, tin, tout) y ri vienen ya
 * calculados (comunes con el escalar), asi que solo queda la aritmetica del contador y la seleccion de umbral segun el estado de cada peso. */
AVX512_INL uint16_t bits16(const uint64_t *w, size_t bit0) {
    const size_t i = bit0 >> 6, b = bit0 & 63;
    uint64_t v = w[i] >> b; if (b > 48) v |= w[i + 1] << (64 - b);
    return (uint16_t)v;
}
AVX512_FN void ternary_avx512_elemnorm_row(uint8_t *cr, float *rmax, float *g, float *v, size_t W, float beta2, float omb, float bc, float eps) {
    const size_t nt = (W + 15) / 16, st = (W + 1) / 2;
    const uint8_t keep_hi = (W & 1) ? (uint8_t)(cr[st - 1] & 0xF0) : 0;
    const __m512 vb = _mm512_set1_ps(beta2), vo = _mm512_set1_ps(omb), vrm = _mm512_set1_ps(*rmax);
    const __m512i sh = _mm512_set_epi32(4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0), m15 = _mm512_set1_epi32(15);
    __m512 mxv = _mm512_setzero_ps();
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(W, t);
        const size_t nb = (st - 8 * t) >= 8 ? 8 : (st - 8 * t);
        const __m128i b8 = _mm_maskz_loadu_epi8((__mmask16)((1u << nb) - 1u), cr + 8 * t);
        const __m512i c = _mm512_and_si512(_mm512_srlv_epi32(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(b8, b8)), sh), m15);
        const __m512 vh = _mm512_scalef_ps(vrm, _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_setzero_si512(), c)));
        const __m512 x = _mm512_maskz_loadu_ps(lim, g + 16 * t);
        const __m512 vn = ADDPS(MULPS(vb, vh), MULPS(vo, MULPS(x, x)));
        const __m512 vm = _mm512_maskz_mov_ps(lim, vn);
        _mm512_mask_storeu_ps(v + 16 * t, lim, vm);
        mxv = _mm512_max_ps(mxv, vm);
    }
    const float mx = _mm512_reduce_max_ps(mxv);
    *rmax = mx;
    const __m512 vmx = _mm512_set1_ps(mx), vbc = _mm512_set1_ps(bc), veps = _mm512_set1_ps(eps);
    const __m512 vlo = _mm512_set1_ps(0x1p-16f), vmt = _mm512_set1_ps(0.70710678f);
    const __m512i i0 = _mm512_setzero_si512(), i15 = m15, one = _mm512_set1_epi32(1);
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(W, t);
        const size_t nb = (st - 8 * t) >= 8 ? 8 : (st - 8 * t);
        const __m512 q = _mm512_div_ps(_mm512_maskz_loadu_ps(lim, v + 16 * t), vmx);
        const __mmask16 ok = _mm512_cmp_ps_mask(q, vlo, _CMP_GE_OQ);
        const __m512 e = _mm512_getexp_ps(q);
        const __m512 m = _mm512_getmant_ps(q, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
        const __mmask16 lt = _mm512_cmp_ps_mask(m, vmt, _CMP_LT_OQ);
        /* r = -(getexp+1) + (m < 2^-0.5) */
        __m512i r = _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_add_epi32(_mm512_cvtps_epi32(e), one));
        r = _mm512_mask_add_epi32(r, lt, r, one);
        r = _mm512_min_epi32(_mm512_max_epi32(r, i0), i15);
        r = _mm512_mask_mov_epi32(i15, ok, r);
        const __m512 vq = _mm512_scalef_ps(vmx, _mm512_cvtepi32_ps(_mm512_sub_epi32(i0, r)));
        const __m512 gx = _mm512_maskz_loadu_ps(lim, g + 16 * t);
        const __m512 den = ADDPS(_mm512_sqrt_ps(_mm512_div_ps(vq, vbc)), veps);
        _mm512_mask_storeu_ps(g + 16 * t, lim, _mm512_div_ps(gx, den));
        const __m128i w = _mm512_cvtepi32_epi8(r);
        const __m128i pk = _mm_packus_epi16(_mm_and_si128(_mm_or_si128(w, _mm_srli_epi16(w, 4)), _mm_set1_epi16(0xFF)), _mm_setzero_si128());
        _mm_mask_storeu_epi8(cr + 8 * t, (__mmask16)((1u << nb) - 1u), pk);
    }
    if (W & 1) cr[st - 1] = (uint8_t)((cr[st - 1] & 0x0F) | keep_hi);
}

AVX512_FN void ternary_avx512_r2_row(uint64_t *mask, uint64_t *sign, int8_t *cnt, size_t base, const float *g, size_t n,
                                     float beta, float inc, int ti, int tin, int tout, int ri, int pre) {
    float inv = 1.0f;                                       /* pre: gradiente ya normalizado */
    if (!pre) { const double ms = ss_row(g, n) / (double)n; inv = 1.0f / ((float)sqrt(ms) + 1e-12f); }
    const float cb = beta, cg = inc * 127.0f;
    const __m512 vinv = _mm512_set1_ps(inv), vcb = _mm512_set1_ps(cb), vcg = _mm512_set1_ps(cg);
    const __m512 c3 = _mm512_set1_ps(3.0f), cm3 = _mm512_set1_ps(-3.0f);
    const __m512i v127 = _mm512_set1_epi32(127), vm127 = _mm512_set1_epi32(-127), vti = _mm512_set1_epi32(ti), vtin = _mm512_set1_epi32(tin), vtout = _mm512_set1_epi32(tout);
    const __m512i vrp = _mm512_set1_epi32(-ri), vrn = _mm512_set1_epi32(ri), vz = _mm512_setzero_si512();
    const size_t nt = (n + 15) / 16;
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(n, t);
        __m512 gh = MULPS(_mm512_maskz_loadu_ps(lim, g + 16 * t), vinv);
        gh = _mm512_mask_mov_ps(gh, _mm512_cmp_ps_mask(gh, c3, _CMP_GT_OQ), c3);
        gh = _mm512_mask_mov_ps(gh, _mm512_cmp_ps_mask(gh, cm3, _CMP_LT_OQ), cm3);
        const __m512 c = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(lim, cnt + 16 * t)));
        __m512 x = SUBPS(MULPS(vcb, c), MULPS(vcg, gh));
        x = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(x, x, _CMP_EQ_OQ), x);                  /* NaN -> 0 */
        x = _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512i v = _mm512_cvtps_epi32(x);
        v = _mm512_min_epi32(_mm512_max_epi32(v, vm127), v127);
        const uint16_t m16 = bits16(mask, base + 16 * t), s16 = bits16(sign, base + 16 * t);
        const __mmask16 zero = (__mmask16)~m16, pos = (__mmask16)(m16 & ~s16), neg = (__mmask16)(m16 & s16);
        const __mmask16 dpos = _mm512_cmpgt_epi32_mask(v, vz), dneg = (__mmask16)~dpos;     /* dir = +1 si v > 0, -1 en otro caso */
        const __mmask16 towards0 = (__mmask16)((dpos & neg) | (dneg & pos));                  /* dir == -cur con cur != 0 */
        __m512i th = vti;
        th = _mm512_mask_mov_epi32(th, towards0, vtout);
        th = _mm512_mask_mov_epi32(th, zero, vtin);
        const __mmask16 F = (_mm512_cmpge_epi32_mask(v, th) | _mm512_cmple_epi32_mask(v, _mm512_sub_epi32(vz, th))) & lim;
        const __mmask16 P = F & dpos, N = F & dneg;
        v = _mm512_mask_mov_epi32(v, P, vrp);                                                 /* v = -dir * ri */
        v = _mm512_mask_mov_epi32(v, N, vrn);
        _mm_mask_storeu_epi8(cnt + 16 * t, lim, _mm512_cvtepi32_epi8(v));
        apply_bits(mask, sign, base + 16 * t, (uint16_t)P, (uint16_t)N);
    }
}


/* ======================================================================================================
 * Elementos sueltos: cuantizacion uint8, residual, softmax entero, RoPE entera y utilidades de GQA.
 * ====================================================================================================== */
AVX512_FN void ternary_avx512_u8_quantize_row(const float *x, size_t n, uint8_t *q, float *scale_out, int32_t *zp_out) {
    const size_t nt = (n + 15) / 16;
    __m512 mn = _mm512_setzero_ps(), mx = _mm512_setzero_ps();             /* el rango incluye el 0 */
    for (size_t t = 0; t < nt; t++) {
        const __m512 v = _mm512_maskz_loadu_ps(tail_lim(n, t), x + 16 * t);
        mn = _mm512_min_ps(v, mn); mx = _mm512_max_ps(v, mx);               /* un NaN no cambia el rango (devuelve el 2o operando) */
    }
    const float lo = _mm512_reduce_min_ps(mn), hi = _mm512_reduce_max_ps(mx);
    const float sc = (hi - lo) / 255.0f;
    if (!(sc > 0.0f) || !isfinite(sc)) { *scale_out = 1.0f; *zp_out = 0; memset(q, 0, n); return; }
    int zp = (int)rintf(-lo / sc);
    if (zp < 0) zp = 0;
    if (zp > 255) zp = 255;
    *scale_out = sc; *zp_out = zp;
    const __m512 vinv = _mm512_set1_ps(1.0f / sc), vzp = _mm512_set1_ps((float)zp);
    const __m512 z0 = _mm512_setzero_ps(), c255 = _mm512_set1_ps(255.0f);
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(n, t);
        __m512 qf = ADDPS(_mm512_roundscale_ps(MULPS(_mm512_maskz_loadu_ps(lim, x + 16 * t), vinv), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), vzp);
        qf = _mm512_mask_mov_ps(qf, _mm512_cmp_ps_mask(qf, z0, _CMP_LT_OQ), z0);
        qf = _mm512_mask_mov_ps(qf, _mm512_cmp_ps_mask(qf, c255, _CMP_GT_OQ), c255);
        _mm_mask_storeu_epi8(q + 16 * t, lim, _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(qf)));
    }
}

AVX512_FN void ternary_avx512_add_f32(float *out, const float *a, const float *b, size_t n) {
    const size_t nt = (n + 15) / 16;
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(n, t);
        _mm512_mask_storeu_ps(out + 16 * t, lim, ADDPS(_mm512_maskz_loadu_ps(lim, a + 16 * t), _mm512_maskz_loadu_ps(lim, b + 16 * t)));
    }
}
AVX512_FN void ternary_avx512_acc_add2(float *dst, const float *a, const float *b, size_t n) {
    const size_t nt = (n + 15) / 16;
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(n, t);
        const __m512 sum = ADDPS(_mm512_maskz_loadu_ps(lim, a + 16 * t), _mm512_maskz_loadu_ps(lim, b + 16 * t));
        _mm512_mask_storeu_ps(dst + 16 * t, lim, ADDPS(_mm512_maskz_loadu_ps(lim, dst + 16 * t), sum));
    }
}

AVX512_FN void ternary_avx512_softmax_fwd(const int32_t *scores, size_t n_valid, const float *table, size_t table_len, float *out) {
    const size_t nt = (n_valid + 15) / 16;
    __m512i mv = _mm512_set1_epi32(INT32_MIN);
    for (size_t t = 0; t < nt; t++) mv = _mm512_max_epi32(mv, _mm512_mask_loadu_epi32(_mm512_set1_epi32(INT32_MIN), tail_lim(n_valid, t), scores + 16 * t));
    const __m512i vm = _mm512_set1_epi32(_mm512_reduce_max_epi32(mv)), vcap = _mm512_set1_epi32((int32_t)table_len - 1);
    __m512 part = _mm512_setzero_ps();
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(n_valid, t);
        const __m512i k = _mm512_min_epi32(_mm512_sub_epi32(vm, _mm512_maskz_loadu_epi32(lim, scores + 16 * t)), vcap);
        const __m512 e = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), lim, k, table, 4);
        _mm512_mask_storeu_ps(out + 16 * t, lim, e);
        part = ADDPS(part, e);
    }
    const __m512 vinv = _mm512_set1_ps(1.0f / sum16_ps(part));
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(n_valid, t);
        _mm512_mask_storeu_ps(out + 16 * t, lim, MULPS(_mm512_maskz_loadu_ps(lim, out + 16 * t), vinv));
    }
}

AVX512_FN void ternary_avx512_softmax_bwd(const float *y, const float *g, size_t n, float inv_temp, float *out_ds) {
    const size_t nt = (n + 15) / 16;
    __m512 part = _mm512_setzero_ps();
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(n, t);
        part = ADDPS(part, MULPS(_mm512_maskz_loadu_ps(lim, g + 16 * t), _mm512_maskz_loadu_ps(lim, y + 16 * t)));
    }
    const __m512 vdot = _mm512_set1_ps(sum16_ps(part)), vt = _mm512_set1_ps(inv_temp);
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(n, t);
        const __m512 yv = _mm512_maskz_loadu_ps(lim, y + 16 * t), gv = _mm512_maskz_loadu_ps(lim, g + 16 * t);
        _mm512_mask_storeu_ps(out_ds + 16 * t, lim, MULPS(MULPS(vt, yv), SUBPS(gv, vdot)));
    }
}

/* ---- RoPE entera y utilidades de GQA ---- */
AVX512_INL __m512i idx_even(void) { return _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30); }
AVX512_INL __m512i idx_odd(void)  { return _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31); }
AVX512_INL __m512i idx_il0(void)  { return _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23); }
AVX512_INL __m512i idx_il1(void)  { return _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31); }

AVX512_FN void ternary_avx512_rope_rotate_int(const int32_t *z, size_t heads, size_t H, size_t P, const int16_t *c, const int16_t *s, int32_t *out) {
    for (size_t h = 0; h < heads; h++)
        for (size_t i0 = 0; i0 < P; i0 += 16) {
            const size_t np = (P - i0 < 16) ? P - i0 : 16, ints = 2 * np;
            const __mmask16 pm = (__mmask16)((1u << np) - 1u);
            const __mmask16 lo = ints >= 16 ? (__mmask16)0xFFFFu : (__mmask16)((1u << ints) - 1u);
            const __mmask16 hi = ints > 16 ? (__mmask16)((1u << (ints - 16)) - 1u) : (__mmask16)0;
            const int32_t *zp = z + h * H + 2 * i0;
            const __m512i v0 = _mm512_maskz_loadu_epi32(lo, zp), v1 = _mm512_maskz_loadu_epi32(hi, zp + 16);
            const __m512i a = _mm512_permutex2var_epi32(v0, idx_even(), v1), b = _mm512_permutex2var_epi32(v0, idx_odd(), v1);
            const __m512i cv = _mm512_cvtepi16_epi32(_mm256_maskz_loadu_epi16(pm, c + i0)), sv = _mm512_cvtepi16_epi32(_mm256_maskz_loadu_epi16(pm, s + i0));
            const __m512i oa = _mm512_sub_epi32(_mm512_mullo_epi32(a, cv), _mm512_mullo_epi32(b, sv));
            const __m512i ob = _mm512_add_epi32(_mm512_mullo_epi32(a, sv), _mm512_mullo_epi32(b, cv));
            int32_t *op = out + h * H + 2 * i0;
            _mm512_mask_storeu_epi32(op, lo, _mm512_permutex2var_epi32(oa, idx_il0(), ob));
            _mm512_mask_storeu_epi32(op + 16, hi, _mm512_permutex2var_epi32(oa, idx_il1(), ob));
        }
}

AVX512_FN void ternary_avx512_rope_unrotate_grad(float *grad, size_t heads, size_t H, size_t P, const int16_t *c, const int16_t *s, float inv) {
    const __m512 vinv = _mm512_set1_ps(inv);
    for (size_t h = 0; h < heads; h++)
        for (size_t i0 = 0; i0 < P; i0 += 16) {
            const size_t np = (P - i0 < 16) ? P - i0 : 16, ints = 2 * np;
            const __mmask16 pm = (__mmask16)((1u << np) - 1u);
            const __mmask16 lo = ints >= 16 ? (__mmask16)0xFFFFu : (__mmask16)((1u << ints) - 1u);
            const __mmask16 hi = ints > 16 ? (__mmask16)((1u << (ints - 16)) - 1u) : (__mmask16)0;
            float *gp = grad + h * H + 2 * i0;
            const __m512 v0 = _mm512_maskz_loadu_ps(lo, gp), v1 = _mm512_maskz_loadu_ps(hi, gp + 16);
            const __m512 ga = _mm512_permutex2var_ps(v0, idx_even(), v1), gb = _mm512_permutex2var_ps(v0, idx_odd(), v1);
            const __m512 cf = MULPS(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm256_maskz_loadu_epi16(pm, c + i0))), vinv);
            const __m512 sf = MULPS(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm256_maskz_loadu_epi16(pm, s + i0))), vinv);
            const __m512 oa = ADDPS(MULPS(ga, cf), MULPS(gb, sf));
            const __m512 ob = ADDPS(MULPS(_mm512_sub_ps(_mm512_setzero_ps(), ga), sf), MULPS(gb, cf));
            _mm512_mask_storeu_ps(gp, lo, _mm512_permutex2var_ps(oa, idx_il0(), ob));
            _mm512_mask_storeu_ps(gp + 16, hi, _mm512_permutex2var_ps(oa, idx_il1(), ob));
        }
}

AVX512_FN void ternary_avx512_shift_round_i32(const int32_t *z, size_t n, int shift, int32_t *out) {
    const __m512i half = _mm512_set1_epi32(1 << (shift - 1));
    const __m128i cnt = _mm_cvtsi32_si128(shift);
    const size_t nt = (n + 15) / 16;
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(n, t);
        _mm512_mask_storeu_epi32(out + 16 * t, lim, _mm512_sra_epi32(_mm512_add_epi32(_mm512_maskz_loadu_epi32(lim, z + 16 * t), half), cnt));
    }
}

AVX512_FN void ternary_avx512_sign_f32(const int32_t *z, size_t n, float *out, size_t stride) {
    const size_t nt = (n + 15) / 16;
    const __m512i iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(n, t);
        const __m512 r = sign_f(_mm512_maskz_loadu_epi32(lim, z + 16 * t));
        if (stride == 1) _mm512_mask_storeu_ps(out + 16 * t, lim, r);
        else _mm512_mask_i32scatter_ps(out, lim, _mm512_mullo_epi32(_mm512_add_epi32(iota, _mm512_set1_epi32((int)(16 * t))), _mm512_set1_epi32((int)stride)), r, 4);
    }
}

AVX512_FN void ternary_avx512_scale_i32_f32(const int32_t *z, size_t n, float k, float *out, size_t stride) {
    const size_t nt = (n + 15) / 16;
    const __m512 vk = _mm512_set1_ps(k);
    const __m512i iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(n, t);
        const __m512 r = MULPS(vk, _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(lim, z + 16 * t)));
        if (stride == 1) _mm512_mask_storeu_ps(out + 16 * t, lim, r);
        else _mm512_mask_i32scatter_ps(out, lim, _mm512_mullo_epi32(_mm512_add_epi32(iota, _mm512_set1_epi32((int)(16 * t))), _mm512_set1_epi32((int)stride)), r, 4);
    }
}


AVX512_FN void ternary_avx512_scatter_i32(const int32_t *src, size_t n, int32_t *dst, size_t stride) {
    const size_t nt = (n + 15) / 16;
    const __m512i iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(n, t);
        _mm512_mask_i32scatter_epi32(dst, lim, _mm512_mullo_epi32(_mm512_add_epi32(iota, _mm512_set1_epi32((int)(16 * t))), _mm512_set1_epi32((int)stride)),
                                     _mm512_maskz_loadu_epi32(lim, src + 16 * t), 4);
    }
}


AVX512_FN void ternary_avx512_smear_k(const int32_t *in, int32_t *out, int32_t *prev, size_t n, int first) {
    const size_t nt = (n + 15) / 16;
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(n, t);
        const __m512i a = _mm512_maskz_loadu_epi32(lim, in + 16 * t);
        const __m512i r = first ? a : _mm512_srai_epi32(_mm512_add_epi32(a, _mm512_maskz_loadu_epi32(lim, prev + 16 * t)), 1);
        _mm512_mask_storeu_epi32(prev + 16 * t, lim, a);
        _mm512_mask_storeu_epi32(out + 16 * t, lim, r);
    }
}

/* Sin FMA ni reasociacion: cada elemento es exactamente own + nxt con own/nxt productos por 1 o 0.5 (exactos), igual que el escalar. */
AVX512_FN void ternary_avx512_smear_grad_rows(float *g, size_t p0, size_t p1, size_t T, size_t w, const float *next_of_last) {
    const size_t nt = (w + 15) / 16;
    const __m512 half = _mm512_set1_ps(0.5f);
    for (size_t p = p0; p < p1; p++) {
        float *row = g + p * w;
        /* la fila p+1 de la ultima posicion del trozo viene de una copia (otro hilo puede estar reescribiendola) */
        const float *nrow = (p + 1 == p1 && next_of_last) ? next_of_last : row + w;
        for (size_t t = 0; t < nt; t++) {
            const __mmask16 lim = tail_lim(w, t);
            const __m512 a = _mm512_maskz_loadu_ps(lim, row + 16 * t);
            const __m512 own = p == 0 ? a : _mm512_mul_ps(a, half);
            const __m512 nxt = p + 1 < T ? _mm512_mul_ps(_mm512_maskz_loadu_ps(lim, nrow + 16 * t), half) : _mm512_setzero_ps();
            _mm512_mask_storeu_ps(row + 16 * t, lim, _mm512_add_ps(own, nxt));
        }
    }
}

/* n trits de los planos mask/sign (desde el bit 0) a floats {-1, 0, +1}. */
AVX512_FN void ternary_avx512_expand_trits(const uint64_t *mask, const uint64_t *sign, size_t n, float *out) {
    const unsigned char *mb = (const unsigned char *)mask, *sb = (const unsigned char *)sign;
    const size_t nt = (n + 15) / 16;
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(n, t);
        const uint16_t m = (uint16_t)(mb[2 * t] | (mb[2 * t + 1] << 8)), sg = (uint16_t)(sb[2 * t] | (sb[2 * t + 1] << 8));
        _mm512_mask_storeu_ps(out + 16 * t, lim, expand_pm((__mmask16)(m & (uint16_t)~sg), (__mmask16)(m & sg)));
    }
}


AVX512_INL void put_bits16(uint64_t *plane, size_t bit, uint16_t val, uint16_t lim) {
    const size_t w = bit >> 6, b = bit & 63;
    plane[w] = (plane[w] & ~((uint64_t)lim << b)) | ((uint64_t)val << b);
    if (b > 48) { const size_t sh = 64 - b; plane[w + 1] = (plane[w + 1] & ~((uint64_t)lim >> sh)) | ((uint64_t)val >> sh); }
}
/* Escribe sign(z[i]) en los bits bit0 + i de los planos (los demas bits no se tocan): igual que set_sign_i32 en bucle. */
AVX512_FN void ternary_avx512_pack_sign_i32(const int32_t *z, size_t n, uint64_t *mask, uint64_t *sign, size_t bit0) {
    const size_t nt = (n + 15) / 16;
    for (size_t t = 0; t < nt; t++) {
        const __mmask16 lim = tail_lim(n, t);
        const __m512i v = _mm512_maskz_loadu_epi32(lim, z + 16 * t);
        put_bits16(mask, bit0 + 16 * t, (uint16_t)(_mm512_cmpneq_epi32_mask(v, _mm512_setzero_si512()) & lim), (uint16_t)lim);
        put_bits16(sign, bit0 + 16 * t, (uint16_t)(_mm512_cmplt_epi32_mask(v, _mm512_setzero_si512()) & lim), (uint16_t)lim);
    }
}

/* ---- busqueda de ids en un rango [lo, hi) (marcadores especiales de ternary_targets) ---------------------------------- */
AVX512_FN size_t ternary_avx512_find_range(const int32_t *ids, size_t from, size_t n, int32_t lo, int32_t hi) {
    if (lo >= hi || from >= n) return n;
    const __m512i vlo = _mm512_set1_epi32(lo), vspan = _mm512_set1_epi32((int32_t)((uint32_t)hi - (uint32_t)lo));
    size_t i = from;
    for (; i + 64 <= n; i += 64) {                       /* 4 vectores por vuelta: una sola decision por 64 ids */
        const __mmask16 k0 = _mm512_cmplt_epu32_mask(_mm512_sub_epi32(_mm512_loadu_si512((const void *)(ids + i)), vlo), vspan);
        const __mmask16 k1 = _mm512_cmplt_epu32_mask(_mm512_sub_epi32(_mm512_loadu_si512((const void *)(ids + i + 16)), vlo), vspan);
        const __mmask16 k2 = _mm512_cmplt_epu32_mask(_mm512_sub_epi32(_mm512_loadu_si512((const void *)(ids + i + 32)), vlo), vspan);
        const __mmask16 k3 = _mm512_cmplt_epu32_mask(_mm512_sub_epi32(_mm512_loadu_si512((const void *)(ids + i + 48)), vlo), vspan);
        if ((k0 | k1 | k2 | k3) == 0) continue;
        if (k0) return i + (size_t)__builtin_ctz((unsigned)k0);
        if (k1) return i + 16 + (size_t)__builtin_ctz((unsigned)k1);
        if (k2) return i + 32 + (size_t)__builtin_ctz((unsigned)k2);
        return i + 48 + (size_t)__builtin_ctz((unsigned)k3);
    }
    for (; i + 16 <= n; i += 16) {
        const __mmask16 k = _mm512_cmplt_epu32_mask(_mm512_sub_epi32(_mm512_loadu_si512((const void *)(ids + i)), vlo), vspan);
        if (k) return i + (size_t)__builtin_ctz((unsigned)k);
    }
    if (i < n) {                                         /* cola: carga enmascarada (los carriles de mas no cuentan) */
        const __mmask16 lm = (__mmask16)((1u << (n - i)) - 1u);
        const __mmask16 k = _mm512_mask_cmplt_epu32_mask(lm, _mm512_sub_epi32(_mm512_maskz_loadu_epi32(lm, ids + i), vlo), vspan);
        if (k) return i + (size_t)__builtin_ctz((unsigned)k);
    }
    return n;
}

#else /* no x86-64: esqueletos, ternary_avx512_enabled() siempre da 0 y nada mas se llama */

int ternary_avx512_enabled(void) { return 0; }
int ternary_avx512_vnni_enabled(void) { return 0; }
void ternary_avx512_expand_i8(const uint64_t *mask, const uint64_t *sign, size_t n, int8_t *out) { (void)mask; (void)sign; (void)n; (void)out; abort(); }
void ternary_vnni_forward(const int8_t *wt, const int32_t *wsum, const int32_t *bias, size_t Opad, size_t O, size_t in, size_t P,
                          const uint8_t *q, size_t qs, const float *sx, const int32_t *zp, float scale,
                          float *out, size_t os, int32_t *raw, size_t rs) { (void)wt; (void)wsum; (void)bias; (void)Opad; (void)O; (void)in; (void)P; (void)q; (void)qs; (void)sx; (void)zp; (void)scale; (void)out; (void)os; (void)raw; (void)rs; abort(); }
size_t ternary_avx512_find_range(const int32_t *ids, size_t from, size_t n, int32_t lo, int32_t hi) { (void)ids; (void)from; (void)n; (void)lo; (void)hi; abort(); }
void ternary_avx512_gradw(float *base, size_t ld, size_t outd, size_t n, size_t S, const float *gT,
                          const uint16_t *xp, const uint16_t *xn) { (void)base; (void)ld; (void)outd; (void)n; (void)S; (void)gT; (void)xp; (void)xn; abort(); }
void ternary_avx512_gradx(float *const *gx, size_t S, size_t outd, size_t n, const float *gs,
                          const uint16_t *wp, const uint16_t *wn) { (void)gx; (void)S; (void)outd; (void)n; (void)gs; (void)wp; (void)wn; abort(); }
void ternary_avx512_gradx_scaled(float *const *gx, size_t S, size_t outd, size_t n, const float *g, float scale,
                                 const uint16_t *wp, const uint16_t *wn) { (void)gx; (void)S; (void)outd; (void)n; (void)g; (void)scale; (void)wp; (void)wn; abort(); }
void ternary_avx512_scale_terms(float *terms, size_t ts, const float *g, size_t gs, const float *sx, const int32_t *z, size_t zs, size_t P, size_t o0, size_t o1) { (void)terms; (void)ts; (void)g; (void)gs; (void)sx; (void)z; (void)zs; (void)P; (void)o0; (void)o1; abort(); }
void ternary_avx512_outer_acc_rows(float *gw, size_t n, size_t o0, size_t o1, size_t P, const float *g, size_t gs,
                                   const float *x, size_t xs) { (void)gw; (void)n; (void)o0; (void)o1; (void)P; (void)g; (void)gs; (void)x; (void)xs; abort(); }
float ternary_avx512_sum_abs(const float *x, size_t n) { (void)x; (void)n; abort(); }
void ternary_avx512_pack_thr(const float *x, size_t n, float thr, uint64_t *mask, uint64_t *sign) { (void)x; (void)n; (void)thr; (void)mask; (void)sign; abort(); }
void ternary_avx512_rmsnorm_forward(size_t rows, size_t D, float eps, float scale, const uint16_t *gp,
                                    const uint16_t *gm, const float *x, float *out) { (void)rows; (void)D; (void)eps; (void)scale; (void)gp; (void)gm; (void)x; (void)out; abort(); }
void ternary_avx512_rmsnorm_backward(size_t rows, size_t D, float eps, float scale, const uint16_t *gp,
                                     const uint16_t *gm, const float *x, const float *go, float *grad_gain_out,
                                     float *grad_x_out, double *gscale) { (void)rows; (void)D; (void)eps; (void)scale; (void)gp; (void)gm; (void)x; (void)go; (void)grad_gain_out; (void)grad_x_out; (void)gscale; abort(); }
void ternary_avx512_bwd1_row(size_t in, const uint16_t *xp, const uint16_t *xn, const uint16_t *wp, const uint16_t *wn,
                             float g, float gs, float *grad_row, float *gx, int accumulate) { (void)in; (void)xp; (void)xn; (void)wp; (void)wn; (void)g; (void)gs; (void)grad_row; (void)gx; (void)accumulate; abort(); }
void ternary_avx512_unpack_row(const uint64_t *mask, const uint64_t *sign, size_t nbits_total, size_t bit0, size_t n,
                               float scale, float *out, int32_t *raw) { (void)mask; (void)sign; (void)nbits_total; (void)bit0; (void)n; (void)scale; (void)out; (void)raw; abort(); }
void ternary_avx512_scores_i32(const int32_t *a, const int32_t *kb, size_t kstride, size_t H, size_t count, int shift,
                               int32_t lim, int32_t *out) { (void)a; (void)kb; (void)kstride; (void)H; (void)count; (void)shift; (void)lim; (void)out; abort(); }
void ternary_avx512_qk_grad(float *gq, const int32_t *qv, const int32_t *kb, float *gk, size_t kstride, size_t H,
                            size_t count, const float *ds) { (void)gq; (void)qv; (void)kb; (void)gk; (void)kstride; (void)H; (void)count; (void)ds; abort(); }
void ternary_avx512_outer_add(float *dst, size_t stride, size_t H, size_t count, const float *p, const float *ga) { (void)dst; (void)stride; (void)H; (void)count; (void)p; (void)ga; abort(); }
void ternary_avx512_attn_scores(const int32_t *const *q, size_t G, const int32_t *kT, size_t ld, size_t H, size_t n,
                                int shift, int32_t lim, int32_t *sc, size_t scs, int32_t *m_out) { (void)q; (void)G; (void)kT; (void)ld; (void)H; (void)n; (void)shift; (void)lim; (void)sc; (void)scs; (void)m_out; abort(); }
void ternary_avx512_attn_pv(size_t G, const int32_t *sc, size_t scs, const int32_t *m, const float *tab, uint32_t K0,
                            size_t n, const float *v, size_t vs, size_t H, float *e, size_t es, float *out, float *inv_out) { (void)G; (void)sc; (void)scs; (void)m; (void)tab; (void)K0; (void)n; (void)v; (void)vs; (void)H; (void)e; (void)es; (void)out; (void)inv_out; abort(); }
void ternary_avx512_attn_bwd_pds(size_t G, size_t n, size_t H, const int32_t *sc, size_t scs, const int32_t *m, const float *inv, const float *tab, uint32_t K0, const float *ga, const float *o, const float *vT, size_t ld, float it, float gam, const int32_t *qr, float *p, float *ds, size_t es, float *qf) { (void)G; (void)n; (void)H; (void)sc; (void)scs; (void)m; (void)inv; (void)tab; (void)K0; (void)ga; (void)o; (void)vT; (void)ld; (void)it; (void)gam; (void)qr; (void)p; (void)ds; (void)es; (void)qf; abort(); }
size_t ternary_avx512_attn_bwd_rows_cap(size_t G, size_t H) { (void)G; (void)H; return 0; }
void ternary_avx512_attn_bwd_rows(size_t R, size_t G, size_t H, const size_t *nr, const float *ga, size_t gas, const float *qf, const float *p, const float *ds, size_t es, size_t rs, const int32_t *kr, size_t ks, float *gq, size_t gqs, float *gk, float *gv, size_t gs, int doA) { (void)R; (void)G; (void)H; (void)nr; (void)ga; (void)gas; (void)qf; (void)p; (void)ds; (void)es; (void)rs; (void)kr; (void)ks; (void)gq; (void)gqs; (void)gk; (void)gv; (void)gs; (void)doA; abort(); }
void ternary_avx512_attn_bwd(size_t G, size_t n, size_t H, const int32_t *sc, size_t scs, const int32_t *m,
                             const float *inv, const float *tab, uint32_t K0, const float *ga, const float *o,
                             const float *vT, size_t ld, float it, float gam, const int32_t *qr, const int32_t *kr,
                             size_t ks, float *gq, float *gk, float *gv, size_t gs, float *p, float *ds, size_t es,
                             float *qf) { (void)G; (void)n; (void)H; (void)sc; (void)scs; (void)m; (void)inv; (void)tab; (void)K0; (void)ga; (void)o; (void)vT; (void)ld; (void)it; (void)gam; (void)qr; (void)kr; (void)ks; (void)gq; (void)gk; (void)gv; (void)gs; (void)p; (void)ds; (void)es; (void)qf; abort(); }
void ternary_avx512_attn_pv2(size_t G, const int32_t *sc, size_t scs, const int32_t *m, const float *tab, uint32_t K0,
                             size_t nA, size_t nB, size_t spA, const float *vA, const float *vB, size_t vs, size_t H,
                             float *e, size_t es, float *out, float *inv_out) { (void)G; (void)sc; (void)scs; (void)m; (void)tab; (void)K0; (void)nA; (void)nB; (void)spA; (void)vA; (void)vB; (void)vs; (void)H; (void)e; (void)es; (void)out; (void)inv_out; abort(); }
void ternary_avx512_attn_bwd2(size_t G, size_t nA, size_t nB, size_t spA, size_t H, const int32_t *sc, size_t scs,
                              const int32_t *m, const float *inv, const float *tab, uint32_t K0, const float *ga,
                              const float *o, const float *vTA, const float *vTB, size_t ld, float it, float gam,
                              const int32_t *qr, const int32_t *krA, const int32_t *krB, size_t ks, float *gq,
                              float *gkA, float *gkB, float *gvA, float *gvB, size_t gs, float *p, float *ds,
                              size_t es, float *qf) { (void)G; (void)nA; (void)nB; (void)spA; (void)H; (void)sc; (void)scs; (void)m; (void)inv; (void)tab; (void)K0; (void)ga; (void)o; (void)vTA; (void)vTB; (void)ld; (void)it; (void)gam; (void)qr; (void)krA; (void)krB; (void)ks; (void)gq; (void)gkA; (void)gkB; (void)gvA; (void)gvB; (void)gs; (void)p; (void)ds; (void)es; (void)qf; abort(); }
int32_t ternary_avx512_dot_u8(const uint64_t *wm, const uint64_t *ws, size_t n, const uint8_t *q, int32_t zp) { (void)wm; (void)ws; (void)n; (void)q; (void)zp; abort(); }
void ternary_avx512_u8_to_f32(const uint8_t *q, size_t n, float sx, int32_t zp, float *out) { (void)q; (void)n; (void)sx; (void)zp; (void)out; abort(); }
void ternary_avx512_outer_acc(float *gw, size_t n, float g, const float *xf) { (void)gw; (void)n; (void)g; (void)xf; abort(); }
void ternary_avx512_gradx_row(float *gx, size_t n, const uint64_t *wm, const uint64_t *ws, float gs) { (void)gx; (void)n; (void)wm; (void)ws; (void)gs; abort(); }
void ternary_avx512_swiglu_prod_z(int mode, size_t H, const int32_t *zg, const int32_t *zu, float su, const float *silu_tab, int32_t D, float *h) { (void)mode; (void)H; (void)zg; (void)zu; (void)su; (void)silu_tab; (void)D; (void)h; abort(); }
void ternary_avx512_swiglu_bwd_z(int mode, size_t H, const int32_t *zg, const int32_t *zu, const float *gt, float su, const float *silu_tab, const float *dsilu_tab, int32_t D, float *gu, float *ga) { (void)mode; (void)H; (void)zg; (void)zu; (void)gt; (void)su; (void)silu_tab; (void)dsilu_tab; (void)D; (void)gu; (void)ga; abort(); }
void ternary_avx512_swiglu_fwd_u8(int mode, size_t H, const float *fa, const float *fu, const int32_t *zg, float *h) { (void)mode; (void)H; (void)fa; (void)fu; (void)zg; (void)h; abort(); }
void ternary_avx512_swiglu_bwd_u8(int mode, size_t H, const int32_t *zg, const int32_t *zu, const float *gt, float kg, float ku, float *gu, float *ga) { (void)mode; (void)H; (void)zg; (void)zu; (void)gt; (void)kg; (void)ku; (void)gu; (void)ga; abort(); }
void ternary_avx512_xent_row(const float *l, size_t V, size_t t, size_t n, float *g, int *ok_out, double *loss_out) { (void)l; (void)V; (void)t; (void)n; (void)g; (void)ok_out; (void)loss_out; abort(); }
double ternary_avx512_dot_gz(const float *g, const int32_t *z, size_t n, int use_sx, double sx) { (void)g; (void)z; (void)n; (void)use_sx; (void)sx; abort(); }
void ternary_avx512_scale_f32(float *x, size_t n, float s) { (void)x; (void)n; (void)s; abort(); }
void ternary_avx512_elemnorm_row(uint8_t *cr, float *rmax, float *g, float *v, size_t W, float beta2, float omb, float bc, float eps) { (void)cr; (void)rmax; (void)g; (void)v; (void)W; (void)beta2; (void)omb; (void)bc; (void)eps; abort(); }
void ternary_avx512_r2_row(uint64_t *mask, uint64_t *sign, int8_t *cnt, size_t base, const float *g, size_t n, float beta, float inc, int ti, int tin, int tout, int ri, int pre) { (void)pre; (void)mask; (void)sign; (void)cnt; (void)base; (void)g; (void)n; (void)beta; (void)inc; (void)ti; (void)tin; (void)tout; (void)ri; abort(); }
void ternary_avx512_u8_quantize_row(const float *x, size_t n, uint8_t *q, float *scale_out, int32_t *zp_out) { (void)x; (void)n; (void)q; (void)scale_out; (void)zp_out; abort(); }
void ternary_avx512_add_f32(float *out, const float *a, const float *b, size_t n) { (void)out; (void)a; (void)b; (void)n; abort(); }
void ternary_avx512_acc_add2(float *dst, const float *a, const float *b, size_t n) { (void)dst; (void)a; (void)b; (void)n; abort(); }
void ternary_avx512_softmax_fwd(const int32_t *scores, size_t n_valid, const float *table, size_t table_len, float *out) { (void)scores; (void)n_valid; (void)table; (void)table_len; (void)out; abort(); }
void ternary_avx512_softmax_bwd(const float *y, const float *g, size_t n, float inv_temp, float *out_ds) { (void)y; (void)g; (void)n; (void)inv_temp; (void)out_ds; abort(); }
void ternary_avx512_rope_rotate_int(const int32_t *z, size_t heads, size_t H, size_t P, const int16_t *c, const int16_t *s, int32_t *out) { (void)z; (void)heads; (void)H; (void)P; (void)c; (void)s; (void)out; abort(); }
void ternary_avx512_rope_unrotate_grad(float *grad, size_t heads, size_t H, size_t P, const int16_t *c, const int16_t *s, float inv) { (void)grad; (void)heads; (void)H; (void)P; (void)c; (void)s; (void)inv; abort(); }
void ternary_avx512_shift_round_i32(const int32_t *z, size_t n, int shift, int32_t *out) { (void)z; (void)n; (void)shift; (void)out; abort(); }
void ternary_avx512_sign_f32(const int32_t *z, size_t n, float *out, size_t stride) { (void)z; (void)n; (void)out; (void)stride; abort(); }
void ternary_avx512_scale_i32_f32(const int32_t *z, size_t n, float k, float *out, size_t stride) { (void)z; (void)n; (void)k; (void)out; (void)stride; abort(); }
void ternary_avx512_scatter_i32(const int32_t *src, size_t n, int32_t *dst, size_t stride) { (void)src; (void)n; (void)dst; (void)stride; abort(); }
void ternary_avx512_smear_k(const int32_t *in, int32_t *out, int32_t *prev, size_t n, int first) { (void)in; (void)out; (void)prev; (void)n; (void)first; abort(); }
void ternary_avx512_smear_grad_rows(float *g, size_t p0, size_t p1, size_t T, size_t w, const float *next_of_last) { (void)g; (void)p0; (void)p1; (void)T; (void)w; (void)next_of_last; abort(); }
void ternary_avx512_expand_trits(const uint64_t *mask, const uint64_t *sign, size_t n, float *out) { (void)mask; (void)sign; (void)n; (void)out; abort(); }
void ternary_avx512_pack_sign_i32(const int32_t *z, size_t n, uint64_t *mask, uint64_t *sign, size_t bit0) { (void)z; (void)n; (void)mask; (void)sign; (void)bit0; abort(); }
#endif
