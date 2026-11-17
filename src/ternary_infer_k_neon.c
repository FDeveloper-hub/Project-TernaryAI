/* Nucleo NEON de 128 bits (ARMv7 / ARMv8 en 32 bits y 64 bits): producto ternario x uint8 y puntajes de atencion, todo entero exacto,
 * asi que da los mismos z, puntajes y tokens que el nucleo escalar. Las partes en float (cuantizador, relu2, softmax, pv) se dejan en NULL
 * (el motor usa las versiones escalares): NEON aplasta los denormales a cero y el escalar en VFP no, y eso cambiaria bits.
 *
 * Pesos (TERNARYAI_INFER_LAYOUT_NEON): fila a fila, cp/4 bytes por fila, en grupos de 64 columnas = 16 bytes: el byte c del grupo g guarda
 * en su campo j (bits 2j..2j+1) la columna 64g + 16j + c. Asi cada campo, extraido con (b >> 2j) & 3, es un vector de 16 columnas contiguas
 * que se multiplica por 16 activaciones u8 contiguas (vmlal_u8: u8 x u8 -> u16, acumulando). Si cp no es multiple de 64, el grupo final
 * tiene n = cp - 64g columnas (16, 32 o 48), B = n/4 bytes, y el campo j guarda la columna 64g + j*B + c (se procesa con un bucle simple).
 * z = sum(codigo*q) - sum(q) - zp*S. Cada carril u16 recibe 8 productos (<= 510) por grupo: se vuelca a u32 cada 16 grupos (16*8*510 < 65536). */
#include "ternary_infer_internal.h"

#if defined(__ARM_NEON) && (defined(__arm__) || defined(__aarch64__))
#include <arm_neon.h>
#include <math.h>
#include <string.h>

#define FLUSH_GROUPS 16

static inline uint32_t hsum_u32(uint32x4_t a) {
    const uint32x2_t t = vadd_u32(vget_low_u32(a), vget_high_u32(a));
    return vget_lane_u32(vpadd_u32(t, t), 0);
}

static inline uint32_t sum_u8(const uint8_t *q, int n) {          /* n multiplo de 16 */
    uint32x4_t a = vdupq_n_u32(0);
    for (int i = 0; i < n; i += 16) a = vpadalq_u16(a, vpaddlq_u8(vld1q_u8(q + i)));
    return hsum_u32(a);
}

/* campo j del grupo final (B bytes) de la fila r: codigos de las columnas 64g + j*B + c */
static inline uint32_t tail_dot(const uint8_t *row, const uint8_t *q, int g, int B) {
    uint32_t s = 0;
    for (int j = 0; j < 4; j++)
        for (int c = 0; c < B; c++) s += (uint32_t)((row[16 * g + c] >> (2 * j)) & 3) * (uint32_t)q[64 * g + j * B + c];
    return s;
}

#define EXPAND4(W, E)                                                                                                        \
    do { (E)[0] = vandq_u8((W), m3); (E)[1] = vandq_u8(vshrq_n_u8((W), 2), m3); (E)[2] = vandq_u8(vshrq_n_u8((W), 4), m3);    \
         (E)[3] = vshrq_n_u8((W), 6); } while (0)

#define MLA8(ACC, E, Q)                                                                                                      \
    do { for (int j_ = 0; j_ < 4; j_++) { (ACC) = vmlal_u8((ACC), vget_low_u8((E)[j_]), vget_low_u8((Q)[j_]));                \
                                           (ACC) = vmlal_u8((ACC), vget_high_u8((E)[j_]), vget_high_u8((Q)[j_])); } } while (0)

/* NR filas a la vez (NR constante: el compilador desenrolla) */
static inline __attribute__((always_inline)) void nk_rows(const TernaryInferMat *m, const uint8_t *q, uint32_t sq, int32_t zp, int32_t *z, int o, const int NR) {
    const int Q = m->cp >> 2, nfull = m->cp >> 6, tailc = m->cp & 63;
    const uint8x16_t m3 = vdupq_n_u8(3);
    const uint8_t *r[4];
    uint32x4_t a32[4];
    for (int k = 0; k < NR; k++) { r[k] = m->w + (size_t)(o + k) * (size_t)Q; a32[k] = vdupq_n_u32(0); }
    for (int g0 = 0; g0 < nfull; g0 += FLUSH_GROUPS) {
        const int g1 = (g0 + FLUSH_GROUPS < nfull) ? g0 + FLUSH_GROUPS : nfull;
        uint16x8_t a16[4];
        for (int k = 0; k < NR; k++) a16[k] = vdupq_n_u16(0);
        for (int g = g0; g < g1; g++) {
            uint8x16_t qv[4];
            for (int j = 0; j < 4; j++) qv[j] = vld1q_u8(q + 64 * g + 16 * j);
            for (int k = 0; k < NR; k++) {
                uint8x16_t e[4];
                const uint8x16_t w = vld1q_u8(r[k] + 16 * g);
                EXPAND4(w, e);
                MLA8(a16[k], e, qv);
            }
        }
        for (int k = 0; k < NR; k++) a32[k] = vpadalq_u16(a32[k], a16[k]);
    }
    for (int k = 0; k < NR; k++) {
        uint32_t s = hsum_u32(a32[k]);
        if (tailc) s += tail_dot(r[k], q, nfull, tailc >> 2);
        z[o + k] = (int32_t)(s - sq) - zp * m->S[o + k];
    }
}

static void nk_matvec(const TernaryInferMat *m, const uint8_t *q, int32_t zp, int32_t *z) {
    const uint32_t sq = sum_u8(q, m->cp);
    int o = 0;
    for (; o + 4 <= m->rows; o += 4) nk_rows(m, q, sq, zp, z, o, 4);
    for (; o < m->rows; o++) nk_rows(m, q, sq, zp, z, o, 1);
}

/* varios carriles: cada fila de pesos se desempaqueta una vez y se usa para NB carriles */
static inline __attribute__((always_inline)) void nk_mm_nb(const TernaryInferMat *m, const uint8_t *const *q, const uint32_t *sq, const int32_t *zp,
                                                          int32_t *const *z, const int NB) {
    const int Q = m->cp >> 2, nfull = m->cp >> 6, tailc = m->cp & 63;
    const uint8x16_t m3 = vdupq_n_u8(3);
    for (int o = 0; o < m->rows; o++) {
        const uint8_t *r = m->w + (size_t)o * (size_t)Q;
        uint32x4_t a32[4];
        for (int b = 0; b < NB; b++) a32[b] = vdupq_n_u32(0);
        for (int g0 = 0; g0 < nfull; g0 += FLUSH_GROUPS) {
            const int g1 = (g0 + FLUSH_GROUPS < nfull) ? g0 + FLUSH_GROUPS : nfull;
            uint16x8_t a16[4];
            for (int b = 0; b < NB; b++) a16[b] = vdupq_n_u16(0);
            for (int g = g0; g < g1; g++) {
                uint8x16_t e[4];
                const uint8x16_t w = vld1q_u8(r + 16 * g);
                EXPAND4(w, e);
                for (int b = 0; b < NB; b++) {
                    uint8x16_t qv[4];
                    for (int j = 0; j < 4; j++) qv[j] = vld1q_u8(q[b] + 64 * g + 16 * j);
                    MLA8(a16[b], e, qv);
                }
            }
            for (int b = 0; b < NB; b++) a32[b] = vpadalq_u16(a32[b], a16[b]);
        }
        for (int b = 0; b < NB; b++) {
            uint32_t s = hsum_u32(a32[b]);
            if (tailc) s += tail_dot(r, q[b], nfull, tailc >> 2);
            z[b][o] = (int32_t)(s - sq[b]) - zp[b] * m->S[o];
        }
    }
}

static void nk_matmul(const TernaryInferMat *m, const uint8_t *const *q, const int32_t *zp, int32_t *const *z, int B) {
    uint32_t sq[TERNARYAI_INFER_MAX_LANES];
    for (int b = 0; b < B; b++) sq[b] = sum_u8(q[b], m->cp);
    int b = 0;
    for (; b + 4 <= B; b += 4) nk_mm_nb(m, q + b, sq + b, zp + b, z + b, 4);
    switch (B - b) {
    case 3: nk_mm_nb(m, q + b, sq + b, zp + b, z + b, 3); break;
    case 2: nk_mm_nb(m, q + b, sq + b, zp + b, z + b, 2); break;
    case 1: nk_mm_nb(m, q + b, sq + b, zp + b, z + b, 1); break;
    default: break;
    }
}

/* primer maximo estricto (el de menor indice en empate) */
static int nk_argmax(const int32_t *z, int n) {
    int bi = 0; int32_t bv = z[0];
    int i = 1;
    for (; i + 4 <= n; i += 4) {
        const int32x4_t v = vld1q_s32(z + i);
        const int32x2_t m2 = vpmax_s32(vget_low_s32(v), vget_high_s32(v));
        if (vget_lane_s32(vpmax_s32(m2, m2), 0) > bv) {
            for (int k = 0; k < 4; k++) if (z[i + k] > bv) { bv = z[i + k]; bi = i + k; }
        }
    }
    for (; i < n; i++) if (z[i] > bv) { bv = z[i]; bi = i; }
    return bi;
}

/* puntajes: (q . k) >> (16+shift) recortado a +-lim. kT: hd filas de paso tp (claves en carriles). int64 exacto (vmlal_s32). Dos cabezas a la vez. */
static void nk_scores(const int32_t *kT, int tp, int hd, int start, int cnt, const int32_t *const *q, int G,
                      int shift, int32_t lim, int32_t **sc, int64_t *scr) {
    (void)scr;
    const int64x2_t vsh = vdupq_n_s64(-(int64_t)(16 + shift));
    const int32x4_t vlim = vdupq_n_s32(lim), vnlim = vdupq_n_s32(-lim);
    for (int b0 = 0; b0 < G; b0 += 2) {
        const int two = (G - b0 >= 2);
        const int32_t *q0 = q[b0], *q1 = two ? q[b0 + 1] : q[b0];
        for (int t = 0; t < cnt; t += 4) {
            const int32_t *kp = kT + start + t;
            const int full = (cnt - t >= 4);
            int32_t tmp[4] = { 0, 0, 0, 0 };
            int64x2_t a0l = vdupq_n_s64(0), a0h = a0l, a1l = a0l, a1h = a0l;
            for (int d = 0; d < hd; d++) {
                int32x4_t kv;
                if (full) kv = vld1q_s32(kp + (size_t)d * tp);
                else { for (int k = 0; k < cnt - t; k++) tmp[k] = kp[(size_t)d * tp + k]; kv = vld1q_s32(tmp); }
                a0l = vmlal_n_s32(a0l, vget_low_s32(kv), q0[d]); a0h = vmlal_n_s32(a0h, vget_high_s32(kv), q0[d]);
                if (two) { a1l = vmlal_n_s32(a1l, vget_low_s32(kv), q1[d]); a1h = vmlal_n_s32(a1h, vget_high_s32(kv), q1[d]); }
            }
            for (int bb = 0; bb < 1 + two; bb++) {
                const int64x2_t l = bb ? a1l : a0l, h = bb ? a1h : a0h;
                /* desplazamiento aritmetico (vshl con cuenta negativa) y estrechado con saturacion: igual que recortar a int32 y luego a +-lim */
                int32x4_t v = vcombine_s32(vqmovn_s64(vshlq_s64(l, vsh)), vqmovn_s64(vshlq_s64(h, vsh)));
                v = vminq_s32(vmaxq_s32(v, vnlim), vlim);
                int32_t *dst = sc[b0 + bb] + t;
                if (full) vst1q_s32(dst, v);
                else { int32_t o4[4]; vst1q_s32(o4, v); for (int k = 0; k < cnt - t; k++) dst[k] = o4[k]; }
            }
        }
    }
}

const TernaryInferKernels ternaryai_infer_k_neon = { "neon", nk_matvec, nk_argmax, nk_scores, NULL, NULL, NULL, NULL, NULL, NULL, nk_matmul, TERNARYAI_INFER_LAYOUT_NEON, NULL };


/* ------------------------------------------------------------------ partes float del motor (mismos bits que el escalar mientras no haya denormales:
 * NEON los aplasta a cero; aceptado, y en el escalar medido no aparece ninguno). Sin FMA: multiplicacion y suma por separado. */
#ifndef TERNARY_NEON_INFER_SOFTMAX
#define TERNARY_NEON_INFER_SOFTMAX 1
#endif
#ifndef TERNARY_NEON_INFER_PV
#define TERNARY_NEON_INFER_PV 1
#endif
#ifndef TERNARY_NEON_INFER_RELU2
#define TERNARY_NEON_INFER_RELU2 1
#endif
#ifndef TERNARY_NEON_INFER_QUANT
#define TERNARY_NEON_INFER_QUANT 1
#endif

/* ee[j] = tab[min(mmax - sc[j], K0)]; suma en el orden canonico de 16 carriles (carril l acumula j = l mod 16 en orden) */
static float nk_softmax(const int32_t *sc, int cnt, const float *tab, uint32_t K0, float *ee) {
    int32x4_t vm = vdupq_n_s32(INT32_MIN);
    int j = 0;
    for (; j + 4 <= cnt; j += 4) vm = vmaxq_s32(vm, vld1q_s32(sc + j));
    int32x2_t m2 = vpmax_s32(vget_low_s32(vm), vget_high_s32(vm));
    int32_t mm = vget_lane_s32(vpmax_s32(m2, m2), 0);
    for (; j < cnt; j++) if (sc[j] > mm) mm = sc[j];
    const int32x4_t vmm = vdupq_n_s32(mm);
    const uint32x4_t vk0 = vdupq_n_u32(K0);
    float32x4_t p0 = vdupq_n_f32(0.0f), p1 = p0, p2 = p0, p3 = p0;
    j = 0;
    for (; j + 16 <= cnt; j += 16) {
        for (int u = 0; u < 4; u++) {
            uint32_t k[4];
            vst1q_u32(k, vminq_u32(vreinterpretq_u32_s32(vsubq_s32(vmm, vld1q_s32(sc + j + 4 * u))), vk0));
            ee[j + 4 * u + 0] = tab[k[0]]; ee[j + 4 * u + 1] = tab[k[1]]; ee[j + 4 * u + 2] = tab[k[2]]; ee[j + 4 * u + 3] = tab[k[3]];
        }
        p0 = vaddq_f32(p0, vld1q_f32(ee + j)); p1 = vaddq_f32(p1, vld1q_f32(ee + j + 4));
        p2 = vaddq_f32(p2, vld1q_f32(ee + j + 8)); p3 = vaddq_f32(p3, vld1q_f32(ee + j + 12));
    }
    float p[16];
    vst1q_f32(p, p0); vst1q_f32(p + 4, p1); vst1q_f32(p + 8, p2); vst1q_f32(p + 12, p3);
    for (; j < cnt; j++) {                    /* cola: igual que el escalar (carril j & 15) */
        const uint32_t k = (uint32_t)(mm - sc[j]);
        const float e = tab[k < K0 ? k : K0];
        ee[j] = e; p[j & 15] += e;
    }
    float t8[8], t4[4];
    for (int l = 0; l < 8; l++) t8[l] = p[l] + p[l + 8];
    for (int l = 0; l < 4; l++) t4[l] = t8[l] + t8[l + 4];
    const float a = t4[0] + t4[2], b = t4[1] + t4[3];
    return a + b;
}

/* acc[c][d] += ee[c][j] * ((float)v[c][j*hd+d] * vs[j]), j en orden; NB cadenas a la vez, 8 columnas por bloque */
#define NK_PV(NB)                                                                                                                  \
    do {                                                                                                                           \
        int d0 = 0;                                                                                                                \
        for (; d0 + 8 <= hd; d0 += 8) {                                                                                            \
            float32x4_t al[NB], ah[NB];                                                                                            \
            for (int c = 0; c < (NB); c++) { al[c] = vld1q_f32(acc[c0 + c] + d0); ah[c] = vld1q_f32(acc[c0 + c] + d0 + 4); }       \
            for (int j = 0; j < cnt; j++) {                                                                                        \
                const float sj = vs[j];                                                                                            \
                for (int c = 0; c < (NB); c++) {                                                                                   \
                    const int16x8_t w = vmovl_s8(vld1_s8(v[c0 + c] + (size_t)j * hd + d0));                                        \
                    const float32x4_t lo = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w))), sj);                             \
                    const float32x4_t hi = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(w))), sj);                            \
                    const float e = ee[c0 + c][j];                                                                                 \
                    al[c] = vaddq_f32(al[c], vmulq_n_f32(lo, e)); ah[c] = vaddq_f32(ah[c], vmulq_n_f32(hi, e));                    \
                }                                                                                                                  \
            }                                                                                                                      \
            for (int c = 0; c < (NB); c++) { vst1q_f32(acc[c0 + c] + d0, al[c]); vst1q_f32(acc[c0 + c] + d0 + 4, ah[c]); }         \
        }                                                                                                                          \
        for (; d0 < hd; d0++)                                                                                                      \
            for (int c = 0; c < (NB); c++) {                                                                                       \
                float a = acc[c0 + c][d0];                                                                                         \
                for (int j = 0; j < cnt; j++) { const float vv = (float)v[c0 + c][(size_t)j * hd + d0] * vs[j]; a += ee[c0 + c][j] * vv; } \
                acc[c0 + c][d0] = a;                                                                                               \
            }                                                                                                                      \
    } while (0)
static void nk_pv_multi(int nc, const float *const *ee, int cnt, const int8_t *const *v, const float *vs, int hd, float *const *acc) {
    int c0 = 0;
    for (; c0 + 4 <= nc; c0 += 4) NK_PV(4);
    switch (nc - c0) {
    case 3: NK_PV(3); break;
    case 2: NK_PV(2); break;
    case 1: NK_PV(1); break;
    default: break;
    }
}

/* h[j] = a > 0 ? a*a : 0, a = k*(float)z[j] */
static void nk_relu2(const int32_t *z, int n, float k, float *h) {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    int j = 0;
    for (; j + 4 <= n; j += 4) {
        const float32x4_t a = vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(z + j)), k);
        const float32x4_t r = vbslq_f32(vcgtq_f32(a, zero), a, zero);
        vst1q_f32(h + j, vmulq_f32(r, r));
    }
    for (; j < n; j++) { const float a = k * (float)z[j]; const float r = a > 0.0f ? a : 0.0f; h[j] = r * r; }
}

/* cuantizador u8 con punto cero: mismas operaciones que quant_scalar (minimo/maximo con la misma semantica de comparacion) */
static void nk_quant(const float *x, int n, uint8_t *q, float *scale, int32_t *zp) {
    float32x4_t lo0 = vdupq_n_f32(0.0f), hi0 = lo0, lo1 = lo0, hi1 = lo0, lo2 = lo0, hi2 = lo0, lo3 = lo0, hi3 = lo0;
    int i = 0;
#define MM(LO, HI, V) do { const float32x4_t v_ = (V); LO = vbslq_f32(vcltq_f32(v_, LO), v_, LO); HI = vbslq_f32(vcgtq_f32(v_, HI), v_, HI); } while (0)
    for (; i + 16 <= n; i += 16) {
        MM(lo0, hi0, vld1q_f32(x + i)); MM(lo1, hi1, vld1q_f32(x + i + 4)); MM(lo2, hi2, vld1q_f32(x + i + 8)); MM(lo3, hi3, vld1q_f32(x + i + 12));
    }
#undef MM
    float lo16[16], hi16[16];
    vst1q_f32(lo16, lo0); vst1q_f32(lo16 + 4, lo1); vst1q_f32(lo16 + 8, lo2); vst1q_f32(lo16 + 12, lo3);
    vst1q_f32(hi16, hi0); vst1q_f32(hi16 + 4, hi1); vst1q_f32(hi16 + 8, hi2); vst1q_f32(hi16 + 12, hi3);
    float lo = 0.0f, hi = 0.0f;
    for (int l = 0; l < 16; l++) { lo = lo16[l] < lo ? lo16[l] : lo; hi = hi16[l] > hi ? hi16[l] : hi; }
    for (; i < n; i++) { const float v = x[i]; lo = v < lo ? v : lo; hi = v > hi ? v : hi; }
    const float sc = (hi - lo) / 255.0f;
    if (!(sc > 0.0f) || !isfinite(sc)) { *scale = 1.0f; *zp = 0; memset(q, 0, (size_t)n); return; }
    int zz = (int)rintf(-lo / sc);
    if (zz < 0) zz = 0;
    if (zz > 255) zz = 255;
    *scale = sc; *zp = zz;
    const float inv = 1.0f / sc, zf = (float)zz;
    const float32x4_t vinv = vdupq_n_f32(inv), vmag = vdupq_n_f32(12582912.0f), vzf = vdupq_n_f32(zf), v0 = vdupq_n_f32(0.0f), v255 = vdupq_n_f32(255.0f);
    int k = 0;
    for (; k + 4 <= n; k += 4) {
        float32x4_t t = vmulq_f32(vld1q_f32(x + k), vinv);
        t = vsubq_f32(vaddq_f32(t, vmag), vmag);
        float32x4_t qf = vaddq_f32(t, vzf);
        qf = vbslq_f32(vcltq_f32(qf, v0), v0, qf);
        qf = vbslq_f32(vcgtq_f32(qf, v255), v255, qf);
        const uint16x4_t h = vmovn_u32(vcvtq_u32_f32(qf));
        const uint8x8_t b = vmovn_u16(vcombine_u16(h, h));
        vst1_lane_u32((uint32_t *)(void *)(q + k), vreinterpret_u32_u8(b), 0);
    }
    for (; k < n; k++) {
        float t = x[k] * inv;
        t = (t + 12582912.0f) - 12582912.0f;
        float qf = t + zf;
        qf = qf < 0.0f ? 0.0f : qf;
        qf = qf > 255.0f ? 255.0f : qf;
        q[k] = (uint8_t)qf;
    }
}

/* puntajes con claves int16 planas: bloques de 8 posiciones en registros (int32 exacto con vmlal_s16) recorriendo todas las dimensiones; hasta 4 cabezas
 * a la vez comparten la carga de las claves. Igual que scores16_planar del motor (el motor ya garantiza que el acumulador int32 no se desborda). */
#ifndef TERNARY_NEON_INFER_SCORES16
#define TERNARY_NEON_INFER_SCORES16 1
#endif
#define NK_S16(NB)                                                                                                                  \
    do {                                                                                                                            \
        int j = 0;                                                                                                                  \
        for (; j + 8 <= cnt; j += 8) {                                                                                              \
            int32x4_t lo[NB], hi[NB];                                                                                               \
            for (int c = 0; c < (NB); c++) { lo[c] = vdupq_n_s32(0); hi[c] = lo[c]; }                                               \
            const int16_t *kp = k16 + start + j;                                                                                    \
            for (int d = 0; d < hd; d++) {                                                                                          \
                const int16x8_t kv = vld1q_s16(kp + (size_t)d * tp);                                                                \
                const int16x4_t kl = vget_low_s16(kv), kh = vget_high_s16(kv);                                                      \
                for (int c = 0; c < (NB); c++) {                                                                                    \
                    const int16_t qd = (int16_t)q[b0 + c][d];                                                                       \
                    lo[c] = vmlal_n_s16(lo[c], kl, qd); hi[c] = vmlal_n_s16(hi[c], kh, qd);                                         \
                }                                                                                                                   \
            }                                                                                                                       \
            for (int c = 0; c < (NB); c++) {                                                                                        \
                const int32x4_t a = vminq_s32(vmaxq_s32(vshlq_s32(lo[c], vsh), vnlim), vlim), b = vminq_s32(vmaxq_s32(vshlq_s32(hi[c], vsh), vnlim), vlim); \
                vst1q_s32(sc[b0 + c] + j, a); vst1q_s32(sc[b0 + c] + j + 4, b);                                                     \
            }                                                                                                                       \
        }                                                                                                                           \
        for (; j < cnt; j++)                                                                                                        \
            for (int c = 0; c < (NB); c++) {                                                                                        \
                int32_t acc = 0;                                                                                                    \
                for (int d = 0; d < hd; d++) acc += (int32_t)(int16_t)q[b0 + c][d] * (int32_t)k16[(size_t)d * tp + start + j];       \
                int32_t v = acc >> sh;                                                                                              \
                if (v > lim) v = lim; else if (v < -lim) v = -lim;                                                                  \
                sc[b0 + c][j] = v;                                                                                                  \
            }                                                                                                                       \
    } while (0)
static void nk_scores16p(const int16_t *k16, int tp, int hd, int start, int cnt, const int32_t *const *q, int G, int shift, int32_t lim, int32_t **sc) {
    const int sh = 16 + shift > 31 ? 31 : 16 + shift;
    const int32x4_t vsh = vdupq_n_s32(-sh), vlim = vdupq_n_s32(lim), vnlim = vdupq_n_s32(-lim);
    int b0 = 0;
    for (; b0 + 4 <= G; b0 += 4) NK_S16(4);
    switch (G - b0) {
    case 3: NK_S16(3); break;
    case 2: NK_S16(2); break;
    case 1: NK_S16(1); break;
    default: break;
    }
}

/* Backend NEON que usa el motor (medido en un dispositivo con Cortex-A53): el matvec/matmul NEON de 2 bits no gana al escalar,
 * asi que el producto ternario usa el escalar (layout ROW, que el compilador ya vectoriza con NEON) y NEON solo hace lo que
 * si gana: puntajes de atencion y argmax. Mismos bits que el escalar. */
static void hy_matvec(const TernaryInferMat *m, const uint8_t *q, int32_t zp, int32_t *z) { ternaryai_infer_k_scalar.matvec(m, q, zp, z); }

const TernaryInferKernels ternaryai_infer_k_neon_hybrid = { "neon", hy_matvec, nk_argmax, nk_scores,
    TERNARY_NEON_INFER_QUANT ? nk_quant : NULL, TERNARY_NEON_INFER_RELU2 ? nk_relu2 : NULL, TERNARY_NEON_INFER_SOFTMAX ? nk_softmax : NULL, NULL, NULL,
    TERNARY_NEON_INFER_PV ? nk_pv_multi : NULL, NULL, TERNARYAI_INFER_LAYOUT_ROW,
    TERNARY_NEON_INFER_SCORES16 ? nk_scores16p : NULL };

#else   /* sin NEON: el nucleo no existe; se reutiliza el escalar para que el enlace funcione */
const TernaryInferKernels ternaryai_infer_k_neon = { .name = "scalar" };
const TernaryInferKernels ternaryai_infer_k_neon_hybrid = { .name = "scalar" };
#endif
