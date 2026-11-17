/* Nucleo AVX-512: producto ternario x uint8 con VNNI (vpdpbusd) o, sin VNNI, con BW (vpmaddubsw + vpmaddwd).
 * Todo entero exacto: mismos z que el nucleo escalar. Se compila siempre (atributos de funcion) y se elige en ejecucion. */
#include "ternary_infer_internal.h"
#include <math.h>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>

#define AVX_BW  __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq")))
#define AVX_VNNI __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")))

static inline uint32_t ld32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* ------------------------------------------------------------------ producto sobre pesos de 2 bits
 * Los pesos se quedan a 2 bits en memoria: cada vector de 64 bytes trae 16 filas x 16 columnas (4 codigos por byte). Se separan los
 * 4 campos en registro (desplazar y enmascarar: 4 vectores de 16 filas x 4 columnas) y cada uno entra en un vpdpbusd con el codigo
 * (0,1,2) como operando sin signo y la activacion con signo (q-128, es decir q^0x80):
 *   acc = sum c*(q-128) = sum c*q - 128*(S+cp)   =>   z = acc + 128*cp - sum(q) + (128-zp)*S    (enteros exactos)
 * Las columnas de relleno tienen codigo 1 y se cancelan solas porque sum(q) se hace sobre las mismas cp posiciones. */
AVX_BW static inline int32_t sum_u8(const uint8_t *q, int n) {     /* n multiplo de 16; lee exactamente n bytes */
    __m512i a = _mm512_setzero_si512();
    const __m512i zero = a;
    int i = 0;
    for (; i + 64 <= n; i += 64) a = _mm512_add_epi64(a, _mm512_sad_epu8(_mm512_loadu_si512((const void *)(q + i)), zero));
    if (i < n) a = _mm512_add_epi64(a, _mm512_sad_epu8(_mm512_maskz_loadu_epi8((__mmask64)((1ULL << (n - i)) - 1), q + i), zero));
    return (int32_t)_mm512_reduce_add_epi64(a);
}

#define EXPAND(V, J) ((J) == 0 ? _mm512_and_si512((V), m3) : _mm512_and_si512(_mm512_srli_epi16((V), 2 * (J)), m3))
#define ACT(Q, G, J) _mm512_set1_epi32((int)(ld32((Q) + 16 * (G) + 4 * (J)) ^ 0x80808080u))
#define FIN(ACC, VK, VC, S32) _mm512_add_epi32(_mm512_add_epi32((ACC), (VK)), _mm512_mullo_epi32((VC), _mm512_loadu_si512((const void *)(S32))))

/* matvec: 4 bloques de 16 filas a la vez (4 acumuladores independientes) */
#define DEFINE_MATVEC(NAME, ATTR, DPB)                                                                                  \
ATTR static void NAME(const TernaryInferMat *m, const uint8_t *q, int32_t zp, int32_t *z) {                              \
    const int G = m->cp / 16;                                                                                           \
    const __m512i m3 = _mm512_set1_epi8(3), ones = _mm512_set1_epi16(1); (void)ones;                                    \
    const __m512i vk = _mm512_set1_epi32(128 * m->cp - sum_u8(q, m->cp)), vc = _mm512_set1_epi32(128 - zp);             \
    const size_t bs = (size_t)G * 64;                                                                                   \
    int rb = 0;                                                                                                         \
    for (; rb + 4 <= m->rb; rb += 4) {                                                                                  \
        const uint8_t *w0 = m->w + (size_t)rb * bs, *w1 = w0 + bs, *w2 = w1 + bs, *w3 = w2 + bs;                       \
        __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;                                                 \
        for (int g = 0; g < G; g++) {                                                                                   \
            const __m512i v0 = _mm512_load_si512((const void *)(w0 + (size_t)g * 64)), v1 = _mm512_load_si512((const void *)(w1 + (size_t)g * 64)); \
            const __m512i v2 = _mm512_load_si512((const void *)(w2 + (size_t)g * 64)), v3 = _mm512_load_si512((const void *)(w3 + (size_t)g * 64)); \
            for (int J = 0; J < 4; J++) {                                                                               \
                const __m512i a = ACT(q, g, J);                                                                         \
                a0 = DPB(a0, EXPAND(v0, J), a); a1 = DPB(a1, EXPAND(v1, J), a); a2 = DPB(a2, EXPAND(v2, J), a); a3 = DPB(a3, EXPAND(v3, J), a); \
            }                                                                                                           \
        }                                                                                                               \
        _mm512_storeu_si512((void *)(z + 16 * (rb + 0)), FIN(a0, vk, vc, m->S + 16 * (rb + 0)));                        \
        _mm512_storeu_si512((void *)(z + 16 * (rb + 1)), FIN(a1, vk, vc, m->S + 16 * (rb + 1)));                        \
        _mm512_storeu_si512((void *)(z + 16 * (rb + 2)), FIN(a2, vk, vc, m->S + 16 * (rb + 2)));                        \
        _mm512_storeu_si512((void *)(z + 16 * (rb + 3)), FIN(a3, vk, vc, m->S + 16 * (rb + 3)));                        \
    }                                                                                                                   \
    for (; rb < m->rb; rb++) {                                                                                          \
        const uint8_t *w0 = m->w + (size_t)rb * bs;                                                                     \
        __m512i a0 = _mm512_setzero_si512(), a1 = a0;                                                                   \
        for (int g = 0; g < G; g++) {                                                                                   \
            const __m512i v0 = _mm512_load_si512((const void *)(w0 + (size_t)g * 64));                                  \
            a0 = DPB(a0, EXPAND(v0, 0), ACT(q, g, 0)); a1 = DPB(a1, EXPAND(v0, 1), ACT(q, g, 1));                       \
            a0 = DPB(a0, EXPAND(v0, 2), ACT(q, g, 2)); a1 = DPB(a1, EXPAND(v0, 3), ACT(q, g, 3));                       \
        }                                                                                                               \
        _mm512_storeu_si512((void *)(z + 16 * rb), FIN(_mm512_add_epi32(a0, a1), vk, vc, m->S + 16 * rb));              \
    }                                                                                                                   \
}

#define DPB_VNNI(acc, u, s) _mm512_dpbusd_epi32((acc), (u), (s))
#define DPB_BW(acc, u, s)   _mm512_add_epi32((acc), _mm512_madd_epi16(_mm512_maddubs_epi16((u), (s)), ones))

DEFINE_MATVEC(mv_vnni, AVX_VNNI, DPB_VNNI)
DEFINE_MATVEC(mv_bw, AVX_BW, DPB_BW)


/* ------------------------------------------------------------------ producto por varios carriles: cada bloque de pesos se lee y se
 * desempaqueta una vez y se usa para NB carriles (NB constante: el compilador lo desenrolla). Mismos enteros que matvec. */
#define DEFINE_MATMUL(NAME, ATTR, DPB)                                                                                        \
ATTR static inline __attribute__((always_inline)) void NAME##_nb(const TernaryInferMat *m, const uint8_t *const *q, const int32_t *zp, \
                                                                  int32_t *const *z, const int NB) {                          \
    const int G = m->cp / 16;                                                                                                  \
    const __m512i m3 = _mm512_set1_epi8(3), ones = _mm512_set1_epi16(1); (void)ones;                                          \
    __m512i vk[4], vc[4];                                                                                                      \
    for (int b = 0; b < NB; b++) { vk[b] = _mm512_set1_epi32(128 * m->cp - sum_u8(q[b], m->cp)); vc[b] = _mm512_set1_epi32(128 - zp[b]); } \
    for (int rb = 0; rb < m->rb; rb++) {                                                                                       \
        const uint8_t *w = m->w + (size_t)rb * (size_t)G * 64;                                                                 \
        __m512i acc[4] = { _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512() };     \
        for (int g = 0; g < G; g++) {                                                                                          \
            const __m512i v = _mm512_load_si512((const void *)(w + (size_t)g * 64));                                           \
            for (int J = 0; J < 4; J++) {                                                                                      \
                const __m512i e = EXPAND(v, J);                                                                                \
                for (int b = 0; b < NB; b++) acc[b] = DPB(acc[b], e, ACT(q[b], g, J));                                         \
            }                                                                                                                  \
        }                                                                                                                      \
        for (int b = 0; b < NB; b++) _mm512_storeu_si512((void *)(z[b] + 16 * rb), FIN(acc[b], vk[b], vc[b], m->S + 16 * rb));  \
    }                                                                                                                          \
}                                                                                                                              \
ATTR static void NAME(const TernaryInferMat *m, const uint8_t *const *q, const int32_t *zp, int32_t *const *z, int B) {         \
    int b = 0;                                                                                                                 \
    for (; b + 4 <= B; b += 4) NAME##_nb(m, q + b, zp + b, z + b, 4);                                                          \
    switch (B - b) {                                                                                                           \
    case 3: NAME##_nb(m, q + b, zp + b, z + b, 3); break;                                                                      \
    case 2: NAME##_nb(m, q + b, zp + b, z + b, 2); break;                                                                      \
    case 1: NAME##_nb(m, q + b, zp + b, z + b, 1); break;                                                                      \
    default: break;                                                                                                            \
    }                                                                                                                          \
}
DEFINE_MATMUL(mm_vnni, AVX_VNNI, DPB_VNNI)
DEFINE_MATMUL(mm_bw, AVX_BW, DPB_BW)

/* ------------------------------------------------------------------ argmax de un vector de int32 (el primero en empate) */
AVX_BW static int avx_argmax(const int32_t *z, int n) {
    __m512i bestv = _mm512_set1_epi32(INT32_MIN), besti = _mm512_setzero_si512();
    __m512i idx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m512i step = _mm512_set1_epi32(16);
    for (int t = 0; t < n; t += 16) {
        const __mmask16 lm = (n - t >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << (n - t)) - 1);
        const __m512i v = _mm512_mask_loadu_epi32(_mm512_set1_epi32(INT32_MIN), lm, z + t);
        const __mmask16 gt = _mm512_mask_cmpgt_epi32_mask(lm, v, bestv);    /* estrictamente mayor: el primero gana */
        bestv = _mm512_mask_mov_epi32(bestv, gt, v);
        besti = _mm512_mask_mov_epi32(besti, gt, idx);
        idx = _mm512_add_epi32(idx, step);
    }
    const int32_t M = _mm512_reduce_max_epi32(bestv);
    const __mmask16 eq = _mm512_cmpeq_epi32_mask(bestv, _mm512_set1_epi32(M));
    /* entre los carriles con el maximo, el de menor indice (cada carril ya guarda su primer indice) */
    const __m512i cand = _mm512_mask_mov_epi32(_mm512_set1_epi32(INT32_MAX), eq, besti);
    return (int)_mm512_reduce_min_epi32(cand);
}

/* ------------------------------------------------------------------ puntajes de atencion: (q . k) >> (16+shift), recortado
 * kT: hd filas de paso tp (claves en carriles). int64 exacto: productos pares/impares con vpmuldq. */
AVX_BW static void avx_scores(const int32_t *kT, int tp, int hd, int start, int cnt, const int32_t *const *q, int G,
                              int shift, int32_t lim, int32_t **sc, int64_t *scr) {
    (void)scr;
    const __m512i vlim = _mm512_set1_epi64(lim), vnlim = _mm512_set1_epi64(-(int64_t)lim);
    const __m512i perm = _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
    const __m128i vsh = _mm_cvtsi32_si128(16 + shift);
    for (int g0 = 0; g0 < G; g0 += 4) {
        const int gb = (G - g0 < 4) ? (G - g0) : 4;
        for (int t = 0; t < cnt; t += 16) {
            const __mmask16 lm = (cnt - t >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << (cnt - t)) - 1);
            __m512i ae[4], ao[4];
            for (int b = 0; b < 4; b++) { ae[b] = _mm512_setzero_si512(); ao[b] = _mm512_setzero_si512(); }
            const int32_t *kp = kT + start + t;
            if (gb == 2) {
                const int32_t *q0 = q[g0], *q1 = q[g0 + 1];
                for (int d = 0; d < hd; d++) {
                    const __m512i kv = _mm512_maskz_loadu_epi32(lm, kp + (size_t)d * tp), ko = _mm512_srli_epi64(kv, 32);
                    const __m512i b0 = _mm512_set1_epi32(q0[d]), b1 = _mm512_set1_epi32(q1[d]);
                    ae[0] = _mm512_add_epi64(ae[0], _mm512_mul_epi32(b0, kv)); ao[0] = _mm512_add_epi64(ao[0], _mm512_mul_epi32(b0, ko));
                    ae[1] = _mm512_add_epi64(ae[1], _mm512_mul_epi32(b1, kv)); ao[1] = _mm512_add_epi64(ao[1], _mm512_mul_epi32(b1, ko));
                }
            } else {
                for (int d = 0; d < hd; d++) {
                    const __m512i kv = _mm512_maskz_loadu_epi32(lm, kp + (size_t)d * tp), ko = _mm512_srli_epi64(kv, 32);
                    for (int b = 0; b < gb; b++) {
                        const __m512i qb = _mm512_set1_epi32(q[g0 + b][d]);
                        ae[b] = _mm512_add_epi64(ae[b], _mm512_mul_epi32(qb, kv));
                        ao[b] = _mm512_add_epi64(ao[b], _mm512_mul_epi32(qb, ko));
                    }
                }
            }
            for (int b = 0; b < gb; b++) {
                __m512i e = _mm512_sra_epi64(ae[b], vsh), o = _mm512_sra_epi64(ao[b], vsh);
                e = _mm512_min_epi64(_mm512_max_epi64(e, vnlim), vlim);
                o = _mm512_min_epi64(_mm512_max_epi64(o, vnlim), vlim);
                const __m512i comb = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi64_epi32(e)), _mm512_cvtepi64_epi32(o), 1);
                _mm512_mask_storeu_epi32(sc[g0 + b] + t, lm, _mm512_permutexvar_epi32(perm, comb));
            }
        }
    }
}

/* ------------------------------------------------------------------ cuantizacion u8 por fila (mismos bits que la escalar) */
AVX_BW static void avx_quant(const float *x, int n, uint8_t *q, float *scale, int32_t *zp) {
    __m512 vlo = _mm512_setzero_ps(), vhi = _mm512_setzero_ps();
    for (int i = 0; i < n; i += 16) {
        const __mmask16 lm = (n - i >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << (n - i)) - 1);
        const __m512 v = _mm512_maskz_loadu_ps(lm, x + i);
        vlo = _mm512_min_ps(vlo, v); vhi = _mm512_max_ps(vhi, v);
    }
    const float lo = _mm512_reduce_min_ps(vlo), hi = _mm512_reduce_max_ps(vhi);
    const float sc = (hi - lo) / 255.0f;
    if (!(sc > 0.0f) || !(sc <= 3.4e38f)) { *scale = 1.0f; *zp = 0; memset(q, 0, (size_t)n); return; }
    int z = (int)rintf(-lo / sc);
    if (z < 0) z = 0;
    if (z > 255) z = 255;
    *scale = sc; *zp = z;
    const __m512 vinv = _mm512_set1_ps(1.0f / sc);
    const __m512i vz = _mm512_set1_epi32(z), v0 = _mm512_setzero_si512(), v255 = _mm512_set1_epi32(255);
    for (int i = 0; i < n; i += 16) {
        const __mmask16 lm = (n - i >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << (n - i)) - 1);
        __m512i r = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_maskz_loadu_ps(lm, x + i), vinv)), vz);
        r = _mm512_min_epi32(_mm512_max_epi32(r, v0), v255);
        _mm_mask_storeu_epi8(q + i, lm, _mm512_cvtepi32_epi8(r));
    }
}

AVX_BW static void avx_relu2(const int32_t *z, int n, float k, float *h) {
    const __m512 vk = _mm512_set1_ps(k);
    for (int i = 0; i < n; i += 16) {
        const __mmask16 lm = (n - i >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << (n - i)) - 1);
        const __m512 a = _mm512_mul_ps(vk, _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(lm, z + i)));
        const __mmask16 pos = _mm512_cmp_ps_mask(a, _mm512_setzero_ps(), _CMP_GT_OQ);
        _mm512_mask_storeu_ps(h + i, lm, _mm512_maskz_mul_ps(pos, a, a));
    }
}


/* ------------------------------------------------------------------ puntajes con claves int16 (vpmaddwd): 2 filas por instruccion */
AVX_BW static void avx_scores16(const int16_t *k16, int tp, int hd, int start, int cnt, const int32_t *const *q, int G,
                                int shift, int32_t lim, int32_t **sc) {
    const int HP = hd / 2;
    int32_t qp[8][128];                              /* pares (q[2i], q[2i+1]) empaquetados en int16 */
    for (int b = 0; b < G; b++)
        for (int i = 0; i < HP; i++) qp[b][i] = (int32_t)(((uint32_t)(uint16_t)q[b][2 * i]) | ((uint32_t)(uint16_t)q[b][2 * i + 1] << 16));
    const __m128i vsh = _mm_cvtsi32_si128(16 + shift);
    const __m512i vlim = _mm512_set1_epi32(lim), vnlim = _mm512_set1_epi32(-lim);
    for (int t = 0; t < cnt; t += 16) {
        const __mmask16 lm = (cnt - t >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << (cnt - t)) - 1);
        const int16_t *kp = k16 + 2 * (size_t)(start + t);
        int b0 = 0;
        for (; b0 + 4 <= G; b0 += 4) {
            __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
            for (int i = 0; i < HP; i++) {
                const __m512i kv = _mm512_maskz_loadu_epi32(lm, kp + (size_t)i * tp * 2);
                a0 = _mm512_add_epi32(a0, _mm512_madd_epi16(kv, _mm512_set1_epi32(qp[b0][i])));
                a1 = _mm512_add_epi32(a1, _mm512_madd_epi16(kv, _mm512_set1_epi32(qp[b0 + 1][i])));
                a2 = _mm512_add_epi32(a2, _mm512_madd_epi16(kv, _mm512_set1_epi32(qp[b0 + 2][i])));
                a3 = _mm512_add_epi32(a3, _mm512_madd_epi16(kv, _mm512_set1_epi32(qp[b0 + 3][i])));
            }
            const __m512i r[4] = { a0, a1, a2, a3 };
            for (int k = 0; k < 4; k++) {
                __m512i v = _mm512_sra_epi32(r[k], vsh);
                v = _mm512_min_epi32(_mm512_max_epi32(v, vnlim), vlim);
                _mm512_mask_storeu_epi32(sc[b0 + k] + t, lm, v);
            }
        }
        for (; b0 < G; b0++) {
            __m512i a0 = _mm512_setzero_si512();
            for (int i = 0; i < HP; i++)
                a0 = _mm512_add_epi32(a0, _mm512_madd_epi16(_mm512_maskz_loadu_epi32(lm, kp + (size_t)i * tp * 2), _mm512_set1_epi32(qp[b0][i])));
            __m512i v = _mm512_sra_epi32(a0, vsh);
            v = _mm512_min_epi32(_mm512_max_epi32(v, vnlim), vlim);
            _mm512_mask_storeu_epi32(sc[b0] + t, lm, v);
        }
    }
}

/* ------------------------------------------------------------------ softmax (tabla de exp) y suma ponderada de V */
AVX_BW static float avx_softmax(const int32_t *sc, int cnt, const float *tab, uint32_t K0, float *ee) {
    __m512i vm = _mm512_set1_epi32(INT32_MIN);
    for (int j = 0; j < cnt; j += 16) {
        const __mmask16 lm = (cnt - j >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << (cnt - j)) - 1);
        vm = _mm512_mask_max_epi32(vm, lm, vm, _mm512_maskz_loadu_epi32(lm, sc + j));
    }
    const __m512i mmax = _mm512_set1_epi32(_mm512_reduce_max_epi32(vm));
    const __m512i vk0 = _mm512_set1_epi32((int)K0);
    __m512 part = _mm512_setzero_ps();
    for (int j = 0; j < cnt; j += 16) {
        const __mmask16 lm = (cnt - j >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << (cnt - j)) - 1);
        const __m512i k = _mm512_min_epu32(_mm512_sub_epi32(mmax, _mm512_maskz_loadu_epi32(lm, sc + j)), vk0);
        const __m512 e = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), lm, k, tab, 4);
        _mm512_mask_storeu_ps(ee + j, lm, e);
        part = _mm512_add_ps(part, e);       /* carril l acumula j = l (mod 16) en orden: igual que el escalar */
    }
    float p[16]; _mm512_storeu_ps(p, part);
    float t8[8], t4[4];
    for (int l = 0; l < 8; l++) t8[l] = p[l] + p[l + 8];
    for (int l = 0; l < 4; l++) t4[l] = t8[l] + t8[l + 4];
    const float a = t4[0] + t4[2], b = t4[1] + t4[3];
    return a + b;
}

#define PVM_NB(NB)                                                                                                       \
    do {                                                                                                                 \
        for (int d0 = 0; d0 < hd; d0 += 16) {                                                                            \
            const __mmask16 lm = (hd - d0 >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << (hd - d0)) - 1);               \
            __m512 a[4];                                                                                                 \
            for (int c = 0; c < (NB); c++) a[c] = _mm512_maskz_loadu_ps(lm, acc[c0 + c] + d0);                            \
            for (int j = 0; j < cnt; j++)                                                                                \
                for (int c = 0; c < (NB); c++)                                                                           \
                    a[c] = _mm512_add_ps(a[c], _mm512_mul_ps(_mm512_set1_ps(ee[c0 + c][j]),                                   \
                           _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(lm, v[c0 + c] + (size_t)j * hd + d0))), _mm512_set1_ps(vs[j])))); \
            for (int c = 0; c < (NB); c++) _mm512_mask_storeu_ps(acc[c0 + c] + d0, lm, a[c]);                              \
        }                                                                                                                \
    } while (0)
AVX_BW static void avx_pv_multi(int nc, const float *const *ee, int cnt, const int8_t *const *v, const float *vs, int hd, float *const *acc) {
    int c0 = 0;
    for (; c0 + 4 <= nc; c0 += 4) PVM_NB(4);
    switch (nc - c0) {
    case 3: PVM_NB(3); break;
    case 2: PVM_NB(2); break;
    case 1: PVM_NB(1); break;
    default: break;
    }
}

const TernaryInferKernels ternaryai_infer_k_avx512_vnni = { "avx512-vnni", mv_vnni, avx_argmax, avx_scores, avx_quant, avx_relu2, avx_softmax, NULL, avx_scores16, avx_pv_multi, mm_vnni, TERNARYAI_INFER_LAYOUT_AVX, NULL };
const TernaryInferKernels ternaryai_infer_k_avx512_bw = { "avx512-bw", mv_bw, avx_argmax, avx_scores, avx_quant, avx_relu2, avx_softmax, NULL, avx_scores16, avx_pv_multi, mm_bw, TERNARYAI_INFER_LAYOUT_AVX, NULL };

#else   /* sin x86-64 / GCC: el nucleo AVX-512 no existe; se reutiliza el escalar para que el enlace funcione */
const TernaryInferKernels ternaryai_infer_k_avx512_vnni = { .name = "scalar" };
const TernaryInferKernels ternaryai_infer_k_avx512_bw = { .name = "scalar" };
#endif
