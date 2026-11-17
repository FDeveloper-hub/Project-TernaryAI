/* Nucleo escalar (C portable, sin intrinsecos). Resultados enteros exactos: iguales a los del nucleo AVX-512. */
#include "ternary_infer_internal.h"

/* Pesos de 2 bits fila a fila (TERNARYAI_INFER_LAYOUT_ROW): cada palabra de 16 bits trae 8 codigos (0,1,2); el campo p es la columna t + p*T.
 * Se usan vectores de 8 carriles de 16 bits (extension de vectores de GCC/clang: SSE2, NEON o escalar segun el destino, sin intrinsecos
 * ni instrucciones especiales): por campo, (palabra >> 2p) & 3 ya es un carril int16 y se multiplica por q (tambien int16, expandido una vez
 * por llamada). Las sumas parciales van en int16: 8 campos x 8 palabras por carril = 64 productos <= 64*510 = 32640, exacto; despues se
 * pasan a int32. z = sum(c q) - sum(q) - zp*S. */
typedef uint16_t v8u __attribute__((vector_size(16), aligned(2)));
typedef int16_t v8s __attribute__((vector_size(16), aligned(2)));
typedef int32_t v4i __attribute__((vector_size(16)));

/* suma de los 8 carriles int16 (exacta: cada carril es un entero con signo de 16 bits) */
static inline __attribute__((always_inline)) int32_t hsum8(v8s a) {
    const v4i x = (v4i)a, t = ((x << 16) >> 16) + (x >> 16);
    return t[0] + t[1] + t[2] + t[3];
}

static inline __attribute__((always_inline)) void sc_rows(const TernaryInferMat *m, const int16_t *q16, int32_t base, int32_t zp, int32_t *z, int o, const int NR) {
    const int T = m->cp >> 3, Q = m->cp >> 2, nfull = T >> 3;
    const uint8_t *r[8];
    int32_t s[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int k = 0; k < NR; k++) r[k] = m->w + (size_t)(o + k) * (size_t)Q;
    for (int c0 = 0; c0 < nfull; c0 += 8) {
        const int c1 = (c0 + 8 < nfull) ? c0 + 8 : nfull;
        v8s acc[8] = { { 0 }, { 0 }, { 0 }, { 0 }, { 0 }, { 0 }, { 0 }, { 0 } };
        for (int blk = c0; blk < c1; blk++) {
            v8s qv[8];
            for (int P = 0; P < 8; P++) memcpy(&qv[P], q16 + (size_t)P * T + (size_t)blk * 8, 16);
            for (int k = 0; k < NR; k++) {
                v8u w; memcpy(&w, r[k] + (size_t)blk * 16, 16);
                v8s a = acc[k];
                a += (v8s)(w & 3) * qv[0];
                a += (v8s)((w >> 2) & 3) * qv[1];
                a += (v8s)((w >> 4) & 3) * qv[2];
                a += (v8s)((w >> 6) & 3) * qv[3];
                a += (v8s)((w >> 8) & 3) * qv[4];
                a += (v8s)((w >> 10) & 3) * qv[5];
                a += (v8s)((w >> 12) & 3) * qv[6];
                a += (v8s)(w >> 14) * qv[7];
                acc[k] = a;
            }
        }
        for (int k = 0; k < NR; k++) s[k] += hsum8(acc[k]);
    }
    for (int t = nfull * 8; t < T; t++)
        for (int k = 0; k < NR; k++) {
            uint16_t w; memcpy(&w, r[k] + 2 * (size_t)t, 2);
            for (int P = 0; P < 8; P++) s[k] += (int32_t)((w >> (2 * P)) & 3) * (int32_t)q16[(size_t)P * T + t];
        }
    for (int k = 0; k < NR; k++) z[o + k] = s[k] - base - zp * m->S[o + k];
}

static void sc_matvec(const TernaryInferMat *m, const uint8_t *q, int32_t zp, int32_t *z) {
    int16_t q16[m->cp];
    int32_t sq = 0;
    for (int i = 0; i < m->cp; i++) { q16[i] = (int16_t)q[i]; sq += q[i]; }
    int o = 0;
    for (; o + 8 <= m->rows; o += 8) sc_rows(m, q16, sq, zp, z, o, 8);
    for (; o + 4 <= m->rows; o += 4) sc_rows(m, q16, sq, zp, z, o, 4);
    for (; o < m->rows; o++) sc_rows(m, q16, sq, zp, z, o, 1);
}

static int sc_argmax(const int32_t *z, int n) {
    int bi = 0; int32_t bv = z[0];
    for (int i = 1; i < n; i++) if (z[i] > bv) { bv = z[i]; bi = i; }
    return bi;
}

static void sc_scores(const int32_t *kT, int tp, int hd, int start, int cnt, const int32_t *const *q, int G,
                      int shift, int32_t lim, int32_t **sc, int64_t *scr) {
    const int sh = 16 + shift;
    for (int b = 0; b < G; b++) {
        memset(scr, 0, (size_t)cnt * sizeof(int64_t));
        for (int d = 0; d < hd; d++) {
            const int64_t qd = (int64_t)q[b][d];
            const int32_t *row = kT + (size_t)d * tp + start;
            for (int j = 0; j < cnt; j++) scr[j] += qd * (int64_t)row[j];
        }
        for (int j = 0; j < cnt; j++) {
            int64_t dot = scr[j] >> sh;
            if (dot > lim) dot = lim; else if (dot < -(int64_t)lim) dot = -(int64_t)lim;
            sc[b][j] = (int32_t)dot;
        }
    }
}

const TernaryInferKernels ternaryai_infer_k_scalar = { "scalar", sc_matvec, sc_argmax, sc_scores, NULL, NULL, NULL, NULL, NULL, NULL, NULL, TERNARYAI_INFER_LAYOUT_ROW, NULL };
