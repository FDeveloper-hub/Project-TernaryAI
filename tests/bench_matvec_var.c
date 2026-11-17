/* Variantes del matvec NEON del motor frente al escalar, en una sola corrida (para decidir en el dispositivo).
 *   bench_matvec_var --check   comprueba que todas las variantes dan los mismos z que el escalar (formas de borde)
 *   bench_matvec_var           tabla de tiempos (us por llamada) y razon frente al escalar
 * Variantes (layout NEON, grupos de 64 columnas): NRxA = filas a la vez x acumuladores independientes por fila; PF = con prefetch;
 * row4/row8 = layout por filas (el del escalar) con intrinsecos NEON explicitos. */
#define _POSIX_C_SOURCE 200809L
#include "../src/ternary_infer_internal.h"
#include <stdio.h>
#include <time.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>

static uint64_t rs = 0x2545F4914F6CDD1DULL;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 32); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static inline uint32_t hsum_u32(uint32x4_t a) { const uint32x2_t t = vadd_u32(vget_low_u32(a), vget_high_u32(a)); return vget_lane_u32(vpadd_u32(t, t), 0); }
static inline uint32_t sum_u8(const uint8_t *q, int n) {
    uint32x4_t a = vdupq_n_u32(0);
    for (int i = 0; i < n; i += 16) a = vpadalq_u16(a, vpaddlq_u8(vld1q_u8(q + i)));
    return hsum_u32(a);
}
static inline uint32_t tail_dot(const uint8_t *row, const uint8_t *q, int g, int B) {
    uint32_t s = 0;
    for (int j = 0; j < 4; j++) for (int c = 0; c < B; c++) s += (uint32_t)((row[16 * g + c] >> (2 * j)) & 3) * (uint32_t)q[64 * g + j * B + c];
    return s;
}

/* ---- layout NEON: NR filas, A acumuladores por fila, PF = prefetch de pesos ---- */
static inline __attribute__((always_inline)) void rows_n(const TernaryInferMat *m, const uint8_t *q, uint32_t sq, int32_t zp, int32_t *z, int o,
                                                         const int NR, const int A, const int PF) {
    const int Q = m->cp >> 2, nfull = m->cp >> 6, tailc = m->cp & 63;
    const uint8x16_t m3 = vdupq_n_u8(3);
    const uint8_t *r[8];
    uint32x4_t a32[8];
    for (int k = 0; k < NR; k++) { r[k] = m->w + (size_t)(o + k) * (size_t)Q; a32[k] = vdupq_n_u32(0); }
    for (int g0 = 0; g0 < nfull; g0 += 16) {
        const int g1 = (g0 + 16 < nfull) ? g0 + 16 : nfull;
        uint16x8_t a16[8][4];
        for (int k = 0; k < NR; k++) for (int a = 0; a < A; a++) a16[k][a] = vdupq_n_u16(0);
        for (int g = g0; g < g1; g++) {
            uint8x16_t qv[4];
            for (int j = 0; j < 4; j++) qv[j] = vld1q_u8(q + 64 * g + 16 * j);
            for (int k = 0; k < NR; k++) {
                if (PF) __builtin_prefetch(r[k] + 16 * (g + 8));
                const uint8x16_t w = vld1q_u8(r[k] + 16 * g);
                uint8x16_t e[4];
                e[0] = vandq_u8(w, m3); e[1] = vandq_u8(vshrq_n_u8(w, 2), m3); e[2] = vandq_u8(vshrq_n_u8(w, 4), m3); e[3] = vshrq_n_u8(w, 6);
                for (int j = 0; j < 4; j++) {
                    const int a = j % A;
                    a16[k][a] = vmlal_u8(a16[k][a], vget_low_u8(e[j]), vget_low_u8(qv[j]));
                    a16[k][a] = vmlal_u8(a16[k][a], vget_high_u8(e[j]), vget_high_u8(qv[j]));
                }
            }
        }
        for (int k = 0; k < NR; k++) for (int a = 0; a < A; a++) a32[k] = vpadalq_u16(a32[k], a16[k][a]);
    }
    for (int k = 0; k < NR; k++) {
        uint32_t s = hsum_u32(a32[k]);
        if (tailc) s += tail_dot(r[k], q, nfull, tailc >> 2);
        z[o + k] = (int32_t)(s - sq) - zp * m->S[o + k];
    }
}
#define DEF_NEON(NAME, NR, A, PF)                                                                                        \
static void NAME(const TernaryInferMat *m, const uint8_t *q, int32_t zp, int32_t *z) {                                   \
    const uint32_t sq = sum_u8(q, m->cp);                                                                                \
    int o = 0;                                                                                                           \
    for (; o + (NR) <= m->rows; o += (NR)) rows_n(m, q, sq, zp, z, o, (NR), (A), (PF));                                  \
    for (; o < m->rows; o++) rows_n(m, q, sq, zp, z, o, 1, 1, 0);                                                        \
}
DEF_NEON(v_4x1, 4, 1, 0)
DEF_NEON(v_4x2, 4, 2, 0)
DEF_NEON(v_2x4, 2, 4, 0)
DEF_NEON(v_8x1, 8, 1, 0)
DEF_NEON(v_8x1_pf, 8, 1, 1)
DEF_NEON(v_4x2_pf, 4, 2, 1)

/* ---- layout por filas (el del escalar) con intrinsecos: campos de 16 bits, vmla de 16 bits, 8 acumuladores independientes ---- */
static inline __attribute__((always_inline)) void rows_row(const TernaryInferMat *m, const uint16_t *q16, uint32_t sq, int32_t zp, int32_t *z, int o, const int NR) {
    const int T = m->cp >> 3, Q = m->cp >> 2, nfull = T >> 3;
    const uint8_t *r[8];
    uint32x4_t a32[8];
    for (int k = 0; k < NR; k++) { r[k] = m->w + (size_t)(o + k) * (size_t)Q; a32[k] = vdupq_n_u32(0); }
    const uint16x8_t m3 = vdupq_n_u16(3);
    for (int c0 = 0; c0 < nfull; c0 += 8) {
        const int c1 = (c0 + 8 < nfull) ? c0 + 8 : nfull;
        uint16x8_t a16[8];
        for (int k = 0; k < NR; k++) a16[k] = vdupq_n_u16(0);
        for (int blk = c0; blk < c1; blk++) {
            uint16x8_t qv[8];
            for (int P = 0; P < 8; P++) qv[P] = vld1q_u16(q16 + (size_t)P * T + (size_t)blk * 8);
            for (int k = 0; k < NR; k++) {
                const uint16x8_t w = vreinterpretq_u16_u8(vld1q_u8(r[k] + (size_t)blk * 16));
                uint16x8_t a = a16[k];
                a = vmlaq_u16(a, vandq_u16(w, m3), qv[0]);
                a = vmlaq_u16(a, vandq_u16(vshrq_n_u16(w, 2), m3), qv[1]);
                a = vmlaq_u16(a, vandq_u16(vshrq_n_u16(w, 4), m3), qv[2]);
                a = vmlaq_u16(a, vandq_u16(vshrq_n_u16(w, 6), m3), qv[3]);
                a = vmlaq_u16(a, vandq_u16(vshrq_n_u16(w, 8), m3), qv[4]);
                a = vmlaq_u16(a, vandq_u16(vshrq_n_u16(w, 10), m3), qv[5]);
                a = vmlaq_u16(a, vandq_u16(vshrq_n_u16(w, 12), m3), qv[6]);
                a = vmlaq_u16(a, vshrq_n_u16(w, 14), qv[7]);
                a16[k] = a;
            }
        }
        for (int k = 0; k < NR; k++) a32[k] = vpadalq_u16(a32[k], a16[k]);
    }
    for (int k = 0; k < NR; k++) {
        uint32_t s = hsum_u32(a32[k]);
        for (int t = nfull * 8; t < T; t++) {
            uint16_t w; memcpy(&w, r[k] + 2 * (size_t)t, 2);
            for (int P = 0; P < 8; P++) s += (uint32_t)((w >> (2 * P)) & 3) * (uint32_t)q16[(size_t)P * T + t];
        }
        z[o + k] = (int32_t)(s - sq) - zp * m->S[o + k];
    }
}
#define DEF_ROW(NAME, NR)                                                                                                \
static void NAME(const TernaryInferMat *m, const uint8_t *q, int32_t zp, int32_t *z) {                                   \
    uint16_t q16[m->cp];                                                                                                 \
    for (int i = 0; i < m->cp; i++) q16[i] = q[i];                                                                       \
    const uint32_t sq = sum_u8(q, m->cp);                                                                                \
    int o = 0;                                                                                                           \
    for (; o + (NR) <= m->rows; o += (NR)) rows_row(m, q16, sq, zp, z, o, (NR));                                         \
    for (; o < m->rows; o++) rows_row(m, q16, sq, zp, z, o, 1);                                                          \
}
DEF_ROW(v_row4, 4)
DEF_ROW(v_row8, 8)

typedef void (*MvFn)(const TernaryInferMat *, const uint8_t *, int32_t, int32_t *);
typedef struct { const char *name; MvFn fn; int layout; } Var;
static const Var VARS[] = {
    { "escalar (referencia)", NULL, TERNARYAI_INFER_LAYOUT_ROW },
    { "neon actual 4x1", v_4x1, TERNARYAI_INFER_LAYOUT_NEON },
    { "neon 4x2", v_4x2, TERNARYAI_INFER_LAYOUT_NEON },
    { "neon 2x4", v_2x4, TERNARYAI_INFER_LAYOUT_NEON },
    { "neon 8x1", v_8x1, TERNARYAI_INFER_LAYOUT_NEON },
    { "neon 8x1 +prefetch", v_8x1_pf, TERNARYAI_INFER_LAYOUT_NEON },
    { "neon 4x2 +prefetch", v_4x2_pf, TERNARYAI_INFER_LAYOUT_NEON },
    { "fila-16bit 4 filas", v_row4, TERNARYAI_INFER_LAYOUT_ROW },
    { "fila-16bit 8 filas", v_row8, TERNARYAI_INFER_LAYOUT_ROW },
};
#define NV ((int)(sizeof VARS / sizeof VARS[0]))

static int check(void) {
    int bad = 0, total = 0;
    const int rows_l[] = { 1, 3, 7, 8, 9, 16, 17, 33, 100 }, cols_l[] = { 1, 15, 16, 48, 63, 64, 65, 130, 192, 300, 1100, 2050 };
    for (int ri = 0; ri < 9; ri++) for (int ci = 0; ci < 12; ci++) for (int mode = 0; mode < 2; mode++) {
        const int rows = rows_l[ri], cols = cols_l[ci];
        int8_t *w = malloc((size_t)rows * cols);
        for (int i = 0; i < rows * cols; i++) w[i] = mode ? 1 : (int8_t)((int)(rnd() % 3) - 1);
        TernaryInferMat mr, mn;
        ternaryai_infer_mat_build(&mr, w, rows, cols, 1.0f, TERNARYAI_INFER_LAYOUT_ROW); ternaryai_infer_mat_build(&mn, w, rows, cols, 1.0f, TERNARYAI_INFER_LAYOUT_NEON);
        uint8_t *q = malloc((size_t)mr.cp + 16); for (int i = 0; i < mr.cp; i++) q[i] = mode ? 255 : (uint8_t)rnd();
        const int32_t zp = (int32_t)(rnd() % 256);
        int32_t *zr = calloc((size_t)mr.rb * 16 + 16, 4), *zv = calloc((size_t)mr.rb * 16 + 16, 4);
        ternaryai_infer_k_scalar.matvec(&mr, q, zp, zr);
        for (int v = 1; v < NV; v++) {
            memset(zv, 0, (size_t)rows * 4);
            VARS[v].fn(VARS[v].layout == TERNARYAI_INFER_LAYOUT_NEON ? &mn : &mr, q, zp, zv);
            total++; if (memcmp(zr, zv, (size_t)rows * 4)) { bad++; if (bad < 10) printf("FALLA %s rows=%d cols=%d mode=%d\n", VARS[v].name, rows, cols, mode); }
        }
        free(w); free(q); free(zr); free(zv); ternaryai_infer_mat_free(&mr); ternaryai_infer_mat_free(&mn);
    }
    printf("%d/%d comprobaciones correctas\n", total - bad, total);
    return bad != 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--check")) return check();
    if (argc > 3 && !strcmp(argv[1], "--count")) {       /* --count <idx variante> <reps>: para contar instrucciones bajo qemu */
        const int v = atoi(argv[2]), reps = atoi(argv[3]);
        int8_t *w = malloc(512 * 512); for (int i = 0; i < 512 * 512; i++) w[i] = (int8_t)((int)(rnd() % 3) - 1);
        TernaryInferMat mm; ternaryai_infer_mat_build(&mm, w, 512, 512, 1.0f, VARS[v].layout);
        uint8_t *q = malloc(mm.cp + 16); for (int i = 0; i < mm.cp; i++) q[i] = (uint8_t)rnd();
        int32_t *z = calloc(mm.rb * 16 + 16, 4);
        for (int i = 0; i < reps; i++) (VARS[v].fn ? VARS[v].fn : ternaryai_infer_k_scalar.matvec)(&mm, q, 128, z);
        return 0;
    }
    const int shp[3][2] = { { 512, 512 }, { 1024, 256 }, { 2048, 1024 } };
    double t[NV][3];
    for (int s = 0; s < 3; s++) {
        const int rows = shp[s][0], cols = shp[s][1];
        int8_t *w = malloc((size_t)rows * cols); for (int i = 0; i < rows * cols; i++) w[i] = (int8_t)((int)(rnd() % 3) - 1);
        TernaryInferMat mr, mn;
        ternaryai_infer_mat_build(&mr, w, rows, cols, 1.0f, TERNARYAI_INFER_LAYOUT_ROW); ternaryai_infer_mat_build(&mn, w, rows, cols, 1.0f, TERNARYAI_INFER_LAYOUT_NEON);
        uint8_t *q = malloc((size_t)mr.cp + 16); for (int i = 0; i < mr.cp; i++) q[i] = (uint8_t)rnd();
        int32_t *z = calloc((size_t)mr.rb * 16 + 16, 4);
        for (int v = 0; v < NV; v++) {
            const TernaryInferMat *mt = VARS[v].layout == TERNARYAI_INFER_LAYOUT_NEON ? &mn : &mr;
            MvFn fn = VARS[v].fn ? VARS[v].fn : ternaryai_infer_k_scalar.matvec;
            int reps = 1; double el;
            for (;;) { const double t0 = now(); for (int i = 0; i < reps; i++) fn(mt, q, 128, z); el = now() - t0; if (el > 0.15) break; reps *= 2; }
            double best = el / reps;
            for (int r = 0; r < 7; r++) { const double t0 = now(); for (int i = 0; i < reps; i++) fn(mt, q, 128, z); const double e = (now() - t0) / reps; if (e < best) best = e; }
            t[v][s] = best * 1e6;
        }
        free(w); free(q); free(z); ternaryai_infer_mat_free(&mr); ternaryai_infer_mat_free(&mn);
    }
    printf("%-22s %12s %12s %12s   (x = escalar/variante; >1 gana al escalar)\n", "variante", "512x512 us", "1024x256 us", "2048x1024 us");
    for (int v = 0; v < NV; v++) printf("%-22s %12.1f %12.1f %12.1f   x%.2f x%.2f x%.2f\n", VARS[v].name, t[v][0], t[v][1], t[v][2], t[0][0] / t[v][0], t[0][1] / t[v][1], t[0][2] / t[v][2]);
    return 0;
}
#else
int main(void) { printf("sin NEON: nada que medir\n"); return 0; }
#endif
