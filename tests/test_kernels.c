/*
 * Nucleos de include/ternary_kernels.h contra el bucle escalar de siempre, BIT A BIT (memcmp), sobre formas al azar:
 * S 1..37 (colas de 1..3 muestras), n 0..300 (colas de columnas), filas de salida 1..41, valores de g con ceros, -0
 * y magnitudes distintas. Implementacion portable (C puro).
 */
#include "ternary_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static uint64_t rs = 88172645463325252ull;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static float frand(void) {
    switch (rnd() % 7) {
        case 0: return 0.0f;
        case 1: return -0.0f;
        case 2: return (float)((double)(rnd() % 2001) - 1000.0) * 1e-3f;
        case 3: return (float)((double)(rnd() % 2001) - 1000.0) * 1e3f;
        default: return (float)((double)(rnd() >> 11) / 9007199254740992.0 - 0.5);
    }
}
static float lm[256][8] __attribute__((aligned(16))), ls[256][8] __attribute__((aligned(16)));
static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FALLO: %s (linea %d)\n", m, __LINE__); fails++; return; } } while (0)

static void expand_ref(const uint64_t *mask, const uint64_t *sign, size_t n, float *out) {
    for (size_t i = 0; i < n; i++) {
        int m = (int)((mask[i >> 6] >> (i & 63)) & 1u), s = (int)((sign[i >> 6] >> (i & 63)) & 1u);
        out[i] = (float)(m * (1 - 2 * s));
    }
}

static void test_expand(void) {
    for (int it = 0; it < 300; it++) {
        size_t n = rnd() % 400;
        size_t words = (n + 63) / 64 + 1;
        uint64_t *m = calloc(words, 8), *s = calloc(words, 8);
        for (size_t w = 0; w < words; w++) { m[w] = rnd(); s[w] = rnd(); }
        float *a = malloc((n + 1) * 4), *b = malloc((n + 1) * 4);
        expand_ref(m, s, n, a);
        tk_expand_trits(m, s, n, b, (const float (*)[8])lm, (const float (*)[8])ls);
        CHECK(memcmp(a, b, n * 4) == 0, "expand_trits distinto de la referencia");
        free(m); free(s); free(a); free(b);
    }
}

static void test_gradw(void) {
    for (int it = 0; it < 400; it++) {
        size_t S = 1 + rnd() % 37, n = rnd() % 300, ld = n + (rnd() % 5);
        float *x = malloc(S * (ld + 1) * 4), *g = malloc(S * 4), *r1 = malloc((n + 1) * 4), *r2 = malloc((n + 1) * 4);
        for (size_t i = 0; i < S * ld; i++) x[i] = (float)((int)(rnd() % 3) - 1);
        for (size_t s = 0; s < S; s++) g[s] = frand();
        for (size_t i = 0; i < n; i++) r1[i] = r2[i] = frand();
        for (size_t i = 0; i < n; i++) { float a = r1[i]; for (size_t s = 0; s < S; s++) a += g[s] * x[s * ld + i]; r1[i] = a; }
        tk_gradw_row(r2, n, S, g, x, ld);
        CHECK(memcmp(r1, r2, n * 4) == 0, "gradw_row distinto de la referencia");
        free(x); free(g); free(r1); free(r2);
    }
}

static void test_gradx(void) {
    for (int it = 0; it < 250; it++) {
        size_t S = 1 + rnd() % 21, outd = (it % 10 == 0) ? (TERNARY_GRADX_OB + 1 + rnd() % 600) : (1 + rnd() % 41), n8 = rnd() % 20, cols = n8 * 8, col = 8 * (rnd() % 3);
        size_t wstride = n8 + 1 + rnd() % 4;
        unsigned char *mb = malloc(outd * wstride + 64), *sb = malloc(outd * wstride + 64);
        for (size_t i = 0; i < outd * wstride + 64; i++) { mb[i] = (unsigned char)rnd(); sb[i] = (unsigned char)rnd(); }
        float *gsT = malloc(outd * S * 4);
        for (size_t i = 0; i < outd * S; i++) gsT[i] = frand();
        float **g1 = malloc(S * sizeof(float *)), **g2 = malloc(S * sizeof(float *));
        for (size_t s = 0; s < S; s++) {
            g1[s] = malloc((col + cols + 8) * 4); g2[s] = malloc((col + cols + 8) * 4);
            for (size_t i = 0; i < col + cols + 8; i++) g1[s][i] = g2[s][i] = frand();
        }
        float **g0 = malloc(S * sizeof(float *));
        for (size_t s = 0; s < S; s++) { g0[s] = malloc((col + cols + 8) * 4); memcpy(g0[s], g1[s], (col + cols + 8) * 4); }
        for (size_t s = 0; s < S; s++)
            for (size_t j = 0; j < cols; j++) {
                float a = g1[s][col + j];
                for (size_t o = 0; o < outd; o++) {
                    int m = (int)((mb[o * wstride + j / 8] >> (j % 8)) & 1u), sg = (int)((sb[o * wstride + j / 8] >> (j % 8)) & 1u);
                    float w = (float)(m * (1 - 2 * sg)); /* (float)int: +0 */
                    a += gsT[o * S + s] * w;
                }
                g1[s][col + j] = a;
            }
        {   /* gsT [o][s] -> orden de grupo de 4 muestras */
            size_t ng = (S + 3) / 4;
            float *gsG = calloc(ng * outd * 4 + 1, 4);
            float *wsc = malloc(TERNARY_GRADX_OB * 8 * 4);
            for (size_t s = 0; s < S; s++) for (size_t o = 0; o < outd; o++) gsG[(s >> 2) * outd * 4 + o * 4 + (s & 3)] = gsT[o * S + s];
            tk_gradx_cols(g2, col, n8, S, outd, gsG, mb, sb, wstride, (const float (*)[8])lm, (const float (*)[8])ls, wsc);
            free(wsc);
            float **g3 = malloc(S * sizeof(float *));
            for (size_t s = 0; s < S; s++) { g3[s] = malloc((col + cols + 8) * 4); memcpy(g3[s], g2[s], 0); }
            /* variante "rows": mismo punto de partida que g1 antes de la referencia -> se recalcula aparte */
            for (size_t s = 0; s < S; s++) for (size_t i = 0; i < col + cols + 8; i++) g3[s][i] = g0[s][i];
            tk_gradx_cols_rows(g3, col, n8, S, outd, gsG, mb, sb, wstride, (const float (*)[8])lm, (const float (*)[8])ls);
            for (size_t s = 0; s < S; s++) { CHECK(memcmp(g1[s], g3[s], (col + cols + 8) * 4) == 0, "gradx_cols_rows distinto de la referencia"); free(g3[s]); }
            free(g3); free(gsG);
        }
        for (size_t s = 0; s < S; s++)
            CHECK(memcmp(g1[s], g2[s], (col + cols + 8) * 4) == 0, "gradx_cols distinto de la referencia");
        for (size_t s = 0; s < S; s++) { free(g1[s]); free(g2[s]); free(g0[s]); }
        free(g0); free(g1); free(g2); free(mb); free(sb); free(gsT);
    }
}

int main(void) {
    for (int b = 0; b < 256; b++) for (int k = 0; k < 8; k++) { lm[b][k] = (float)((b >> k) & 1); ls[b][k] = (float)(1 - 2 * ((b >> k) & 1)); }
    const char *impl = "portable";
    test_expand(); test_gradw(); test_gradx();
    printf("test_kernels [%s]: expand 300, gradw 400, gradx 250 casos -- %s\n", impl, fails ? "FALLAS" : "identico bit a bit a la referencia escalar");
    return fails ? 1 : 0;
}
