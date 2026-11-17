/* Activaciones uint8 con escala y zero-point por fila (include/ternary_u8.h): cuantizador, producto entero exacto
 * contra una referencia elemento a elemento, y Dense con entrada uint8 (forward EXACTO contra una referencia entera;
 * backward contra una referencia en double calculada desde las definiciones). */
#include "ternary_u8.h"
#include "ternary_dense.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fallas = 0;
#define CHECK(c, ...) do { if (!(c)) { fallas++; printf("FALLO (linea %d): ", __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint32_t rs = 12345u;
static uint32_t rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }
static float rndf(void) { return ((float)(rnd() % 20001) / 10000.0f) - 1.0f; }   /* [-1, 1] */

static void test_cuantizador(void) {
    int peor_fuera = 0; double peor_err = 0;
    for (int t = 0; t < 200; t++) {
        size_t rows = 1 + rnd() % 4, n = 1 + rnd() % 150;
        float *x = malloc(rows * n * sizeof(float)), *d = malloc(rows * n * sizeof(float));
        float amp = 0.001f + (float)(rnd() % 1000);
        int modo = (int)(rnd() % 4);   /* 0 mixto, 1 solo positivos, 2 solo negativos, 3 con ceros */
        for (size_t i = 0; i < rows * n; i++) {
            float v = rndf() * amp;
            if (modo == 1) v = fabsf(v);
            if (modo == 2) v = -fabsf(v);
            if (modo == 3 && rnd() % 3 == 0) v = 0.0f;
            x[i] = v;
        }
        TernaryU8 *q = ternary_u8_alloc(rows, n);
        ternary_u8_quantize(q, x);
        ternary_u8_dequant(q, d);
        for (size_t r = 0; r < rows; r++) {
            CHECK(q->zp[r] >= 0 && q->zp[r] <= 255, "zp fuera de rango: %d", q->zp[r]);
            for (size_t i = 0; i < n; i++) {
                double e = fabs((double)x[r * n + i] - (double)d[r * n + i]);
                if (e > q->scale[r] * 0.5001) peor_fuera++;
                if (e > peor_err) peor_err = e;
                if (x[r * n + i] == 0.0f) CHECK(d[r * n + i] == 0.0f, "el 0 no es exacto: %g", d[r * n + i]);
                if (x[r * n + i] > 0.0f && d[r * n + i] < 0.0f) CHECK(0, "cambio de signo (+ -> -)");
                if (x[r * n + i] < 0.0f && d[r * n + i] > 0.0f) CHECK(0, "cambio de signo (- -> +)");
            }
        }
        ternary_u8_free(q); free(x); free(d);
    }
    CHECK(peor_fuera == 0, "%d valores con error mayor que escala/2", peor_fuera);
    /* fila de ceros */
    float z[5] = {0, 0, 0, 0, 0}, dz[5];
    TernaryU8 *q = ternary_u8_alloc(1, 5);
    ternary_u8_quantize(q, z); ternary_u8_dequant(q, dz);
    for (int i = 0; i < 5; i++) CHECK(dz[i] == 0.0f, "fila de ceros");
    CHECK(q->scale[0] == 1.0f && q->zp[0] == 0, "fila de ceros: escala/zp");
    ternary_u8_free(q);
    printf("[u8 cuantizador] 200 casos (mixtos, solo +, solo -, con ceros): 0 cambios de signo, ceros exactos, error max %.3g <= escala/2\n", peor_err);
}

static void test_dot(void) {
    int malos = 0;
    for (int t = 0; t < 3000; t++) {
        size_t n = 1 + rnd() % 300, words = (n + 63) / 64;
        int8_t *w = malloc(n); uint8_t *q = malloc(words * 64);
        uint64_t *wm = calloc(words, 8), *ws = calloc(words, 8);
        for (size_t i = 0; i < words * 64; i++) q[i] = (uint8_t)(rnd() & 255);
        for (size_t i = 0; i < n; i++) {
            int v = (int)(rnd() % 3) - 1; w[i] = (int8_t)v;
            if (v) { wm[i >> 6] |= (uint64_t)1 << (i & 63); if (v < 0) ws[i >> 6] |= (uint64_t)1 << (i & 63); }
        }
        int32_t zp = (int32_t)(rnd() & 255);
        if (ternary_dot_u8_scalar(wm, ws, n, q, zp) != ternary_dot_u8_reference(w, n, q, zp)) malos++;
        free(w); free(q); free(wm); free(ws);
    }
    CHECK(malos == 0, "%d productos distintos de la referencia", malos);
    printf("[u8 producto] 3000 casos (n 1..300, zp 0..255): 0 discrepancias contra la referencia elemento a elemento\n");
}

static void dense_u8_caso(size_t in, size_t out, int bias) {
    TernaryDense *L = ternary_dense_alloc(in, out, bias);
    L->scale = 0.37f;
    for (size_t o = 0; o < out; o++)
        for (size_t i = 0; i < in; i++) ternary_dense_set_weight(L, o, i, (int)(rnd() % 3) - 1);
    if (bias) for (size_t o = 0; o < out; o++) ternary_vec_set(L->bias, o, (int)(rnd() % 3) - 1);
    float *x = malloc(in * sizeof(float));
    for (size_t i = 0; i < in; i++) x[i] = rndf() * 3.0f + 0.2f;
    TernaryU8 *q = ternary_u8_alloc(1, in);
    ternary_u8_quantize(q, x);

    float *o1 = malloc(out * sizeof(float)); int32_t *raw = malloc(out * sizeof(int32_t));
    ternary_dense_forward_u8(L, q, o1, raw);
    const float k = L->scale * q->scale[0];
    int malos = 0;
    for (size_t o = 0; o < out; o++) {
        int64_t z = 0;
        for (size_t i = 0; i < in; i++) z += (int64_t)ternary_dense_get_weight(L, o, i) * ((int64_t)q->q[i] - q->zp[0]);
        if (bias) z += ternary_vec_get(L->bias, o);
        if ((int32_t)z != raw[o] || o1[o] != k * (float)(int32_t)z) malos++;
    }
    CHECK(malos == 0, "forward uint8 in=%zu out=%zu: %d salidas distintas de la referencia entera", in, out, malos);

    /* backward contra double desde las definiciones */
    float *g = malloc(out * sizeof(float));
    for (size_t o = 0; o < out; o++) g[o] = rndf();
    float *gw = calloc(in * out, sizeof(float)), *gx = calloc(in, sizeof(float)), *gb = bias ? calloc(out, sizeof(float)) : NULL;
    float gs = 0.0f;
    ternary_dense_backward_u8(L, q, g, gw, gx, &gs, gb, raw);
    float *xd = malloc(in * sizeof(float)); ternary_u8_dequant(q, xd);
    double max_rel = 0.0; double gs_ref = 0.0;
    for (size_t o = 0; o < out; o++) {
        gs_ref += (double)g[o] * (double)q->scale[0] * (double)raw[o];
        if (bias) { double e = fabs(gb[o] - (double)g[o] * q->scale[0]); if (e > 1e-6 * (1 + fabs(gb[o]))) malos++; }
        for (size_t i = 0; i < in; i++) {
            double ref = (double)g[o] * (double)xd[i];
            double e = fabs(gw[o * in + i] - ref) / (1e-6 + fabs(ref));
            if (e > max_rel) max_rel = e;
        }
    }
    for (size_t i = 0; i < in; i++) {
        double ref = 0;
        for (size_t o = 0; o < out; o++) ref += (double)g[o] * L->scale * ternary_dense_get_weight(L, o, i);
        double e = fabs(gx[i] - ref) / (1e-5 + fabs(ref));
        if (e > max_rel) max_rel = e;
    }
    CHECK(max_rel < 1e-4, "backward uint8 in=%zu out=%zu: error relativo max %.3g", in, out, max_rel);
    CHECK(fabs(gs - gs_ref) <= 1e-4 * (1 + fabs(gs_ref)), "grad_escala %g vs %g", gs, gs_ref);
    CHECK(malos == 0, "bias");
    /* z_raw_opt = NULL (recalcula) da lo mismo */
    float *gw2 = calloc(in * out, sizeof(float)), *gx2 = calloc(in, sizeof(float)); float gs2 = 0.0f;
    float *gb2 = bias ? calloc(out, sizeof(float)) : NULL;
    ternary_dense_backward_u8(L, q, g, gw2, gx2, &gs2, gb2, NULL);
    CHECK(memcmp(gw, gw2, in * out * sizeof(float)) == 0 && memcmp(gx, gx2, in * sizeof(float)) == 0 && gs == gs2,
          "backward con z_raw y recalculado difieren");
    free(gw2); free(gx2); free(gb2);
    free(gw); free(gx); free(gb); free(g); free(xd); free(o1); free(raw); free(x);
    ternary_u8_free(q); ternary_dense_free(L);
}

static void test_dense(void) {
    const size_t dims[][2] = { {64, 8}, {128, 16}, {70, 9}, {5, 3}, {200, 33}, {1, 4}, {63, 7}, {65, 5} };
    for (size_t d = 0; d < sizeof(dims) / sizeof(dims[0]); d++)
        for (int b = 0; b < 2; b++) dense_u8_caso(dims[d][0], dims[d][1], b);
    printf("[u8 dense] 8 formas (alineadas y no) x con/sin bias: forward exacto contra la referencia entera; backward contra double\n");
}

int main(void) {
    test_cuantizador();
    test_dot();
    test_dense();
    printf("fallas: %d\n", fallas);
    if (fallas == 0) printf("TODAS LAS PRUEBAS DE ACTIVACIONES UINT8 PASARON\n");
    return fallas != 0;
}
