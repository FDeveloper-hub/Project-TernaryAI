/*
 * Leyes de escala de GQA con ventana (el objetivo: contextos de millones de tokens):
 *  - MEMORIA: lo que reserva la capa crece ~32 bytes por token (los 2 bits por elemento de la salida de atencion que
 *    el backward necesita), no kilobytes: se mide con mallinfo2 (heap + mmap en uso) entre dos longitudes.
 *  - TIEMPO: con ventana, duplicar T duplica el tiempo (pendiente log-log ~1); sin ventana lo cuadruplica (~2).
 * Las cotas son holgadas (el reloj de una maquina compartida es ruidoso) pero separan 1 de 2 con claridad.
 */
#define _POSIX_C_SOURCE 200809L
#include "ternary_gqa.h"
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("FALLA: " __VA_ARGS__); printf("\n"); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
#if defined(__ANDROID__)      /* bionic: mallinfo2 no existe en las APIs bajas */
static size_t heap_in_use(void) { struct mallinfo m = mallinfo(); return (size_t)m.uordblks + (size_t)m.hblkhd; }
#else
static size_t heap_in_use(void) { struct mallinfo2 m = mallinfo2(); return m.uordblks + m.hblkhd; }
#endif

#define E 128
#define NH 8
#define NKV 2

static TernaryGQA *make(size_t T, size_t W, size_t S) {
    TernaryGQA *g = ternary_gqa_alloc_window(E, T, NH, NKV, W, S);
    if (!g || ternary_gqa_enable_rope_int(g, 10.0f, 2) != 0) { printf("alloc fallo\n"); exit(1); }
    srand(1);
    TernaryDense *ws[4] = {g->wq, g->wk, g->wv, g->wo};
    for (int m = 0; m < 4; m++) for (size_t o = 0; o < ws[m]->output_dim; o++) for (size_t i = 0; i < ws[m]->input_dim; i++)
        ternary_dense_set_weight(ws[m], o, i, (rand() % 3) - 1);
    return g;
}

/* tiempo de forward+backward (mejor de 2) */
static double run(size_t T, size_t W, size_t S) {
    TernaryGQA *g = make(T, W, S);
    TernaryVec *x = ternary_vec_alloc(T * E);
    for (size_t i = 0; i < T * E; i++) ternary_vec_set(x, i, (rand() % 3) - 1);
    float *out = (float *)malloc(T * E * 4), *go = (float *)malloc(T * E * 4), *gx = (float *)calloc(T * E, 4);
    for (size_t i = 0; i < T * E; i++) go[i] = (float)((int)(i % 11) - 5) * 0.1f;
    TernaryDense *ws[4] = {g->wq, g->wk, g->wv, g->wo};
    float *gw[4]; for (int m = 0; m < 4; m++) gw[m] = (float *)malloc(ws[m]->output_dim * ws[m]->input_dim * 4);
    float gs[4] = {0};
    double best = 1e30;
    for (int r = 0; r < 2; r++) {
        double t0 = now();
        ternary_gqa_forward(g, x, out, NULL);
        ternary_gqa_backward(g, x, go, gw, gx, gs, NULL);
        const double dt = now() - t0;
        if (dt < best) best = dt;
    }
    for (int m = 0; m < 4; m++) free(gw[m]);
    free(out); free(go); free(gx); ternary_vec_free(x); ternary_gqa_free(g);
    return best;
}

static size_t layer_bytes(size_t T, size_t W, size_t S) {
    const size_t b0 = heap_in_use();
    TernaryGQA *g = make(T, W, S);
    const size_t b1 = heap_in_use();
    ternary_gqa_free(g);
    return b1 - b0;
}

int main(void) {
    /* memoria: bytes por token marginales, ventana vs completa */
    const size_t W = 128, S = 4;
    const size_t m1 = layer_bytes(16384, W, S), m2 = layer_bytes(65536, W, S), m3 = layer_bytes(262144, W, S);
    const double bpt_w = (double)(m3 - m2) / (262144 - 65536);
    const size_t f1 = layer_bytes(2048, 0, 0), f2 = layer_bytes(8192, 0, 0);
    const double bpt_f = (double)(f2 - f1) / (8192 - 2048);
    printf("[memoria] con ventana W=%zu sinks=%zu: %.1f bytes/token marginales (capa a 16K: %.2f MB, a 64K: %.2f MB, a 256K: %.2f MB)\n",
           W, S, bpt_w, m1 / 1048576.0, m2 / 1048576.0, m3 / 1048576.0);
    printf("[memoria] atencion completa: %.0f bytes/token marginales\n", bpt_f);
    CHECK(bpt_w < 40.0, "la capa con ventana deberia usar <40 bytes/token, usa %.1f", bpt_w);
    CHECK(bpt_f > 50.0 * bpt_w, "la atencion completa deberia usar mucho mas por token (%.0f vs %.1f)", bpt_f, bpt_w);

    /* tiempo */
    double tw[3], Tw[3] = {4096, 8192, 16384};
    for (int k = 0; k < 3; k++) tw[k] = run((size_t)Tw[k], W, S);
    const double slope_w = log(tw[2] / tw[0]) / log(Tw[2] / Tw[0]);
    double tf[3], Tf[3] = {512, 1024, 2048};
    for (int k = 0; k < 3; k++) tf[k] = run((size_t)Tf[k], 0, 0);
    const double slope_f = log(tf[2] / tf[0]) / log(Tf[2] / Tf[0]);
    printf("[tiempo] con ventana: T=4096/8192/16384 -> %.3f/%.3f/%.3f s, pendiente log-log = %.2f (lineal = 1)\n", tw[0], tw[1], tw[2], slope_w);
    printf("[tiempo] completa:    T=512/1024/2048   -> %.3f/%.3f/%.3f s, pendiente log-log = %.2f (cuadratica = 2)\n", tf[0], tf[1], tf[2], slope_f);
    CHECK(slope_w < 1.25, "con ventana el tiempo deberia ser ~lineal, pendiente %.2f", slope_w);
    CHECK(slope_f > 1.5, "sin ventana el tiempo deberia ser ~cuadratico, pendiente %.2f", slope_f);
    if (fails) { printf("%d FALLAS\n", fails); return 1; }
    printf("OK test_gqa_scaling\n");
    return 0;
}
