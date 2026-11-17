/* Escalado de GQA con la longitud: tiempo de forward+backward y memoria por token.
 * Uso: bench_gqa_scale T W SINKS   (W=0: atencion completa). Imprime: T W t_fwd t_bwd bytes_capa bytes_por_token rss_kb */
#define _POSIX_C_SOURCE 200809L
#include "ternary_gqa.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
int main(int argc, char **argv) {
    if (argc < 4) return 2;
    const size_t T = strtoull(argv[1], 0, 10), W = strtoull(argv[2], 0, 10), S = strtoull(argv[3], 0, 10);
    const size_t E = 128, NH = 8, NKV = 2;
    TernaryGQA *g = ternary_gqa_alloc_window(E, T, NH, NKV, W, S);
    if (!g || ternary_gqa_enable_rope_int(g, 10.0f, 2) != 0) { puts("alloc fallo"); return 1; }
    srand(1);
    TernaryDense *ws[4] = {g->wq, g->wk, g->wv, g->wo};
    for (int m = 0; m < 4; m++) for (size_t o = 0; o < ws[m]->output_dim; o++) for (size_t i = 0; i < ws[m]->input_dim; i++)
        ternary_dense_set_weight(ws[m], o, i, (rand() % 3) - 1);
    TernaryVec *x = ternary_vec_alloc(T * E);
    for (size_t i = 0; i < T * E; i++) ternary_vec_set(x, i, (rand() % 3) - 1);
    float *out = (float *)malloc(T * E * 4), *go = (float *)malloc(T * E * 4), *gx = (float *)calloc(T * E, 4);
    for (size_t i = 0; i < T * E; i++) go[i] = (float)((int)(i % 11) - 5) * 0.1f;
    float *gw[4]; for (int m = 0; m < 4; m++) gw[m] = (float *)malloc(ws[m]->output_dim * ws[m]->input_dim * 4);
    float gs[4] = {0};
    double t0 = now(); ternary_gqa_forward(g, x, out, NULL); double t1 = now();
    ternary_gqa_backward(g, x, go, gw, gx, gs, NULL); double t2 = now();
    /* memoria PROPIA de la capa GQA (sin pesos, sin entrada/salida del llamador): lo que reserva ternary_gqa_alloc* */
    size_t qw = NH * (E / NH), kvw = NKV * (E / NH), bytes;
    if (W) {
        size_t slots = g->w_C0 + g->w_Lb + 16;
        bytes = slots * (kvw * 4 * 5 + qw * 8 + kvw * 4 + (g->w_kT ? kvw * 8 : 0)) /*kr,vf,kz,vz,gk,gv + qz,gq*/ + g->chunk * (qw * 8 + E * 4) + g->w_n * 12 + T * qw / 4;
    } else {
        bytes = T * (qw * 4 * 5 + kvw * 4 * 6 + E * 4 + NH * 8 + 8) + T * qw / 4 + T * (qw + 2 * kvw) / 4;
    }
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    printf("%zu %zu %.4f %.4f %zu %.1f %ld\n", T, W, t1 - t0, t2 - t1, bytes, (double)bytes / T, ru.ru_maxrss);
    return 0;
}
