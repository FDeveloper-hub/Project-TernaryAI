/* Nucleo NEON del motor contra el escalar, por operacion.
 *   bench_infer_kern <matvec|matmul|scores|argmax> <n|s> <reps>      -> ejecuta 'reps' veces (para contar instrucciones bajo qemu: diferencia entre reps)
 *   bench_infer_kern --time                                           -> tabla de tiempos reales neon vs escalar (usar en un dispositivo ARM real) */
#define _POSIX_C_SOURCE 200809L
#include "../src/ternary_infer_internal.h"
#include <stdio.h>
#include <time.h>

static uint64_t rs = 88172645463325252ULL;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 32); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

typedef struct { TernaryInferMat m; uint8_t *q[8]; int32_t zp[8], *z[8]; int32_t *kT; int tp, hd, cnt; int32_t *qq[8], *sc[8]; int64_t *scr; const TernaryInferKernels *k; int rows, cols; } Ctx;

static void setup(Ctx *c, const TernaryInferKernels *k, int rows, int cols) {
    memset(c, 0, sizeof *c); c->k = k; c->rows = rows; c->cols = cols;
    int8_t *w = malloc((size_t)rows * cols); for (int i = 0; i < rows * cols; i++) w[i] = (int8_t)((int)(rnd() % 3) - 1);
    ternaryai_infer_mat_build(&c->m, w, rows, cols, 1.0f, k->layout); free(w);
    for (int b = 0; b < 8; b++) { c->q[b] = malloc((size_t)c->m.cp + 16); for (int i = 0; i < c->m.cp; i++) c->q[b][i] = (uint8_t)rnd(); c->zp[b] = 128; c->z[b] = calloc((size_t)c->m.rb * 16 + 16, 4); }
    c->hd = 64; c->cnt = 512; c->tp = 528;
    c->kT = malloc((size_t)c->hd * c->tp * 4); for (int i = 0; i < c->hd * c->tp; i++) c->kT[i] = (int32_t)(rnd() % 2001) - 1000;
    for (int g = 0; g < 4; g++) { c->qq[g] = malloc(c->hd * 4); for (int d = 0; d < c->hd; d++) c->qq[g][d] = (int32_t)(rnd() % 2001) - 1000; c->sc[g] = malloc((size_t)c->cnt * 4); }
    c->scr = malloc((size_t)c->cnt * 8);
}
static void op(Ctx *c, const char *name) {
    if (!strcmp(name, "matvec")) c->k->matvec(&c->m, c->q[0], c->zp[0], c->z[0]);
    else if (!strcmp(name, "matmul")) { if (c->k->matmul) c->k->matmul(&c->m, (const uint8_t *const *)c->q, c->zp, c->z, 4); else for (int b = 0; b < 4; b++) c->k->matvec(&c->m, c->q[b], c->zp[b], c->z[b]); }
    else if (!strcmp(name, "scores")) c->k->scores(c->kT, c->tp, c->hd, 0, c->cnt, (const int32_t *const *)c->qq, 4, 1, 100000, c->sc, c->scr);
    else if (!strcmp(name, "argmax")) { volatile int r = c->k->argmax(c->z[0], c->rows); (void)r; }
}
int main(int argc, char **argv) {
    const TernaryInferKernels *N = &ternaryai_infer_k_neon, *S = &ternaryai_infer_k_scalar;
    if (argc >= 2 && !strcmp(argv[1], "--scores16")) {          /* puntajes int16 planos: kernel NEON contra el portable del motor (copia de scores16_planar) */
        const TernaryInferKernels *Hy = &ternaryai_infer_k_neon_hybrid;
        const int cfg[][3] = { { 16, 2, 16 }, { 16, 2, 64 }, { 16, 2, 160 }, { 16, 2, 320 }, { 32, 2, 64 }, { 32, 2, 160 }, { 32, 2, 320 }, { 32, 4, 320 }, { 64, 2, 320 } };   /* hd, G, cnt */
        printf("%-22s %12s %12s %8s\n", "forma (hd,G,cnt)", "neon ns", "portable ns", "x");
        for (int ci = 0; ci < 9; ci++) {
            const int hd = cfg[ci][0], G = cfg[ci][1], cnt = cfg[ci][2], tp = 352;
            int16_t *k16 = malloc((size_t)hd * tp * 2); int32_t *q[8], *a[8], *b[8];
            for (int i = 0; i < hd * tp; i++) k16[i] = (int16_t)((i * 37) % 201 - 100);
            for (int g = 0; g < G; g++) { q[g] = malloc(hd * 4); a[g] = malloc((cnt + 8) * 4); b[g] = malloc((cnt + 8) * 4); for (int d = 0; d < hd; d++) q[g][d] = (d * 13 + g) % 201 - 100; }
            const int sh = 17; const int32_t lim = 100000; double t[2];
            for (int w = 0; w < 2; w++) {
                int reps = 1; double el;
                for (;;) {
                    double t0 = now();
                    for (int r = 0; r < reps; r++) {
                        if (w == 0) Hy->scores16p(k16, tp, hd, 0, cnt, (const int32_t *const *)q, G, 1, lim, b);
                        else for (int g = 0; g < G; g++) {
                            int32_t *acc = a[g];
                            for (int j = 0; j < cnt; j++) acc[j] = 0;
                            for (int d = 0; d < hd; d++) { const int32_t qd = (int16_t)q[g][d]; const int16_t *row = k16 + (size_t)d * tp; for (int j = 0; j < cnt; j++) acc[j] += qd * (int32_t)row[j]; }
                            for (int j = 0; j < cnt; j++) { int32_t v = acc[j] >> sh; if (v > lim) v = lim; else if (v < -lim) v = -lim; acc[j] = v; }
                        }
                    }
                    el = now() - t0; if (el > 0.2) break; reps *= 2;
                }
                double best = el / reps;
                for (int r = 0; r < 5; r++) {
                    double t0 = now();
                    for (int i = 0; i < reps; i++) {
                        if (w == 0) Hy->scores16p(k16, tp, hd, 0, cnt, (const int32_t *const *)q, G, 1, lim, b);
                        else for (int g = 0; g < G; g++) {
                            int32_t *acc = a[g];
                            for (int j = 0; j < cnt; j++) acc[j] = 0;
                            for (int d = 0; d < hd; d++) { const int32_t qd = (int16_t)q[g][d]; const int16_t *row = k16 + (size_t)d * tp; for (int j = 0; j < cnt; j++) acc[j] += qd * (int32_t)row[j]; }
                            for (int j = 0; j < cnt; j++) { int32_t v = acc[j] >> sh; if (v > lim) v = lim; else if (v < -lim) v = -lim; acc[j] = v; }
                        }
                    }
                    const double e = (now() - t0) / reps; if (e < best) best = e;
                }
                t[w] = best * 1e9;
            }
            for (int g = 0; g < G; g++) if (memcmp(a[g], b[g], (size_t)cnt * 4)) { printf("DIFERENCIA en hd=%d G=%d cnt=%d\n", hd, G, cnt); return 1; }
            char nm[32]; snprintf(nm, sizeof nm, "hd%d G%d cnt%d", hd, G, cnt);
            printf("%-22s %12.0f %12.0f %8.2f\n", nm, t[0], t[1], t[1] / t[0]);
            for (int g = 0; g < G; g++) { free(q[g]); free(a[g]); free(b[g]); }
            free(k16);
        }
        return 0;
    }
    if (argc >= 2 && !strcmp(argv[1], "--time")) {
        const char *ops[] = { "matvec", "matmul", "scores", "argmax" }; const int shp[][2] = { { 512, 512 }, { 1024, 256 }, { 2048, 1024 } };
        printf("%-8s %-10s %12s %12s %8s\n", "op", "forma", "neon us", "escalar us", "x");
        for (int o = 0; o < 4; o++) for (int s = 0; s < 3; s++) {
            if (o >= 2 && s) continue;
            double t[2];
            for (int w = 0; w < 2; w++) {
                Ctx c; setup(&c, w ? S : N, shp[s][0], shp[s][1]);
                int reps = 1; double el;
                for (;;) { double t0 = now(); for (int i = 0; i < reps; i++) op(&c, ops[o]); el = now() - t0; if (el > 0.2) break; reps *= 2; }
                double best = el / reps;
                for (int r = 0; r < 5; r++) { double t0 = now(); for (int i = 0; i < reps; i++) op(&c, ops[o]); double e = (now() - t0) / reps; if (e < best) best = e; }
                t[w] = best * 1e6;
            }
            printf("%-8s %4dx%-5d %12.1f %12.1f %8.2f\n", ops[o], shp[s][0], shp[s][1], t[0], t[1], t[1] / t[0]);
        }
        return 0;
    }
    if (argc < 4) return 2;
    const int reps = atoi(argv[3]);
    Ctx c; setup(&c, argv[2][0] == 'n' ? N : S, 512, 512);
    for (int i = 0; i < reps; i++) op(&c, argv[1]);
    return 0;
}
