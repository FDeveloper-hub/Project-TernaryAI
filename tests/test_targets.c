#define _POSIX_C_SOURCE 200809L
/*
 * Objetivos con mascara de perdida (ternary_targets.h): casos a mano, contra una referencia ingenua token a token (azar),
 * lote en serie y repartido en hilos, y los nucleos vectoriales de busqueda (AVX-512 / NEON) contra el escalar.
 */
#include "ternary_targets.h"
#include "ternary_threadpool.h"
#include "ternary_neon.h"
#include "ternary_avx512.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { U = 256, A = 257, END = 258, TH = 259, SYS = 260, TH2 = 261, NSPEC = 6 };
static const int8_t KIND[NSPEC] = { TERNARY_TARGETS_MASK, TERNARY_TARGETS_TRAIN, TERNARY_TARGETS_END, TERNARY_TARGETS_OTHER,
                                    TERNARY_TARGETS_MASK, TERNARY_TARGETS_OTHER };

static TernaryTargetsCfg cfg(int unmarked) { TernaryTargetsCfg c = { 256, 256 + NSPEC, KIND, unmarked }; return c; }

static void expect(const int32_t *ids, size_t n, int unmarked, const int32_t *want) {
    int32_t y[256];
    TernaryTargetsCfg c = cfg(unmarked);
    assert(n <= 256);
    for (size_t i = 0; i < n; i++) y[i] = 12345;
    assert(ternary_targets_build(ids, n, &c, y));
    for (size_t i = 0; i < n; i++)
        if (y[i] != want[i]) { printf("FALLA en i=%zu: y=%d esperado=%d\n", i, y[i], want[i]); assert(0); }
}

static void test_manual(void) {
    /* conversacion de dos turnos: solo se entrena el asistente y el marcador que cierra su turno */
    const int32_t c1[] = { U, 10, 11, A, 20, 21, 22, U, 12, A, 23, END };
    const int32_t w1[] = { -1, -1, -1, 20, 21, 22, U, -1, -1, 23, END, -1 };
    expect(c1, 12, 1, w1);
    expect(c1, 12, 0, w1);            /* unmarked_train no cambia nada si hay marcadores desde el principio */
    /* texto plano */
    const int32_t t1[] = { 1, 2, 3, 4 };
    const int32_t wt[] = { 2, 3, 4, -1 }, wm[] = { -1, -1, -1, -1 };
    expect(t1, 4, 1, wt);
    expect(t1, 4, 0, wm);
    /* especiales normales (think) dentro del turno del asistente: se entrenan */
    const int32_t c2[] = { U, 1, A, TH, 5, TH2, 7, END };
    const int32_t w2[] = { -1, -1, TH, 5, TH2, 7, END, -1 };
    expect(c2, 8, 1, w2);
    /* empieza en medio de un turno de asistente (ventana de un ejemplo largo) */
    const int32_t c3[] = { 5, 6, U, 7, A, 8 };
    const int32_t w3[] = { 6, U, -1, -1, 8, -1 };
    expect(c3, 6, 1, w3);
    /* <|end|> devuelve al estado inicial */
    const int32_t c4[] = { A, 5, END, 6, 7 };
    const int32_t w4a[] = { 5, END, 6, 7, -1 }, w4b[] = { 5, END, -1, -1, -1 };
    expect(c4, 5, 1, w4a);
    expect(c4, 5, 0, w4b);
    /* system y otros roles enmascarados; un marcador al principio abre su tramo */
    const int32_t c5[] = { SYS, 1, U, 2, A, 3, END };
    const int32_t w5[] = { -1, -1, -1, -1, 3, END, -1 };
    expect(c5, 7, 1, w5);
    /* n = 0 y n = 1 */
    int32_t y[2] = { 7, 7 };
    TernaryTargetsCfg c = cfg(1);
    assert(ternary_targets_build(c1, 0, &c, y) && y[0] == 7);
    assert(ternary_targets_build(c1, 1, &c, y) && y[0] == -1);
    /* sin especiales en la configuracion */
    TernaryTargetsCfg none = { 0, 0, NULL, 1 };
    int32_t yy[4];
    assert(ternary_targets_build(c1, 4, &none, yy) && yy[0] == 10 && yy[1] == 11 && yy[2] == A && yy[3] == -1);
    printf("[targets] casos a mano (turnos, texto plano, think, ventana, <|end|>, system, n=0/1): OK\n");
}

/* referencia ingenua: una pasada token a token, sin saltos */
static void naive(const int32_t *ids, size_t n, int unmarked, int32_t *y) {
    int cur = unmarked ? 1 : 0;
    if (n == 0) return;
    int kd0 = (ids[0] >= 256 && ids[0] < 256 + NSPEC) ? KIND[ids[0] - 256] : -1;
    if (kd0 == TERNARY_TARGETS_TRAIN) cur = 1; else if (kd0 == TERNARY_TARGETS_MASK) cur = 0; else if (kd0 == TERNARY_TARGETS_END) cur = unmarked ? 1 : 0;
    for (size_t j = 1; j < n; j++) {
        y[j - 1] = cur ? ids[j] : -1;
        int kd = (ids[j] >= 256 && ids[j] < 256 + NSPEC) ? KIND[ids[j] - 256] : -1;
        if (kd == TERNARY_TARGETS_TRAIN) cur = 1; else if (kd == TERNARY_TARGETS_MASK) cur = 0; else if (kd == TERNARY_TARGETS_END) cur = unmarked ? 1 : 0;
    }
    y[n - 1] = -1;
}

static uint32_t rs = 12345;
static uint32_t rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }

static void test_azar(void) {
    for (int it = 0; it < 4000; it++) {
        size_t n = rnd() % 300;
        int32_t *ids = malloc((n + 1) * 4), *y1 = malloc((n + 1) * 4), *y2 = malloc((n + 1) * 4);
        const unsigned dens = 1 + rnd() % 40;                      /* densidad de especiales: de muchos a pocos */
        for (size_t i = 0; i < n; i++) ids[i] = (rnd() % dens == 0) ? (int32_t)(256 + rnd() % NSPEC) : (int32_t)(rnd() % 1000 + (rnd() % 2 ? 262 : 0));
        const int un = (int)(rnd() & 1);
        TernaryTargetsCfg c = cfg(un);
        naive(ids, n, un, y1);
        ternary_targets_build(ids, n, &c, y2);
        assert(n == 0 || memcmp(y1, y2, n * 4) == 0);
        free(ids); free(y1); free(y2);
    }
    printf("[targets] 4000 ejemplos al azar iguales a la referencia token a token: OK\n");
}

static void test_lote_y_hilos(void) {
    const size_t n_ex = 600;
    size_t *offs = malloc((n_ex + 1) * sizeof(size_t));
    offs[0] = 0;
    for (size_t e = 0; e < n_ex; e++) offs[e + 1] = offs[e] + (rnd() % 400);        /* ~120 mil tokens: pasa el umbral de reparto */
    const size_t tot = offs[n_ex];
    assert(tot >= 65536);
    int32_t *ids = malloc(tot * 4 + 4), *ya = malloc(tot * 4 + 4), *yb = malloc(tot * 4 + 4);
    for (size_t i = 0; i < tot; i++) ids[i] = (rnd() % 12 == 0) ? (int32_t)(256 + rnd() % NSPEC) : (int32_t)(rnd() % 900);
    TernaryTargetsCfg c = cfg(1);
    ternary_set_num_threads(1);
    ternary_targets_build_batch(ids, offs, n_ex, &c, ya);
    for (size_t e = 0; e < n_ex; e++) {                                               /* el lote == ejemplo por ejemplo */
        int32_t *t = malloc((offs[e + 1] - offs[e] + 1) * 4);
        naive(ids + offs[e], offs[e + 1] - offs[e], 1, t);
        assert(memcmp(t, ya + offs[e], (offs[e + 1] - offs[e]) * 4) == 0);
        free(t);
    }
    for (int th = 2; th <= 8; th += 3) {
        ternary_set_num_threads(th);
        ternary_targets_build_batch(ids, offs, n_ex, &c, yb);
        assert(memcmp(ya, yb, tot * 4) == 0);
    }
    ternary_set_num_threads(1);
    printf("[targets] lote de %zu ejemplos (%zu tokens): igual al ejemplo por ejemplo y con 1/2/5/8 hilos: OK\n", n_ex, tot);
    free(offs); free(ids); free(ya); free(yb);
}

/* nucleos de busqueda: el despachado (AVX-512 / NEON / escalar) contra la referencia escalar */
static void test_busqueda(void) {
    for (int it = 0; it < 20000; it++) {
        size_t n = rnd() % 200;
        int32_t *ids = malloc((n + 16) * 4);
        const unsigned dens = 1 + rnd() % 200;
        for (size_t i = 0; i < n; i++) ids[i] = (rnd() % dens == 0) ? (int32_t)(250 + rnd() % 20) : (int32_t)(rnd() % 3000) - (rnd() % 7 == 0 ? 3000 : 0);
        int32_t lo = (int32_t)(rnd() % 300) - 20, hi = lo + (int32_t)(rnd() % 40) - 3;     /* incluye rango vacio / invertido */
        if (rnd() % 50 == 0) { lo = -5; hi = 2147483647; }
        if (rnd() % 50 == 0) { lo = -2147483647 - 1; hi = 0; }
        size_t from = n ? rnd() % (n + 1) : 0;
        size_t a = ternary_targets_find_range_scalar(ids, from, n, lo, hi), b = ternary_targets_find_range(ids, from, n, lo, hi);
        if (a != b) { printf("FALLA busqueda: n=%zu from=%zu lo=%d hi=%d scalar=%zu despachado=%zu\n", n, from, lo, hi, a, b); assert(0); }
        if (ternary_avx512_enabled()) {
            size_t c = ternary_avx512_find_range(ids, from, n, lo, hi);
            if (a != c) { printf("FALLA AVX-512: n=%zu from=%zu lo=%d hi=%d scalar=%zu avx512=%zu\n", n, from, lo, hi, a, c); assert(0); }
        }
        if (ternary_neon_enabled()) {
            size_t c = ternary_neon_find_range(ids, from, n, lo, hi);
            if (a != c) { printf("FALLA NEON: n=%zu from=%zu lo=%d hi=%d scalar=%zu neon=%zu\n", n, from, lo, hi, a, c); assert(0); }
        }
        free(ids);
    }
    printf("[targets] 20000 busquedas de rango: despachado%s == escalar: OK\n", ternary_avx512_enabled() ? " y AVX-512" : (ternary_neon_enabled() ? " y NEON" : ""));
}

static void test_busqueda_exhaustiva(void) {
    enum { MAXN = 200 };
    int32_t ids[MAXN + 8];
    for (size_t n = 0; n <= MAXN; n++) {
        for (size_t pos = 0; pos <= n; pos++) {                 /* pos == n: no hay ninguno */
            for (size_t i = 0; i < n; i++) ids[i] = 7;
            if (pos < n) ids[pos] = 258;
            for (size_t from = 0; from <= n; from += (n < 40 ? 1 : 7)) {
                size_t want = pos >= from ? pos : n;
                assert(ternary_targets_find_range_scalar(ids, from, n, 256, 262) == want);
                assert(ternary_targets_find_range(ids, from, n, 256, 262) == want);
                if (ternary_avx512_enabled()) assert(ternary_avx512_find_range(ids, from, n, 256, 262) == want);
                if (ternary_neon_enabled()) assert(ternary_neon_find_range(ids, from, n, 256, 262) == want);
            }
        }
    }
    printf("[targets] busqueda exhaustiva (todas las posiciones y largos hasta %d, cruzando vectores de 16/64): OK\n", MAXN);
}

/* --time: velocidad de la busqueda y del armado completo (ids por segundo), escalar frente al nucleo vectorial */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec; }
static void bench(void) {
    const size_t N = 4u << 20;
    int32_t *ids = malloc(N * 4), *y = malloc(N * 4);
    for (int dens = 8; dens <= 512; dens *= 8) {                 /* un especial cada ~dens ids */
        for (size_t i = 0; i < N; i++) ids[i] = (rnd() % (unsigned)dens == 0) ? (int32_t)(256 + rnd() % NSPEC) : (int32_t)(rnd() % 900);
        size_t sink = 0;
        double t0 = now_s();
        for (int r = 0; r < 20; r++) for (size_t i = 0; i < N;) { i = ternary_targets_find_range_scalar(ids, i, N, 256, 256 + NSPEC) + 1; sink++; }
        double ts = now_s() - t0;
        t0 = now_s();
        for (int r = 0; r < 20; r++) for (size_t i = 0; i < N;) { i = ternary_targets_find_range(ids, i, N, 256, 256 + NSPEC) + 1; sink++; }
        double tv = now_s() - t0;
        TernaryTargetsCfg c = cfg(1);
        t0 = now_s();
        for (int r = 0; r < 20; r++) ternary_targets_build(ids, N, &c, y);
        double tb = now_s() - t0;
        printf("[bench] 1 especial cada ~%3d ids: busqueda escalar %6.0f Mids/s | despachada %6.0f Mids/s (x%.2f) | armado completo %6.0f Mids/s (%d)\n",
               dens, 20.0 * N / ts / 1e6, 20.0 * N / tv / 1e6, ts / tv, 20.0 * N / tb / 1e6, sink > 0);
    }
    free(ids); free(y);
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--time") == 0) { bench(); return 0; }
    test_manual();
    test_azar();
    test_lote_y_hilos();
    test_busqueda();
    test_busqueda_exhaustiva();
    printf("\ntest_targets: TODO OK\n");
    return 0;
}
