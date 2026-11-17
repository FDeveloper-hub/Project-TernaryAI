/*
 * Regla R2 (ver include/ternary_dense.h). Verifica:
 *  1. la fila (ternary_r2_update_row, AVX-512 si esta activo) contra una referencia escrita aparte desde la definicion:
 *     pesos y contadores identicos tras varias rondas (anchos 7, 40, 64, 130; tau/beta/inc/objetivo de ceros/reinicio distintos);
 *  2. invariancia de escala: multiplicar el gradiente por 1024 (exacto en float) no cambia nada;
 *  3. casos limite: fila de ceros (el contador solo decae), gradientes enormes (recorte a +-3), NaN/Inf sin romper nada,
 *     el contador siempre en [-127, 127] y el peso siempre en {-1,0,1};
 *  4. histeresis: tras un volteo el contador queda en el lado contrario; el objetivo de ceros baja los ceros de una fila muy dispersa;
 *  5. Dense: apply_matrix_gradients repartido entre 4 hilos == fila por fila, bit a bit, en formas con ancho no multiplo de 64;
 *  6. Embedding: la regla se aplica por fila igual que en Dense;
 *  7. un stack nuevo arranca en R2 (0.7/0.999); la regla de un stack rige solo mientras dura ternary_stack_apply_gradients.
 */
#include "ternary_dense.h"
#include "ternary_embedding.h"
#include "ternary_stack.h"
#include "ternary_threadpool.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fallas = 0;
#define CHECK(c, ...) do { if (!(c)) { fallas++; printf("FALLO (linea %d): ", __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint64_t rs = 88172645463325252ull;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static float rndf(void) { return (float)((double)(rnd() >> 11) / 9007199254740992.0 * 2.0 - 1.0); }

/* Referencia: la definicion de la regla sobre arreglos planos (aritmetica float como el codigo: los umbrales y el contador dependen de ella). */
static void ref_row(int8_t *w, int *c, const float *g, size_t n, float tau, float beta, float inc, float zt, float rst) {
    double ss = 0;
    for (size_t i = 0; i < n; i++) ss += (double)g[i] * (double)g[i];
    const float inv = 1.0f / ((float)sqrt(ss / (double)n) + 1e-12f), cg = inc * 127.0f;
    size_t nz = 0; for (size_t i = 0; i < n; i++) nz += w[i] == 0;
    float fin = 1.0f, fout = 1.0f;
    if (zt >= 0.0f) { const float d = ((float)nz / (float)n - zt) * 2.0f; float a = 1.0f - d, b = 1.0f + d;
        a = a < 0.4f ? 0.4f : (a > 1.6f ? 1.6f : a); b = b < 0.6f ? 0.6f : (b > 2.5f ? 2.5f : b); fin = a; fout = b; }
    int t0 = (int)lrintf(tau * 127.0f), t1 = (int)lrintf(tau * fin * 127.0f), t2 = (int)lrintf(tau * fout * 127.0f);
    if (t0 < 1) t0 = 1;
    if (t0 > 127) t0 = 127;
    if (t1 < 1) t1 = 1;
    if (t1 > 127) t1 = 127;
    if (t2 < 1) t2 = 1;
    if (t2 > 127) t2 = 127;
    const int ri = (int)lrintf(rst * (float)t0);
    for (size_t i = 0; i < n; i++) {
        float gh = g[i] * inv; if (gh > 3.0f) gh = 3.0f; if (gh < -3.0f) gh = -3.0f;
        float x = beta * (float)c[i] - cg * gh; if (!(x == x)) x = 0.0f;
        int v = (int)nearbyintf(x); if (v > 127) v = 127; if (v < -127) v = -127;
        const int cur = w[i], dir = v > 0 ? 1 : -1, th = cur == 0 ? t1 : (dir == -cur ? t2 : t0);
        if (v >= th || v <= -th) { int nw = cur + dir; if (nw > 1) nw = 1; if (nw < -1) nw = -1; w[i] = (int8_t)nw; v = -dir * ri; }
        c[i] = v;
    }
}

static void test_rows(size_t n, float tau, float beta, float inc, float zt, float rst) {
    ternary_r2_set_params(inc, zt, rst);
    const size_t rows = 8;
    TernaryDense *d = ternary_dense_alloc(n, rows, 0);
    int8_t *w = malloc(rows * n); int *c = calloc(rows * n, sizeof(int));
    for (size_t o = 0; o < rows; o++) for (size_t i = 0; i < n; i++) { int v = (int)(rnd() % 3) - 1; ternary_dense_set_weight(d, o, i, v); w[o * n + i] = (int8_t)v; }
    TernaryUpdateRule r = { tau, beta, inc, zt, rst, 0.0f };
    ternary_update_rule_set(&r);
    float *g = malloc(rows * n * sizeof(float));
    size_t mism = 0, changed = 0;
    for (int round = 0; round < 40; round++) {
        for (size_t o = 0; o < rows; o++) { float amp = 0.001f + (float)(rnd() % 1000) / 100.0f; for (size_t i = 0; i < n; i++) g[o * n + i] = (rndf() + ((i + o) % 3 == 0 ? 1.5f : ((i + o) % 3 == 1 ? -1.5f : 0.0f))) * amp; }
        for (size_t o = 0; o < rows; o++) {
            int8_t before[512]; memcpy(before, w + o * n, n);
            ternary_dense_apply_row_gradients(d, o, g + o * n);
            ref_row(w + o * n, c + o * n, g + o * n, n, tau, beta, inc, zt, rst);
            for (size_t i = 0; i < n; i++) changed += before[i] != w[o * n + i];
        }
    }
    for (size_t o = 0; o < rows; o++) for (size_t i = 0; i < n; i++) {
        mism += ternary_dense_get_weight(d, o, i) != w[o * n + i];
        mism += d->counter_W[o * n + i] != (int8_t)c[o * n + i];
        CHECK(d->counter_W[o * n + i] >= -127 && d->counter_W[o * n + i] <= 127, "contador fuera de rango");
    }
    CHECK(mism == 0, "n=%zu tau=%.2f beta=%.3f: %zu diferencias contra la referencia", n, tau, beta, mism);
    CHECK(changed > 0, "ningun peso cambio (la prueba no ejercita la regla)");
    printf("  fila n=%zu tau=%.2f beta=%.3f inc=%.2f zt=%.2f rst=%.2f: 40 rondas x 8 filas, %zu cambios de peso, %zu diferencias\n", n, tau, beta, inc, zt, rst, changed, mism);
    ternary_update_rule_set(NULL);
    free(g); free(w); free(c); ternary_dense_free(d);
}

static void test_scale_and_limits(void) {
    const size_t n = 70, rows = 6;
    TernaryDense *a = ternary_dense_alloc(n, rows, 0), *b = ternary_dense_alloc(n, rows, 0);
    for (size_t o = 0; o < rows; o++) for (size_t i = 0; i < n; i++) { int v = (int)(rnd() % 3) - 1; ternary_dense_set_weight(a, o, i, v); ternary_dense_set_weight(b, o, i, v); }
    TernaryUpdateRule r = { 0.7f, 0.999f, 0.03f, 0.2f, 0.5f, 0.0f };
    ternary_update_rule_set(&r); ternary_r2_set_params(0.03f, 0.2f, 0.5f);
    float *g = malloc(n * sizeof(float)), *g2 = malloc(n * sizeof(float));
    for (int round = 0; round < 10; round++) {
        for (size_t o = 0; o < rows; o++) {
            for (size_t i = 0; i < n; i++) { g[i] = rndf() * 0.37f; g2[i] = g[i] * 1024.0f; }
            ternary_dense_apply_row_gradients(a, o, g);
            ternary_dense_apply_row_gradients(b, o, g2);
        }
    }
    CHECK(!memcmp(a->W->mask, b->W->mask, a->W->words * 8) && !memcmp(a->W->sign, b->W->sign, a->W->words * 8) &&
          !memcmp(a->counter_W, b->counter_W, rows * n), "la regla debe ser invariante a la escala del gradiente (x1024)");

    /* limites: ceros, enormes, NaN, Inf */
    TernaryDense *z = ternary_dense_alloc(n, 4, 0);
    memset(z->counter_W, 7, 4 * n);
    for (size_t o = 0; o < 4; o++) for (size_t i = 0; i < n; i++) ternary_dense_set_weight(z, o, i, 0);
    for (size_t i = 0; i < n; i++) g[i] = 0.0f;
    ternary_dense_apply_row_gradients(z, 0, g);
    int ok0 = 1; for (size_t i = 0; i < n; i++) ok0 &= (z->counter_W[i] == 7) && ternary_dense_get_weight(z, 0, i) == 0;   /* rint(0.999*7) = 7: con enteros no decae, y sin gradiente no hay movimiento */
    CHECK(ok0, "fila de ceros: el contador no debe moverse (7) ni mover pesos");
    for (size_t i = 0; i < n; i++) g[i] = (i % 2 ? 1.0f : -1.0f) * 1e30f;
    ternary_dense_apply_row_gradients(z, 1, g);
    for (size_t i = 0; i < n; i++) g[i] = (i == 3) ? NAN : (i == 5 ? INFINITY : 0.25f);
    ternary_dense_apply_row_gradients(z, 2, g);
    int rango = 1;
    for (size_t o = 0; o < 4; o++) for (size_t i = 0; i < n; i++) {
        rango &= z->counter_W[o * n + i] >= -127;
        int wv = ternary_dense_get_weight(z, o, i); rango &= wv >= -1 && wv <= 1;
    }
    CHECK(rango, "contador o peso fuera de rango tras gradientes enormes / NaN / Inf");
    ternary_update_rule_set(NULL);
    free(g); free(g2); ternary_dense_free(a); ternary_dense_free(b); ternary_dense_free(z);
    printf("  invariancia de escala (x1024), fila de ceros, gradientes enormes, NaN/Inf: OK\n");
}

/* Histeresis y objetivo de ceros. */
static void test_hysteresis_and_zeros(void) {
    const size_t n = 32;
    TernaryDense *d = ternary_dense_alloc(n, 1, 0);
    for (size_t i = 0; i < n; i++) ternary_dense_set_weight(d, 0, i, 0);
    TernaryUpdateRule r = { 0.7f, 0.999f, 0.03f, -1.0f, 0.5f, 0.0f };
    ternary_update_rule_set(&r);
    float g[32]; for (size_t i = 0; i < n; i++) g[i] = -1.0f;     /* gradiente negativo constante: el peso sube */
    int pasos = 0;
    while (ternary_dense_get_weight(d, 0, 0) == 0 && pasos < 1000) { ternary_dense_apply_row_gradients(d, 0, g); pasos++; }
    CHECK(ternary_dense_get_weight(d, 0, 0) == 1, "el peso debe subir a +1 con gradiente negativo sostenido");
    /* sin decaimiento: |c| llega a T=89 con paso 0.03*127*1 = 3.81 -> ~24 pasos */
    CHECK(pasos >= 20 && pasos <= 30, "pasos hasta el primer volteo: %d (esperado ~24)", pasos);
    CHECK(d->counter_W[0] == -(int)lrintf(0.5f * 89.0f), "tras el volteo el contador debe quedar en el lado contrario (-%d), esta en %d", (int)lrintf(0.5f * 89.0f), d->counter_W[0]);
    ternary_dense_free(d);
    /* objetivo de ceros: una fila de ceros con objetivo 0.2 baja el umbral de entrada: se activan antes que sin objetivo */
    int p_sin = 0, p_con = 0;
    for (int k = 0; k < 2; k++) {
        ternary_r2_set_params(0.03f, k ? 0.2f : -1.0f, 0.5f);
        TernaryDense *z = ternary_dense_alloc(n, 1, 0);
        for (size_t i = 0; i < n; i++) ternary_dense_set_weight(z, 0, i, 0);
        int p = 0; while (ternary_dense_get_weight(z, 0, 0) == 0 && p < 1000) { ternary_dense_apply_row_gradients(z, 0, g); p++; }
        if (k) p_con = p; else p_sin = p;
        ternary_dense_free(z);
    }
    CHECK(p_con < p_sin, "con objetivo de ceros 0.2 una fila de ceros debe activarse antes (%d frente a %d)", p_con, p_sin);
    ternary_update_rule_set(NULL); ternary_r2_set_params(0.03f, 0.2f, 0.5f);
    printf("  histeresis (reinicio al lado contrario) y objetivo de ceros: OK\n");
}

static void test_parallel(void) {
    TernaryThreadPool *pool = ternary_threadpool_create(4);
    ternary_set_compute_pool(pool);
    const size_t shapes[][2] = { {1200, 100}, {800, 256}, {1500, 130}, {2000, 64}, {1500, 70}, {400, 500} };
    TernaryUpdateRule r = { 0.7f, 0.999f, 0.03f, 0.2f, 0.5f, 0.0f };
    ternary_update_rule_set(&r); ternary_r2_set_params(0.03f, 0.2f, 0.5f);
    for (size_t sh = 0; sh < sizeof(shapes) / sizeof(shapes[0]); sh++) {
        const size_t rows = shapes[sh][0], n = shapes[sh][1];
        TernaryDense *a = ternary_dense_alloc(n, rows, 0), *b = ternary_dense_alloc(n, rows, 0);
        for (size_t o = 0; o < rows; o++) for (size_t i = 0; i < n; i++) { int v = (int)(rnd() % 3) - 1; ternary_dense_set_weight(a, o, i, v); ternary_dense_set_weight(b, o, i, v); }
        float *g = malloc(rows * n * sizeof(float));
        int igual = 1; size_t moved = 0;
        for (int round = 0; round < 6 && igual; round++) {
            for (size_t o = 0; o < rows; o++) { float amp = 0.05f + (float)(rnd() % 100) / 50.0f; for (size_t i = 0; i < n; i++) g[o * n + i] = rndf() * amp; }
            ternary_dense_apply_matrix_gradients(a, g);
            for (size_t o = 0; o < rows; o++) ternary_dense_apply_row_gradients(b, o, g + o * n);
            igual = !memcmp(a->W->mask, b->W->mask, a->W->words * 8) && !memcmp(a->W->sign, b->W->sign, a->W->words * 8) && !memcmp(a->counter_W, b->counter_W, rows * n);
        }
        for (size_t k = 0; k < rows * n; k++) moved += a->counter_W[k] != 0;
        CHECK(igual, "forma %zux%zu: 4 hilos distinto de fila por fila", rows, n);
        CHECK(moved > 0, "forma %zux%zu: contadores todos en 0", rows, n);
        free(g); ternary_dense_free(a); ternary_dense_free(b);
    }
    ternary_update_rule_set(NULL);
    ternary_set_compute_pool(NULL);
    ternary_threadpool_destroy(pool);
    printf("  Dense 4 hilos == fila por fila (6 formas, 6 rondas): OK\n");
}

static void test_embedding(void) {
    const size_t V = 20, E = 40;
    TernaryEmbedding *l = ternary_embedding_alloc(V, E, 1);
    int8_t *w = malloc(V * E); int *c = calloc(V * E, sizeof(int));
    for (size_t v = 0; v < V; v++) for (size_t d = 0; d < E; d++) { int x = (int)(rnd() % 3) - 1; ternary_embedding_set_weight(l, v, d, x); w[v * E + d] = (int8_t)x; }
    TernaryUpdateRule r = { 0.7f, 0.999f, 0.03f, 0.2f, 0.5f, 0.0f };
    ternary_update_rule_set(&r); ternary_r2_set_params(0.03f, 0.2f, 0.5f);
    float g[40]; size_t mism = 0;
    for (int round = 0; round < 12; round++) for (size_t v = 0; v < V; v++) {
        float amp = 0.01f + (float)(rnd() % 500) / 100.0f;
        for (size_t d = 0; d < E; d++) g[d] = rndf() * amp;
        ternary_embedding_apply_row_gradients(l, v, g);
        ref_row(w + v * E, c + v * E, g, E, 0.7f, 0.999f, 0.03f, 0.2f, 0.5f);
    }
    for (size_t v = 0; v < V; v++) for (size_t d = 0; d < E; d++) { mism += ternary_embedding_get_weight(l, v, d) != w[v * E + d]; mism += l->counter_W[v * E + d] != (int8_t)c[v * E + d]; }
    CHECK(mism == 0, "embedding: %zu diferencias contra la referencia", mism);
    ternary_update_rule_set(NULL);
    free(w); free(c); ternary_embedding_free(l);
    printf("  Embedding contra la referencia: OK\n");
}

static void test_rule_scope(void) {
    /* el default del hilo es R2 (tau 0.7, beta 0.999): NULL y la regla explicita coinciden */
    TernaryUpdateRule d = ternary_update_rule_get();
    CHECK(d.tau == 0.7f && d.beta == 0.999f, "el default del hilo debe ser R2 (tau 0.7, beta 0.999)");
    const size_t n = 50, rows = 6;
    TernaryDense *a = ternary_dense_alloc(n, rows, 0), *b = ternary_dense_alloc(n, rows, 0);
    for (size_t o = 0; o < rows; o++) for (size_t i = 0; i < n; i++) { int v = (int)(rnd() % 3) - 1; ternary_dense_set_weight(a, o, i, v); ternary_dense_set_weight(b, o, i, v); }
    float *g = malloc(rows * n * sizeof(float));
    TernaryUpdateRule rx = { 0.7f, 0.999f, 0.03f, 0.2f, 0.5f, 0.0f };
    for (int round = 0; round < 10; round++) {
        for (size_t k = 0; k < rows * n; k++) g[k] = rndf();
        ternary_update_rule_set(NULL); ternary_dense_apply_matrix_gradients(a, g);
        ternary_update_rule_set(&rx);  ternary_dense_apply_matrix_gradients(b, g);
    }
    ternary_update_rule_set(NULL);
    CHECK(!memcmp(a->counter_W, b->counter_W, rows * n) && !memcmp(a->W->mask, b->W->mask, a->W->words * 8), "NULL y la regla explicita deben coincidir");

    /* la regla de un stack rige solo durante ternary_stack_apply_gradients */
    TernaryStack *s = ternary_stack_create();
    { TernaryUpdateRule d0 = ternary_stack_get_update_rule(s);
      CHECK(d0.tau == 0.7f && d0.beta == 0.999f, "un stack nuevo usa R2 (tau 0.7, beta 0.999)"); }
    ternary_stack_set_update_rule(s, 0.6f, 0.8f);
    TernaryUpdateRule got = ternary_stack_get_update_rule(s);
    CHECK(got.tau == 0.6f && got.beta == 0.8f, "set/get de la regla del stack");
    TernaryUpdateRule prev = { 0.9f, 0.5f, 0.03f, 0.2f, 0.5f, 0.0f };
    ternary_update_rule_set(&prev);
    ternary_stack_apply_gradients(s, 0.02f);   /* stack vacio: solo prueba que la regla del hilo se restaura */
    CHECK(ternary_update_rule_get().tau == 0.9f && ternary_update_rule_get().beta == 0.5f, "la regla previa del hilo se restaura");
    ternary_update_rule_set(NULL);
    ternary_stack_free(s);
    free(g); ternary_dense_free(a); ternary_dense_free(b);
    printf("  default R2, regla del stack acotada a apply_gradients: OK\n");
}

int main(void) {
    printf("-- regla R2\n");
    test_rows(7, 0.7f, 0.999f, 0.03f, 0.2f, 0.5f); test_rows(40, 0.7f, 0.999f, 0.03f, 0.2f, 0.5f); test_rows(64, 0.5f, 0.99f, 0.05f, 0.3f, 0.25f);
    test_rows(130, 0.7f, 0.999f, 0.03f, -1.0f, 0.5f); test_rows(130, 0.6f, 0.9f, 0.08f, 0.2f, 1.0f);
    test_hysteresis_and_zeros();
    test_scale_and_limits();
    test_parallel();
    test_embedding();
    test_rule_scope();
    if (fallas) { printf("%d fallas\n", fallas); return 1; }
    printf("OK\n");
    return 0;
}
