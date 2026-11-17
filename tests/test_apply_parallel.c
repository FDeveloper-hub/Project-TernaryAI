/*
 * ternary_dense_apply_matrix_gradients repartido entre hilos (filas en bloques cuyo limite cae en multiplo de 64 bits
 * de W) contra el camino serial fila por fila (ternary_dense_apply_row_gradients): pesos y votos IDENTICOS bit a bit
 * tras varias rondas, en formas con ancho no multiplo de 64 (donde varias filas comparten una palabra de W).
 */
#include "ternary_dense.h"
#include "ternary_threadpool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 88172645463325252ull;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

int main(void) {
    TernaryThreadPool *pool = ternary_threadpool_create(4);
    ternary_set_compute_pool(pool);
    const size_t shapes[][2] = { {1200, 100}, {800, 256}, {1500, 130}, {2000, 64}, {1500, 70}, {400, 500}, {1000, 129} };
    int fails = 0;
    for (size_t sh = 0; sh < sizeof(shapes) / sizeof(shapes[0]); sh++) {
        const size_t rows = shapes[sh][0], n = shapes[sh][1];
        TernaryDense *a = ternary_dense_alloc(n, rows, 0), *b = ternary_dense_alloc(n, rows, 0);
        for (size_t o = 0; o < rows; o++) for (size_t i = 0; i < n; i++) {
            int v = (int)(rnd() % 3) - 1;
            ternary_dense_set_weight(a, o, i, v); ternary_dense_set_weight(b, o, i, v);
        }
        float *g = malloc(rows * n * sizeof(float));
        for (int round = 0; round < 8; round++) {
            for (size_t o = 0; o < rows; o++) {
                float amp = 0.05f + (float)(rnd() % 100) / 50.0f;
                for (size_t i = 0; i < n; i++) g[o * n + i] = ((float)(rnd() % 2001) - 1000.0f) / 1000.0f * amp;
            }
            ternary_dense_apply_matrix_gradients(a, g);
            for (size_t o = 0; o < rows; o++) ternary_dense_apply_row_gradients(b, o, g + o * n);
            if (memcmp(a->W->mask, b->W->mask, a->W->words * 8) || memcmp(a->W->sign, b->W->sign, a->W->words * 8) ||
                memcmp(a->counter_W, b->counter_W, rows * n)) {
                fprintf(stderr, "FALLO: forma %zux%zu ronda %d: pesos o votos distintos del camino serial\n", rows, n, round);
                fails++; break;
            }
        }
        free(g); ternary_dense_free(a); ternary_dense_free(b);
    }
    ternary_set_compute_pool(NULL);
    ternary_threadpool_destroy(pool);
    printf("test_apply_parallel: %s (7 formas, 8 rondas, 4 hilos)\n", fails ? "FALLAS" : "identico bit a bit al camino serial");
    return fails ? 1 : 0;
}
