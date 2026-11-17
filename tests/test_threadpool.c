#include "ternary_dense.h"
#include "ternary_threadpool.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

static void llenar_random_matriz(TernaryDense *layer) {
    for (size_t o = 0; o < layer->output_dim; o++)
        for (size_t i = 0; i < layer->input_dim; i++)
            ternary_dense_set_weight(layer, o, i, (rand() % 3) - 1);
}

static void llenar_random_vec(TernaryVec *v, size_t n) {
    for (size_t i = 0; i < n; i++) ternary_vec_set(v, i, (rand() % 3) - 1);
}

static void test_forward_paralelo_vs_serial(size_t input_dim, size_t output_dim, int num_threads) {
    srand((unsigned)(input_dim * 1000 + output_dim * 10 + num_threads));
    TernaryDense *layer = ternary_dense_alloc(input_dim, output_dim, 1);
    llenar_random_matriz(layer);
    llenar_random_vec(layer->bias, output_dim);

    TernaryVec *x = ternary_vec_alloc(input_dim);
    llenar_random_vec(x, input_dim);

    float *out_serial = malloc(output_dim * sizeof(float));
    float *out_paralelo = malloc(output_dim * sizeof(float));

    ternary_dense_forward(layer, x, out_serial, NULL);

    TernaryThreadPool *pool = ternary_threadpool_create(num_threads);
    ternary_dense_forward_parallel(layer, x, out_paralelo, NULL, pool);

    int discrepancias = 0;
    for (size_t o = 0; o < output_dim; o++) {
        if (fabsf(out_serial[o] - out_paralelo[o]) > 1e-6f) {
            discrepancias++;
            if (discrepancias <= 5) {
                printf("[discrepancia o=%zu] serial=%.4f paralelo=%.4f\n", o, out_serial[o], out_paralelo[o]);
            }
        }
    }
    printf("[input=%zu output=%zu hilos=%d] discrepancias=%d\n",
           input_dim, output_dim, num_threads, discrepancias);
    assert(discrepancias == 0);

    ternary_threadpool_destroy(pool);
    free(out_serial); free(out_paralelo);
    ternary_vec_free(x);
    ternary_dense_free(layer);
}

int main(void) {
    int hilos[] = {1, 2, 3, 4, 8, 16};
    size_t configs[][2] = {
        {2, 3}, {16, 1}, {2, 16}, {128, 64}, {784, 256}, {1000, 1}, {5, 200}
    };
    for (size_t c = 0; c < sizeof(configs)/sizeof(configs[0]); c++) {
        for (size_t h = 0; h < sizeof(hilos)/sizeof(hilos[0]); h++) {
            test_forward_paralelo_vs_serial(configs[c][0], configs[c][1], hilos[h]);
        }
    }
    printf("\nTODAS LAS PRUEBAS DEL THREAD POOL PASARON\n");
    return 0;
}
