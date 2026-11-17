#include "ternary_dense.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

static void test_forward_manual(void) {
    /* Capa 2->3 con bias, pesos y bias elegidos a mano, scale=1 (default) */
    TernaryDense *layer = ternary_dense_alloc(2, 3, 1);
    ternary_dense_set_weight(layer, 0, 0, 1);  ternary_dense_set_weight(layer, 0, 1, -1);
    ternary_dense_set_weight(layer, 1, 0, 0);  ternary_dense_set_weight(layer, 1, 1, 1);
    ternary_dense_set_weight(layer, 2, 0, -1); ternary_dense_set_weight(layer, 2, 1, -1);
    ternary_vec_set(layer->bias, 0, 1);
    ternary_vec_set(layer->bias, 1, -1);
    ternary_vec_set(layer->bias, 2, 0);

    TernaryVec *x = ternary_vec_alloc(2);
    ternary_vec_set(x, 0, 1); ternary_vec_set(x, 1, -1);

    float out[3];
    int32_t out_raw[3];
    ternary_dense_forward(layer, x, out, out_raw);
    /* scale=1 -> out == out_raw. fila0: 1*1+(-1)*(-1)=2,+1=3; fila1: -1,-1=-2; fila2: 0,+0=0 */
    printf("[forward manual] out_raw = [%d, %d, %d] (esperado [3, -2, 0])\n", out_raw[0], out_raw[1], out_raw[2]);
    assert(out_raw[0] == 3 && out_raw[1] == -2 && out_raw[2] == 0);
    assert(fabsf(out[0]-3.0f) < 1e-6f && fabsf(out[1]-(-2.0f)) < 1e-6f && fabsf(out[2]-0.0f) < 1e-6f);

    /* Con escala != 1, out debe escalar out_raw proporcionalmente. */
    layer->scale = 0.25f;
    ternary_dense_forward(layer, x, out, out_raw);
    printf("[forward con escala 0.25] out = [%.3f, %.3f, %.3f] (esperado [0.75,-0.5,0.0])\n", out[0],out[1],out[2]);
    assert(fabsf(out[0]-0.75f) < 1e-6f && fabsf(out[1]-(-0.5f)) < 1e-6f && fabsf(out[2]) < 1e-6f);

    ternary_vec_free(x);
    ternary_dense_free(layer);
}

/* La regla R2 (ver tests/test_r2.c para la comparacion exacta contra una referencia): un gradiente constante hacia +1
 * lleva los pesos de 0 a +1 y se queda ahi (saturado), sin salirse de {-1,0,1}. */
static void test_update_r2_sanidad(void) {
    const size_t n = 300;
    TernaryDense *layer = ternary_dense_alloc(n, 1, 0);
    float *grad = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) grad[i] = -1.0f;      /* dL/dW < 0: el peso debe subir */
    for (int step = 0; step < 2000; step++) ternary_dense_apply_row_gradients(layer, 0, grad);
    size_t pos = 0, bad = 0;
    for (size_t i = 0; i < n; i++) {
        int w = ternary_dense_get_weight(layer, 0, i);
        pos += (w == 1); bad += (w < -1 || w > 1);
    }
    printf("[update R2 sanidad] 2000 pasos x %zu pesos con gradiente constante: %zu a +1, %zu fuera de rango\n", n, pos, bad);
    assert(bad == 0 && pos == n);
    free(grad);
    ternary_dense_free(layer);
}

int main(void) {
    test_forward_manual();
    test_update_r2_sanidad();
    printf("\nTODAS LAS PRUEBAS DE LA CAPA DENSA PASARON\n");
    return 0;
}
