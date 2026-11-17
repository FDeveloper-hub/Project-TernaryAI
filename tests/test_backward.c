#include "ternary_dense.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

static void backward_reference(int8_t *W_flat, int8_t *x_flat,
                                size_t in_dim, size_t out_dim, float scale,
                                const float *grad_out, float *grad_w_out,
                                float *grad_x_out, float *grad_scale_out,
                                float *grad_bias_out) {
    for (size_t o = 0; o < out_dim; o++) {
        int32_t z_raw = 0;
        for (size_t i = 0; i < in_dim; i++) z_raw += (int32_t)W_flat[o*in_dim+i] * (int32_t)x_flat[i];
        float g = grad_out[o];
        *grad_scale_out += g * (float)z_raw;
        grad_bias_out[o] += g;   /* G_b = sum(G_Z), sin multiplicar por nada */
        for (size_t i = 0; i < in_dim; i++) {
            grad_w_out[o*in_dim+i] = g * (float)x_flat[i];             /* SIN escala */
            grad_x_out[i] += g * scale * (float)W_flat[o*in_dim+i];    /* CON escala */
        }
    }
}

int main(void) {
    srand(555);
    size_t in_dim = 40, out_dim = 25;
    TernaryDense *layer = ternary_dense_alloc(in_dim, out_dim, 0);
    layer->scale = 0.37f; /* escala no trivial, para que el test la ejercite de verdad */
    int8_t *W_flat = malloc(out_dim * in_dim);
    for (size_t o = 0; o < out_dim; o++)
        for (size_t i = 0; i < in_dim; i++) {
            int v = (rand()%3)-1;
            ternary_dense_set_weight(layer, o, i, v);
            W_flat[o*in_dim+i] = (int8_t)v;
        }

    TernaryVec *x = ternary_vec_alloc(in_dim);
    int8_t x_flat[40];
    for (size_t i = 0; i < in_dim; i++) {
        int v = (rand()%3)-1;
        ternary_vec_set(x, i, v);
        x_flat[i] = (int8_t)v;
    }

    float grad_out[25];
    for (size_t o = 0; o < out_dim; o++) grad_out[o] = ((float)(rand()%2000)/1000.0f - 1.0f);

    float grad_w_bits[25*40], grad_w_ref[25*40];
    float grad_x_bits[40] = {0}, grad_x_ref[40] = {0};
    float grad_scale_bits = 0.0f, grad_scale_ref = 0.0f;
    float grad_bias_bits[25] = {0}, grad_bias_ref[25] = {0};

    ternary_dense_backward(layer, x, grad_out, grad_w_bits, grad_x_bits, &grad_scale_bits, grad_bias_bits, NULL);
    backward_reference(W_flat, x_flat, in_dim, out_dim, layer->scale, grad_out, grad_w_ref, grad_x_ref, &grad_scale_ref, grad_bias_ref);

    int discrepancias = 0;
    for (size_t k = 0; k < out_dim*in_dim; k++)
        if (fabsf(grad_w_bits[k]-grad_w_ref[k]) > 1e-4f) discrepancias++;
    for (size_t i = 0; i < in_dim; i++)
        if (fabsf(grad_x_bits[i]-grad_x_ref[i]) > 1e-4f) discrepancias++;
    if (fabsf(grad_scale_bits - grad_scale_ref) > 1e-3f) discrepancias++;
    for (size_t o = 0; o < out_dim; o++)
        if (fabsf(grad_bias_bits[o]-grad_bias_ref[o]) > 1e-4f) discrepancias++;

    printf("[backward vs referencia] discrepancias=%d (esperado 0)\n", discrepancias);
    printf("[grad_scale] bits=%.4f ref=%.4f\n", grad_scale_bits, grad_scale_ref);
    assert(discrepancias == 0);
    printf("TODAS LAS PRUEBAS DE BACKWARD PASARON\n");

    free(W_flat);
    ternary_vec_free(x);
    ternary_dense_free(layer);
    return 0;
}
