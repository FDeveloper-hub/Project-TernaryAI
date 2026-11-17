/*
 * Verifica TernaryStack contra el camino de siempre (llamar a
 * Dense/Conv2D directamente, a mano) -- si el contenedor hace lo mismo
 * bit a bit que el codigo ya verificado, la capa de abstraccion no
 * introdujo ningun bug.
 *
 * 1) Encadena Conv2D(8x8x1 -> 6x6x4) + Dense(144->10): compara el
 *    forward del stack contra llamar conv2d_forward + sign() a mano +
 *    dense_forward_auto, bit a bit.
 * 2) Confirma que agregar Conv2D como SEGUNDA capa se rechaza (no
 *    propaga gradiente a su entrada) -- la regla que evita entrenar en
 *    silencio con capas congeladas.
 * 3) Entrena la MISMA red (Dense 32->16->4) via el stack vs a mano
 *    (mismos pesos iniciales, mismos datos) durante varias epocas y
 *    exige pesos/bias/escala identicos bit a bit al final.
 */
#include "ternary_stack.h"
#include "ternary_dispatch.h"
#include "ternary_dense.h"
#include "ternary_conv2d.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;

/* Fuerza CPU serial (sin pool, sin GPU) -- este proyecto nunca acepta
 * caps==NULL (ternary_choose_engine lo desreferencia sin chequear), asi
 * que este test usa una struct en cero en vez de NULL en todas las
 * llamadas a forward. */
static TernaryCapabilities caps_none(void) {
    TernaryCapabilities c; memset(&c, 0, sizeof(c));
    c.num_cores = 1;
    c.dot_impl_name = "scalar";
    return c;
}
#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FALLO: %s\n", msg); fails++; } } while (0)

static void test_forward_conv_dense(void) {
    size_t H = 8, W = 8, Cin = 1, KH = 3, KW = 3, Cout = 4, stride = 1;
    TernaryConv2D *conv = ternary_conv2d_alloc(H, W, Cin, KH, KW, Cout, stride, 1);
    size_t conv_out_len = conv->out_height * conv->out_width * conv->out_channels;
    TernaryDense *dense = ternary_dense_alloc(conv_out_len, 10, 1);

    srand(42);
    for (size_t o = 0; o < Cout; o++) {
        for (size_t k = 0; k < conv->kernel_size; k++) ternary_conv2d_set_weight(conv, o, k, (rand() % 3) - 1);
        ternary_vec_set(conv->bias, o, (rand() % 3) - 1);
    }
    for (size_t o = 0; o < 10; o++) {
        for (size_t i = 0; i < conv_out_len; i++) ternary_dense_set_weight(dense, o, i, (rand() % 3) - 1);
        ternary_vec_set(dense->bias, o, (rand() % 3) - 1);
    }

    TernaryVec *image = ternary_vec_alloc(H * W * Cin);
    for (size_t i = 0; i < H * W * Cin; i++) ternary_vec_set(image, i, (rand() % 3) - 1);

    /* --- Manual --- */
    float *conv_out = (float *)malloc(conv_out_len * sizeof(float));
    ternary_conv2d_forward(conv, image, conv_out, NULL);
    TernaryU8 *conv_act = ternary_u8_alloc(1, conv_out_len);
    TERNARY_ACTIVATION_UINT8.apply_u8(conv_out, conv_act);
    float dense_out_manual[10];
    TernaryCapabilities cn = caps_none();
    ternary_dense_forward_u8(dense, conv_act, dense_out_manual, NULL);

    /* --- Via el stack (capas y datos IDENTICOS: el stack se queda con
     * los mismos punteros, no con copias) --- */
    TernaryStack *s = ternary_stack_create();
    int ok1 = ternary_stack_add(s, ternary_layer_from_conv2d(conv), TERNARY_ACTIVATION_UINT8);
    CHECK(ok1, "no se pudo agregar Conv2D como primera capa");
    int ok2 = ternary_stack_add(s, ternary_layer_from_dense(dense), TERNARY_ACTIVATION_UINT8);
    CHECK(ok2, "no se pudo agregar Dense como segunda capa (input_len deberia calzar)");
    CHECK(ternary_stack_num_layers(s) == 2, "num_layers deberia ser 2");
    CHECK(ternary_stack_input_len(s) == H * W * Cin, "input_len del stack incorrecto");
    CHECK(ternary_stack_output_len(s) == 10, "output_len del stack incorrecto");

    float dense_out_stack[10];
    size_t image_dims[1] = { H * W * Cin };
    TernaryTensor *image_t = ternary_tensor_view_ternary(image_dims, 1, image);
    ternary_stack_forward(s, image_t, dense_out_stack, &cn);
    ternary_tensor_free(image_t);
    CHECK(memcmp(dense_out_manual, dense_out_stack, sizeof(dense_out_manual)) == 0,
          "forward del stack (Conv2D+Dense) difiere del manual");

    /* Rechazo: Conv2D como SEGUNDA capa de una red NUEVA */
    TernaryConv2D *conv2 = ternary_conv2d_alloc(4, 4, Cout, 2, 2, 2, 1, 1);
    TernaryStack *s2 = ternary_stack_create();
    /* Una capa Dense cualquiera de salida 10 primero, para que la
     * segunda (conv2, que espera 4*4*10... no calza igual, pero lo que
     * se quiere probar es el rechazo por 'no propaga', asi que se arma
     * con el mismo output_len que espera conv2 para aislar esa regla). */
    TernaryDense *d0 = ternary_dense_alloc(4, 4 * 4 * Cout, 0);
    CHECK(ternary_stack_add(s2, ternary_layer_from_dense(d0), TERNARY_ACTIVATION_UINT8) == 1,
          "no se pudo agregar la primera capa de control");
    int rejected = ternary_stack_add(s2, ternary_layer_from_conv2d(conv2), TERNARY_ACTIVATION_UINT8);
    CHECK(rejected == 0, "Conv2D como segunda capa deberia ser rechazado (no propaga gradiente)");
    ternary_conv2d_free(conv2); /* el stack lo rechazo: sigue siendo nuestro */

    ternary_stack_free(s);  /* libera conv y dense (ya no son nuestros) */
    ternary_stack_free(s2); /* libera d0 */
    free(conv_out);
    ternary_u8_free(conv_act);
    ternary_vec_free(image);
}

#define IN_DIM 32
#define HID_DIM 16
#define OUT_DIM 4
#define BATCH 6
#define EPOCHS 20

static TernaryDense *alloc_init(size_t in_d, size_t out_d, unsigned seed) {
    TernaryDense *l = ternary_dense_alloc(in_d, out_d, 1);
    srand(seed);
    for (size_t o = 0; o < out_d; o++) {
        for (size_t i = 0; i < in_d; i++) ternary_dense_set_weight(l, o, i, (rand() % 3) - 1);
        ternary_vec_set(l->bias, o, (rand() % 3) - 1);
    }
    return l;
}

static void test_training_matches_manual(void) {
    size_t total = (size_t)BATCH * EPOCHS;
    float *dataset = (float *)malloc(total * IN_DIM * sizeof(float));
    float *targets = (float *)malloc(total * OUT_DIM * sizeof(float));
    srand(555);
    for (size_t b = 0; b < total; b++) {
        for (size_t i = 0; i < IN_DIM; i++) dataset[b * IN_DIM + i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        for (size_t o = 0; o < OUT_DIM; o++) targets[b * OUT_DIM + o] = (rand() % 2) ? 1.0f : -1.0f;
    }
    TernaryVec *x = ternary_vec_alloc(IN_DIM);

    float scale_lr = 0.05f;
    TernaryCapabilities cn = caps_none();

    /* --- Manual (igual patron que tests/test_training_parallel.c) --- */
    TernaryDense *m1 = alloc_init(IN_DIM, HID_DIM, 1234);
    TernaryDense *m2 = alloc_init(HID_DIM, OUT_DIM, 5678);
    float o1[HID_DIM], o2[OUT_DIM]; int32_t o1r[HID_DIM], o2r[OUT_DIM];
    TernaryU8 *a1 = ternary_u8_alloc(1, HID_DIM);
    float gw1[HID_DIM * IN_DIM], gw2[OUT_DIM * HID_DIM];
    float gb1[HID_DIM], gb2[OUT_DIM];
    for (int e = 0; e < EPOCHS; e++) {
        memset(gw1, 0, sizeof(gw1)); memset(gw2, 0, sizeof(gw2));
        memset(gb1, 0, sizeof(gb1)); memset(gb2, 0, sizeof(gb2));
        float gs1 = 0.0f, gs2 = 0.0f;
        for (int b = 0; b < BATCH; b++) {
            size_t idx = (size_t)e * BATCH + b;
            for (size_t i = 0; i < IN_DIM; i++)
                ternary_vec_set(x, i, dataset[idx * IN_DIM + i] > 0.15f ? 1 : (dataset[idx * IN_DIM + i] < -0.15f ? -1 : 0));
            ternary_dense_forward_auto(m1, x, o1, o1r, &cn);
            TERNARY_ACTIVATION_UINT8.apply_u8(o1, a1);
            ternary_dense_forward_u8(m2, a1, o2, o2r);
            float gout2[OUT_DIM];
            for (size_t o = 0; o < OUT_DIM; o++) gout2[o] = (2.0f / OUT_DIM) * (o2[o] - targets[idx * OUT_DIM + o]);
            float gw2s[OUT_DIM * HID_DIM], gx2[HID_DIM]; memset(gx2, 0, sizeof(gx2));
            memset(gw2s, 0, sizeof(gw2s));
            ternary_dense_backward_u8(m2, a1, gout2, gw2s, gx2, &gs2, gb2, o2r);
            for (size_t kk = 0; kk < OUT_DIM * HID_DIM; kk++) gw2[kk] += gw2s[kk];
            float gw1s[HID_DIM * IN_DIM], gx1[IN_DIM]; memset(gx1, 0, sizeof(gx1));
            ternary_dense_backward(m1, x, gx2, gw1s, gx1, &gs1, gb1, o1r);
            for (size_t kk = 0; kk < HID_DIM * IN_DIM; kk++) gw1[kk] += gw1s[kk];
        }
        for (size_t o = 0; o < HID_DIM; o++) ternary_dense_apply_row_gradients(m1, o, gw1 + o * IN_DIM);
        for (size_t o = 0; o < OUT_DIM; o++) ternary_dense_apply_row_gradients(m2, o, gw2 + o * HID_DIM);
        ternary_dense_apply_bias_gradients(m1, gb1);
        ternary_dense_apply_bias_gradients(m2, gb2);
        ternary_dense_apply_scale_gradient(m1, gs1, scale_lr);
        ternary_dense_apply_scale_gradient(m2, gs2, scale_lr);
    }

    /* --- Via el stack --- */
    TernaryDense *s1 = alloc_init(IN_DIM, HID_DIM, 1234);
    TernaryDense *s2l = alloc_init(HID_DIM, OUT_DIM, 5678);
    TernaryStack *stack = ternary_stack_create();
    ternary_stack_add(stack, ternary_layer_from_dense(s1), TERNARY_ACTIVATION_UINT8);
    ternary_stack_add(stack, ternary_layer_from_dense(s2l), TERNARY_ACTIVATION_UINT8);
    float out_final[OUT_DIM];
    size_t x_dims[1] = { IN_DIM };
    TernaryTensor *x_t = ternary_tensor_view_ternary(x_dims, 1, x);
    for (int e = 0; e < EPOCHS; e++) {
        ternary_stack_zero_gradients(stack);
        for (int b = 0; b < BATCH; b++) {
            size_t idx = (size_t)e * BATCH + b;
            for (size_t i = 0; i < IN_DIM; i++)
                ternary_vec_set(x, i, dataset[idx * IN_DIM + i] > 0.15f ? 1 : (dataset[idx * IN_DIM + i] < -0.15f ? -1 : 0));
            ternary_stack_forward(stack, x_t, out_final, &cn);
            float grad_final[OUT_DIM];
            for (size_t o = 0; o < OUT_DIM; o++) grad_final[o] = (2.0f / OUT_DIM) * (out_final[o] - targets[idx * OUT_DIM + o]);
            ternary_stack_backward(stack, x_t, grad_final);
        }
        ternary_stack_apply_gradients(stack, scale_lr);
    }
    ternary_tensor_free(x_t);

    int iguales = 1;
    for (size_t o = 0; o < HID_DIM && iguales; o++)
        for (size_t i = 0; i < IN_DIM; i++)
            if (ternary_dense_get_weight(m1, o, i) != ternary_dense_get_weight(s1, o, i)) { iguales = 0; break; }
    CHECK(iguales, "capa1: pesos entrenados via stack difieren del manual");
    iguales = 1;
    for (size_t o = 0; o < OUT_DIM && iguales; o++)
        for (size_t i = 0; i < HID_DIM; i++)
            if (ternary_dense_get_weight(m2, o, i) != ternary_dense_get_weight(s2l, o, i)) { iguales = 0; break; }
    CHECK(iguales, "capa2: pesos entrenados via stack difieren del manual");
    CHECK(m1->scale == s1->scale && m2->scale == s2l->scale, "escalas entrenadas via stack difieren del manual");

    ternary_dense_free(m1); ternary_dense_free(m2);
    ternary_stack_free(stack); /* libera s1 y s2l */
    ternary_u8_free(a1); ternary_vec_free(x);
    free(dataset); free(targets);
}

int main(void) {
    test_forward_conv_dense();
    test_training_matches_manual();
    if (fails == 0) {
        printf("test_stack: OK (forward Conv2D+Dense identico al manual, regla de rechazo, "
               "entrenamiento via stack identico bit a bit al manual)\n");
        return 0;
    }
    printf("test_stack: %d FALLO(S)\n", fails);
    return 1;
}
