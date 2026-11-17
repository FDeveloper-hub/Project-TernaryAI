/*
 * Verifica que TernaryStack entrena capas con VARIAS matrices de pesos
 * (vtable->num_matrices > 1) -- el mecanismo que GQA necesita (Wq, Wk,
 * Wv, Wo) -- sin haber escrito GQA todavia.
 *
 * Capa de prueba "fusionada": out = W2 . sign(W1 . x), dos matrices Dense
 * dentro de UNA sola capa (num_matrices = 2, una escala y un bias por
 * matriz). Se compara contra la MISMA red armada como dos capas Dense
 * separadas encadenadas por el stack de siempre: mismos pesos iniciales,
 * mismos datos, y se exigen pesos, bias, escalas y salidas identicos bit
 * a bit tras entrenar. Si el contenedor manejara mal el gradiente de la
 * segunda matriz (acumulador, orden de aplicacion, escala propia), las
 * dos redes divergirian.
 */
#include "ternary_stack.h"
#include "ternary_dense.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FALLO: %s (linea %d)\n", msg, __LINE__); return 1; } } while (0)
#define IN 32
#define HID 16
#define OUT 4
#define BATCH 6
#define EPOCHS 25

typedef struct {
    TernaryDense *d[2];
    /* estado de forward que backward necesita (el stack garantiza que
     * backward va justo despues de forward con el mismo x) */
    float *o1; int32_t *raw1; TernaryU8 *h; int32_t *raw2; float *gh;
} Fused;

static Fused *fused_new(TernaryDense *d1, TernaryDense *d2) {
    Fused *f = (Fused *)calloc(1, sizeof(Fused));
    f->d[0] = d1; f->d[1] = d2;
    f->o1 = (float *)malloc(HID * sizeof(float));
    f->raw1 = (int32_t *)malloc(HID * sizeof(int32_t));
    f->h = ternary_u8_alloc(1, HID);
    f->raw2 = (int32_t *)malloc(OUT * sizeof(int32_t));
    f->gh = (float *)malloc(HID * sizeof(float));
    return f;
}

static size_t f_in(const void *s) { (void)s; return IN; }
static size_t f_out(const void *s) { (void)s; return OUT; }
static size_t f_nm(const void *s) { (void)s; return 2; }
static size_t f_rows(const void *s, size_t m) { return ((const Fused *)s)->d[m]->output_dim; }
static size_t f_rw(const void *s, size_t m) { return ((const Fused *)s)->d[m]->input_dim; }
static int f_bias(const void *s, size_t m) { return ((const Fused *)s)->d[m]->bias != NULL; }
static int f_prop(const void *s) { (void)s; return 1; }
static TernaryDType f_dtype(const void *s) { (void)s; return TERNARY_DTYPE_TERNARY; }

static void f_forward(const void *s_, const TernaryTensor *x, float *out, int32_t *out_raw, const TernaryCapabilities *caps) {
    Fused *f = (Fused *)s_;
    ternary_dense_forward_auto(f->d[0], x->data_t, f->o1, f->raw1, caps);
    TERNARY_ACTIVATION_UINT8.apply_u8(f->o1, f->h);
    ternary_dense_forward_u8(f->d[1], f->h, out, f->raw2);
    memcpy(out_raw, f->raw2, OUT * sizeof(int32_t));
}

static void f_backward(const void *s_, const TernaryTensor *x, const float *grad_out,
                       float *const *gw, float *grad_x_out, float *gs, float *const *gb, const int32_t *raw) {
    (void)raw;
    Fused *f = (Fused *)s_;
    memset(f->gh, 0, HID * sizeof(float));
    memset(gw[1], 0, f_rows(f, 1) * f_rw(f, 1) * sizeof(float));
    if (gb[1]) memset(gb[1], 0, f_rows(f, 1) * sizeof(float));
    ternary_dense_backward_u8(f->d[1], f->h, grad_out, gw[1], f->gh, &gs[1], gb[1], f->raw2);
    ternary_dense_backward(f->d[0], x->data_t, f->gh, gw[0], grad_x_out, &gs[0], gb[0], f->raw1);
}

static void f_apply_row(void *s_, size_t m, size_t r, const float *g) {
    ternary_dense_apply_row_gradients(((Fused *)s_)->d[m], r, g);
}
static void f_apply_bias(void *s_, size_t m, const float *g) {
    ternary_dense_apply_bias_gradients(((Fused *)s_)->d[m], g);
}
static void f_apply_scale(void *s_, size_t m, float g, float lr) {
    ternary_dense_apply_scale_gradient(((Fused *)s_)->d[m], g, lr);
}
static void f_destroy(void *s_) {
    Fused *f = (Fused *)s_;
    ternary_dense_free(f->d[0]); ternary_dense_free(f->d[1]);
    free(f->o1); free(f->raw1); ternary_u8_free(f->h); free(f->raw2); free(f->gh); free(f);
}

static const TernaryLayerVTable FUSED_VT = {
    "fused_test", f_in, f_out, f_nm, f_rows, f_rw, f_bias, f_prop, f_dtype,
    f_forward, f_backward, f_apply_row, f_apply_bias, f_apply_scale,
    f_destroy,
    NULL, /* backward_accumulate: no soportado */
    NULL, /* backward_block: no soportado */
    NULL, /* apply_matrix_gradients: no soportado */
    NULL, /* forward_block: no soportado */
    NULL, /* state_bytes: sin estado compartible */
    NULL, /* bind_state */
    NULL, /* accepts_u8 */
    NULL, /* output_row_len */
    NULL, /* export_layer */
    NULL, /* import_layer */
};

static TernaryDense *make_dense(size_t in, size_t out, unsigned seed) {
    TernaryDense *d = ternary_dense_alloc(in, out, 1);
    srand(seed);
    for (size_t o = 0; o < out; o++) {
        for (size_t i = 0; i < in; i++) ternary_dense_set_weight(d, o, i, (rand() % 3) - 1);
        ternary_vec_set(d->bias, o, (rand() % 3) - 1);
    }
    return d;
}

static int same_dense(const TernaryDense *a, const TernaryDense *b) {
    for (size_t o = 0; o < a->output_dim; o++) {
        for (size_t i = 0; i < a->input_dim; i++)
            if (ternary_dense_get_weight(a, o, i) != ternary_dense_get_weight(b, o, i)) return 0;
        if (ternary_vec_get(a->bias, o) != ternary_vec_get(b->bias, o)) return 0;
    }
    return a->scale == b->scale;
}

int main(void) {
    TernaryCapabilities cn; memset(&cn, 0, sizeof(cn)); cn.num_cores = 1; cn.dot_impl_name = "scalar";
    const float scale_lr = 0.05f;

    TernaryDense *a1 = make_dense(IN, HID, 5), *a2 = make_dense(HID, OUT, 6);   /* red fusionada */
    TernaryDense *b1 = make_dense(IN, HID, 5), *b2 = make_dense(HID, OUT, 6);   /* red de dos capas */
    Fused *fu = fused_new(a1, a2);
    TernaryLayer fl; fl.impl = fu; fl.vt = &FUSED_VT;

    TernaryStack *sa = ternary_stack_create();
    CHECK(ternary_stack_add(sa, fl, TERNARY_ACTIVATION_UINT8) == 1, "no se pudo agregar la capa de 2 matrices");
    TernaryStack *sb = ternary_stack_create();
    ternary_stack_add(sb, ternary_layer_from_dense(b1), TERNARY_ACTIVATION_UINT8);
    ternary_stack_add(sb, ternary_layer_from_dense(b2), TERNARY_ACTIVATION_UINT8);

    TernaryVec *x = ternary_vec_alloc(IN);
    size_t dims[1] = { IN };
    TernaryTensor *xt = ternary_tensor_view_ternary(dims, 1, x);
    float oa[OUT], ob[OUT];
    srand(99);
    for (int ep = 0; ep < EPOCHS; ep++) {
        ternary_stack_zero_gradients(sa); ternary_stack_zero_gradients(sb);
        for (int b = 0; b < BATCH; b++) {
            for (size_t i = 0; i < IN; i++) ternary_vec_set(x, i, (rand() % 3) - 1);
            float tg[OUT]; for (size_t o = 0; o < OUT; o++) tg[o] = (rand() % 2) ? 1.0f : -1.0f;
            ternary_stack_forward(sa, xt, oa, &cn);
            ternary_stack_forward(sb, xt, ob, &cn);
            CHECK(memcmp(oa, ob, sizeof(oa)) == 0, "forward de la capa fusionada difiere del de las dos capas");
            float ga[OUT], gb_[OUT];
            for (size_t o = 0; o < OUT; o++) { ga[o] = (2.0f / OUT) * (oa[o] - tg[o]); gb_[o] = (2.0f / OUT) * (ob[o] - tg[o]); }
            ternary_stack_backward(sa, xt, ga);
            ternary_stack_backward(sb, xt, gb_);
        }
        ternary_stack_apply_gradients(sa, scale_lr);
        ternary_stack_apply_gradients(sb, scale_lr);
    }
    CHECK(same_dense(a1, b1), "matriz 0 (pesos/bias/escala) difiere tras entrenar");
    CHECK(same_dense(a2, b2), "matriz 1 (pesos/bias/escala) difiere tras entrenar");

    /* que el entrenamiento haya movido algo (si no, la igualdad no prueba nada) */
    TernaryDense *r1 = make_dense(IN, HID, 5), *r2 = make_dense(HID, OUT, 6);
    int moved = !same_dense(a1, r1) || !same_dense(a2, r2);
    CHECK(moved, "el entrenamiento no cambio ningun peso: la comparacion no prueba nada");
    printf("[2 matrices] %d epocas: capa fusionada (num_matrices=2) == dos capas Dense, pesos/bias/escalas identicos bit a bit, y el entrenamiento si cambio pesos\n", EPOCHS);

    ternary_dense_free(r1); ternary_dense_free(r2);
    ternary_tensor_free(xt); ternary_vec_free(x);
    ternary_stack_free(sa); ternary_stack_free(sb);
    printf("\ntest_multimatrix: TODO OK\n");
    return 0;
}
