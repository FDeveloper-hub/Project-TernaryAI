/* Activacion uint8 dentro del contenedor (etapa 1): Dense -> uint8 -> Dense, Embedding -> RMSNorm(uint8) -> Head.
 *  - forward del stack == composicion manual con las mismas funciones (bits identicos),
 *  - el entrenamiento con uint8 entre capas baja la perdida,
 *  - rechazos: capa que no acepta uint8, residual con uint8. */
#include "ternary_stack.h"
#include "ternary_layer.h"
#include "ternary_embedding.h"
#include "ternary_rmsnorm.h"
#include "ternary_head.h"
#include "ternary_residual.h"
#include "ternary_u8.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fallas = 0;
#define CHECK(c, ...) do { if (!(c)) { fallas++; printf("FALLO (linea %d): ", __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static TernaryCapabilities caps_none(void) { TernaryCapabilities c; memset(&c, 0, sizeof(c)); c.num_cores = 1; c.dot_impl_name = "scalar"; return c; }

static TernaryDense *mk_dense(size_t in, size_t out, unsigned seed) {
    TernaryDense *d = ternary_dense_alloc(in, out, 1);
    srand(seed);
    for (size_t o = 0; o < out; o++) for (size_t i = 0; i < in; i++) ternary_dense_set_weight(d, o, i, (rand() % 3) - 1);
    return d;
}

static void test_dense_dense(void) {
    const size_t IN = 24, H = 16, OUT = 3;
    TernaryCapabilities cn = caps_none();
    TernaryStack *s = ternary_stack_create();
    CHECK(ternary_stack_add(s, ternary_layer_from_dense(mk_dense(IN, H, 1)), TERNARY_ACTIVATION_UINT8) == 1, "add 1");
    CHECK(ternary_stack_add(s, ternary_layer_from_dense(mk_dense(H, OUT, 2)), TERNARY_ACTIVATION_UINT8) == 1, "add 2");
    TernaryDense *d1 = mk_dense(IN, H, 1), *d2 = mk_dense(H, OUT, 2);
    for (int t = 0; t < 50; t++) {
        TernaryVec *x = ternary_vec_alloc(IN);
        for (size_t i = 0; i < IN; i++) ternary_vec_set(x, i, (rand() % 3) - 1);
        size_t dims[1] = { IN };
        TernaryTensor *xt = ternary_tensor_view_ternary(dims, 1, x);
        float out[3], h[16], ref[3];
        ternary_stack_forward(s, xt, out, &cn);
        ternary_dense_forward(d1, x, h, NULL);
        TernaryU8 *q = ternary_u8_alloc(1, H);
        ternary_u8_quantize(q, h);
        ternary_dense_forward_u8(d2, q, ref, NULL);
        CHECK(memcmp(out, ref, sizeof(ref)) == 0, "forward distinto (t=%d)", t);
        ternary_u8_free(q); ternary_tensor_free(xt); ternary_vec_free(x);
    }
    ternary_dense_free(d1); ternary_dense_free(d2); ternary_stack_free(s);
}

static double aprende(TernaryActivation act, double *l0o) {
    const size_t IN = 32, H = 32, OUT = 4, N = 16;
    TernaryCapabilities cn = caps_none();
    TernaryStack *s = ternary_stack_create();
    TernaryDense *da = mk_dense(IN, H, 3), *db = mk_dense(H, OUT, 4);
    da->scale = 0.05f; db->scale = 0.05f;
    ternary_stack_add(s, ternary_layer_from_dense(da), act);
    ternary_stack_add(s, ternary_layer_from_dense(db), TERNARY_ACTIVATION_UINT8);
    TernaryVec *xs[16]; float ys[16][4];
    srand(9);
    for (size_t n = 0; n < N; n++) {
        xs[n] = ternary_vec_alloc(IN);
        for (size_t i = 0; i < IN; i++) ternary_vec_set(xs[n], i, (rand() % 3) - 1);
        for (size_t o = 0; o < OUT; o++) ys[n][o] = ((float)(rand() % 200) / 100.0f - 1.0f);
    }
    size_t dims[1] = { IN };
    double l0 = 0, l1 = 0;
    for (int ep = 0; ep < 60; ep++) {
        double l = 0;
        ternary_stack_zero_gradients(s);
        for (size_t n = 0; n < N; n++) {
            TernaryTensor *xt = ternary_tensor_view_ternary(dims, 1, xs[n]);
            float out[4], g[4];
            ternary_stack_forward(s, xt, out, &cn);
            for (size_t o = 0; o < OUT; o++) { float e = out[o] - ys[n][o]; l += e * e; g[o] = 2.0f * e / (float)(N * OUT); }
            ternary_stack_backward(s, xt, g);
            ternary_tensor_free(xt);
        }
        ternary_stack_apply_gradients(s, 0.0f);
        if (ep == 0) l0 = l;
        l1 = l;
    }
    for (size_t n = 0; n < N; n++) ternary_vec_free(xs[n]);
    ternary_stack_free(s);
    *l0o = l0;
    return l1;
}

static void test_aprende(void) {
    double l0, l1 = aprende(TERNARY_ACTIVATION_UINT8, &l0);
    double c0, c1 = aprende(TERNARY_ACTIVATION_UINT8, &c0);
    printf("  uint8: perdida %.4f -> %.4f\n", l0, l1); (void)c0; (void)c1;
    CHECK(l1 < 0.8 * l0, "no aprende con uint8: %.4f -> %.4f", l0, l1);
}

static void test_emb_rms_head(void) {
    const size_t V = 12, E = 8, T = 5, VOC = 7;
    TernaryCapabilities cn = caps_none();
    TernaryStack *s = ternary_stack_create();
    TernaryEmbedding *e = ternary_embedding_alloc(V, E, T), *e2 = ternary_embedding_alloc(V, E, T);
    srand(5);
    for (size_t v = 0; v < V; v++) for (size_t d = 0; d < E; d++) { int w = (rand() % 3) - 1; ternary_embedding_set_weight(e, v, d, w); ternary_embedding_set_weight(e2, v, d, w); }
    TernaryRMSNorm *n = ternary_rmsnorm_alloc(T, E, 1e-5f), *n2 = ternary_rmsnorm_alloc(T, E, 1e-5f);
    TernaryHead *h = ternary_head_alloc(T, E, VOC), *h2 = ternary_head_alloc(T, E, VOC);
    srand(6);
    for (size_t o = 0; o < VOC; o++) for (size_t i = 0; i < E; i++) { int w = (rand() % 3) - 1; ternary_dense_set_weight(h->w, o, i, w); ternary_dense_set_weight(h2->w, o, i, w); }
    CHECK(ternary_stack_add(s, ternary_layer_from_embedding(e), TERNARY_ACTIVATION_UINT8) == 1, "add emb");
    CHECK(ternary_stack_add(s, ternary_layer_from_rmsnorm(n), TERNARY_ACTIVATION_UINT8) == 1, "add rms");
    CHECK(ternary_stack_add(s, ternary_layer_from_head(h), TERNARY_ACTIVATION_UINT8) == 1, "add head");
    for (int t = 0; t < 30; t++) {
        int32_t ids[5];
        for (size_t i = 0; i < T; i++) ids[i] = rand() % V;
        size_t dims[1] = { T };
        TernaryTensor *x = ternary_tensor_view_id(dims, 1, ids);
        float *out = malloc(T * VOC * sizeof(float)), *ref = malloc(T * VOC * sizeof(float));
        float xe[40], xn[40];
        ternary_stack_forward(s, x, out, &cn);
        ternary_embedding_forward(e2, ids, xe, NULL);
        ternary_rmsnorm_forward(n2, xe, xn);
        TernaryU8 *q = ternary_u8_alloc(T, E);
        ternary_u8_quantize(q, xn);
        ternary_head_forward_u8(h2, q, ref, NULL);
        CHECK(memcmp(out, ref, T * VOC * sizeof(float)) == 0, "forward emb/rms/head distinto (t=%d)", t);
        ternary_u8_free(q); free(out); free(ref); ternary_tensor_free(x);
    }
    ternary_embedding_free(e2); ternary_rmsnorm_free(n2); ternary_head_free(h2);
    ternary_stack_free(s);
}

static void test_rechazos(void) {
    TernaryStack *s = ternary_stack_create();
    ternary_stack_add(s, ternary_layer_from_dense(mk_dense(8, 8, 1)), TERNARY_ACTIVATION_UINT8);
    TernaryRMSNorm *n = ternary_rmsnorm_alloc(1, 8, 1e-5f);
    /* RMSNorm lee FLOAT: tras una Dense se acepta y la salida de la Dense no se cuantiza */
    CHECK(ternary_stack_add(s, ternary_layer_from_rmsnorm(n), TERNARY_ACTIVATION_UINT8) == 1, "rmsnorm tras dense se acepta");
    /* una Dense de entrada distinta no encaja: se rechaza y el llamador conserva la capa */
    TernaryDense *bad = mk_dense(4, 4, 3);
    CHECK(ternary_stack_add(s, ternary_layer_from_dense(bad), TERNARY_ACTIVATION_UINT8) == 0, "largo incompatible se rechaza");
    ternary_dense_free(bad);
    ternary_stack_free(s);

    TernaryResidual *r = ternary_residual_create();
    TernaryDense *d = mk_dense(8, 8, 1);
    CHECK(ternary_residual_add(r, ternary_layer_from_dense(d), TERNARY_ACTIVATION_UINT8) == 0, "residual con primera sub-capa uint8 deberia rechazarse");
    ternary_dense_free(d);
    ternary_residual_free(r);
}

int main(void) {
    printf("-- uint8 en el stack\n");
    test_dense_dense();
    test_aprende();
    test_emb_rms_head();
    test_rechazos();
    if (fallas) { printf("%d fallas\n", fallas); return 1; }
    printf("OK\n");
    return 0;
}
