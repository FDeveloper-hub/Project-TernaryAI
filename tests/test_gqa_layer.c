/*
 * GQA como capa del TernaryStack (type "gqa", 4 matrices: Wq, Wk, Wv, Wo).
 * Verifica:
 *  1. registro: se construye por nombre desde la tabla, describe() devuelve
 *     los mismos parametros, y los errores de parametros/forma dan mensajes;
 *  2. entrenar a traves del stack == entrenar a mano con ternary_gqa_* y la
 *     misma regla (filas de las 4 matrices + SOLO la escala de Wo): pesos y
 *     escala de Wo identicos bit a bit; las escalas de Wq/Wk/Wv siguen en 1.0
 *     (no afectan la salida) y el entrenamiento si movio pesos;
 *  3. checkpoint completo: guardar/cargar/seguir entrenando == entrenar de un
 *     tiron (bit a bit), y la configuracion de RoPE sobrevive;
 *  4. export empaquetado: la inferencia del modelo cargado es identica.
 */
#include "tmp_path.h"
#include "ternary_stack.h"
#include "ternary_layer.h"
#include "ternary_rng.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FALLO: %s (linea %d)\n", msg, __LINE__); return 1; } } while (0)
#define E 16
#define SEQ 6
#define POSD 6
#define BATCH 8
#define EPOCHS 12

static const double PARAMS[5] = { 2 /*n_heads*/, 1 /*n_kv_heads*/,
                                  1 /*rope: entero*/, 10.0 /*rope_base*/, 2 /*score_shift*/ };

static const float SCALE_LR = 0.001f;

static TernaryGQA *gqa_of(TernaryStack *s) { return (TernaryGQA *)ternary_stack_layer(s, 0).impl; }

static void make_sample(TernaryVec *x, int content[SEQ][E]) {
    for (size_t p = 0; p < SEQ; p++) for (size_t i = 0; i < E; i++) {
        int v = (i < POSD) ? 1 : (rand() % 2) * 2 - 1;
        content[p][i] = v; ternary_vec_set(x, p * E + i, v);
    }
}
static void make_grad(const float *out, int content[SEQ][E], float *go) {
    memset(go, 0, SEQ * E * sizeof(float));
    for (size_t d = POSD; d < E; d++) go[(SEQ - 1) * E + d] = (2.0f / E) * (out[(SEQ - 1) * E + d] - (float)content[SEQ - 2][d]);
}

static int same_dense(const TernaryDense *a, const TernaryDense *b) {
    if (a->input_dim != b->input_dim || a->output_dim != b->output_dim || a->scale != b->scale) return 0;
    for (size_t o = 0; o < a->output_dim; o++) for (size_t i = 0; i < a->input_dim; i++)
        if (ternary_dense_get_weight(a, o, i) != ternary_dense_get_weight(b, o, i)) return 0;
    return 1;
}
static int same_gqa(const TernaryGQA *a, const TernaryGQA *b) {
    return same_dense(a->wq, b->wq) && same_dense(a->wk, b->wk) && same_dense(a->wv, b->wv) && same_dense(a->wo, b->wo);
}

/* Entrena el stack `epochs` epocas; todas las muestras salen de rand() (sembrado por el llamador). */
static void train_stack(TernaryStack *s, TernaryTensor *xt, TernaryVec *x, int epochs, const TernaryCapabilities *cn) {
    float out[SEQ * E], go[SEQ * E]; int content[SEQ][E];
    for (int ep = 0; ep < epochs; ep++) {
        ternary_stack_zero_gradients(s);
        for (int b = 0; b < BATCH; b++) {
            make_sample(x, content);
            ternary_stack_forward(s, xt, out, cn);
            make_grad(out, content, go);
            ternary_stack_backward(s, xt, go);
        }
        ternary_stack_apply_gradients(s, SCALE_LR);
    }
}

static TernaryStack *new_stack(unsigned seed, char *err, size_t cap) {
    const TernaryLayerType *t = ternary_layer_type_by_name("gqa");
    if (!t) return NULL;
    TernaryRng rng; ternary_rng_seed(&rng, seed);
    size_t shape[2] = { SEQ, E }; TernaryLayer l;
    if (!ternary_layer_type_build(t, shape, 2, PARAMS, 5, &rng, &l, err, cap)) return NULL;
    TernaryStack *s = ternary_stack_create();
    if (!ternary_stack_add(s, l, TERNARY_ACTIVATION_UINT8)) return NULL;
    return s;
}

int main(void) {
    TernaryCapabilities cn; memset(&cn, 0, sizeof(cn)); cn.num_cores = 1; cn.dot_impl_name = "scalar";
    char err[256] = "";

    /* --- 1. registro, describe y errores --- */
    const TernaryLayerType *t = ternary_layer_type_by_name("gqa");
    CHECK(t && strcmp(t->class_name, "GQA") == 0 && ternary_layer_type_num_slots(t) == 9, "tipo gqa en el registro");
    TernaryStack *sa = new_stack(42, err, sizeof(err));
    CHECK(sa != NULL, err);
    TernaryGQA *ga = gqa_of(sa);
    double back[TERNARY_MAX_LAYER_SLOTS]; size_t oshape[TERNARY_MAX_SHAPE_DIMS];
    size_t nd = t->describe(ga, back, oshape);
    CHECK(nd == 2 && oshape[0] == SEQ && oshape[1] == E, "forma de salida (seq, embed)");
    for (int i = 0; i < 5; i++) CHECK(back[i] == PARAMS[i], "describe devuelve los parametros con que se construyo");
    CHECK(ga->qk_int == 1 && ga->qk_shift == 2 && ga->rope_base == 10.0f, "RoPE entero configurado");
    CHECK(ternary_stack_input_len(sa) == SEQ * E && ternary_stack_output_len(sa) == SEQ * E, "largos de entrada/salida");

    TernaryRng rng; ternary_rng_seed(&rng, 1); TernaryLayer tmp;
    size_t shape2[2] = { SEQ, E }, shape1[1] = { SEQ * E };
    double bad_div[5] = { 3, 1, 1, 10.0, 2 }, bad_kv[5] = { 4, 3, 1, 10.0, 2 };
    double odd_hd[5] = { 16, 1, 1, 10.0, 2 }, bad_base[5] = { 2, 1, 1, 1.0, 2 }, bad_rope[5] = { 2, 1, 3, 10.0, 2 };
    CHECK(!ternary_layer_type_build(t, shape2, 2, bad_div, 5, &rng, &tmp, err, sizeof(err)) && strstr(err, "divisible"), "n_heads no divide embed_dim");
    CHECK(!ternary_layer_type_build(t, shape2, 2, bad_kv, 5, &rng, &tmp, err, sizeof(err)) && strstr(err, "divisible"), "n_kv_heads no divide n_heads");
    CHECK(!ternary_layer_type_build(t, shape2, 2, odd_hd, 5, &rng, &tmp, err, sizeof(err)) && strstr(err, "even"), "head_dim impar con RoPE");
    CHECK(!ternary_layer_type_build(t, shape2, 2, bad_base, 5, &rng, &tmp, err, sizeof(err)) && strstr(err, "rope_base"), "rope_base <= 1");
    CHECK(!ternary_layer_type_build(t, shape2, 2, bad_rope, 5, &rng, &tmp, err, sizeof(err)) && strstr(err, "rope"), "rope fuera de rango");
    CHECK(!ternary_layer_type_build(t, shape1, 1, PARAMS, 5, &rng, &tmp, err, sizeof(err)) && strstr(err, "GQA needs"), "entrada de 1 dimension");
    printf("[registro] GQA se construye por nombre, describe() devuelve los parametros, 6 errores de parametros/forma con mensaje OK\n");

    /* --- 2. stack == manual (mismas reglas) --- */
    TernaryGQA *gb = ternary_gqa_alloc(E, SEQ, 2, 1);
    CHECK(gb && ternary_gqa_enable_rope_int(gb, 10.0f, 2) == 0, "gqa manual");
    TernaryDense *src[4] = { ga->wq, ga->wk, ga->wv, ga->wo }, *dst[4] = { gb->wq, gb->wk, gb->wv, gb->wo };
    for (int m = 0; m < 4; m++) for (size_t o = 0; o < src[m]->output_dim; o++) for (size_t i = 0; i < src[m]->input_dim; i++)
        ternary_dense_set_weight(dst[m], o, i, ternary_dense_get_weight(src[m], o, i));
    TernaryGQA *g0 = ternary_gqa_alloc(E, SEQ, 2, 1);   /* copia inicial para comprobar que el entrenamiento movio algo */
    TernaryDense *d0[4] = { g0->wq, g0->wk, g0->wv, g0->wo };
    for (int m = 0; m < 4; m++) for (size_t o = 0; o < src[m]->output_dim; o++) for (size_t i = 0; i < src[m]->input_dim; i++)
        ternary_dense_set_weight(d0[m], o, i, ternary_dense_get_weight(src[m], o, i));

    TernaryVec *x = ternary_vec_alloc(SEQ * E); size_t xd[2] = { SEQ, E };
    TernaryTensor *xt = ternary_tensor_view_ternary(xd, 2, x);
    size_t gsz[4] = { dst[0]->output_dim * dst[0]->input_dim, dst[1]->output_dim * dst[1]->input_dim,
                      dst[2]->output_dim * dst[2]->input_dim, dst[3]->output_dim * dst[3]->input_dim };
    float *acc[4], *gw[4];
    for (int m = 0; m < 4; m++) { acc[m] = malloc(gsz[m] * 4); gw[m] = malloc(gsz[m] * 4); }
    float out[SEQ * E], go[SEQ * E], gx[SEQ * E], gs[4]; int content[SEQ][E]; int32_t raw[SEQ * E];
    srand(7);
    for (int ep = 0; ep < EPOCHS; ep++) {
        for (int m = 0; m < 4; m++) memset(acc[m], 0, gsz[m] * 4);
        float gs3 = 0.0f;
        ternary_stack_zero_gradients(sa);
        for (int b = 0; b < BATCH; b++) {
            make_sample(x, content);
            float so[SEQ * E];
            ternary_stack_forward(sa, xt, so, &cn);
            ternary_gqa_forward(gb, x, out, raw);
            CHECK(memcmp(so, out, sizeof(out)) == 0, "forward del stack difiere del manual");
            make_grad(out, content, go);
            ternary_stack_backward(sa, xt, go);
            memset(gx, 0, sizeof(gx)); memset(gs, 0, sizeof(gs));
            ternary_gqa_backward(gb, x, go, gw, gx, gs, raw);
            for (int m = 0; m < 4; m++) for (size_t i = 0; i < gsz[m]; i++) acc[m][i] += gw[m][i];
            gs3 += gs[3];
        }
        ternary_stack_apply_gradients(sa, SCALE_LR);
        for (int m = 0; m < 4; m++)
            for (size_t r = 0; r < dst[m]->output_dim; r++)
                ternary_dense_apply_row_gradients(dst[m], r, acc[m] + r * dst[m]->input_dim);
        ternary_dense_apply_scale_gradient(gb->wo, gs3, SCALE_LR);
    }
    CHECK(same_gqa(ga, gb), "pesos/escala de Wo difieren entre entrenar por el stack y entrenar a mano");
    CHECK(ga->wq->scale == 1.0f && ga->wk->scale == 1.0f && ga->wv->scale == 1.0f, "las escalas de Wq/Wk/Wv no deben cambiar");
    CHECK(ga->wo->scale != 1.0f, "la escala de Wo deberia haberse actualizado");
    CHECK(!same_gqa(ga, g0), "el entrenamiento no cambio ningun peso: la comparacion no prueba nada");
    printf("[stack vs manual] %d epocas x %d muestras: pesos de Wq/Wk/Wv/Wo y escala de Wo identicos bit a bit; escalas de Wq/Wk/Wv intactas (1.0); Wo.scale=%.4f\n", EPOCHS, BATCH, ga->wo->scale);

    /* --- 3. checkpoint completo: pausar/retomar == de un tiron --- */
    const char *path = tmp_path("test_gqa_layer_ckpt.bin"), *path_exp = tmp_path("test_gqa_layer_export.bin");
    TernaryStack *s1 = new_stack(42, err, sizeof(err)), *s2 = NULL;
    srand(11); train_stack(s1, xt, x, 6, &cn);
    CHECK(ternary_stack_save(s1, path, 1) == 1, "guardar checkpoint");
    s2 = ternary_stack_load(path);
    CHECK(s2 != NULL, "cargar checkpoint");
    CHECK(same_gqa(gqa_of(s1), gqa_of(s2)), "pesos/escala difieren tras guardar y cargar");
    CHECK(gqa_of(s2)->qk_int == 1 && gqa_of(s2)->qk_shift == 2 && gqa_of(s2)->rope_base == 10.0f && gqa_of(s2)->n_kv_heads == 1, "configuracion (RoPE, cabezas) no sobrevivio");
    srand(12); train_stack(s1, xt, x, 6, &cn);
    srand(12); train_stack(s2, xt, x, 6, &cn);
    CHECK(same_gqa(gqa_of(s1), gqa_of(s2)), "retomar tras cargar no da lo mismo que seguir de un tiron");
    printf("[checkpoint] guardar tras 6 epocas, cargar y entrenar 6 mas == entrenar 12 de un tiron (bit a bit), RoPE y cabezas preservados\n");

    /* --- 4. export empaquetado: inferencia identica --- */
    CHECK(ternary_stack_save(s1, path_exp, 0) == 1, "exportar");
    TernaryStack *s3 = ternary_stack_load(path_exp);
    CHECK(s3 != NULL, "cargar export");
    srand(13);
    for (int n = 0; n < 30; n++) {
        make_sample(x, content);
        float o1[SEQ * E], o3[SEQ * E];
        ternary_stack_forward(s1, xt, o1, &cn); ternary_stack_forward(s3, xt, o3, &cn);
        CHECK(memcmp(o1, o3, sizeof(o1)) == 0, "la inferencia del export difiere");
    }
    printf("[export] modelo empaquetado: 30 inferencias identicas a la original\n");

    model_dir_remove(path); model_dir_remove(path_exp);
    for (int m = 0; m < 4; m++) { free(acc[m]); free(gw[m]); }
    ternary_gqa_free(gb); ternary_gqa_free(g0);
    ternary_tensor_free(xt); ternary_vec_free(x);
    ternary_stack_free(sa); ternary_stack_free(s1); ternary_stack_free(s2); ternary_stack_free(s3);
    printf("\ntest_gqa_layer: TODO OK\n");
    return 0;
}
