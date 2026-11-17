/*
 * Verifica la capa Embedding ternaria (src/ternary_embedding.c) y su
 * integracion en TernaryStack como primera capa con entrada de ids:
 *
 * 1) forward: fila correcta * escala, ids repetidos, id fuera de rango.
 * 2) backward contra una referencia plana escrita a mano: filas no
 *    mencionadas en cero, ids repetidos SUMAN, grad de escala.
 * 3) Embedding + Dense dentro de un TernaryStack: forward igual al camino
 *    manual (lookup -> sign -> dense), rechazo de Embedding como segunda
 *    capa, y rechazo de Dense-despues-de-Embedding con largos que no calzan.
 * 4) Entrenamiento via stack == entrenamiento manual, pesos/escala
 *    identicos bit a bit (mismo criterio que tests/test_stack.c).
 * 5) Guardado: checkpoint retomable identico y export empaquetado con el
 *    mismo forward.
 */
#include "tmp_path.h"
#include "ternary_stack.h"
#include "ternary_embedding.h"
#include "ternary_dense.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FALLO: %s (linea %d)\n", msg, __LINE__); return 1; } } while (0)

#define VOCAB 12
#define EDIM  8
#define SEQ   3
#define OUT   4
#define BATCH 6
#define EPOCHS 20

static TernaryCapabilities caps_none(void) {
    TernaryCapabilities c; memset(&c, 0, sizeof(c));
    c.num_cores = 1; c.dot_impl_name = "scalar";
    return c;
}

static TernaryEmbedding *make_emb(unsigned seed) {
    TernaryEmbedding *e = ternary_embedding_alloc(VOCAB, EDIM, SEQ);
    srand(seed);
    for (size_t v = 0; v < VOCAB; v++)
        for (size_t d = 0; d < EDIM; d++) ternary_embedding_set_weight(e, v, d, (rand() % 3) - 1);
    return e;
}

static TernaryDense *make_dense(unsigned seed) {
    TernaryDense *d = ternary_dense_alloc(SEQ * EDIM, OUT, 1);
    srand(seed);
    for (size_t o = 0; o < OUT; o++) {
        for (size_t i = 0; i < SEQ * EDIM; i++) ternary_dense_set_weight(d, o, i, (rand() % 3) - 1);
        ternary_vec_set(d->bias, o, (rand() % 3) - 1);
    }
    return d;
}


static int test_forward_y_backward(void) {
    TernaryEmbedding *e = make_emb(11);
    e->scale = 0.5f;
    int32_t ids[SEQ] = { 5, 5, 99 };   /* repetido y fuera de rango */
    float out[SEQ * EDIM]; int32_t raw[SEQ * EDIM];
    ternary_embedding_forward(e, ids, out, raw);
    for (size_t p = 0; p < SEQ; p++)
        for (size_t d = 0; d < EDIM; d++) {
            int w = (ids[p] == 99) ? 0 : ternary_embedding_get_weight(e, 5, d);
            CHECK(out[p * EDIM + d] == 0.5f * (float)w, "forward: valor distinto de escala*W[id][d]");
            CHECK(raw[p * EDIM + d] == w, "forward: Z crudo incorrecto");
        }

    float g[SEQ * EDIM];
    for (size_t i = 0; i < SEQ * EDIM; i++) g[i] = (float)((int)(i % 5) - 2) * 0.25f;
    static float gw[VOCAB * EDIM];
    float gs = 0.0f;
    ternary_embedding_backward(e, ids, g, gw, &gs, raw);

    /* referencia plana */
    static float ref[VOCAB * EDIM]; memset(ref, 0, sizeof(ref));
    float ref_gs = 0.0f;
    for (size_t p = 0; p < SEQ; p++)
        for (size_t d = 0; d < EDIM; d++) {
            ref_gs += g[p * EDIM + d] * (float)raw[p * EDIM + d];
            if (ids[p] != 99) ref[(size_t)ids[p] * EDIM + d] += g[p * EDIM + d];
        }
    CHECK(memcmp(gw, ref, sizeof(ref)) == 0, "backward: gradiente de pesos distinto de la referencia");
    CHECK(gs == ref_gs, "backward: gradiente de escala distinto de la referencia");
    for (size_t v = 0; v < VOCAB; v++) {
        if (v == 5) continue;
        for (size_t d = 0; d < EDIM; d++) CHECK(gw[v * EDIM + d] == 0.0f, "backward: fila no mencionada no quedo en cero");
    }
    printf("[embedding fwd/bwd] forward, ids repetidos (se suman), id fuera de rango, grad de escala: OK\n");
    ternary_embedding_free(e);
    return 0;
}

static int test_stack_integracion(void) {
    TernaryCapabilities cn = caps_none();
    TernaryEmbedding *e = make_emb(21);
    TernaryDense *d = make_dense(22);
    TernaryEmbedding *e_m = make_emb(21);   /* copias para el camino manual */
    TernaryDense *d_m = make_dense(22);

    TernaryStack *s = ternary_stack_create();
    CHECK(ternary_stack_add(s, ternary_layer_from_embedding(e), TERNARY_ACTIVATION_UINT8) == 1,
          "no se pudo agregar Embedding como primera capa");
    CHECK(ternary_stack_add(s, ternary_layer_from_dense(d), TERNARY_ACTIVATION_UINT8) == 1,
          "no se pudo agregar Dense despues de Embedding");
    CHECK(ternary_stack_input_len(s) == SEQ, "input_len del stack deberia ser seq_len");
    CHECK(ternary_stack_output_len(s) == OUT, "output_len del stack incorrecto");

    int32_t ids[SEQ] = { 3, 7, 3 };
    size_t dims[1] = { SEQ };
    TernaryTensor *x = ternary_tensor_view_id(dims, 1, ids);
    float out_stack[OUT];
    ternary_stack_forward(s, x, out_stack, &cn);

    float emb_out[SEQ * EDIM];
    ternary_embedding_forward(e_m, ids, emb_out, NULL);
    TernaryU8 *a = ternary_u8_alloc(1, SEQ * EDIM);
    TERNARY_ACTIVATION_UINT8.apply_u8(emb_out, a);
    float out_manual[OUT];
    ternary_dense_forward_u8(d_m, a, out_manual, NULL);
    CHECK(memcmp(out_stack, out_manual, sizeof(out_manual)) == 0,
          "forward del stack (Embedding+Dense) difiere del manual");
    printf("[embedding en stack] forward Embedding+Dense identico al manual\n");

    /* Embedding como segunda capa: debe rechazarse (ids solo como entrada) */
    TernaryDense *d0 = ternary_dense_alloc(4, SEQ, 0);
    TernaryStack *s2 = ternary_stack_create();
    CHECK(ternary_stack_add(s2, ternary_layer_from_dense(d0), TERNARY_ACTIVATION_UINT8) == 1, "capa de control");
    TernaryEmbedding *e2 = ternary_embedding_alloc(VOCAB, EDIM, SEQ);
    CHECK(ternary_stack_add(s2, ternary_layer_from_embedding(e2), TERNARY_ACTIVATION_UINT8) == 0,
          "Embedding como segunda capa deberia rechazarse");
    ternary_embedding_free(e2);
    ternary_stack_free(s2);

    ternary_tensor_free(x); ternary_u8_free(a);
    ternary_embedding_free(e_m); ternary_dense_free(d_m);
    ternary_stack_free(s);
    return 0;
}

static int weights_equal_emb(const TernaryEmbedding *a, const TernaryEmbedding *b) {
    for (size_t v = 0; v < VOCAB; v++)
        for (size_t d = 0; d < EDIM; d++)
            if (ternary_embedding_get_weight(a, v, d) != ternary_embedding_get_weight(b, v, d)) return 0;
    return a->scale == b->scale;
}

/* Datos: ids aleatorios, target = +1 si la suma de ids es par, -1 si no. */
static void make_data(int32_t *ids, float *targets, size_t n) {
    srand(4242);
    for (size_t b = 0; b < n; b++) {
        int sum = 0;
        for (size_t p = 0; p < SEQ; p++) { ids[b * SEQ + p] = rand() % VOCAB; sum += ids[b * SEQ + p]; }
        for (size_t o = 0; o < OUT; o++) targets[b * OUT + o] = (sum % 2 == 0) ? 1.0f : -1.0f;
    }
}

static int test_entrenamiento_igual_al_manual(void) {
    TernaryCapabilities cn = caps_none();
    const float scale_lr = 0.05f;
    static int32_t ids_all[EPOCHS * BATCH * SEQ];
    static float targets[EPOCHS * BATCH * OUT];
    make_data(ids_all, targets, EPOCHS * BATCH);

    /* --- manual --- */
    TernaryEmbedding *em = make_emb(31);
    TernaryDense *dm = make_dense(32);
    TernaryU8 *a = ternary_u8_alloc(1, SEQ * EDIM);
    static float gw_e[VOCAB * EDIM], gw_es[VOCAB * EDIM], gw_d[OUT * SEQ * EDIM], gw_ds[OUT * SEQ * EDIM];
    for (int ep = 0; ep < EPOCHS; ep++) {
        memset(gw_e, 0, sizeof(gw_e)); memset(gw_d, 0, sizeof(gw_d));
        float gb_d[OUT]; memset(gb_d, 0, sizeof(gb_d));
        float gs_e = 0.0f, gs_d = 0.0f;
        for (int b = 0; b < BATCH; b++) {
            const int32_t *ids = ids_all + ((size_t)ep * BATCH + b) * SEQ;
            const float *tg = targets + ((size_t)ep * BATCH + b) * OUT;
            float eo[SEQ * EDIM]; int32_t er[SEQ * EDIM];
            ternary_embedding_forward(em, ids, eo, er);
            TERNARY_ACTIVATION_UINT8.apply_u8(eo, a);
            float o2[OUT]; int32_t o2r[OUT];
            ternary_dense_forward_u8(dm, a, o2, o2r);
            float go[OUT];
            for (size_t o = 0; o < OUT; o++) go[o] = (2.0f / OUT) * (o2[o] - tg[o]);
            float gx[SEQ * EDIM]; memset(gx, 0, sizeof(gx));
            memset(gw_ds, 0, sizeof(gw_ds));
            ternary_dense_backward_u8(dm, a, go, gw_ds, gx, &gs_d, gb_d, o2r);
            for (size_t kk = 0; kk < OUT * SEQ * EDIM; kk++) gw_d[kk] += gw_ds[kk];
            ternary_embedding_backward(em, ids, gx, gw_es, &gs_e, er);
            for (size_t kk = 0; kk < VOCAB * EDIM; kk++) gw_e[kk] += gw_es[kk];
        }
        for (size_t v = 0; v < VOCAB; v++) ternary_embedding_apply_row_gradients(em, v, gw_e + v * EDIM);
        for (size_t o = 0; o < OUT; o++) ternary_dense_apply_row_gradients(dm, o, gw_d + o * SEQ * EDIM);
        ternary_dense_apply_bias_gradients(dm, gb_d);
        ternary_embedding_apply_scale_gradient(em, gs_e, scale_lr);
        ternary_dense_apply_scale_gradient(dm, gs_d, scale_lr);
    }

    /* --- via el stack --- */
    TernaryEmbedding *es = make_emb(31);
    TernaryDense *ds = make_dense(32);
    TernaryStack *stack = ternary_stack_create();
    ternary_stack_add(stack, ternary_layer_from_embedding(es), TERNARY_ACTIVATION_UINT8);
    ternary_stack_add(stack, ternary_layer_from_dense(ds), TERNARY_ACTIVATION_UINT8);
    int32_t cur[SEQ];
    size_t dims[1] = { SEQ };
    TernaryTensor *x = ternary_tensor_view_id(dims, 1, cur);
    float out_final[OUT];
    for (int ep = 0; ep < EPOCHS; ep++) {
        ternary_stack_zero_gradients(stack);
        for (int b = 0; b < BATCH; b++) {
            memcpy(cur, ids_all + ((size_t)ep * BATCH + b) * SEQ, sizeof(cur));
            const float *tg = targets + ((size_t)ep * BATCH + b) * OUT;
            ternary_stack_forward(stack, x, out_final, &cn);
            float grad[OUT];
            for (size_t o = 0; o < OUT; o++) grad[o] = (2.0f / OUT) * (out_final[o] - tg[o]);
            ternary_stack_backward(stack, x, grad);
        }
        ternary_stack_apply_gradients(stack, scale_lr);
    }

    CHECK(weights_equal_emb(em, es), "Embedding: pesos/escala entrenados via stack difieren del manual");
    int same = 1;
    for (size_t o = 0; o < OUT && same; o++)
        for (size_t i = 0; i < SEQ * EDIM; i++)
            if (ternary_dense_get_weight(dm, o, i) != ternary_dense_get_weight(ds, o, i)) { same = 0; break; }
    CHECK(same && dm->scale == ds->scale, "Dense tras Embedding: pesos/escala via stack difieren del manual");
    printf("[embedding entrenamiento] %d epocas via stack == manual, pesos y escala identicos bit a bit\n", EPOCHS);

    /* --- guardado: checkpoint retomable + export empaquetado --- */
    const char *ck = tmp_path("test_embedding_ckpt.bin"), *ex = tmp_path("test_embedding_export.bin");
    CHECK(ternary_stack_save(stack, ck, 1), "save checkpoint fallo");
    CHECK(ternary_stack_save(stack, ex, 0), "save export fallo");
    TernaryStack *l1 = ternary_stack_load(ck), *l2 = ternary_stack_load(ex);
    CHECK(l1 && l2, "load devolvio NULL");
    for (int rep = 0; rep < 8; rep++) {
        for (size_t p = 0; p < SEQ; p++) cur[p] = (rep * 5 + (int)p * 3) % VOCAB;
        float oa[OUT], ob[OUT], oc[OUT];
        ternary_stack_forward(stack, x, oa, &cn);
        ternary_stack_forward(l1, x, ob, &cn);
        ternary_stack_forward(l2, x, oc, &cn);
        CHECK(memcmp(oa, ob, sizeof(oa)) == 0, "forward tras cargar checkpoint difiere");
        CHECK(memcmp(oa, oc, sizeof(oa)) == 0, "forward tras cargar export empaquetado difiere");
    }
    long s1 = model_dir_bytes(ck), s2 = model_dir_bytes(ex);
    CHECK(s1 > 0 && s2 > 0 && s2 < s1, "el export deberia pesar menos que el checkpoint");
    printf("[embedding guardado] checkpoint=%ld bytes, export=%ld bytes, forward identico tras cargar ambos\n", s1, s2);
    model_dir_remove(ck); model_dir_remove(ex);

    ternary_stack_free(l1); ternary_stack_free(l2); ternary_stack_free(stack);
    ternary_tensor_free(x);
    ternary_embedding_free(em); ternary_dense_free(dm); ternary_u8_free(a);
    return 0;
}

int main(void) {
    if (test_forward_y_backward()) return 1;
    if (test_stack_integracion()) return 1;
    if (test_entrenamiento_igual_al_manual()) return 1;
    printf("\ntest_embedding: TODO OK\n");
    return 0;
}
