/*
 * Cabeza de salida ternaria ("lm_head", src/ternary_head.c) y perdida de entropia cruzada
 * (src/ternary_loss.c). Verifica:
 *  1. registro por nombre, describe() de vuelta, formas 1D/2D y errores de parametros con mensaje;
 *  2. forward IDENTICO a una referencia escrita aparte (pesos leidos con get_weight): Z crudo exacto y
 *     logits = escala * Z, por posicion (la matriz es compartida, las posiciones no se mezclan);
 *  3. perdida: contra una referencia en double, gradiente contra diferencias finitas centrales, filas
 *     ignoradas (target < 0), target fuera de rango -> NaN, estabilidad con logits de +-1e4;
 *  4. backward igual a una referencia en double (gradiente de W, de la entrada y de la escala);
 *  5. capa dentro del TernaryStack: forward del stack == directo; checkpoint completo (20+20 == 40) y export
 *     empaquetado (inferencia identica);
 *  6. APRENDIZAJE con entropia cruzada: (a) la cabeza sola contra las etiquetas de un maestro, (b)
 *     Embedding -> cabeza aprende un "siguiente token" fijo desde ids (el camino de un modelo de lenguaje).
 */
#include "tmp_path.h"
#include "ternary_stack.h"
#include "ternary_layer.h"
#include "ternary_activation.h"
#include "ternary_embedding.h"
#include "ternary_loss.h"
#include "ternary_rng.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FALLO: %s (linea %d)\n", msg, __LINE__); return 1; } } while (0)

#define ROWS 4
#define DIM 24
#define VOC 11

static uint64_t rs = 88172645463325252ull;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) / 9007199254740992.0; }
static double nrand(void) { double u = urand() + 1e-12, v = urand(); return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v); }

static void random_x(TernaryVec *x, size_t n) {
    for (size_t i = 0; i < n; i++) ternary_vec_set(x, i, (int)(urand() * 3.0) - 1);
}

static TernaryHead *make_head(size_t rows, unsigned seed) {
    TernaryRng rng; ternary_rng_seed(&rng, seed);
    size_t in[2] = { rows, DIM };
    double p[1] = { VOC };
    TernaryLayer l; char err[200];
    if (!ternary_layer_type_build(ternary_layer_type_by_name("lm_head"), in, rows == 1 ? 1 : 2,
                                  p, 1, &rng, &l, err, sizeof(err))) {
        fprintf(stderr, "build: %s\n", err); return NULL;
    }
    return (TernaryHead *)l.impl;
}

static int test_registry(void) {
    const TernaryLayerType *ty = ternary_layer_type_by_name("lm_head");
    CHECK(ty != NULL, "registro: lm_head");
    CHECK(strcmp(ty->class_name, "LMHead") == 0, "nombre de clase");
    CHECK(ternary_layer_type_num_slots(ty) == 1, "1 parametro");

    TernaryRng rng; ternary_rng_seed(&rng, 1);
    char err[200]; TernaryLayer l;
    size_t in1[1] = { DIM }, in2[2] = { ROWS, DIM };
    double bad_vocab[1] = { 0 };
    CHECK(!ternary_layer_type_build(ty, in2, 2, bad_vocab, 1, &rng, &l, err, sizeof(err)) && strstr(err, "vocab"), "vocab 0");

    double p[1] = { VOC };
    CHECK(ternary_layer_type_build(ty, in2, 2, p, 1, &rng, &l, err, sizeof(err)), "build 2D");
    double back[2]; size_t shape[TERNARY_MAX_SHAPE_DIMS];
    size_t nd = ty->describe(l.impl, back, shape);
    CHECK(nd == 2 && shape[0] == ROWS && shape[1] == VOC, "describe: forma 2D");
    CHECK(back[0] == VOC, "describe: parametros");
    CHECK(l.vt->input_len(l.impl) == ROWS * DIM && l.vt->output_len(l.impl) == ROWS * VOC, "largos");
    CHECK(l.vt->num_matrices(l.impl) == 1 && l.vt->grad_rows(l.impl, 0) == VOC && l.vt->grad_row_width(l.impl, 0) == DIM, "forma de la matriz");
    CHECK(l.vt->input_dtype(l.impl) == TERNARY_DTYPE_TERNARY && l.vt->has_bias(l.impl, 0) == 0, "dtype ternario, sin bias");
    l.vt->destroy(l.impl);

    CHECK(ternary_layer_type_build(ty, in1, 1, p, 1, &rng, &l, err, sizeof(err)), "build 1D");
    nd = ty->describe(l.impl, back, shape);
    CHECK(nd == 1 && shape[0] == VOC, "describe: forma 1D");
    l.vt->destroy(l.impl);
    return 0;
}

static int test_forward_reference(void) {
    TernaryHead *h = make_head(ROWS, 11);
    CHECK(h, "build");
    h->w->scale = 0.37f;
    TernaryVec *x = ternary_vec_alloc(ROWS * DIM);
    random_x(x, ROWS * DIM);
    float out[ROWS * VOC]; int32_t raw[ROWS * VOC];
    ternary_head_forward(h, x, out, raw);
    for (size_t p = 0; p < ROWS; p++)
        for (size_t v = 0; v < VOC; v++) {
            int32_t z = 0;
            for (size_t i = 0; i < DIM; i++) z += ternary_dense_get_weight(h->w, v, i) * ternary_vec_get(x, p * DIM + i);
            CHECK(raw[p * VOC + v] == z, "forward: Z crudo distinto de la referencia");
            CHECK(out[p * VOC + v] == h->w->scale * (float)z, "forward: logit distinto de escala*Z");
        }
    ternary_vec_free(x); ternary_head_free(h);
    return 0;
}

static int test_loss(void) {
    /* contra una referencia en double + argmax/aciertos */
    enum { R = 5, V = 7 };
    float logits[R * V], grad[R * V];
    int32_t tg[R] = { 3, 0, -1, 6, 2 };
    for (int i = 0; i < R * V; i++) logits[i] = (float)(nrand() * 2.0);
    size_t nv = 0, nc = 0;
    double loss = ternary_xent(logits, R, V, tg, grad, &nv, &nc);
    double ref = 0.0; size_t cnt = 0, correct = 0;
    for (int p = 0; p < R; p++) {
        if (tg[p] < 0) continue;
        double m = logits[p * V]; int arg = 0;
        for (int v = 1; v < V; v++) if (logits[p * V + v] > m) { m = logits[p * V + v]; arg = v; }
        double sum = 0; for (int v = 0; v < V; v++) sum += exp((double)logits[p * V + v] - m);
        ref += log(sum) - ((double)logits[p * V + tg[p]] - m);
        cnt++; if (arg == tg[p]) correct++;
    }
    ref /= (double)cnt;
    CHECK(nv == cnt && nc == correct && nv == 4, "filas validas / aciertos");
    CHECK(fabs(loss - ref) < 1e-6, "perdida distinta de la referencia en double");

    /* fila ignorada: gradiente exactamente 0; las validas suman 0 por fila (softmax - onehot) */
    for (int v = 0; v < V; v++) CHECK(grad[2 * V + v] == 0.0f, "fila ignorada: gradiente != 0");
    for (int p = 0; p < R; p++) {
        double s = 0; for (int v = 0; v < V; v++) s += grad[p * V + v];
        CHECK(fabs(s) < 1e-6, "el gradiente de cada fila suma 0");
    }
    /* gradiente vs diferencias finitas centrales */
    double worst = 0.0;
    for (int k = 0; k < R * V; k++) {
        const float e = 1e-2f, keep = logits[k];
        logits[k] = keep + e; double lp = ternary_xent(logits, R, V, tg, NULL, NULL, NULL);
        logits[k] = keep - e; double lm = ternary_xent(logits, R, V, tg, NULL, NULL, NULL);
        logits[k] = keep;
        const double fd = (lp - lm) / (2.0 * (double)e);
        worst = fmax(worst, fabs(fd - (double)grad[k]));
    }
    CHECK(worst < 2e-4, "gradiente de la perdida distinto de diferencias finitas");

    /* solo perdida (grad NULL) da lo mismo */
    CHECK(ternary_xent(logits, R, V, tg, NULL, NULL, NULL) == loss, "perdida sin gradiente != con gradiente");

    /* target fuera de rango -> NaN y gradiente en cero; nunca se ignora en silencio */
    int32_t bad[R] = { 3, 0, 1, V, 2 };
    for (int i = 0; i < R * V; i++) grad[i] = 5.0f;
    CHECK(isnan(ternary_xent(logits, R, V, bad, grad, &nv, &nc)) && nv == 0, "target fuera de rango debe dar NaN");
    for (int i = 0; i < R * V; i++) CHECK(grad[i] == 0.0f, "target invalido: gradiente en cero");

    /* sin filas validas */
    int32_t none[R] = { -1, -1, -1, -1, -1 };
    CHECK(ternary_xent(logits, R, V, none, grad, &nv, &nc) == 0.0 && nv == 0, "sin filas validas: perdida 0");

    /* estabilidad: logits enormes no producen inf/NaN */
    float big[2 * V];
    for (int v = 0; v < V; v++) { big[v] = 1e4f * (float)v; big[V + v] = -1e4f * (float)v; }
    int32_t tb[2] = { V - 1, 0 };
    double lb = ternary_xent(big, 2, V, tb, grad, &nv, &nc);
    CHECK(isfinite(lb) && lb == 0.0 && nc == 2, "logits +-1e4: perdida finita (0: el objetivo es el maximo)");
    for (int i = 0; i < 2 * V; i++) CHECK(isfinite(grad[i]), "logits +-1e4: gradiente finito");
    tb[0] = 0;
    lb = ternary_xent(big, 2, V, tb, NULL, NULL, NULL);
    CHECK(isfinite(lb) && lb > 1e4, "objetivo muy improbable: perdida grande pero finita");
    return 0;
}

static int test_backward_reference(void) {
    TernaryHead *h = make_head(ROWS, 31);
    CHECK(h, "build");
    h->w->scale = 0.8f;
    TernaryVec *x = ternary_vec_alloc(ROWS * DIM);
    random_x(x, ROWS * DIM);
    float out[ROWS * VOC], go[ROWS * VOC];
    ternary_head_forward(h, x, out, NULL);
    for (size_t i = 0; i < ROWS * VOC; i++) go[i] = (float)nrand();
    go[5] = 0.0f; go[17] = 0.0f;    /* gradientes exactamente 0: el atajo de saltearlos no debe cambiar nada */

    float *gw = (float *)malloc(VOC * DIM * sizeof(float));
    float gx[ROWS * DIM]; memset(gx, 0, sizeof(gx));
    float gs = 0.0f;
    ternary_head_backward(h, x, go, gw, gx, &gs);

    double maxd = 0, maxref = 0, ref_gs = 0;
    double rgx[ROWS * DIM]; memset(rgx, 0, sizeof(rgx));
    double *rgw = (double *)calloc(VOC * DIM, sizeof(double));
    for (size_t p = 0; p < ROWS; p++)
        for (size_t v = 0; v < VOC; v++) {
            int32_t z = 0;
            for (size_t i = 0; i < DIM; i++) {
                const int xi = ternary_vec_get(x, p * DIM + i), wi = ternary_dense_get_weight(h->w, v, i);
                z += wi * xi;
                rgw[v * DIM + i] += (double)go[p * VOC + v] * xi;
                rgx[p * DIM + i] += (double)go[p * VOC + v] * (double)h->w->scale * wi;
            }
            ref_gs += (double)go[p * VOC + v] * (double)z;
        }
    for (size_t k = 0; k < VOC * DIM; k++) { maxd = fmax(maxd, fabs(rgw[k] - gw[k])); maxref = fmax(maxref, fabs(rgw[k])); }
    for (size_t k = 0; k < ROWS * DIM; k++) { maxd = fmax(maxd, fabs(rgx[k] - gx[k])); maxref = fmax(maxref, fabs(rgx[k])); }
    CHECK(maxref > 1e-2, "los gradientes de referencia no deben ser ~0");
    CHECK(maxd < 1e-4 * (1.0 + maxref), "backward: gradientes (W, x) distintos de la referencia");
    CHECK(fabs(ref_gs - gs) < 1e-3 * (1.0 + fabs(ref_gs)), "backward: gradiente de escala");

    /* el gradiente de la escala tambien coincide con diferencias finitas sobre la perdida real */
    {
        int32_t tg[ROWS] = { 1, 4, 0, 7 };
        float lg[ROWS * VOC], gl[ROWS * VOC], gw2[VOC * DIM], gx2[ROWS * DIM]; float gs2 = 0.0f;
        memset(gx2, 0, sizeof(gx2));
        ternary_head_forward(h, x, lg, NULL);
        ternary_xent(lg, ROWS, VOC, tg, gl, NULL, NULL);
        ternary_head_backward(h, x, gl, gw2, gx2, &gs2);
        const float s0 = h->w->scale, e = 1e-3f;
        float lp[ROWS * VOC], lm[ROWS * VOC];
        h->w->scale = s0 + e; ternary_head_forward(h, x, lp, NULL);
        h->w->scale = s0 - e; ternary_head_forward(h, x, lm, NULL);
        h->w->scale = s0;
        double fd = (ternary_xent(lp, ROWS, VOC, tg, NULL, NULL, NULL) - ternary_xent(lm, ROWS, VOC, tg, NULL, NULL, NULL)) / (2.0 * (double)e);
        CHECK(fabs(fd - (double)gs2) < 2e-3 * (1.0 + fabs(fd)), "d(perdida)/d(escala) analitico distinto de diferencias finitas");
    }
    free(gw); free(rgw); ternary_vec_free(x); ternary_head_free(h);
    return 0;
}

/* ---------------- stack ---------------- */
#define NIN (ROWS * DIM)
#define NOUT (ROWS * VOC)
static const float SCALE_LR = 0.01f;
#define BATCH 8

static TernaryStack *new_head_stack(unsigned seed) {
    TernaryRng rng; ternary_rng_seed(&rng, seed);
    size_t in[2] = { ROWS, DIM };
    double p[1] = { VOC };
    TernaryLayer l; char err[200];
    if (!ternary_layer_type_build(ternary_layer_type_by_name("lm_head"), in, 2, p, 1, &rng, &l, err, sizeof(err))) return NULL;
    TernaryStack *s = ternary_stack_create();
    if (!ternary_stack_add(s, l, TERNARY_ACTIVATION_UINT8)) return NULL;
    return s;
}
static const TernaryHead *head_at(const TernaryStack *s) { return (const TernaryHead *)ternary_stack_layer(s, 0).impl; }
static int same_dense(const TernaryDense *a, const TernaryDense *b) {
    if (a->input_dim != b->input_dim || a->output_dim != b->output_dim || a->scale != b->scale) return 0;
    for (size_t o = 0; o < a->output_dim; o++) for (size_t i = 0; i < a->input_dim; i++)
        if (ternary_dense_get_weight(a, o, i) != ternary_dense_get_weight(b, o, i)) return 0;
    return 1;
}

static TernaryStack *g_teacher;
static TernaryCapabilities g_cn;

/* Muestra: x aleatorio; etiqueta de cada posicion = argmax de los logits del maestro. */
static void make_sample(TernaryVec *x, int32_t *targets) {
    random_x(x, NIN);
    size_t xd[1] = { NIN }; TernaryTensor *xt = ternary_tensor_view_ternary(xd, 1, x);
    float t[NOUT];
    ternary_stack_forward(g_teacher, xt, t, &g_cn);
    ternary_tensor_free(xt);
    for (size_t p = 0; p < ROWS; p++) {
        size_t arg = 0;
        for (size_t v = 1; v < VOC; v++) if (t[p * VOC + v] > t[p * VOC + arg]) arg = v;
        targets[p] = (int32_t)arg;
    }
}

/* Una epoca = BATCH muestras con entropia cruzada. Devuelve la perdida media del lote. */
static double train_epoch(TernaryStack *s) {
    TernaryVec *x = ternary_vec_alloc(NIN); size_t xd[1] = { NIN };
    TernaryTensor *xt = ternary_tensor_view_ternary(xd, 1, x);
    float out[NOUT], go[NOUT]; int32_t tg[ROWS];
    double loss = 0.0;
    ternary_stack_zero_gradients(s);
    for (int b = 0; b < BATCH; b++) {
        make_sample(x, tg);
        ternary_stack_forward(s, xt, out, &g_cn);
        loss += ternary_xent(out, ROWS, VOC, tg, go, NULL, NULL);
        for (size_t i = 0; i < NOUT; i++) go[i] /= (float)BATCH;
        ternary_stack_backward(s, xt, go);
    }
    ternary_stack_apply_gradients(s, SCALE_LR);
    ternary_tensor_free(xt); ternary_vec_free(x);
    return loss / BATCH;
}

/* Perdida y precision (aciertos por posicion) sobre un conjunto FIJO; no altera el generador de entrenamiento. */
static void eval_stack(TernaryStack *s, int n, double *loss_out, double *acc_out) {
    const uint64_t keep = rs;
    rs = 0xD1B54A32D192ED03ull;
    TernaryVec *x = ternary_vec_alloc(NIN); size_t xd[1] = { NIN };
    TernaryTensor *xt = ternary_tensor_view_ternary(xd, 1, x);
    float out[NOUT]; int32_t tg[ROWS];
    double loss = 0.0; size_t hits = 0, total = 0;
    for (int i = 0; i < n; i++) {
        make_sample(x, tg);
        ternary_stack_forward(s, xt, out, &g_cn);
        size_t nv, nc;
        loss += ternary_xent(out, ROWS, VOC, tg, NULL, &nv, &nc);
        hits += nc; total += nv;
    }
    ternary_tensor_free(xt); ternary_vec_free(x);
    rs = keep;
    *loss_out = loss / n; *acc_out = (double)hits / (double)total;
}

static int test_stack_and_persistence(void) {
    memset(&g_cn, 0, sizeof(g_cn)); g_cn.num_cores = 1; g_cn.dot_impl_name = "scalar";
    g_teacher = new_head_stack(777);
    CHECK(g_teacher, "maestro");

    {   /* forward del stack == forward directo */
        TernaryStack *s = new_head_stack(5);
        CHECK(s, "stack");
        TernaryVec *x = ternary_vec_alloc(NIN); random_x(x, NIN);
        size_t xd[1] = { NIN }; TernaryTensor *xt = ternary_tensor_view_ternary(xd, 1, x);
        float a[NOUT], b[NOUT];
        ternary_stack_forward(s, xt, a, &g_cn);
        ternary_head_forward((TernaryHead *)head_at(s), x, b, NULL);
        CHECK(memcmp(a, b, sizeof(a)) == 0, "forward del stack distinto del directo");
        ternary_tensor_free(xt); ternary_vec_free(x); ternary_stack_free(s);
    }

    const char *ck = tmp_path("test_head_ck.bin"), *ex = tmp_path("test_head_ex.bin");
    const uint64_t rs0 = 123456789012345ull;
    rs = rs0;
    TernaryStack *a = new_head_stack(9);
    for (int e = 0; e < 40; e++) train_epoch(a);
    rs = rs0;
    TernaryStack *b = new_head_stack(9);
    for (int e = 0; e < 20; e++) train_epoch(b);
    CHECK(ternary_stack_save(b, ck, 1), "guardar checkpoint");
    TernaryStack *c = ternary_stack_load(ck);
    CHECK(c, "cargar checkpoint");
    CHECK(same_dense(head_at(b)->w, head_at(c)->w) && head_at(c)->rows == ROWS && head_at(c)->vocab == VOC, "el checkpoint cargado difiere del guardado");
    for (int e = 0; e < 20; e++) train_epoch(c);
    CHECK(same_dense(head_at(a)->w, head_at(c)->w), "checkpoint: 20+20 distinto de 40 de un tiron");

    CHECK(ternary_stack_save(a, ex, 0), "guardar export");
    TernaryStack *e2 = ternary_stack_load(ex);
    CHECK(e2, "cargar export");
    TernaryVec *x = ternary_vec_alloc(NIN); size_t xd[1] = { NIN };
    TernaryTensor *xt = ternary_tensor_view_ternary(xd, 1, x);
    for (int k = 0; k < 10; k++) {
        random_x(x, NIN);
        float o1[NOUT], o2[NOUT];
        ternary_stack_forward(a, xt, o1, &g_cn);
        ternary_stack_forward(e2, xt, o2, &g_cn);
        CHECK(memcmp(o1, o2, sizeof(o1)) == 0, "export: inferencia distinta");
    }
    ternary_tensor_free(xt); ternary_vec_free(x);
    ternary_stack_free(a); ternary_stack_free(b); ternary_stack_free(c); ternary_stack_free(e2);
    ternary_stack_free(g_teacher); g_teacher = NULL;
    model_dir_remove(ck); model_dir_remove(ex);
    return 0;
}

static int test_learning_head_alone(void) {
    memset(&g_cn, 0, sizeof(g_cn)); g_cn.num_cores = 1; g_cn.dot_impl_name = "scalar";
    g_teacher = new_head_stack(4242);
    CHECK(g_teacher, "maestro");
    rs = 0x9E3779B97F4A7C15ull;
    TernaryStack *s = new_head_stack(17);
    CHECK(s, "alumno");
    double l0, a0, l1, a1;
    eval_stack(s, 64, &l0, &a0);
    for (int e = 0; e < 300; e++) train_epoch(s);
    eval_stack(s, 64, &l1, &a1);
    printf("APRENDIZAJE cabeza sola (entropia cruzada, 300 epocas): perdida %.3f -> %.3f | precision %.1f%% -> %.1f%% (azar %.1f%%)\n",
           l0, l1, 100.0 * a0, 100.0 * a1, 100.0 / VOC);
    CHECK(l1 < 0.6 * l0, "la perdida de la cabeza sola no baja lo suficiente");
    CHECK(a1 > 3.0 * (1.0 / VOC) && a1 > a0 + 0.30, "la precision de la cabeza sola no supera claramente al azar");
    ternary_stack_free(s); ternary_stack_free(g_teacher); g_teacher = NULL;
    return 0;
}

/* Embedding(ids) -> cabeza: aprende un "siguiente token" fijo (tabla aleatoria) -- el camino de un modelo de lenguaje. */
#define EV 16     /* vocabulario */
#define ED 32     /* dimension del embedding */
#define ES 6      /* posiciones por muestra */
static int32_t g_next[EV];

static TernaryStack *new_lm(unsigned seed) {
    TernaryRng rng; ternary_rng_seed(&rng, seed);
    size_t in0[1] = { ES }, in1[2] = { ES, ED };
    double pe[2] = { EV, ED }, ph[1] = { EV };
    TernaryLayer l1, l2; char err[200];
    if (!ternary_layer_type_build(ternary_layer_type_by_name("embedding"), in0, 1, pe, 2, &rng, &l1, err, sizeof(err))) return NULL;
    if (!ternary_layer_type_build(ternary_layer_type_by_name("lm_head"), in1, 2, ph, 1, &rng, &l2, err, sizeof(err))) return NULL;
    TernaryStack *s = ternary_stack_create();
    if (!ternary_stack_add(s, l1, TERNARY_ACTIVATION_UINT8) || !ternary_stack_add(s, l2, TERNARY_ACTIVATION_UINT8)) return NULL;
    return s;
}

static int test_learning_embedding_head(void) {
    memset(&g_cn, 0, sizeof(g_cn)); g_cn.num_cores = 1; g_cn.dot_impl_name = "scalar";
    rs = 0xA24BAED4963EE407ull;
    for (int v = 0; v < EV; v++) g_next[v] = (int32_t)(urand() * EV);
    TernaryStack *s = new_lm(23);
    CHECK(s, "modelo");
    size_t xd[1] = { ES };
    int32_t ids[ES], tg[ES];
    float out[ES * EV], go[ES * EV];
    double first = 0.0, last = 0.0; double acc = 0.0;
    for (int ep = 0; ep < 600; ep++) {
        ternary_stack_zero_gradients(s);
        double loss = 0.0;
        for (int b = 0; b < BATCH; b++) {
            for (int p = 0; p < ES; p++) { ids[p] = (int32_t)(urand() * EV); tg[p] = g_next[ids[p]]; }
            TernaryTensor *xt = ternary_tensor_view_id(xd, 1, ids);
            ternary_stack_forward(s, xt, out, &g_cn);
            loss += ternary_xent(out, ES, EV, tg, go, NULL, NULL);
            for (int i = 0; i < ES * EV; i++) go[i] /= (float)BATCH;
            ternary_stack_backward(s, xt, go);
            ternary_tensor_free(xt);
        }
        ternary_stack_apply_gradients(s, SCALE_LR);
        loss /= BATCH;
        if (ep == 0) first = loss;
        last = loss;
    }
    /* precision sobre TODOS los ids (cada uno por separado, en cada posicion) */
    size_t hits = 0, total = 0;
    for (int v = 0; v < EV; v++) {
        for (int p = 0; p < ES; p++) ids[p] = (int32_t)v;
        for (int p = 0; p < ES; p++) tg[p] = g_next[v];
        TernaryTensor *xt = ternary_tensor_view_id(xd, 1, ids);
        ternary_stack_forward(s, xt, out, &g_cn);
        size_t nv, nc; ternary_xent(out, ES, EV, tg, NULL, &nv, &nc);
        hits += nc; total += nv;
        ternary_tensor_free(xt);
    }
    acc = (double)hits / (double)total;
    printf("APRENDIZAJE Embedding->cabeza (siguiente token fijo, %d ids, 600 epocas): perdida %.3f -> %.3f | precision %.1f%% (azar %.1f%%)\n",
           EV, first, last, 100.0 * acc, 100.0 / EV);
    CHECK(last < 0.5 * first, "Embedding->cabeza: la perdida no baja lo suficiente");
    CHECK(acc > 0.5, "Embedding->cabeza: la precision no supera el 50%");
    ternary_stack_free(s);
    return 0;
}

/* ternary_dense_backward_accumulate: sumar dos muestras directo == pisar y sumar a mano (bit a bit). Es la funcion
 * con la que GQA y SwiGLU (y toda capa nueva) acumulan sin buffers propios del tamano de la matriz. */
static int test_dense_accumulate(void) {
    TernaryRng rng; ternary_rng_seed(&rng, 5);
    TernaryDense *d = ternary_dense_alloc(DIM, VOC, 0);
    CHECK(d, "alloc");
    for (size_t o = 0; o < VOC; o++) for (size_t i = 0; i < DIM; i++) ternary_dense_set_weight(d, o, i, ternary_rng_sample_faninit(&rng, DIM));
    d->scale = 0.6f;
    TernaryVec *x1 = ternary_vec_alloc(DIM), *x2 = ternary_vec_alloc(DIM);
    random_x(x1, DIM); random_x(x2, DIM);
    float g1[VOC], g2[VOC];
    for (size_t o = 0; o < VOC; o++) { g1[o] = (float)nrand(); g2[o] = (float)nrand(); }
    float a[VOC * DIM], b1[VOC * DIM], b2[VOC * DIM], gx_a[DIM] = { 0 }, gx_b[DIM] = { 0 };
    float gs_a = 0.0f, gs_b = 0.0f;
    memset(a, 0, sizeof(a));
    ternary_dense_backward_accumulate(d, x1, g1, a, gx_a, &gs_a, NULL, NULL);
    ternary_dense_backward_accumulate(d, x2, g2, a, gx_a, &gs_a, NULL, NULL);
    ternary_dense_backward(d, x1, g1, b1, gx_b, &gs_b, NULL, NULL);
    ternary_dense_backward(d, x2, g2, b2, gx_b, &gs_b, NULL, NULL);
    for (size_t k = 0; k < VOC * DIM; k++) CHECK(a[k] == b1[k] + b2[k], "accumulate: distinto de pisar y sumar a mano");
    CHECK(memcmp(gx_a, gx_b, sizeof(gx_a)) == 0 && gs_a == gs_b, "accumulate: grad_x y grad_scale iguales");
    ternary_vec_free(x1); ternary_vec_free(x2); ternary_dense_free(d);
    return 0;
}

/* Forma "dificil": dim que no es multiplo de 64 (las posiciones empiezan en bits no alineados) y mas posiciones
 * que un bloque (48) con un resto. Referencia: el calculo elemento a elemento de siempre (get_weight / vec_get). */
static int test_backward_unaligned_many_rows(void) {
    const size_t R = 101, D = 70, V = 37;
    TernaryHead *h = ternary_head_alloc(R, D, V);
    CHECK(h != NULL, "alloc");
    for (size_t v = 0; v < V; v++) for (size_t i = 0; i < D; i++) ternary_dense_set_weight(h->w, v, i, (int)(urand() * 3.0) - 1);
    h->w->scale = 0.41f;
    TernaryVec *x = ternary_vec_alloc(R * D);
    random_x(x, R * D);
    float *out = (float *)malloc(R * V * sizeof(float)), *g = (float *)malloc(R * V * sizeof(float));
    for (size_t i = 0; i < R * V; i++) g[i] = (float)(urand() - 0.5) * (urand() < 0.2 ? 0.0f : 1.0f);
    int32_t *raw = (int32_t *)malloc(R * V * sizeof(int32_t));
    ternary_head_forward(h, x, out, raw);
    double maxrel = 0.0;
    for (size_t p = 0; p < R; p++) for (size_t v = 0; v < V; v++) {
        long z = 0;
        for (size_t i = 0; i < D; i++) z += (long)ternary_dense_get_weight(h->w, v, i) * ternary_vec_get(x, p * D + i);
        CHECK(raw[p * V + v] == (int32_t)z, "forward: Z crudo por posicion (dim 70, 101 posiciones)");
        CHECK(out[p * V + v] == h->w->scale * (float)z, "forward: logits = escala*Z");
    }
    float *gw = (float *)malloc(V * D * sizeof(float)), *gx = (float *)calloc(R * D, sizeof(float));
    float gs = 0.0f;
    ternary_head_backward(h, x, g, gw, gx, &gs);
    double *rw = (double *)calloc(V * D, sizeof(double)), *rx = (double *)calloc(R * D, sizeof(double)), rs_ = 0.0;
    for (size_t p = 0; p < R; p++) for (size_t v = 0; v < V; v++) {
        const double gv = g[p * V + v];
        rs_ += gv * (double)raw[p * V + v];
        for (size_t i = 0; i < D; i++) {
            rw[v * D + i] += gv * ternary_vec_get(x, p * D + i);
            rx[p * D + i] += gv * h->w->scale * ternary_dense_get_weight(h->w, v, i);
        }
    }
    for (size_t k = 0; k < V * D; k++) { double e = fabs(gw[k] - rw[k]) / (fabs(rw[k]) + 1.0); if (e > maxrel) maxrel = e; }
    for (size_t k = 0; k < R * D; k++) { double e = fabs(gx[k] - rx[k]) / (fabs(rx[k]) + 1.0); if (e > maxrel) maxrel = e; }
    CHECK(maxrel < 1e-4, "backward (dim 70, 101 posiciones): grad_w y grad_x contra la referencia en double");
    CHECK(fabs(gs - rs_) / (fabs(rs_) + 1.0) < 1e-5, "backward: grad de escala");
    printf("[head: forma dificil] dim 70, 101 posiciones (2 bloques + resto): Z exacto, grad_w/grad_x error relativo max %.2e\n", maxrel);
    free(out); free(g); free(raw); free(gw); free(gx); free(rw); free(rx); ternary_vec_free(x); ternary_head_free(h);
    return 0;
}

int main(void) {
    if (test_registry()) return 1;
    printf("OK: registro, describe, formas 1D/2D y errores de parametros\n");
    if (test_forward_reference()) return 1;
    printf("OK: forward identico a la referencia (Z crudo exacto, logits = escala*Z)\n");
    if (test_loss()) return 1;
    printf("OK: entropia cruzada: referencia en double, gradiente por diferencias finitas, filas ignoradas, NaN con target invalido, estable con +-1e4\n");
    if (test_backward_reference()) return 1;
    printf("OK: backward igual a la referencia en double; d(perdida)/d(escala) por diferencias finitas\n");
    if (test_backward_unaligned_many_rows()) return 1;
    if (test_dense_accumulate()) return 1;
    printf("OK: dense_backward_accumulate == pisar y sumar a mano (bit a bit)\n");
    if (test_stack_and_persistence()) return 1;
    printf("OK: stack == forward directo; checkpoint 20+20 == 40; export identico\n");
    if (test_learning_head_alone()) return 1;
    if (test_learning_embedding_head()) return 1;
    printf("OK: test_head\n");
    return 0;
}
