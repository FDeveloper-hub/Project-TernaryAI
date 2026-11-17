/*
 * RMSNorm como capa del TernaryStack ("rmsnorm"). Verifica:
 *  1. registro por nombre, describe() de vuelta, formas 1D/2D/3D y errores de parametros con mensaje;
 *  2. red Dense -> RMSNorm -> Dense: el forward del stack es IDENTICO bit a bit al calculo manual
 *     (en particular, RMSNorm recibe el FLOAT crudo de la Dense, no el cuantizado), y entrenar por el
 *     stack deja pesos, ganancia y las tres escalas identicos bit a bit a entrenar a mano (el gradiente
 *     de RMSNorm llega bien a la Dense anterior);
 *  2a. la escala de RMSNorm usa su propio learning rate fijo (0.001) e ignora el del stack;
 *  2b. aprendizaje de la red Dense -> RMSNorm -> Dense contra un maestro (el error baja y la Dense anterior aprende a traves de RMSNorm);
 *  3. checkpoint completo: guardar tras 20 epocas, cargar y seguir == entrenar 40 de un tiron;
 *  4. export empaquetado: inferencia identica;
 *  5. APRENDIZAJE: RMSNorm como unica capa, entrada float, aprende por votos una ganancia -1/0/+1 y una
 *     escala que le pone un "maestro" (los pesos arrancan en +1 / escala 1).
 */
#include "tmp_path.h"
#include "ternary_stack.h"
#include "ternary_layer.h"
#include "ternary_activation.h"
#include "ternary_rng.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FALLO: %s (linea %d)\n", msg, __LINE__); return 1; } } while (0)
#define N 24
#define D 16
#define M 8
#define BATCH 8
static const float SCALE_LR = 0.01f;
static const double P_DENSE1[2] = { D, 0 }, P_DENSE2[2] = { M, 0 }, P_RMS[1] = { 1e-5 };

static uint64_t rs = 88172645463325252ull;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) / 9007199254740992.0; }
static double nrand(void) { double u = urand() + 1e-12, v = urand(); return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v); }

static int same_dense(const TernaryDense *a, const TernaryDense *b) {
    if (a->input_dim != b->input_dim || a->output_dim != b->output_dim || a->scale != b->scale) return 0;
    for (size_t o = 0; o < a->output_dim; o++) for (size_t i = 0; i < a->input_dim; i++)
        if (ternary_dense_get_weight(a, o, i) != ternary_dense_get_weight(b, o, i)) return 0;
    return 1;
}
static const TernaryDense *dense_at(const TernaryStack *s, size_t i) { return (const TernaryDense *)ternary_stack_layer(s, i).impl; }
static const TernaryRMSNorm *rms_at(const TernaryStack *s, size_t i) { return (const TernaryRMSNorm *)ternary_stack_layer(s, i).impl; }
static int same_stack(const TernaryStack *a, const TernaryStack *b) {
    return same_dense(dense_at(a, 0), dense_at(b, 0)) && same_dense(rms_at(a, 1)->gain, rms_at(b, 1)->gain) && same_dense(dense_at(a, 2), dense_at(b, 2));
}

static TernaryStack *new_stack(unsigned seed, char *err, size_t cap) {
    TernaryRng rng; ternary_rng_seed(&rng, seed);
    size_t in1[1] = { N }, in2[1] = { D };
    TernaryLayer l1, l2, l3;
    if (!ternary_layer_type_build(ternary_layer_type_by_name("dense"), in1, 1, P_DENSE1, 2, &rng, &l1, err, cap)) return NULL;
    if (!ternary_layer_type_build(ternary_layer_type_by_name("rmsnorm"), in2, 1, P_RMS, 1, &rng, &l2, err, cap)) return NULL;
    if (!ternary_layer_type_build(ternary_layer_type_by_name("dense"), in2, 1, P_DENSE2, 2, &rng, &l3, err, cap)) return NULL;
    TernaryStack *s = ternary_stack_create();
    if (!ternary_stack_add(s, l1, TERNARY_ACTIVATION_UINT8) || !ternary_stack_add(s, l2, TERNARY_ACTIVATION_UINT8) ||
        !ternary_stack_add(s, l3, TERNARY_ACTIVATION_UINT8)) return NULL;
    return s;
}

static TernaryStack *g_teacher;   /* mismo esquema Dense->RMSNorm->Dense con otros pesos: el objetivo SI es aprendible */
static TernaryCapabilities g_cn;
static void make_sample(TernaryVec *x, float *target) {
    for (size_t i = 0; i < N; i++) ternary_vec_set(x, i, (int)(urand() * 3.0) - 1);
    size_t xd[1] = { N }; TernaryTensor *xt = ternary_tensor_view_ternary(xd, 1, x);
    ternary_stack_forward(g_teacher, xt, target, &g_cn);
    ternary_tensor_free(xt);
}
static void make_grad(const float *out, const float *target, float *go) {
    for (size_t o = 0; o < M; o++) go[o] = (2.0f / M) * (out[o] - target[o]);
}

static void train_stack(TernaryStack *s, int epochs, const TernaryCapabilities *cn) {
    TernaryVec *x = ternary_vec_alloc(N); size_t xd[1] = { N };
    TernaryTensor *xt = ternary_tensor_view_ternary(xd, 1, x);
    float out[M], go[M], target[M];
    for (int ep = 0; ep < epochs; ep++) {
        ternary_stack_zero_gradients(s);
        for (int b = 0; b < BATCH; b++) {
            make_sample(x, target);
            ternary_stack_forward(s, xt, out, cn);
            make_grad(out, target, go);
            ternary_stack_backward(s, xt, go);
        }
        ternary_stack_apply_gradients(s, SCALE_LR);
    }
    ternary_tensor_free(xt); ternary_vec_free(x);
}

int main(void) {
    TernaryCapabilities cn; memset(&cn, 0, sizeof(cn)); cn.num_cores = 1; cn.dot_impl_name = "scalar";
    char err[256] = "";
    TernaryRng rng; ternary_rng_seed(&rng, 1);
    g_cn = cn;
    g_teacher = new_stack(1000, err, sizeof(err));
    if (!g_teacher) { fprintf(stderr, "FALLO: maestro: %s\n", err); return 1; }

    /* --- 1. registro, describe, formas y errores --- */
    const TernaryLayerType *t = ternary_layer_type_by_name("rmsnorm");
    CHECK(t && strcmp(t->class_name, "RMSNorm") == 0 && ternary_layer_type_num_slots(t) == 1, "tipo rmsnorm en el registro");
    {
        TernaryLayer l; size_t sh1[1] = { 10 }, sh2[2] = { 6, 10 }, sh3[3] = { 2, 3, 10 };
        double p[1] = { 1e-3 }, back[2]; size_t oshape[TERNARY_MAX_SHAPE_DIMS];
        CHECK(ternary_layer_type_build(t, sh1, 1, p, 1, &rng, &l, err, sizeof(err)), "build 1D");
        CHECK(l.vt->input_len(l.impl) == 10 && l.vt->output_len(l.impl) == 10 && l.vt->input_dtype(l.impl) == TERNARY_DTYPE_FLOAT, "1D: largo/dtype");
        CHECK(t->describe(l.impl, back, oshape) == 1 && oshape[0] == 10 && fabs(back[0] - 1e-3) < 1e-9, "1D: describe");
        l.vt->destroy(l.impl);
        CHECK(ternary_layer_type_build(t, sh2, 2, p, 1, &rng, &l, err, sizeof(err)), "build 2D");
        CHECK(l.vt->input_len(l.impl) == 60 && ((TernaryRMSNorm *)l.impl)->rows == 6 && ((TernaryRMSNorm *)l.impl)->dim == 10, "2D: filas/dim");
        CHECK(t->describe(l.impl, back, oshape) == 2 && oshape[0] == 6 && oshape[1] == 10, "2D: describe");
        l.vt->destroy(l.impl);
        CHECK(ternary_layer_type_build(t, sh3, 3, p, 1, &rng, &l, err, sizeof(err)), "build 3D");
        CHECK(((TernaryRMSNorm *)l.impl)->rows == 6 && ((TernaryRMSNorm *)l.impl)->dim == 10, "3D: normaliza sobre la ultima dimension (6 filas)");
        l.vt->destroy(l.impl);
        double bad_eps[1] = { 0.0 }, bad_neg[1] = { -1.0 };
        CHECK(!ternary_layer_type_build(t, sh1, 1, bad_eps, 1, &rng, &l, err, sizeof(err)) && strstr(err, "eps"), "eps = 0");
        CHECK(!ternary_layer_type_build(t, sh1, 1, bad_neg, 1, &rng, &l, err, sizeof(err)) && strstr(err, "eps") , "eps < 0");
        CHECK(!ternary_layer_type_build(t, sh1, 0, p, 1, &rng, &l, err, sizeof(err)) && strstr(err, "invalid"), "entrada sin dimensiones");
    }
    printf("[registro] RMSNorm se construye por nombre, describe() la devuelve, formas 1D/2D/3D (normaliza la ultima dim) y 3 errores (eps=0, eps<0, entrada sin dimensiones) con mensaje OK\n");

    /* --- 2. stack == manual: forward y entrenamiento --- */
    TernaryStack *sa = new_stack(42, err, sizeof(err));
    CHECK(sa != NULL, err);
    TernaryDense *m1 = ternary_dense_alloc(N, D, 0), *m3 = ternary_dense_alloc(D, M, 0);
    TernaryRMSNorm *mr = ternary_rmsnorm_alloc(1, D, 1e-5f);
    for (size_t o = 0; o < D; o++) for (size_t i = 0; i < N; i++) ternary_dense_set_weight(m1, o, i, ternary_dense_get_weight(dense_at(sa, 0), o, i));
    for (size_t o = 0; o < M; o++) for (size_t i = 0; i < D; i++) ternary_dense_set_weight(m3, o, i, ternary_dense_get_weight(dense_at(sa, 2), o, i));
    TernaryStack *s0 = new_stack(42, err, sizeof(err));   /* copia inicial: el entrenamiento debe mover algo */
    TernaryVec *x = ternary_vec_alloc(N), *avec_unused = NULL; TernaryU8 *avec = ternary_u8_alloc(1, D); (void)avec_unused; size_t xd[1] = { N };
    TernaryTensor *xt = ternary_tensor_view_ternary(xd, 1, x);
    float *acc1 = malloc(D * N * 4), *acc3 = malloc(M * D * 4), *accg = malloc(D * 4);
    float *gw1 = malloc(D * N * 4), *gw3 = malloc(M * D * 4), *gwg = malloc(D * 4);
    const int EPOCHS = 40;   /* con vote_limit 16 hacen falta >= 16 epocas de votos consistentes para que un peso cambie */
    rs = 99;
    for (int ep = 0; ep < EPOCHS; ep++) {
        memset(acc1, 0, D*N*4); memset(acc3, 0, M*D*4); memset(accg, 0, D*4);
        float gs1 = 0, gs3 = 0, gsn = 0;
        ternary_stack_zero_gradients(sa);
        for (int b = 0; b < BATCH; b++) {
            float target[M], so[M], go[M], f1[D], y[D], f2[M]; int32_t raw1[D], raw2[M];
            make_sample(x, target);
            ternary_stack_forward(sa, xt, so, &cn);
            ternary_dense_forward_auto(m1, x, f1, raw1, &cn);
            ternary_rmsnorm_forward(mr, f1, y);
            TERNARY_ACTIVATION_UINT8.apply_u8(y, avec);
            ternary_dense_forward_u8(m3, avec, f2, raw2);
            CHECK(memcmp(so, f2, sizeof(f2)) == 0, "el forward del stack difiere del manual (RMSNorm debe recibir el float crudo de la Dense)");
            make_grad(f2, target, go);
            ternary_stack_backward(sa, xt, go);
            float gxa[D], gx1[D], gxx[N]; memset(gxa, 0, sizeof gxa); memset(gx1, 0, sizeof gx1); memset(gxx, 0, sizeof gxx);
            memset(gw3, 0, M * D * 4);
            ternary_dense_backward_u8(m3, avec, go, gw3, gxa, &gs3, NULL, raw2);
            ternary_rmsnorm_backward(mr, f1, gxa, gwg, gx1, &gsn);
            ternary_dense_backward(m1, x, gx1, gw1, gxx, &gs1, NULL, raw1);
            for (int i = 0; i < D*N; i++) acc1[i] += gw1[i];
            for (int i = 0; i < M*D; i++) acc3[i] += gw3[i];
            for (int i = 0; i < D; i++) accg[i] += gwg[i];
        }
        ternary_stack_apply_gradients(sa, SCALE_LR);
        for (size_t r = 0; r < D; r++) ternary_dense_apply_row_gradients(m1, r, acc1 + r * N);
        ternary_dense_apply_scale_gradient(m1, gs1, SCALE_LR);
        ternary_dense_apply_row_gradients(mr->gain, 0, accg);
        ternary_dense_apply_scale_gradient(mr->gain, gsn, TERNARY_RMSNORM_SCALE_LR);   /* la capa usa su propio lr, no el del stack */
        for (size_t r = 0; r < M; r++) ternary_dense_apply_row_gradients(m3, r, acc3 + r * D);
        ternary_dense_apply_scale_gradient(m3, gs3, SCALE_LR);
    }
    CHECK(same_dense(dense_at(sa, 0), m1) && same_dense(rms_at(sa, 1)->gain, mr->gain) && same_dense(dense_at(sa, 2), m3),
          "pesos/ganancia/escalas difieren entre entrenar por el stack y a mano");
    CHECK(!same_stack(sa, s0), "el entrenamiento no cambio nada: la comparacion no prueba nada");
    int d1 = 0, dg = 0, d3 = 0;
    for (size_t o = 0; o < D; o++) for (size_t i = 0; i < N; i++) d1 += ternary_dense_get_weight(dense_at(sa, 0), o, i) != ternary_dense_get_weight(dense_at(s0, 0), o, i);
    for (size_t j = 0; j < D; j++) dg += ternary_dense_get_weight(rms_at(sa, 1)->gain, 0, j) != 1;
    for (size_t o = 0; o < M; o++) for (size_t i = 0; i < D; i++) d3 += ternary_dense_get_weight(dense_at(sa, 2), o, i) != ternary_dense_get_weight(dense_at(s0, 2), o, i);
    int votes1 = 0; for (size_t i = 0; i < D * N; i++) votes1 += dense_at(sa, 0)->counter_W[i] != 0;
    CHECK(votes1 > 0, "la Dense ANTERIOR a RMSNorm no acumulo ningun voto: el backward de RMSNorm no propaga gradiente");
    CHECK(d1 > 0, "la Dense ANTERIOR a RMSNorm acumulo votos pero ningun peso cambio en 40 epocas");
    printf("[stack vs manual] Dense->RMSNorm->Dense, %d epocas x %d muestras: forward y entrenamiento identicos bit a bit. Pesos que cambiaron: %d de la Dense anterior (con %d votos acumulados: el gradiente atraviesa RMSNorm), %d ganancias, %d de la Dense final; escala RMSNorm %.4f\n",
           EPOCHS, BATCH, d1, votes1, dg, d3, rms_at(sa, 1)->gain->scale);

    /* --- 2a. learning rate propio de la escala de RMSNorm: ignora el scale_lr que pasa el stack --- */
    {
        TernaryRMSNorm *a = ternary_rmsnorm_alloc(1, 8, 1e-5f);
        TernaryLayer la = ternary_layer_from_rmsnorm(a);
        CHECK(TERNARY_RMSNORM_SCALE_LR == 0.001f, "el lr de la escala de RMSNorm debe ser 0.001");
        la.vt->apply_scale_gradient(la.impl, 0, 10.0f, 0.5f);      /* el stack pasaria 0.5: se ignora */
        float expected = 1.0f - 0.001f * 10.0f;
        CHECK(a->gain->scale == expected, "la escala debia moverse con lr 0.001 (0.99), no con el lr recibido");
        la.vt->apply_scale_gradient(la.impl, 0, 10.0f, 0.0f);      /* lr recibido 0: tampoco importa */
        CHECK(fabsf(a->gain->scale - (expected - 0.01f)) < 1e-6f, "el lr recibido (0) no debe congelar la escala de RMSNorm");
        la.vt->apply_scale_gradient(la.impl, 0, 1e9f, 0.01f);      /* piso 0.02 como en Dense */
        CHECK(a->gain->scale == 0.02f, "la escala debe respetar el piso 0.02");
        printf("[lr de la escala] RMSNorm usa 0.001 fijo e ignora el scale_lr del stack (0.5 y 0.0 dan el mismo paso); respeta el piso 0.02\n");
        la.vt->destroy(la.impl);
    }

    /* --- 2b. Dense -> RMSNorm -> Dense APRENDE de un maestro (el gradiente atraviesa RMSNorm de verdad) --- */
    {
        TernaryStack *st = new_stack(77, err, sizeof(err));
        TernaryStack *st_i = new_stack(77, err, sizeof(err));   /* la misma red SIN entrenar, para medir el punto de partida */
        float out[M], target[M]; double mse0 = 0, mse1 = 0; const int NS = 300;
        rs = 31337; train_stack(st, 400, &cn);
        rs = 4242;
        for (int k = 0; k < NS; k++) {
            make_sample(x, target);
            ternary_stack_forward(st_i, xt, out, &cn); for (size_t o = 0; o < M; o++) mse0 += (double)(out[o]-target[o])*(out[o]-target[o]);
            ternary_stack_forward(st, xt, out, &cn);   for (size_t o = 0; o < M; o++) mse1 += (double)(out[o]-target[o])*(out[o]-target[o]);
        }
        mse0 /= (double)NS * M; mse1 /= (double)NS * M;
        int fl1 = 0; for (size_t o = 0; o < D; o++) for (size_t i = 0; i < N; i++) fl1 += ternary_dense_get_weight(dense_at(st, 0), o, i) != ternary_dense_get_weight(dense_at(st_i, 0), o, i);
        printf("[aprendizaje de la red] Dense->RMSNorm->Dense contra un maestro, 400 epocas: MSE %.3f -> %.3f (%.0f%% menos); %d pesos de la Dense ANTERIOR a RMSNorm cambiaron\n",
               mse0, mse1, 100.0 * (1.0 - mse1 / mse0), fl1);
        CHECK(mse1 < 0.7 * mse0, "la red con RMSNorm no redujo el error contra el maestro");
        CHECK(fl1 > 0, "ningun peso de la Dense anterior a RMSNorm cambio: no aprende a traves de RMSNorm");
        ternary_stack_free(st); ternary_stack_free(st_i);
    }

    /* --- 3. checkpoint completo --- */
    const char *path = tmp_path("test_rmsnorm_layer_ckpt.bin"), *path_exp = tmp_path("test_rmsnorm_layer_export.bin");
    model_dir_remove(path); model_dir_remove(path_exp);   /* restos de una corrida anterior */
    TernaryStack *s1 = new_stack(7, err, sizeof(err)), *s2 = NULL, *sb = new_stack(7, err, sizeof(err));
    rs = 5; train_stack(sb, 40, &cn);
    rs = 5; train_stack(s1, 20, &cn);
    CHECK(ternary_stack_save(s1, path, 1) == 1, "guardar checkpoint");
    s2 = ternary_stack_load(path);
    CHECK(s2 != NULL && same_stack(s1, s2), "cargar checkpoint: pesos distintos");
    CHECK(strcmp(ternary_stack_layer(s2, 1).vt->type_name, "rmsnorm") == 0 && rms_at(s2, 1)->eps == 1e-5f, "tipo/eps tras cargar");
    /* el generador de muestras es global: para comparar de forma justa se repite el protocolo con semilla fija */
    TernaryStack *sc = new_stack(7, err, sizeof(err)), *sd = NULL;
    rs = 5; train_stack(sc, 20, &cn);
    CHECK(ternary_stack_save(sc, path, 1) == 1, "guardar checkpoint (2)");
    sd = ternary_stack_load(path);
    uint64_t rs_mid = rs;
    train_stack(sc, 20, &cn);
    rs = rs_mid; train_stack(sd, 20, &cn);
    CHECK(same_stack(sb, sc), "entrenar 20+20 (sin guardar) difiere de 40 de un tiron");
    CHECK(same_stack(sc, sd), "pausar/retomar desde checkpoint difiere de seguir sin pausa");
    printf("[checkpoint] guardar tras 20 epocas, cargar y entrenar 20 mas == 40 de un tiron (bit a bit), ganancia/escala/eps preservados\n");

    /* --- 4. export empaquetado --- */
    CHECK(ternary_stack_save(sc, path_exp, 0) == 1, "exportar");
    TernaryStack *se = ternary_stack_load(path_exp);
    CHECK(se != NULL, "cargar export");
    for (int k = 0; k < 30; k++) {
        float target[M], o1[M], o3[M];
        make_sample(x, target);
        ternary_stack_forward(sc, xt, o1, &cn); ternary_stack_forward(se, xt, o3, &cn);
        CHECK(memcmp(o1, o3, sizeof(o1)) == 0, "la inferencia del export difiere");
    }
    printf("[export] modelo empaquetado: 30 inferencias identicas a la original\n");

    /* --- 5. aprendizaje: RMSNorm sola, entrada float, ganancia y escala de un maestro --- */
    {
        const size_t ROWS = 4, DIM = 32, LEN = ROWS * DIM;
        const double TEACHER_SCALE = 0.7;
        TernaryRng r2; ternary_rng_seed(&r2, 3);
        size_t shp[2] = { ROWS, DIM }; TernaryLayer l; double p[1] = { 1e-5 };
        CHECK(ternary_layer_type_build(t, shp, 2, p, 1, &r2, &l, err, sizeof(err)), "build maestro");
        TernaryStack *st = ternary_stack_create();
        CHECK(ternary_stack_add(st, l, TERNARY_ACTIVATION_UINT8), "RMSNorm como primera capa (entrada float)");
        int teacher[32]; int nneg = 0, nzero = 0;
        rs = 2024;
        for (size_t j = 0; j < DIM; j++) { teacher[j] = (int)(urand() * 3.0) - 1; nneg += teacher[j] < 0; nzero += teacher[j] == 0; }
        float xf[128], tgt[128], out[128], go[128]; size_t xdm[2] = { ROWS, DIM };
        TernaryTensor *xtf = ternary_tensor_view_float(xdm, 2, xf);
        TernaryRMSNorm *stud = (TernaryRMSNorm *)ternary_stack_layer(st, 0).impl;
        for (int ep = 0; ep < 400; ep++) {
            ternary_stack_zero_gradients(st);
            for (int b = 0; b < 16; b++) {
                for (size_t i = 0; i < ROWS; i++) {
                    double mag = pow(10.0, -1.0 + 2.0 * urand());       /* magnitud por fila de 0.1 a 10: la normalizacion la quita */
                    double ms = 0; for (size_t j = 0; j < DIM; j++) { xf[i*DIM+j] = (float)(nrand() * mag); ms += (double)xf[i*DIM+j] * xf[i*DIM+j]; }
                    double r = 1.0 / sqrt(ms / DIM + 1e-5);
                    for (size_t j = 0; j < DIM; j++) tgt[i*DIM+j] = (float)(TEACHER_SCALE * teacher[j] * xf[i*DIM+j] * r);
                }
                ternary_stack_forward(st, xtf, out, &cn);
                for (size_t k = 0; k < LEN; k++) go[k] = (2.0f / LEN) * (out[k] - tgt[k]);
                ternary_stack_backward(st, xtf, go);
            }
            ternary_stack_apply_gradients(st, SCALE_LR);   /* 0.01 para Dense; RMSNorm usa el suyo (0.001) */
        }
        int wrong = 0; for (size_t j = 0; j < DIM; j++) wrong += ternary_dense_get_weight(stud->gain, 0, j) != teacher[j];
        double mse = 0; for (size_t k = 0; k < LEN; k++) mse += (double)(out[k] - tgt[k]) * (out[k] - tgt[k]); mse /= LEN;
        printf("[aprendizaje] maestro: %d dims con ganancia -1, %d con 0, %d con +1; escala maestra %.2f. Tras 400 epocas: %d/%zu ganancias distintas, escala aprendida %.3f, MSE ultima muestra %.2e\n",
               nneg, nzero, (int)DIM - nneg - nzero, TEACHER_SCALE, wrong, DIM, stud->gain->scale, mse);
        CHECK(nneg > 0 && nzero > 0, "el maestro debe tener ganancias -1 y 0 para que la prueba sea exigente");
        CHECK(wrong == 0, "RMSNorm no aprendio la ganancia del maestro");
        CHECK(fabs(stud->gain->scale - TEACHER_SCALE) < 0.05, "RMSNorm no aprendio la escala del maestro");
        ternary_stack_free(st); ternary_tensor_free(xtf);
    }

    printf("test_rmsnorm_layer: TODO OK\n");
    ternary_stack_free(sa); ternary_stack_free(s0); ternary_stack_free(s1); ternary_stack_free(s2);
    ternary_stack_free(sb); ternary_stack_free(sc); ternary_stack_free(sd); ternary_stack_free(se);
    model_dir_remove(path); model_dir_remove(path_exp);
    return 0;
}
