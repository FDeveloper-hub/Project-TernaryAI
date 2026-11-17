/*
 * Verifica fit()/predict() con ids y entropia cruzada por la API plana de libternaryai
 * (ternaryai_model_train_ids / ternaryai_model_predict_ids):
 *
 *  1) Camino de la API == camino manual con ternary_stack_* (mismo rng, mismas capas, mismo
 *     calendario): perdida por epoca y logits finales IDENTICOS bit a bit, con posiciones
 *     ignoradas (target negativo) e ids de relleno (negativos) incluidos.
 *  2) Aprende de verdad: "siguiente token" fijo sobre 12 ids (azar 8%) llega a >= 95%.
 *  3) Pausar+retomar (checkpoint) da los mismos logits que entrenar de un tiron.
 *  4) Errores: target >= vocab, id >= vocab del Embedding, todo ignorado, modelo sin ids por
 *     la API de ids y modelo de ids por la API de ternarios; el modelo no cambia tras un error.
 */
#include "ternaryai.h"
#include "tmp_path.h"
#include "ternary_dispatch.h"
#include "ternary_layer.h"
#include "ternary_loss.h"
#include "ternary_rng.h"
#include "ternary_stack.h"
#include "ternary_tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FALLO: %s\n", msg); fails++; } } while (0)

enum { VOCAB = 12, DIM = 16, SEQ = 8, N = 24 };
static const double HYPER[TERNARYAI_NUM_HYPER] = { 0.02 };
/* Para entropia cruzada el gradiente es mucho mas chico que el del MSE de XOR (se reparte entre
 * N muestras, seq posiciones y vocab logits): con el piso plano 0.3 no pasa ningun voto. Mismo
 * calendario calibrado, pero con el valle del recorrido en 5e-6 (el valor de los digitos). */
static const double HYPER_LM[TERNARYAI_NUM_HYPER] = { 0.02 };
static char err[256];

static unsigned g_state = 12345;
static int urand(int n) { g_state = g_state * 1103515245u + 12345u; return (int)((g_state >> 16) % (unsigned)n); }

static TernaryAIModel *make_api_model(unsigned seed) {
    size_t in[1] = { SEQ };
    TernaryAIModel *m = ternaryai_model_create(in, 1, seed, err, sizeof(err));
    double pe[2] = { VOCAB, DIM }, ph[1] = { VOCAB };
    size_t sh[8]; int nd;
    if (!m || !ternaryai_model_add_layer(m, "embedding", pe, 2, "uint8", sh, &nd, err, sizeof(err)) ||
        !ternaryai_model_add_layer(m, "lm_head", ph, 1, "uint8", sh, &nd, err, sizeof(err))) {
        fprintf(stderr, "no se pudo armar el modelo: %s\n", err);
        exit(2);
    }
    return m;
}

static TernaryLayer build(const char *type, const size_t *in, size_t nd, const double *p, size_t np, TernaryRng *rng) {
    TernaryLayer l;
    if (!ternary_layer_type_build(ternary_layer_type_by_name(type), in, nd, p, np, rng, &l, err, sizeof(err))) {
        fprintf(stderr, "build %s: %s\n", type, err); exit(2);
    }
    return l;
}

static TernaryStack *make_manual(unsigned seed) {
    TernaryRng rng; ternary_rng_seed(&rng, seed);
    size_t in0[1] = { SEQ }, in1[2] = { SEQ, DIM };
    double pe[2] = { VOCAB, DIM }, ph[1] = { VOCAB };
    TernaryStack *s = ternary_stack_create();
    ternary_stack_add(s, build("embedding", in0, 1, pe, 2, &rng), TERNARY_ACTIVATION_UINT8);
    ternary_stack_add(s, build("lm_head", in1, 2, ph, 1, &rng), TERNARY_ACTIVATION_UINT8);
    return s;
}

/* datos: siguiente token fijo (g_next), con relleno y posiciones ignoradas */
static int32_t g_next[VOCAB];
static void make_data(int32_t *ids, int32_t *tg, int with_holes) {
    for (int s = 0; s < N; s++)
        for (int p = 0; p < SEQ; p++) {
            int32_t id = urand(VOCAB);
            int pad = with_holes && urand(7) == 0;
            ids[s * SEQ + p] = pad ? -1 : id;
            tg[s * SEQ + p] = (pad || (with_holes && urand(5) == 0)) ? -1 : g_next[id];
        }
}

static void test_matches_manual(void) {
    for (int i = 0; i < VOCAB; i++) g_next[i] = (i * 5 + 3) % VOCAB;
    int32_t ids[N * SEQ], tg[N * SEQ];
    make_data(ids, tg, 1);
    const size_t EPOCHS = 25;
    int bad_loss = 0, bad_logit = 0;
    for (unsigned seed = 1; seed <= 4; seed++) {
        TernaryAIModel *m = make_api_model(seed);
        ternaryai_model_set_restore_best(m, 0);     /* el camino manual no vuelve al mejor estado */
        double loss[25], acc[25];
        CHECK(ternaryai_model_train_ids(m, ids, tg, N, 0, EPOCHS, HYPER, loss, acc, err, sizeof(err)), err);

        /* camino manual: mismo calendario (recorrido plano: piso = 0.3 siempre) */
        TernaryStack *s = make_manual(seed);
        TernaryCapabilities caps = ternary_init_cpu();
        float out[SEQ * VOCAB], go[SEQ * VOCAB];
        size_t xd[1] = { SEQ };
        for (size_t e = 0; e < EPOCHS; e++) {
            float k = 0.5f * powf(0.85f, (float)e); if (k < 0.05f) k = 0.05f;
            ternary_stack_zero_gradients(s);
            double l = 0.0;
            for (int smp = 0; smp < N; smp++) {
                TernaryTensor *xt = ternary_tensor_view_id(xd, 1, ids + smp * SEQ);
                ternary_stack_forward(s, xt, out, &caps);
                l += ternary_xent(out, SEQ, VOCAB, tg + smp * SEQ, go, NULL, NULL);
                for (int i = 0; i < SEQ * VOCAB; i++) go[i] *= 1.0f / (float)N;
                ternary_stack_backward(s, xt, go);
                ternary_tensor_free(xt);
            }
            ternary_stack_apply_gradients(s, 0.02f);
            if (loss[e] != l / N) bad_loss++;
        }
        static float a[N * SEQ * VOCAB];
        CHECK(ternaryai_model_predict_ids(m, ids, N, a, err, sizeof(err)), err);
        for (int smp = 0; smp < N; smp++) {
            TernaryTensor *xt = ternary_tensor_view_id(xd, 1, ids + smp * SEQ);
            ternary_stack_forward(s, xt, out, &caps);
            ternary_tensor_free(xt);
            if (memcmp(out, a + (size_t)smp * SEQ * VOCAB, sizeof(out)) != 0) bad_logit++;
        }
        ternary_stack_free(s);
        ternary_shutdown(&caps);
        ternaryai_model_free(m);
    }
    CHECK(bad_loss == 0, "la perdida por epoca difiere del camino manual");
    CHECK(bad_logit == 0, "los logits difieren del camino manual");
    printf("[ids: API vs manual] 4 semillas x 25 epocas, con relleno y posiciones ignoradas: %d perdidas y %d muestras de logits distintas\n",
           bad_loss, bad_logit);
}

static void test_learns(void) {
    int32_t ids[N * SEQ], tg[N * SEQ];
    make_data(ids, tg, 0);
    int ok = 0;
    double worst = 1.0;
    for (unsigned seed = 1; seed <= 5; seed++) {   /* srand(0) == srand(1) en glibc */
        TernaryAIModel *m = make_api_model(seed);
        double loss[400], acc[400];
        CHECK(ternaryai_model_train_ids(m, ids, tg, N, 0, 400, HYPER_LM, loss, acc, err, sizeof(err)), err);
        if (getenv("IDS_DBG")) printf("seed %u: loss %.3f -> %.3f  acc %.3f -> %.3f\n", seed, loss[0], loss[399], acc[0], acc[399]);
        if (acc[399] >= 0.95 && loss[399] < loss[0]) ok++;
        if (acc[399] < worst) worst = acc[399];
        ternaryai_model_free(m);
    }
    CHECK(ok == 5, "el siguiente token fijo no llego a 95% en las 5 semillas");
    printf("[ids: aprende] siguiente token fijo, %d ids (azar %.0f%%): %d/5 semillas >= 95%% (peor %.1f%%)\n",
           VOCAB, 100.0 / VOCAB, ok, 100.0 * worst);
}

static void test_resume(void) {
    int32_t ids[N * SEQ], tg[N * SEQ];
    make_data(ids, tg, 1);
    static float a[N * SEQ * VOCAB], b[N * SEQ * VOCAB];
    double l1[40], l2[20], l3[20];
    TernaryAIModel *m1 = make_api_model(7), *m2 = make_api_model(7);
    CHECK(ternaryai_model_train_ids(m1, ids, tg, N, 0, 40, HYPER, l1, NULL, err, sizeof(err)), err);
    CHECK(ternaryai_model_train_ids(m2, ids, tg, N, 0, 20, HYPER, l2, NULL, err, sizeof(err)), err);
    const char *path = tmp_path("ternaryai_ids_ckpt.tn");
    CHECK(ternaryai_model_save(m2, path, 1, err, sizeof(err)), err);
    TernaryAIModel *m3 = ternaryai_model_load(path, err, sizeof(err));
    CHECK(m3 != NULL, err);
    /* los hiperparametros miran first_epoch+count de cada llamada; con recorrido plano
     * (piso constante) 20+20 == 40 solo cambia por el calendario de k, que depende de la epoca */
    CHECK(ternaryai_model_train_ids(m3, ids, tg, N, 20, 20, HYPER, l3, NULL, err, sizeof(err)), err);
    CHECK(ternaryai_model_predict_ids(m1, ids, N, a, err, sizeof(err)), err);
    CHECK(ternaryai_model_predict_ids(m3, ids, N, b, err, sizeof(err)), err);
    CHECK(memcmp(a, b, sizeof(a)) == 0, "pausar+retomar no da los mismos logits que entrenar de un tiron");
    int same_loss = 1;
    for (int e = 0; e < 20; e++) if (l1[20 + e] != l3[e]) same_loss = 0;
    CHECK(same_loss, "la perdida de las epocas retomadas difiere");
    model_dir_remove(path);
    printf("[ids: checkpoint] 20+20 epocas == 40 de un tiron (logits y perdida identicos)\n");
    ternaryai_model_free(m1); ternaryai_model_free(m2); ternaryai_model_free(m3);
}

static void test_errors(void) {
    int32_t ids[N * SEQ], tg[N * SEQ];
    make_data(ids, tg, 0);
    double loss[2];
    static float a[N * SEQ * VOCAB], b[N * SEQ * VOCAB];
    TernaryAIModel *m = make_api_model(1);
    CHECK(ternaryai_model_predict_ids(m, ids, N, a, err, sizeof(err)), err);

    int32_t id9 = ids[9];
    tg[5] = VOCAB;
    CHECK(!ternaryai_model_train_ids(m, ids, tg, N, 0, 2, HYPER, loss, NULL, err, sizeof(err)) && strstr(err, "targets[5]"), "target >= vocab");
    tg[5] = 0;
    ids[9] = VOCAB;
    CHECK(!ternaryai_model_train_ids(m, ids, tg, N, 0, 2, HYPER, loss, NULL, err, sizeof(err)) && strstr(err, "ids[9]"), "id >= vocab (train)");
    CHECK(!ternaryai_model_predict_ids(m, ids, N, b, err, sizeof(err)) && strstr(err, "ids[9]"), "id >= vocab (predict)");
    ids[9] = id9;
    int32_t none[N * SEQ];
    for (int i = 0; i < N * SEQ; i++) none[i] = -1;
    CHECK(!ternaryai_model_train_ids(m, ids, none, N, 0, 2, HYPER, loss, NULL, err, sizeof(err)) && strstr(err, "nothing to learn"), "todo ignorado");
    double bad[TERNARYAI_NUM_HYPER]; memcpy(bad, HYPER, sizeof(bad)); bad[0] = -1;
    CHECK(!ternaryai_model_train_ids(m, ids, tg, N, 0, 2, bad, loss, NULL, err, sizeof(err)), "hiperparametro invalido");
    int8_t xt[N * SEQ] = {0}; float fo[N * SEQ * VOCAB]; double ly[1]; float yv[SEQ * VOCAB] = {0};
    CHECK(!ternaryai_model_predict(m, xt, 1, fo, err, sizeof(err)) && strstr(err, "ids"), "predict ternario sobre modelo de ids");
    CHECK(!ternaryai_model_train(m, xt, yv, 1, 0, 1, HYPER, ly, err, sizeof(err)) && strstr(err, "ids"), "train ternario sobre modelo de ids");
    /* tras todos los errores el modelo sigue igual */
    CHECK(ternaryai_model_predict_ids(m, ids, N, b, err, sizeof(err)), err);
    CHECK(memcmp(a, b, sizeof(a)) == 0, "un error cambio el modelo");
    ternaryai_model_free(m);

    /* modelo ternario por la API de ids */
    size_t in[1] = { 4 }; size_t sh[8]; int nd; 
    TernaryAIModel *d = ternaryai_model_create(in, 1, 0, err, sizeof(err));
    double pdense[2] = { 3, 1 };
    CHECK(ternaryai_model_add_layer(d, "dense", pdense, 2, "uint8", sh, &nd, err, sizeof(err)), err);
    int32_t di[4] = {0, 1, 2, 3}, dt[4] = {0, 1, 2, 0}; float dout[12];
    CHECK(!ternaryai_model_train_ids(d, di, dt, 1, 0, 1, HYPER, loss, NULL, err, sizeof(err)) && strstr(err, "does not take token ids"), "modelo ternario por train_ids");
    CHECK(!ternaryai_model_predict_ids(d, di, 1, dout, err, sizeof(err)), "modelo ternario por predict_ids");
    ternaryai_model_free(d);
    printf("[ids: errores] 9 casos rechazados con mensaje, el modelo no cambia\n");
}

int main(void) {
    test_matches_manual();
    test_learns();
    test_resume();
    test_errors();
    printf("fallas: %d\n", fails);
    if (fails == 0) printf("TODAS LAS PRUEBAS DE IDS + ENTROPIA CRUZADA (libternaryai) PASARON\n");
    return fails ? 1 : 0;
}
