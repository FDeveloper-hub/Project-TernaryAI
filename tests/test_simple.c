/*
 * Verifica la API plana de libternaryai (include/ternaryai.h) -- la que usa la
 * fachada Python -- contra el camino de siempre, bit a bit:
 *
 *  1) ternary_rng == srand()/rand() de glibc (12 semillas x 200000 valores).
 *  2) Registro de capas/activaciones y validacion de parametros.
 *  3) XOR 2->16->1, 100 epocas, semillas 0..24: la red armada y entrenada
 *     por ternaryai_model_* da EXACTAMENTE las mismas salidas que armarla a
 *     mano con Dense + srand/rand + el bucle de bench_full_step.c (mismos
 *     4 hiperparametros calibrados). Ademas exige el resultado documentado
 *     en el README: 100% de precision media, 25/25 semillas al 100% (tarea AND, regla R2).
 *  4) Una capa rechazada deja el modelo (y su generador de pesos) intacto.
 *  5) Guardado/carga por la API: checkpoint retomable (pausar+retomar da
 *     el mismo archivo, byte a byte, que entrenar de un tiron), export
 *     mas chico, forma de entrada recordada, y archivos del formato viejo
 *     (sin forma de entrada) siguen cargando.
 */
#include "tmp_path.h"
#include "ternaryai.h"
#include "ternary_dense.h"
#include "ternary_activation.h"
#include "ternary_dispatch.h"
#include "ternary_rng.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FALLO: %s\n", msg); fails++; } } while (0)

/* threshold_valley = threshold_floor (0.3): recorrido plano, identico al
 * comportamiento de siempre. threshold_num_cycles=1: un solo ciclo --
 * estas pruebas no ejercitan el recorrido ni los ciclos. */
static const double HYPER[TERNARYAI_NUM_HYPER] = { 0.02 };
static char err[256];

/* ---- 1) RNG ---- */
static void test_rng(void) {
    unsigned seeds[] = {0,1,2,3,24,42,1234,5678,2147483647u,2147483648u,4294967295u,123456789u};
    int bad = 0;
    for (size_t s = 0; s < sizeof(seeds)/sizeof(seeds[0]); s++) {
        TernaryRng r; ternary_rng_seed(&r, seeds[s]); srand(seeds[s]);
        for (int i = 0; i < 200000; i++) if (ternary_rng_next(&r) != rand()) { bad++; break; }
    }
    CHECK(bad == 0, "ternary_rng difiere de rand() de glibc");
    printf("[rng vs rand()] 12 semillas x 200000 valores, %d discrepancias\n", bad);
}

/* ---- 2) registro ---- */
static void test_registry(void) {
    CHECK(ternaryai_abi_version() == TERNARYAI_ABI_VERSION, "abi");
    CHECK(ternaryai_layer_type_count() == 7, "deberia haber 7 tipos de capa");
    CHECK(strcmp(ternaryai_layer_type_name(0), "dense") == 0 && strcmp(ternaryai_layer_type_class(0), "Dense") == 0, "dense");
    CHECK(strcmp(ternaryai_layer_type_name(1), "conv2d") == 0 && strcmp(ternaryai_layer_type_class(1), "Conv2D") == 0, "conv2d");
    CHECK(ternaryai_layer_param_count(0) == 2 && ternaryai_layer_param_count(1) == 4, "cantidad de parametros");
    CHECK(strcmp(ternaryai_layer_param_name(0, 0), "units") == 0 && ternaryai_layer_param_required(0, 0) == 1 &&
          ternaryai_layer_param_kind(0, 0) == TERNARYAI_PARAM_INT && ternaryai_layer_param_has_min(0, 0) == 1 &&
          ternaryai_layer_param_min(0, 0) == 1.0, "Dense.units");
    CHECK(ternaryai_layer_param_name(0, 2) == NULL, "Dense: sin vote_limit");
    CHECK(ternaryai_layer_param_kind(1, 1) == TERNARYAI_PARAM_PAIR, "Conv2D.kernel es pair");
    CHECK(strcmp(ternaryai_layer_type_name(3), "gqa") == 0 && strcmp(ternaryai_layer_type_class(3), "GQA") == 0 && ternaryai_layer_param_count(3) == 9, "gqa");
    CHECK(ternaryai_layer_param_kind(3, 3) == TERNARYAI_PARAM_FLOAT && ternaryai_layer_param_default(3, 3) == 10.0 && ternaryai_layer_param_kind(3, 8) == TERNARYAI_PARAM_BOOL, "GQA.rope_base es float, smear es bool");
    CHECK(strcmp(ternaryai_layer_type_name(4), "rmsnorm") == 0 && strcmp(ternaryai_layer_type_class(4), "RMSNorm") == 0 && ternaryai_layer_param_count(4) == 1, "rmsnorm");
    CHECK(strcmp(ternaryai_layer_param_name(4, 0), "eps") == 0 && ternaryai_layer_param_kind(4, 0) == TERNARYAI_PARAM_FLOAT && ternaryai_layer_param_required(4, 0) == 0 &&
          ternaryai_layer_param_default(4, 0) == 1e-5, "RMSNorm.eps es float con defecto 1e-5");
    CHECK(strcmp(ternaryai_layer_type_name(5), "swiglu") == 0 && strcmp(ternaryai_layer_type_class(5), "SwiGLU") == 0, "swiglu");
    CHECK(strcmp(ternaryai_layer_param_name(5, 0), "hidden") == 0 && ternaryai_layer_param_required(5, 0) == 1 &&
          strcmp(ternaryai_layer_param_name(5, 1), "gate_mode") == 0 && ternaryai_layer_param_default(5, 1) == 0.0 &&
          strcmp(ternaryai_layer_param_name(5, 2), "band") == 0 && ternaryai_layer_param_kind(5, 2) == TERNARYAI_PARAM_FLOAT &&
          ternaryai_layer_param_default(5, 2) == 0.5 && strcmp(ternaryai_layer_param_name(5, 3), "out_scale") == 0, "SwiGLU: hidden requerido, gate_mode 0, band 0.5 (float), out_scale");
    CHECK(strcmp(ternaryai_layer_type_name(6), "lm_head") == 0 && strcmp(ternaryai_layer_type_class(6), "LMHead") == 0, "lm_head");
    CHECK(strcmp(ternaryai_layer_param_name(6, 0), "vocab") == 0 && ternaryai_layer_param_required(6, 0) == 1 &&
          ternaryai_layer_param_name(6, 1) == NULL, "LMHead: solo vocab (requerido)");
    CHECK(ternaryai_layer_type_name(7) == NULL && ternaryai_layer_param_name(0, 3) == NULL, "fuera de rango -> NULL");
    CHECK(ternaryai_activation_count() == 1 && strcmp(ternaryai_activation_name(0), "uint8") == 0, "activaciones");
    printf("[registro] 7 tipos de capa, 3 activaciones OK\n");
}

static void test_param_validation(void) {
    size_t shape[1] = {4};
    TernaryAIModel *m = ternaryai_model_create(shape, 1, 0, err, sizeof(err));
    CHECK(m != NULL, "create");
    size_t out[4]; int nd = 0;
    double zero_units[2] = { 0, 1 }, frac[2] = { 2.5, 1 }, bad_bool[2] = { 2, 2 }, nan_v[2] = { NAN, 1 };
    double ok[2] = { 3, 1 };
    CHECK(!ternaryai_model_add_layer(m, "dense", zero_units, 2, "uint8", out, &nd, err, sizeof(err)) && strstr(err, ">="), "units=0 deberia fallar");
    CHECK(!ternaryai_model_add_layer(m, "dense", frac, 2, "uint8", out, &nd, err, sizeof(err)) && strstr(err, "integer"), "units fraccionario");
    CHECK(!ternaryai_model_add_layer(m, "dense", bad_bool, 2, "uint8", out, &nd, err, sizeof(err)) && strstr(err, "0 or 1"), "bool invalido");
    CHECK(!ternaryai_model_add_layer(m, "dense", nan_v, 2, "uint8", out, &nd, err, sizeof(err)), "NaN");
    CHECK(!ternaryai_model_add_layer(m, "dense", ok, 3, "uint8", out, &nd, err, sizeof(err)) && strstr(err, "expects 2"), "cantidad de valores");
    CHECK(!ternaryai_model_add_layer(m, "nope", ok, 2, "uint8", out, &nd, err, sizeof(err)) && strstr(err, "unknown layer"), "tipo desconocido");
    CHECK(!ternaryai_model_add_layer(m, "dense", ok, 2, "relu", out, &nd, err, sizeof(err)) && strstr(err, "unknown activation"), "activacion desconocida");
    double conv[5] = { 2, 3, 3, 1, 1 };
    CHECK(!ternaryai_model_add_layer(m, "conv2d", conv, 5, "uint8", out, &nd, err, sizeof(err)) && strstr(err, "height, width, channels"), "Conv2D sobre entrada plana");
    CHECK(ternaryai_model_num_layers(m) == 0, "ningun intento fallido debe agregar capas");
    CHECK(ternaryai_model_add_layer(m, "dense", ok, 2, "uint8", out, &nd, err, sizeof(err)) && nd == 1 && out[0] == 3, "dense valida");
    ternaryai_model_free(m);

    size_t img[3] = {4, 4, 1};
    m = ternaryai_model_create(img, 3, 0, err, sizeof(err));
    double big_kernel[5] = { 2, 5, 5, 1, 1 };
    CHECK(!ternaryai_model_add_layer(m, "conv2d", big_kernel, 5, "uint8", out, &nd, err, sizeof(err)) && strstr(err, "larger"), "kernel mas grande que la imagen");
    ternaryai_model_free(m);
    CHECK(ternaryai_model_create(NULL, 1, 0, err, sizeof(err)) == NULL, "forma nula");
    size_t zero_dim[2] = {3, 0};
    CHECK(ternaryai_model_create(zero_dim, 2, 0, err, sizeof(err)) == NULL, "dimension cero");
    printf("[validacion de parametros] OK\n");
}

/* ---- 3) XOR contra el camino manual ---- */
static int sample(size_t fan_in) {   /* copia independiente de bench_full_step.c (usa rand() de libc) */
    double p_nonzero = 1.0 / sqrt((double)fan_in);
    if (p_nonzero < 0.15) p_nonzero = 0.15;
    if (p_nonzero > 0.9) p_nonzero = 0.9;
    double p_each = p_nonzero / 2.0;
    double r = (double)rand() / ((double)RAND_MAX + 1.0);
    if (r < p_each) return -1;
    if (r < p_each + (1.0 - p_nonzero)) return 0;
    return 1;
}

static const int8_t XOR_X[8] = { -1,-1, -1,1, 1,-1, 1,1 };
/* Con la activacion uint8 (afin: solo redondea) la red Dense->Dense es lineal: XOR es imposible, asi que la tarea es un AND
 * (linealmente separable). La no linealidad de verdad la ponen SwiGLU / el gate (ver test_u8_block). */
static const float XOR_Y[4] = { -1, -1, -1, 1 };

/* Red XOR armada y entrenada a mano; deja las 4 salidas finales en out4. */
static void manual_xor(unsigned seed, int epochs, float *out4) {
    TernaryCapabilities cn; memset(&cn, 0, sizeof(cn)); cn.num_cores = 1; cn.dot_impl_name = "scalar";
    TernaryDense *l1 = ternary_dense_alloc(2, 16, 1), *l2 = ternary_dense_alloc(16, 1, 1);
    srand(seed);
    for (size_t o = 0; o < 16; o++) {
        for (size_t i = 0; i < 2; i++) ternary_dense_set_weight(l1, o, i, sample(2));
        ternary_vec_set(l1->bias, o, sample(2));
    }
    for (size_t i = 0; i < 16; i++) ternary_dense_set_weight(l2, 0, i, sample(16));
    ternary_vec_set(l2->bias, 0, sample(16));

    TernaryVec *xv[4];
    for (int s = 0; s < 4; s++) {
        xv[s] = ternary_vec_alloc(2);
        ternary_vec_set(xv[s], 0, XOR_X[2*s]); ternary_vec_set(xv[s], 1, XOR_X[2*s+1]);
    }
    float o1[16], o2[1], gw1[32], gw2[16], gx1[2], gx2[16], acc_w1[32], acc_w2[16], acc_b1[16], acc_b2[1];
    float acc_s1, acc_s2;
    int32_t r1[16], r2[1];
    TernaryU8 *a1 = ternary_u8_alloc(1, 16);
    const float loss_norm = 1.0f / (4.0f * 1.0f);
    for (int e = 0; e < epochs; e++) {
        memset(acc_w1, 0, sizeof(acc_w1)); memset(acc_w2, 0, sizeof(acc_w2));
        memset(acc_b1, 0, sizeof(acc_b1)); memset(acc_b2, 0, sizeof(acc_b2));
        acc_s1 = acc_s2 = 0.0f;
        for (int s = 0; s < 4; s++) {
            ternary_dense_forward_auto(l1, xv[s], o1, r1, &cn);
            TERNARY_ACTIVATION_UINT8.apply_u8(o1, a1);
            ternary_dense_forward_u8(l2, a1, o2, r2);
            float g2[1] = { 2.0f * loss_norm * (o2[0] - XOR_Y[s]) };
            memset(gx2, 0, sizeof(gx2));
            memset(gw2, 0, sizeof(gw2));
            ternary_dense_backward_u8(l2, a1, g2, gw2, gx2, &acc_s2, acc_b2, r2);
            for (int q = 0; q < 16; q++) acc_w2[q] += gw2[q];
            memset(gx1, 0, sizeof(gx1));
            ternary_dense_backward(l1, xv[s], gx2, gw1, gx1, &acc_s1, acc_b1, r1);
            for (int q = 0; q < 32; q++) acc_w1[q] += gw1[q];
        }
        for (int o = 0; o < 16; o++) ternary_dense_apply_row_gradients(l1, (size_t)o, acc_w1 + o*2);
        ternary_dense_apply_row_gradients(l2, 0, acc_w2);
        ternary_dense_apply_scale_gradient(l1, acc_s1, 0.02f);
        ternary_dense_apply_scale_gradient(l2, acc_s2, 0.02f);
        ternary_dense_apply_bias_gradients(l1, acc_b1);
        ternary_dense_apply_bias_gradients(l2, acc_b2);
    }
    for (int s = 0; s < 4; s++) {
        ternary_dense_forward_auto(l1, xv[s], o1, r1, &cn);
        TERNARY_ACTIVATION_UINT8.apply_u8(o1, a1);
        ternary_dense_forward_u8(l2, a1, o2, r2);
        out4[s] = o2[0];
    }
    for (int s = 0; s < 4; s++) ternary_vec_free(xv[s]);
    ternary_u8_free(a1); ternary_dense_free(l1); ternary_dense_free(l2);
}

static TernaryAIModel *xor_model(unsigned seed) {
    size_t shape[1] = {2};
    TernaryAIModel *m = ternaryai_model_create(shape, 1, seed, err, sizeof(err));
    double d16[2] = { 16, 1 }, d1[2] = { 1, 1 };
    size_t out[4]; int nd;
    if (!m || !ternaryai_model_add_layer(m, "dense", d16, 2, "uint8", out, &nd, err, sizeof(err)) ||
        !ternaryai_model_add_layer(m, "dense", d1, 2, "uint8", out, &nd, err, sizeof(err))) {
        fprintf(stderr, "xor_model: %s\n", err); exit(1);
    }

    return m;
}

static void test_xor_matches_manual(void) {
    int at100 = 0, mismatches = 0;
    double acc_sum = 0.0;
    for (unsigned seed = 0; seed < 25; seed++) {
        float manual[4], facade[4];
        manual_xor(seed, 100, manual);
        TernaryAIModel *m = xor_model(seed);
        double loss[100];
        CHECK(ternaryai_model_train(m, XOR_X, XOR_Y, 4, 0, 100, HYPER, loss, err, sizeof(err)), "train");
        CHECK(ternaryai_model_predict(m, XOR_X, 4, facade, err, sizeof(err)), "predict");
        if (memcmp(manual, facade, sizeof(manual)) != 0) mismatches++;
        int correct = 0;
        for (int s = 0; s < 4; s++) {
            int pred = facade[s] > 0 ? 1 : (facade[s] < 0 ? -1 : 0);
            if (pred == 0) pred = 1;
            if (pred == (int)XOR_Y[s]) correct++;
        }
        acc_sum += correct / 4.0;
        if (correct == 4) at100++;
        ternaryai_model_free(m);
    }
    CHECK(mismatches == 0, "la red de la fachada difiere del camino manual");
    CHECK(fabs(acc_sum / 25.0 - 1.0) < 1e-9 && at100 == 25, "R2 debe aprender AND en las 25 semillas (100%, 25/25)");
    printf("[XOR fachada vs manual] 25 semillas, %d salidas distintas | precision media %.1f%%, %d/25 al 100%%\n",
           mismatches, 100.0 * acc_sum / 25.0, at100);
}

/* ---- helpers para comparar archivos ---- */
static __attribute__((unused)) inline char *slurp(const char *path, long *len) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); *len = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)*len + 1);
    if (fread(buf, 1, (size_t)*len, f) != (size_t)*len) { fclose(f); free(buf); return NULL; }
    fclose(f); return buf;
}
static int same_file(const char *a, const char *b) { return model_dirs_equal(a, b); }

/* ---- 4) rechazo que no deja rastro ---- */
static void test_rejected_layer_leaves_no_trace(void) {
    size_t img[3] = {8, 8, 1}, out[4]; int nd;
    double conv1[5] = { 4, 3, 3, 1, 1 }, conv2[5] = { 2, 3, 3, 1, 1 }, d10[2] = { 10, 1 };
    TernaryAIModel *a = ternaryai_model_create(img, 3, 7, err, sizeof(err));
    TernaryAIModel *b = ternaryai_model_create(img, 3, 7, err, sizeof(err));
    CHECK(ternaryai_model_add_layer(a, "conv2d", conv1, 5, "uint8", out, &nd, err, sizeof(err)), "a: conv1");
    CHECK(ternaryai_model_add_layer(b, "conv2d", conv1, 5, "uint8", out, &nd, err, sizeof(err)), "b: conv1");
    int rejected = !ternaryai_model_add_layer(a, "conv2d", conv2, 5, "uint8", out, &nd, err, sizeof(err));
    CHECK(rejected && strstr(err, "first layer"), "Conv2D como segunda capa deberia rechazarse con mensaje claro");
    CHECK(ternaryai_model_num_layers(a) == 1, "el rechazo no debe agregar la capa");
    CHECK(ternaryai_model_add_layer(a, "dense", d10, 2, "uint8", out, &nd, err, sizeof(err)) && nd == 1 && out[0] == 10, "a: dense");
    CHECK(ternaryai_model_add_layer(b, "dense", d10, 2, "uint8", out, &nd, err, sizeof(err)), "b: dense");
    ternaryai_model_save(a, tmp_path("ternaryai_a.tn"), 1, err, sizeof(err));
    ternaryai_model_save(b, tmp_path("ternaryai_b.tn"), 1, err, sizeof(err));
    CHECK(same_file(tmp_path("ternaryai_a.tn"), tmp_path("ternaryai_b.tn")), "el intento rechazado altero los pesos iniciales de las capas siguientes");
    ternaryai_model_free(a); ternaryai_model_free(b);
    model_dir_remove(tmp_path("ternaryai_a.tn")); model_dir_remove(tmp_path("ternaryai_b.tn"));
    printf("[capa rechazada] el modelo queda igual que si nunca se hubiera intentado\n");
}

/* ---- 5) guardado / carga ---- */
static void test_save_load(void) {
    const char *ck = tmp_path("ternaryai_ck.tn"), *ex = tmp_path("ternaryai_ex.tn"), *ck2 = tmp_path("ternaryai_ck2.tn");
    double loss_a[20], loss_b1[8], loss_b2[12];

    /* pausar+retomar == entrenar de un tiron (byte a byte) */
    TernaryAIModel *a = xor_model(3);
    CHECK(ternaryai_model_train(a, XOR_X, XOR_Y, 4, 0, 20, HYPER, loss_a, err, sizeof(err)), "train a");
    CHECK(ternaryai_model_save(a, ck, 1, err, sizeof(err)), "save a");

    /* los parametros de R2 viajan en el checkpoint y NO en el export */
    {
        TernaryAIModel *r = xor_model(3); double q[5];
        CHECK(ternaryai_model_get_update_rule(r, q) && q[0] == (double)0.7f && q[2] == (double)0.03f && q[3] == (double)0.2f && q[4] == 0.5, "R2 por defecto");
        CHECK(!ternaryai_model_set_r2_params(r, 0.0, 0.2, 0.5, err, sizeof(err)) && !ternaryai_model_set_r2_params(r, 0.03, 1.0, 0.5, err, sizeof(err)) &&
              !ternaryai_model_set_r2_params(r, 0.03, 0.2, 1.5, err, sizeof(err)), "r2 params invalidos");
        CHECK(ternaryai_model_set_update_rule(r, 0.5, 0.99, err, sizeof(err)) && ternaryai_model_set_r2_params(r, 0.05, -1.0, 0.25, err, sizeof(err)), "fijar R2");
        const char *rck = tmp_path("ternaryai_r2ck.tn"), *rex = tmp_path("ternaryai_r2ex.tn");
        CHECK(ternaryai_model_save(r, rck, 1, err, sizeof(err)) && ternaryai_model_save(r, rex, 0, err, sizeof(err)), "save r2");
        ternaryai_model_free(r);
        r = ternaryai_model_load(rck, err, sizeof(err));
        CHECK(r && ternaryai_model_get_update_rule(r, q) && q[0] == (double)0.5f && q[1] == (double)0.99f && q[2] == (double)0.05f && q[3] == -1.0 && q[4] == 0.25, "el checkpoint restaura R2");
        ternaryai_model_free(r);
        r = ternaryai_model_load(rex, err, sizeof(err));
        CHECK(r && ternaryai_model_get_update_rule(r, q) && q[0] == (double)0.7f && q[2] == (double)0.03f, "el export no lleva R2 (quedan los defaults)");
        ternaryai_model_free(r);
        model_dir_remove(rck); model_dir_remove(rex);
    }

    TernaryAIModel *b = xor_model(3);
    CHECK(ternaryai_model_train(b, XOR_X, XOR_Y, 4, 0, 8, HYPER, loss_b1, err, sizeof(err)), "train b1");
    CHECK(ternaryai_model_save(b, ck2, 1, err, sizeof(err)), "save b");
    ternaryai_model_free(b);
    b = ternaryai_model_load(ck2, err, sizeof(err));
    CHECK(b != NULL, "load checkpoint");

    CHECK(ternaryai_model_train(b, XOR_X, XOR_Y, 4, 8, 12, HYPER, loss_b2, err, sizeof(err)), "train b2");
    CHECK(memcmp(loss_a, loss_b1, sizeof(loss_b1)) == 0 && memcmp(loss_a + 8, loss_b2, sizeof(loss_b2)) == 0,
          "las perdidas de pausar+retomar difieren de entrenar de un tiron");
    ternaryai_model_save(b, ck2, 1, err, sizeof(err));
    CHECK(same_file(ck, ck2), "pausar+retomar no da el mismo checkpoint que entrenar de un tiron");

    /* export: mas chico, misma inferencia, misma arquitectura */
    CHECK(ternaryai_model_save(a, ex, 0, err, sizeof(err)), "export");
    long lck = model_dir_bytes(ck), lex = model_dir_bytes(ex);
    CHECK(lex < lck, "el export deberia pesar menos que el checkpoint");
    TernaryAIModel *e = ternaryai_model_load(ex, err, sizeof(err));
    CHECK(e != NULL, "load export");
    float pa[4], pe[4];
    ternaryai_model_predict(a, XOR_X, 4, pa, err, sizeof(err));
    ternaryai_model_predict(e, XOR_X, 4, pe, err, sizeof(err));
    CHECK(memcmp(pa, pe, sizeof(pa)) == 0, "el export cargado no infiere igual");
    CHECK(ternaryai_model_num_layers(e) == 2, "capas del export");
    char tn[TERNARYAI_NAME_CAP], act[TERNARYAI_NAME_CAP]; double slots[TERNARYAI_MAX_SLOTS]; int ns, nd; size_t sh[4];
    CHECK(ternaryai_model_layer_info(e, 0, tn, slots, &ns, act, sh, &nd) && strcmp(tn, "dense") == 0 &&
          ns == 2 && slots[0] == 16 && slots[1] == 1 && strcmp(act, "uint8") == 0 && nd == 1 && sh[0] == 16, "layer_info 0");
    CHECK(ternaryai_model_layer_info(e, 1, tn, slots, &ns, act, sh, &nd) && slots[0] == 1 && sh[0] == 1, "layer_info 1");
    CHECK(!ternaryai_model_layer_info(e, 2, tn, slots, &ns, act, sh, &nd), "layer_info fuera de rango");
    ternaryai_model_free(a); ternaryai_model_free(b); ternaryai_model_free(e);
    model_dir_remove(ck); model_dir_remove(ck2); model_dir_remove(ex);

    /* la forma de entrada se recuerda (Dense sobre una imagen) y Conv2D se describe completo */
    size_t img[3] = {8, 8, 1}, out[4]; int nd2;
    double c[5] = { 4, 3, 2, 2, 0 }, d[2] = { 5, 1 };
    TernaryAIModel *m = ternaryai_model_create(img, 3, 1, err, sizeof(err));
    CHECK(ternaryai_model_add_layer(m, "conv2d", c, 5, "uint8", out, &nd2, err, sizeof(err)) && nd2 == 3 &&
          out[0] == 3 && out[1] == 4 && out[2] == 4, "forma de salida de conv2d 8x8 k(3,2) s2");
    CHECK(ternaryai_model_add_layer(m, "dense", d, 2, "uint8", out, &nd2, err, sizeof(err)), "dense tras conv");
    CHECK(ternaryai_model_save(m, ck, 1, err, sizeof(err)), "save conv");
    ternaryai_model_free(m);
    m = ternaryai_model_load(ck, err, sizeof(err));
    size_t in[4]; int ni;
    CHECK(m && ternaryai_model_input_shape(m, in, &ni) && ni == 3 && in[0] == 8 && in[1] == 8 && in[2] == 1, "forma de entrada tras cargar");
    CHECK(ternaryai_model_layer_info(m, 0, tn, slots, &ns, act, sh, &nd) && ns == 5 && slots[0] == 4 && slots[1] == 3 &&
          slots[2] == 2 && slots[3] == 2 && slots[4] == 0 && nd == 3, "layer_info conv2d");
    ternaryai_model_free(m);

    size_t flat_in[3] = {8, 8, 1};
    m = ternaryai_model_create(flat_in, 3, 1, err, sizeof(err));
    ternaryai_model_add_layer(m, "dense", d, 2, "uint8", out, &nd2, err, sizeof(err));
    ternaryai_model_save(m, ck, 1, err, sizeof(err));
    ternaryai_model_free(m);
    m = ternaryai_model_load(ck, err, sizeof(err));
    CHECK(m && ternaryai_model_input_shape(m, in, &ni) && ni == 3 && in[0] == 8 && in[2] == 1, "Dense sobre imagen: la forma (8,8,1) se pierde");
    ternaryai_model_free(m);

    /* un archivo suelto (formato viejo de un solo archivo) ya no es un modelo: se rechaza con un mensaje claro */
    {
        FILE *f0 = fopen(tmp_path("ternaryai_old.tn"), "wb"); fwrite("TNC2\0\0", 1, 6, f0); fclose(f0);
        CHECK(ternaryai_model_load(tmp_path("ternaryai_old.tn"), err, sizeof(err)) == NULL && strstr(err, "cannot read") &&
              strstr(err, "carpeta"), "un archivo suelto no es un modelo");
        remove(tmp_path("ternaryai_old.tn"));
    }
    model_dir_remove(ck);

    /* archivos inservibles */
    CHECK(ternaryai_model_load(tmp_path("ternaryai_no_existe.tn"), err, sizeof(err)) == NULL && strstr(err, "cannot read"), "archivo inexistente");
    FILE *f = fopen(tmp_path("ternaryai_bad.tn"), "wb"); fputs("no soy un modelo", f); fclose(f);
    CHECK(ternaryai_model_load(tmp_path("ternaryai_bad.tn"), err, sizeof(err)) == NULL, "archivo corrupto");
    remove(tmp_path("ternaryai_bad.tn"));
    TernaryAIModel *empty = ternaryai_model_create(flat_in, 3, 0, err, sizeof(err));
    CHECK(!ternaryai_model_save(empty, tmp_path("ternaryai_empty.tn"), 1, err, sizeof(err)), "guardar un modelo sin capas");
    CHECK(!ternaryai_model_train(empty, XOR_X, XOR_Y, 4, 0, 1, HYPER, loss_a, err, sizeof(err)), "entrenar un modelo sin capas");
    ternaryai_model_free(empty);
    ternaryai_model_free(NULL);
    printf("[guardado/carga] checkpoint retomable byte a byte, export mas chico, forma recordada, carpeta model.ternary OK\n");
}

static void test_train_validation(void) {
    TernaryAIModel *m = xor_model(0);
    int8_t bad_x[8] = { -1,-1, -1,2, 1,-1, 1,1 };
    double loss[1], bad_h[TERNARYAI_NUM_HYPER] = { 0.0 };
    CHECK(!ternaryai_model_train(m, bad_x, XOR_Y, 4, 0, 1, HYPER, loss, err, sizeof(err)) && strstr(err, "-1, 0 and 1"), "x fuera de {-1,0,1}");
    CHECK(!ternaryai_model_train(m, XOR_X, XOR_Y, 0, 0, 1, HYPER, loss, err, sizeof(err)), "n=0");
    CHECK(!ternaryai_model_train(m, XOR_X, XOR_Y, 4, 0, 0, HYPER, loss, err, sizeof(err)), "count=0");
    CHECK(!ternaryai_model_train(m, XOR_X, XOR_Y, 4, 0, 1, bad_h, loss, err, sizeof(err)), "hiperparametro en 0");
    float out[4];
    CHECK(!ternaryai_model_predict(m, bad_x, 4, out, err, sizeof(err)), "predict con x invalido");
    ternaryai_model_free(m);
    printf("[validacion de entrenamiento] OK\n");
}

int main(void) {
    test_rng();
    test_registry();
    test_param_validation();
    test_xor_matches_manual();
    test_rejected_layer_leaves_no_trace();
    test_save_load();
    test_train_validation();
    printf("fallas: %d\n", fails);
    if (fails == 0) printf("TODAS LAS PRUEBAS DE LA API PLANA (libternaryai) PASARON\n");
    return fails ? 1 : 0;
}
