/*
 * SwiGLU ternario (capa "swiglu"). Verifica, para LAS DOS compuertas (0: silu por tabla, 1: sign(gate)*up):
 *  1. registro por nombre, describe() de vuelta y errores de parametros;
 *  2. forward IDENTICO a una referencia escrita aparte (pesos leidos con get_weight, mismas operaciones
 *     float en el mismo orden), con escalas distintas de 1 y tres valores de banda;
 *  3. la tabla de silu' coincide con la derivada numerica, y d(h)/d(escala_gate) analitico == diferencias
 *     finitas sobre el producto h (lo unico derivable antes de cuantizar);
 *  4. backward IGUAL a una referencia en double de la cadena completa (estimador directo en la
 *     cuantizacion): gradientes de las tres matrices, de la entrada y de las escalas (la de up, siempre 0;
 *     la de gate, 0 en la compuerta sign);
 *  5. capa dentro del TernaryStack: forward del stack == forward directo; checkpoint completo (entrenar
 *     20+20 epocas == 40 de un tiron) y export empaquetado (inferencia identica);
 *  6. APRENDIZAJE contra un maestro con la misma arquitectura, y COMPARACION de las dos compuertas
 *     (se imprime el error de cada una: es la medicion que decide si vale la modulacion suave de silu).
 */
#include "ternary_stack.h"
#include "tmp_path.h"
#include "ternary_layer.h"
#include "ternary_activation.h"
#include "ternary_rng.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FALLO: %s (linea %d)\n", msg, __LINE__); return 1; } } while (0)

#define ROWS 3
#define DIM 24
#define HID 40

static uint64_t rs = 88172645463325252ull;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) / 9007199254740992.0; }
static double nrand(void) { double u = urand() + 1e-12, v = urand(); return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v); }

static int sgn_i(int32_t v) { return v > 0 ? 1 : (v < 0 ? -1 : 0); }

static TernarySwiGLU *make_layer(int gate_mode, float band, unsigned seed) {
    TernaryRng rng; ternary_rng_seed(&rng, seed);
    size_t in[2] = { ROWS, DIM };
    double p[3] = { HID, (double)gate_mode, (double)band };
    TernaryLayer l; char err[200];
    if (!ternary_layer_type_build(ternary_layer_type_by_name("swiglu"), in, 2, p, 3, &rng, &l, err, sizeof(err))) {
        fprintf(stderr, "build: %s\n", err); return NULL;
    }
    return (TernarySwiGLU *)l.impl;
}

static void random_x(TernaryVec *x, size_t n) {
    for (size_t i = 0; i < n; i++) ternary_vec_set(x, i, (int)(urand() * 3.0) - 1);
}

/* Referencia del forward de UNA fila: llena zg, zu, h, t y devuelve zd. Mismas operaciones float que la capa. */
static void ref_row(const TernarySwiGLU *s, const TernaryVec *x, size_t p, int32_t *zg, int32_t *zu, float *h, int *t, int32_t *zd) {
    const size_t D = s->dim, H = s->hidden;
    for (size_t j = 0; j < H; j++) {
        int32_t a = 0, b = 0;
        for (size_t i = 0; i < D; i++) {
            int xi = ternary_vec_get(x, p * D + i);
            a += ternary_dense_get_weight(s->wg, j, i) * xi;
            b += ternary_dense_get_weight(s->wu, j, i) * xi;
        }
        zg[j] = a; zu[j] = b;
    }
    for (size_t j = 0; j < H; j++) {
        if (s->gate_mode == 0) {
            const double aa = (double)s->wg->scale * (double)zg[j];
            const double sig = 1.0 / (1.0 + exp(-aa));
            const float silu = (float)(aa * sig);
            h[j] = silu * (s->wu->scale * (float)zu[j]);
        } else {
            h[j] = (float)(sgn_i(zg[j]) * zu[j]);
        }
    }
    float sum = 0.0f;
    for (size_t j = 0; j < H; j++) sum += fabsf(h[j]);
    const float thr = s->band * (sum / (float)H);
    for (size_t j = 0; j < H; j++) t[j] = (fabsf(h[j]) > thr) ? (h[j] > 0.0f ? 1 : -1) : 0;
    for (size_t o = 0; o < D; o++) {
        int32_t z = 0;
        for (size_t j = 0; j < H; j++) z += ternary_dense_get_weight(s->wd, o, j) * t[j];
        zd[o] = z;
    }
}

static int test_registry(void) {
    const TernaryLayerType *ty = ternary_layer_type_by_name("swiglu");
    CHECK(ty != NULL, "registro: swiglu");
    CHECK(strcmp(ty->class_name, "SwiGLU") == 0, "nombre de clase");
    CHECK(ternary_layer_type_num_slots(ty) == 4, "4 parametros");

    TernaryRng rng; ternary_rng_seed(&rng, 1);
    char err[200]; TernaryLayer l;
    size_t in1[1] = { DIM }, in2[2] = { ROWS, DIM };
    double bad_mode[3] = { HID, 3, 0.5 }, bad_hid[3] = { 0, 0, 0.5 }, bad_band[3] = { HID, 0, -1 };
    CHECK(!ternary_layer_type_build(ty, in2, 2, bad_mode, 3, &rng, &l, err, sizeof(err)) && strstr(err, "gate_mode"), "gate_mode invalido");
    CHECK(!ternary_layer_type_build(ty, in2, 2, bad_hid, 3, &rng, &l, err, sizeof(err)) && strstr(err, "hidden"), "hidden 0");
    CHECK(!ternary_layer_type_build(ty, in2, 2, bad_band, 3, &rng, &l, err, sizeof(err)) && strstr(err, "band"), "band negativa");

    double p[3] = { HID, 1, 0.25 };
    CHECK(ternary_layer_type_build(ty, in2, 2, p, 3, &rng, &l, err, sizeof(err)), "build 2D");
    double back[4]; size_t shape[TERNARY_MAX_SHAPE_DIMS];
    size_t nd = ty->describe(l.impl, back, shape);
    CHECK(nd == 2 && shape[0] == ROWS && shape[1] == DIM, "describe: forma 2D");
    CHECK(back[0] == HID && back[1] == 1 && back[2] == 0.25 && back[3] == 1.0, "describe: parametros");
    CHECK(l.vt->input_len(l.impl) == ROWS * DIM && l.vt->output_len(l.impl) == ROWS * DIM, "largos");
    CHECK(l.vt->num_matrices(l.impl) == 3, "3 matrices");
    CHECK(l.vt->grad_rows(l.impl, 0) == HID && l.vt->grad_row_width(l.impl, 0) == DIM, "forma W_gate");
    CHECK(l.vt->grad_rows(l.impl, 2) == DIM && l.vt->grad_row_width(l.impl, 2) == HID, "forma W_down");
    l.vt->destroy(l.impl);

    CHECK(ternary_layer_type_build(ty, in1, 1, p, 3, &rng, &l, err, sizeof(err)), "build 1D");
    nd = ty->describe(l.impl, back, shape);
    CHECK(nd == 1 && shape[0] == DIM, "describe: forma 1D");
    l.vt->destroy(l.impl);
    return 0;
}

static int test_forward_reference(int gate_mode, float band) {
    TernarySwiGLU *s = make_layer(gate_mode, band, 11 + (unsigned)gate_mode);
    CHECK(s, "build");
    s->wg->scale = 0.7f; s->wu->scale = 1.3f; s->wd->scale = 0.45f;
    ternary_swiglu_refresh_tables(s);
    TernaryVec *x = ternary_vec_alloc(ROWS * DIM);
    random_x(x, ROWS * DIM);
    float out[ROWS * DIM]; int32_t raw[ROWS * DIM];
    ternary_swiglu_forward(s, x, out, raw);
    int nonzero_t = 0, total_t = 0;
    for (size_t p = 0; p < ROWS; p++) {
        int32_t zg[HID], zu[HID], zd[DIM]; float h[HID]; int t[HID];
        ref_row(s, x, p, zg, zu, h, t, zd);
        for (size_t j = 0; j < HID; j++) { total_t++; if (t[j]) nonzero_t++; }
        for (size_t o = 0; o < DIM; o++) {
            CHECK(raw[p * DIM + o] == zd[o], "forward: Z crudo de W_down distinto de la referencia");
            CHECK(out[p * DIM + o] == s->wd->scale * (float)zd[o], "forward: salida escalada distinta de la referencia");
        }
    }
    if (band >= 100.0f) CHECK(nonzero_t == 0, "banda enorme: t debe ser todo cero");
    else CHECK(nonzero_t > 0 && (band > 0.0f ? nonzero_t < total_t : 1), "banda normal: ni todo cero ni (con banda>0) todo distinto de cero");
    ternary_vec_free(x); ternary_swiglu_free(s);
    return 0;
}

static int test_tables_and_product(void) {
    TernarySwiGLU *s = make_layer(0, 0.5f, 21);
    CHECK(s, "build");
    s->wg->scale = 0.6f; ternary_swiglu_refresh_tables(s);
    /* silu' de la tabla == derivada numerica de silu */
    for (int z = -(int)DIM; z <= (int)DIM; z += 5) {
        const double a = 0.6 * z, e = 1e-5;
        double f1 = (a + e) / (1.0 + exp(-(a + e))), f0 = (a - e) / (1.0 + exp(-(a - e)));
        double num = (f1 - f0) / (2.0 * e);
        CHECK(fabs(num - (double)s->dsilu_tab[z + DIM]) < 1e-4, "tabla de silu': distinta de la derivada numerica");
        CHECK(fabs(a / (1.0 + exp(-a)) - (double)s->silu_tab[z + DIM]) < 1e-5, "tabla de silu: distinta de la funcion");
    }
    /* dh/d(escala_gate) analitico == diferencias finitas sobre el producto */
    TernaryVec *x = ternary_vec_alloc(ROWS * DIM);
    random_x(x, ROWS * DIM);
    float h0[HID], hp[HID], hm[HID];
    ternary_swiglu_product(s, x, 1, h0);
    const float eps = 0.01f, sg = s->wg->scale;
    s->wg->scale = sg + eps; ternary_swiglu_refresh_tables(s); ternary_swiglu_product(s, x, 1, hp);
    s->wg->scale = sg - eps; ternary_swiglu_refresh_tables(s); ternary_swiglu_product(s, x, 1, hm);
    s->wg->scale = sg; ternary_swiglu_refresh_tables(s);
    ternary_swiglu_product(s, x, 1, h0);   /* deja zg/zu de la fila 1 vigentes */
    int checked = 0;
    for (size_t j = 0; j < HID; j++) {
        const double fd = ((double)hp[j] - (double)hm[j]) / (2.0 * eps);
        const int32_t z = s->zg[1 * HID + j];
        const double an = (double)s->dsilu_tab[z + DIM] * (double)z * (double)(s->wu->scale * (float)s->zu[1 * HID + j]);
        CHECK(fabs(fd - an) < 0.05 + 0.02 * fabs(an), "dh/d(escala_gate): analitico distinto de diferencias finitas");
        if (fabs(an) > 0.1) checked++;
    }
    CHECK(checked > 5, "el chequeo de diferencias finitas debe ejercitar varias posiciones no triviales");
    ternary_vec_free(x); ternary_swiglu_free(s);
    return 0;
}

static int test_backward_reference(int gate_mode) {
    TernarySwiGLU *s = make_layer(gate_mode, 0.5f, 31 + (unsigned)gate_mode);
    CHECK(s, "build");
    s->wg->scale = 0.8f; s->wu->scale = 1.2f; s->wd->scale = 0.5f;
    ternary_swiglu_refresh_tables(s);
    TernaryVec *x = ternary_vec_alloc(ROWS * DIM);
    random_x(x, ROWS * DIM);
    float out[ROWS * DIM], go[ROWS * DIM];
    ternary_swiglu_forward(s, x, out, NULL);
    for (size_t i = 0; i < ROWS * DIM; i++) go[i] = (float)nrand();

    float *gw0 = (float *)malloc(HID * DIM * sizeof(float)), *gw1 = (float *)malloc(HID * DIM * sizeof(float)), *gw2 = (float *)malloc(DIM * HID * sizeof(float));
    float *gw[3] = { gw0, gw1, gw2 };
    float gx[ROWS * DIM]; memset(gx, 0, sizeof(gx));
    float gs[3] = { 0, 0, 0 };
    ternary_swiglu_backward(s, x, go, gw, gx, gs);

    /* Referencia en double */
    double *rg0 = (double *)calloc(HID * DIM, sizeof(double)), *rg1 = (double *)calloc(HID * DIM, sizeof(double)), *rg2 = (double *)calloc(DIM * HID, sizeof(double));
    double rgx[ROWS * DIM]; memset(rgx, 0, sizeof(rgx));
    double rs_g = 0, rs_d = 0;
    for (size_t p = 0; p < ROWS; p++) {
        int32_t zg[HID], zu[HID], zd[DIM]; float h[HID]; int t[HID];
        ref_row(s, x, p, zg, zu, h, t, zd);
        double gt[HID];
        for (size_t j = 0; j < HID; j++) {
            double a = 0;
            for (size_t o = 0; o < DIM; o++) a += (double)go[p * DIM + o] * (double)s->wd->scale * ternary_dense_get_weight(s->wd, o, j);
            gt[j] = a;
        }
        for (size_t o = 0; o < DIM; o++) {
            rs_d += (double)go[p * DIM + o] * (double)zd[o];
            for (size_t j = 0; j < HID; j++) rg2[o * HID + j] += (double)go[p * DIM + o] * (double)t[j];
        }
        for (size_t j = 0; j < HID; j++) {
            const double u = (double)s->wu->scale * (double)zu[j];
            double gu, ga;
            if (gate_mode == 0) {
                const double aa = (double)s->wg->scale * (double)zg[j];
                const double sig = 1.0 / (1.0 + exp(-aa));
                gu = gt[j] * (aa * sig);
                ga = gt[j] * u * (sig * (1.0 + aa * (1.0 - sig)));
                rs_g += ga * (double)zg[j];
            } else {
                gu = gt[j] * (double)sgn_i(zg[j]);
                ga = gt[j] * u;
            }
            for (size_t i = 0; i < DIM; i++) {
                const double xi = (double)ternary_vec_get(x, p * DIM + i);
                rg0[j * DIM + i] += ga * xi;
                rg1[j * DIM + i] += gu * xi;
                rgx[p * DIM + i] += (double)s->wg->scale * ga * ternary_dense_get_weight(s->wg, j, i)
                                  + (double)s->wu->scale * gu * ternary_dense_get_weight(s->wu, j, i);
            }
        }
    }
    double maxd = 0, maxref = 0;
    for (size_t k = 0; k < HID * DIM; k++) {
        maxd = fmax(maxd, fabs(rg0[k] - gw0[k])); maxd = fmax(maxd, fabs(rg1[k] - gw1[k]));
        maxref = fmax(maxref, fmax(fabs(rg0[k]), fabs(rg1[k])));
    }
    for (size_t k = 0; k < DIM * HID; k++) { maxd = fmax(maxd, fabs(rg2[k] - gw2[k])); maxref = fmax(maxref, fabs(rg2[k])); }
    for (size_t k = 0; k < ROWS * DIM; k++) { maxd = fmax(maxd, fabs(rgx[k] - gx[k])); maxref = fmax(maxref, fabs(rgx[k])); }
    CHECK(maxref > 1e-3, "los gradientes de referencia no deben ser todos ~0");
    CHECK(maxd < 1e-4 * (1.0 + maxref), "backward: gradientes (W, x) distintos de la referencia");
    CHECK(fabs(rs_d - gs[2]) < 1e-3 * (1.0 + fabs(rs_d)), "backward: gradiente de escala de W_down");
    CHECK(gs[1] == 0.0f, "la escala de up no recibe gradiente");
    if (gate_mode == 0) CHECK(fabs(rs_g - gs[0]) < 1e-3 * (1.0 + fabs(rs_g)) && gs[0] != 0.0f, "backward: gradiente de escala de gate (silu)");
    else CHECK(gs[0] == 0.0f, "compuerta sign: la escala de gate no recibe gradiente");
    free(gw0); free(gw1); free(gw2); free(rg0); free(rg1); free(rg2);
    ternary_vec_free(x); ternary_swiglu_free(s);
    return 0;
}

/* ---------------- stack ---------------- */
#define NIN (ROWS * DIM)
static const float SCALE_LR = 0.01f;
#define BATCH 8

static TernaryStack *new_stack(int gate_mode, unsigned seed) {
    TernaryRng rng; ternary_rng_seed(&rng, seed);
    size_t in[2] = { ROWS, DIM };
    double p[3] = { HID, (double)gate_mode, 0.5 };
    TernaryLayer l; char err[200];
    if (!ternary_layer_type_build(ternary_layer_type_by_name("swiglu"), in, 2, p, 3, &rng, &l, err, sizeof(err))) return NULL;
    TernaryStack *s = ternary_stack_create();
    if (!ternary_stack_add(s, l, TERNARY_ACTIVATION_UINT8)) return NULL;
    return s;
}

static TernaryStack *g_teacher;
static TernaryCapabilities g_cn;
static void make_sample(TernaryVec *x, float *target) {
    random_x(x, NIN);
    size_t xd[1] = { NIN }; TernaryTensor *xt = ternary_tensor_view_ternary(xd, 1, x);
    ternary_stack_forward(g_teacher, xt, target, &g_cn);
    ternary_tensor_free(xt);
}

/* Entrena y devuelve el error cuadratico medio de la ULTIMA epoca (promedio del lote). */
static double train_stack(TernaryStack *s, int epochs) {
    TernaryVec *x = ternary_vec_alloc(NIN); size_t xd[1] = { NIN };
    TernaryTensor *xt = ternary_tensor_view_ternary(xd, 1, x);
    float out[NIN], go[NIN], target[NIN];
    double last = 0.0;
    for (int ep = 0; ep < epochs; ep++) {
        ternary_stack_zero_gradients(s);
        double loss = 0.0;
        for (int b = 0; b < BATCH; b++) {
            make_sample(x, target);
            ternary_stack_forward(s, xt, out, &g_cn);
            for (size_t o = 0; o < NIN; o++) {
                float d = out[o] - target[o];
                go[o] = (2.0f / (float)NIN) * d / (float)BATCH;
                loss += (double)d * (double)d / (double)NIN;
            }
            ternary_stack_backward(s, xt, go);
        }
        ternary_stack_apply_gradients(s, SCALE_LR);
        last = loss / BATCH;
    }
    ternary_tensor_free(xt); ternary_vec_free(x);
    return last;
}

/* Evalua sobre un conjunto FIJO (semilla propia) y deja intacto el generador de entrenamiento: asi el
 * antes/despues se mide con las mismas muestras y evaluar no altera la secuencia que ve el entrenamiento. */
static double eval_stack(TernaryStack *s, int n) {
    const uint64_t keep = rs;
    rs = 0xD1B54A32D192ED03ull;
    TernaryVec *x = ternary_vec_alloc(NIN); size_t xd[1] = { NIN };
    TernaryTensor *xt = ternary_tensor_view_ternary(xd, 1, x);
    float out[NIN], target[NIN]; double loss = 0.0;
    for (int i = 0; i < n; i++) {
        make_sample(x, target);
        ternary_stack_forward(s, xt, out, &g_cn);
        for (size_t o = 0; o < NIN; o++) { double d = (double)out[o] - (double)target[o]; loss += d * d / (double)NIN; }
    }
    ternary_tensor_free(xt); ternary_vec_free(x);
    rs = keep;
    return loss / n;
}

static int same_dense(const TernaryDense *a, const TernaryDense *b) {
    if (a->input_dim != b->input_dim || a->output_dim != b->output_dim || a->scale != b->scale) return 0;
    for (size_t o = 0; o < a->output_dim; o++) for (size_t i = 0; i < a->input_dim; i++)
        if (ternary_dense_get_weight(a, o, i) != ternary_dense_get_weight(b, o, i)) return 0;
    return 1;
}
static const TernarySwiGLU *sw_at(const TernaryStack *s) { return (const TernarySwiGLU *)ternary_stack_layer(s, 0).impl; }
static int same_stack(const TernaryStack *a, const TernaryStack *b) {
    const TernarySwiGLU *x = sw_at(a), *y = sw_at(b);
    return x->gate_mode == y->gate_mode && x->band == y->band && same_dense(x->wg, y->wg) && same_dense(x->wu, y->wu) && same_dense(x->wd, y->wd);
}

static int test_stack_and_persistence(int gate_mode) {
    memset(&g_cn, 0, sizeof(g_cn)); g_cn.num_cores = 1; g_cn.dot_impl_name = "scalar";
    g_teacher = new_stack(gate_mode, 777);
    CHECK(g_teacher, "maestro");

    /* forward del stack == forward directo de la capa */
    {
        TernaryStack *s = new_stack(gate_mode, 5);
        CHECK(s, "stack");
        TernaryVec *x = ternary_vec_alloc(NIN); random_x(x, NIN);
        size_t xd[1] = { NIN }; TernaryTensor *xt = ternary_tensor_view_ternary(xd, 1, x);
        float a[NIN], b[NIN];
        ternary_stack_forward(s, xt, a, &g_cn);
        ternary_swiglu_forward(sw_at(s), x, b, NULL);
        CHECK(memcmp(a, b, sizeof(a)) == 0, "forward del stack distinto del directo");
        ternary_tensor_free(xt); ternary_vec_free(x); ternary_stack_free(s);
    }

    /* checkpoint completo: 20 + (guardar, cargar) + 20 == 40 de un tiron. El generador de muestras se
     * reinicia para que ambos caminos vean exactamente las mismas muestras. */
    char path[1100]; snprintf(path, sizeof(path), "%s/test_swiglu_ck_%d.bin", tmp_dir(), gate_mode);
    const uint64_t rs0 = 123456789012345ull;
    rs = rs0;
    TernaryStack *a = new_stack(gate_mode, 9);
    CHECK(a, "stack a");
    train_stack(a, 40);
    rs = rs0;
    TernaryStack *b = new_stack(gate_mode, 9);
    train_stack(b, 20);
    CHECK(ternary_stack_save(b, path, 1), "guardar checkpoint");
    TernaryStack *c = ternary_stack_load(path);
    CHECK(c, "cargar checkpoint");
    CHECK(same_stack(b, c), "el checkpoint cargado difiere del guardado");
    train_stack(c, 20);
    CHECK(same_stack(a, c), "checkpoint: 20+20 distinto de 40 de un tiron");

    /* export empaquetado: inferencia identica */
    snprintf(path, sizeof(path), "%s/test_swiglu_ex_%d.bin", tmp_dir(), gate_mode);
    CHECK(ternary_stack_save(a, path, 0), "guardar export");
    TernaryStack *e = ternary_stack_load(path);
    CHECK(e, "cargar export");
    TernaryVec *x = ternary_vec_alloc(NIN); size_t xd[1] = { NIN };
    TernaryTensor *xt = ternary_tensor_view_ternary(xd, 1, x);
    for (int k = 0; k < 10; k++) {
        random_x(x, NIN);
        float o1[NIN], o2[NIN];
        ternary_stack_forward(a, xt, o1, &g_cn);
        ternary_stack_forward(e, xt, o2, &g_cn);
        CHECK(memcmp(o1, o2, sizeof(o1)) == 0, "export: inferencia distinta");
    }
    ternary_tensor_free(xt); ternary_vec_free(x);
    ternary_stack_free(a); ternary_stack_free(b); ternary_stack_free(c); ternary_stack_free(e);
    ternary_stack_free(g_teacher); g_teacher = NULL;
    model_dir_remove(tmp_path("test_swiglu_ck_0.bin")); model_dir_remove(tmp_path("test_swiglu_ck_1.bin"));
    model_dir_remove(tmp_path("test_swiglu_ex_0.bin")); model_dir_remove(tmp_path("test_swiglu_ex_1.bin"));
    return 0;
}

/* Aprendizaje contra un maestro de la misma arquitectura + comparacion de compuertas. */
static int test_learning(int gate_mode, double *loss0_out, double *loss1_out) {
    memset(&g_cn, 0, sizeof(g_cn)); g_cn.num_cores = 1; g_cn.dot_impl_name = "scalar";
    g_teacher = new_stack(gate_mode, 4242);
    CHECK(g_teacher, "maestro");
    rs = 0x9E3779B97F4A7C15ull;
    TernaryStack *s = new_stack(gate_mode, 17);
    CHECK(s, "alumno");
    double before = eval_stack(s, 64);
    train_stack(s, 400);
    double after = eval_stack(s, 64);
    *loss0_out = before; *loss1_out = after;
    ternary_stack_free(s); ternary_stack_free(g_teacher); g_teacher = NULL;
    return 0;
}

/* ---------------- activacion uint8 (la unica) ---------------- */
static int test_band_activation(void) {
    const TernaryActivation *a = ternary_activation_by_name("uint8");
    CHECK(a == &TERNARY_ACTIVATION_UINT8 && strcmp(a->name, "uint8") == 0, "registro de la activacion uint8");
    CHECK(ternary_activation_by_name("sign") == NULL && ternary_activation_by_name("band") == NULL, "sign y band ya no existen");
    const float v[8] = { 4.0f, -2.0f, 1.0f, 0.5f, 0.0f, -0.1f, -3.0f, 0.2f };
    TernaryU8 *t = ternary_u8_alloc(1, 8);
    a->apply_u8(v, t);
    float dq[8]; ternary_u8_dequant(t, dq);
    for (int i = 0; i < 8; i++) CHECK(fabsf(dq[i] - v[i]) <= 0.5f * t->scale[0] + 1e-6f, "uint8: error de cuantizacion mayor que medio paso");
    CHECK(dq[4] == 0.0f, "uint8: el cero se representa exacto");
    const float zero[4] = { 0, 0, 0, 0 };
    TernaryU8 *z = ternary_u8_alloc(1, 4);
    a->apply_u8(zero, z); ternary_u8_dequant(z, dq);
    for (int i = 0; i < 4; i++) CHECK(dq[i] == 0.0f, "uint8: entrada todo cero -> todo cero");
    ternary_u8_free(t); ternary_u8_free(z);
    return 0;
}

/* Dense -> RMSNorm [activacion band] -> SwiGLU: el bloque de uso real de la activacion. */
static TernaryStack *new_block(unsigned seed, char *err, size_t cap) {
    TernaryRng rng; ternary_rng_seed(&rng, seed);
    size_t in0[1] = { NIN }, in2[2] = { ROWS, DIM };
    double pd[2] = { ROWS * DIM, 0 }, pr[1] = { 1e-5 }, ps[3] = { HID, 0, 0.5 };
    TernaryLayer l1, l2, l3;
    if (!ternary_layer_type_build(ternary_layer_type_by_name("dense"), in0, 1, pd, 2, &rng, &l1, err, cap)) return NULL;
    if (!ternary_layer_type_build(ternary_layer_type_by_name("rmsnorm"), in2, 2, pr, 1, &rng, &l2, err, cap)) return NULL;
    if (!ternary_layer_type_build(ternary_layer_type_by_name("swiglu"), in2, 2, ps, 3, &rng, &l3, err, cap)) return NULL;
    TernaryStack *s = ternary_stack_create();
    if (!ternary_stack_add(s, l1, TERNARY_ACTIVATION_UINT8) || !ternary_stack_add(s, l2, TERNARY_ACTIVATION_UINT8) ||
        !ternary_stack_add(s, l3, TERNARY_ACTIVATION_UINT8)) return NULL;
    return s;
}
static int same_block(const TernaryStack *a, const TernaryStack *b) {
    const TernaryDense *d0 = (const TernaryDense *)ternary_stack_layer(a, 0).impl, *e0 = (const TernaryDense *)ternary_stack_layer(b, 0).impl;
    const TernaryRMSNorm *r0 = (const TernaryRMSNorm *)ternary_stack_layer(a, 1).impl, *r1 = (const TernaryRMSNorm *)ternary_stack_layer(b, 1).impl;
    const TernarySwiGLU *s0 = (const TernarySwiGLU *)ternary_stack_layer(a, 2).impl, *s1 = (const TernarySwiGLU *)ternary_stack_layer(b, 2).impl;
    return same_dense(d0, e0) && same_dense(r0->gain, r1->gain) && same_dense(s0->wg, s1->wg) && same_dense(s0->wu, s1->wu) && same_dense(s0->wd, s1->wd);
}

static int test_band_block(void) {
    memset(&g_cn, 0, sizeof(g_cn)); g_cn.num_cores = 1; g_cn.dot_impl_name = "scalar";
    char err[256] = "";
    g_teacher = new_block(555, err, sizeof(err));
    CHECK(g_teacher, "maestro del bloque");
    CHECK(strcmp(ternary_stack_activation(g_teacher, 1)->name, "uint8") == 0, "la activacion de RMSNorm es uint8");

    /* forward del stack == calculo manual: Dense -> RMSNorm sobre el FLOAT crudo -> band -> SwiGLU */
    {
        TernaryStack *s = new_block(5, err, sizeof(err));
        CHECK(s, "bloque");
        TernaryVec *x = ternary_vec_alloc(NIN); random_x(x, NIN);
        size_t xd[1] = { NIN }; TernaryTensor *xt = ternary_tensor_view_ternary(xd, 1, x);
        float got[NIN], z[NIN], y[NIN], want[NIN];
        ternary_stack_forward(s, xt, got, &g_cn);
        ternary_dense_forward((const TernaryDense *)ternary_stack_layer(s, 0).impl, x, z, NULL);
        ternary_rmsnorm_forward((const TernaryRMSNorm *)ternary_stack_layer(s, 1).impl, z, y);
        TernaryU8 *tv = ternary_u8_alloc(ROWS, DIM);
        TERNARY_ACTIVATION_UINT8.apply_u8(y, tv);
        ternary_swiglu_forward_u8((const TernarySwiGLU *)ternary_stack_layer(s, 2).impl, tv, want, NULL);
        CHECK(memcmp(got, want, sizeof(got)) == 0, "bloque: forward del stack distinto del calculo manual");
        ternary_u8_free(tv); ternary_tensor_free(xt); ternary_vec_free(x); ternary_stack_free(s);
    }

    /* checkpoint: guardar/cargar conserva la activacion por NOMBRE y 20+20 == 40 de un tiron */
    const uint64_t rs0 = 987654321098765ull;
    rs = rs0;
    TernaryStack *a = new_block(9, err, sizeof(err));
    double a0 = eval_stack(a, 32);
    train_stack(a, 40);
    double a1 = eval_stack(a, 32);
    rs = rs0;
    TernaryStack *b = new_block(9, err, sizeof(err));
    train_stack(b, 20);
    const char *path = tmp_path("test_swiglu_block.bin");
    CHECK(ternary_stack_save(b, path, 1), "guardar bloque");
    TernaryStack *c = ternary_stack_load(path);
    CHECK(c, "cargar bloque");
    CHECK(strcmp(ternary_stack_activation(c, 1)->name, "uint8") == 0 && strcmp(ternary_stack_activation(c, 0)->name, "uint8") == 0,
          "el checkpoint conserva la activacion band de RMSNorm");
    train_stack(c, 20);
    CHECK(same_block(a, c), "bloque: checkpoint 20+20 distinto de 40 de un tiron");
    model_dir_remove(path);

    /* el gradiente atraviesa SwiGLU -> band (estimador directo) -> RMSNorm hasta la Dense: sus pesos cambian */
    TernaryStack *fresh = new_block(9, err, sizeof(err));
    CHECK(!same_dense((const TernaryDense *)ternary_stack_layer(a, 0).impl, (const TernaryDense *)ternary_stack_layer(fresh, 0).impl),
          "la Dense anterior debe haber aprendido a traves de band/RMSNorm");
    printf("APRENDIZAJE bloque Dense->RMSNorm[band]->SwiGLU (ECM, 40 epocas): %.3f -> %.3f\n", a0, a1);
    CHECK(a1 < a0, "el bloque con band no baja el error");
    ternary_stack_free(a); ternary_stack_free(b); ternary_stack_free(c); ternary_stack_free(fresh);
    ternary_stack_free(g_teacher); g_teacher = NULL;
    return 0;
}

int main(void) {
    if (test_registry()) return 1;
    static const float bands[3] = { 0.0f, 0.5f, 1000.0f };
    for (int gm = 0; gm < 2; gm++)
        for (int b = 0; b < 3; b++)
            if (test_forward_reference(gm, bands[b])) return 1;
    printf("OK: forward identico a la referencia (2 compuertas x 3 bandas)\n");
    if (test_tables_and_product()) return 1;
    printf("OK: tabla de silu/silu' y d(h)/d(escala_gate) por diferencias finitas\n");
    for (int gm = 0; gm < 2; gm++) if (test_backward_reference(gm)) return 1;
    printf("OK: backward igual a la referencia en double (2 compuertas)\n");
    for (int gm = 0; gm < 2; gm++) if (test_stack_and_persistence(gm)) return 1;
    printf("OK: stack == forward directo; checkpoint 20+20 == 40; export identico (2 compuertas)\n");

    double l0[2], l1[2];
    for (int gm = 0; gm < 2; gm++) if (test_learning(gm, &l0[gm], &l1[gm])) return 1;
    printf("APRENDIZAJE (ECM contra maestro, 400 epocas): silu   %.3f -> %.3f  (final/inicial = %.3f)\n", l0[0], l1[0], l1[0] / l0[0]);
    printf("APRENDIZAJE (ECM contra maestro, 400 epocas): sign   %.3f -> %.3f  (final/inicial = %.3f)\n", l0[1], l1[1], l1[1] / l0[1]);
    for (int gm = 0; gm < 2; gm++)
        if (!(l1[gm] < 0.85 * l0[gm])) { fprintf(stderr, "FALLO: la compuerta %d no aprende (%.3f -> %.3f)\n", gm, l0[gm], l1[gm]); return 1; }
    if (test_band_activation()) return 1;
    printf("OK: activacion band (valores exactos, registro por nombre)\n");
    if (test_band_block()) return 1;
    printf("OK: bloque Dense->RMSNorm[band]->SwiGLU (forward == manual, checkpoint conserva band, gradiente llega a la Dense)\n");
    printf("OK: test_swiglu\n");
    return 0;
}
