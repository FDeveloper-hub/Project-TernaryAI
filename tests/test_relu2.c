/*
 * ReLU² SIN compuerta (gate_mode = 2 de la capa "swiglu", estilo BitNet): h_j = relu(escala_up * z_up_j)^2, solo DOS
 * matrices (W_up y W_down; sin W_gate). Verifica:
 *  1. estructura: wg == NULL, num_matrices = 2 (3 en las otras compuertas), estado mas chico que con compuerta,
 *     gate_mode 3 rechazado, y que el default de la capa sigue siendo silu (gate_mode 0);
 *  2. ruta TERNARIA: forward contra una referencia aparte, y backward contra una referencia en double
 *     (grad_w_out[0] = W_up, [1] = W_down; escala de up en 0, escala de down con gradiente);
 *  3. ruta UINT8: lo mismo con la entrada cuantizada a uint8 (producto entero exacto + estimador directo);
 *  4. capa dentro del TernaryStack: forward del stack == forward directo, entrenamiento sin NaN que mueve pesos
 *     (el stack maneja 2 matrices), checkpoint completo y export empaquetado recuperan la capa exactamente.
 */
#include "ternary_stack.h"
#include "ternary_swiglu.h"
#include "ternary_layer.h"
#include "ternary_activation.h"
#include "ternary_rng.h"
#include "ternary_u8.h"
#include "tmp_path.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fallas = 0;
#define CHECK(c, ...) do { if (!(c)) { fallas++; printf("FALLO (linea %d): ", __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint32_t rs = 4242u;
static uint32_t rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }
static float rndf(void) { return ((float)(rnd() % 20001) / 10000.0f) - 1.0f; }

static TernarySwiGLU *build(size_t rows, size_t dim, size_t hidden, int mode, unsigned seed, TernaryLayer *l_out) {
    TernaryRng rng; ternary_rng_seed(&rng, seed);
    size_t in[2] = { rows, dim };
    double p[4] = { (double)hidden, (double)mode, 0.5, 1.0 };
    TernaryLayer l; char err[200];
    if (!ternary_layer_type_build(ternary_layer_type_by_name("swiglu"), in, 2, p, 4, &rng, &l, err, sizeof(err))) {
        printf("build: %s\n", err); return NULL;
    }
    if (l_out) *l_out = l;
    return (TernarySwiGLU *)l.impl;
}

static int8_t *weights(const TernaryDense *d) {
    int8_t *w = malloc(d->output_dim * d->input_dim);
    for (size_t o = 0; o < d->output_dim; o++) for (size_t i = 0; i < d->input_dim; i++) w[o * d->input_dim + i] = (int8_t)ternary_dense_get_weight(d, o, i);
    return w;
}

/* ---------- 1. estructura ---------- */
static void test_structure(void) {
    TernarySwiGLU *a = build(4, 32, 64, 2, 1, NULL), *b = build(4, 32, 64, 0, 1, NULL), *c = build(4, 32, 64, 1, 1, NULL);
    CHECK(a && b && c, "build");
    if (!a || !b || !c) return;
    CHECK(a->wg == NULL && b->wg != NULL && c->wg != NULL, "wg solo existe con compuerta");
    CHECK(ternary_swiglu_num_matrices(a) == 2 && ternary_swiglu_num_matrices(b) == 3 && ternary_swiglu_num_matrices(c) == 3, "num_matrices");
    CHECK(ternary_swiglu_matrix(a, 0) == a->wu && ternary_swiglu_matrix(a, 1) == a->wd && ternary_swiglu_matrix(a, 2) == NULL, "indices de matriz sin compuerta");
    CHECK(ternary_swiglu_matrix(b, 0) == b->wg && ternary_swiglu_matrix(b, 2) == b->wd, "indices de matriz con compuerta");
    CHECK(ternary_swiglu_state_bytes(a) < ternary_swiglu_state_bytes(b), "el estado sin compuerta debe ser menor (%zu vs %zu)",
          ternary_swiglu_state_bytes(a), ternary_swiglu_state_bytes(b));
    CHECK(ternary_swiglu_alloc(2, 8, 8, 3, 0.5f) == NULL, "gate_mode 3 debe rechazarse");
    /* el default del registro sigue siendo silu */
    const TernaryLayerType *ty = ternary_layer_type_by_name("swiglu");
    TernaryRng rng; ternary_rng_seed(&rng, 1); size_t in[2] = { 2, 16 }; TernaryLayer l; char err[200];
    double bad[4] = { 8, 3, 0.5, 1.0 };
    CHECK(!ternary_layer_type_build(ty, in, 2, bad, 4, &rng, &l, err, sizeof(err)) && strstr(err, "gate_mode"), "registro: gate_mode 3 invalido");
    double ok[4] = { 8, 2, 0.5, 1.0 }, p_out[8]; size_t shp[4];
    CHECK(ternary_layer_type_build(ty, in, 2, ok, 4, &rng, &l, err, sizeof(err)), "registro: gate_mode 2 valido (%s)", err);
    if (ty->describe) { ty->describe(l.impl, p_out, shp); CHECK((int)p_out[1] == 2 && p_out[2] == 0.5, "describe de vuelta"); }
    l.vt->destroy(l.impl);
    printf("  estructura: wg NULL, 2 matrices, estado %zu B (con compuerta %zu B)\n", ternary_swiglu_state_bytes(a), ternary_swiglu_state_bytes(b));
    ternary_swiglu_free(a); ternary_swiglu_free(b); ternary_swiglu_free(c);
}

/* ---------- 2. ruta ternaria ---------- */
static void test_ternary(size_t rows, size_t dim, size_t hidden, unsigned seed) {
    TernarySwiGLU *s = build(rows, dim, hidden, 2, seed, NULL);
    CHECK(s != NULL, "build");
    if (!s) return;
    rs = seed * 77u;
    s->wu->scale = 0.9f; s->wd->scale = 0.5f;
    TernaryVec *x = ternary_vec_alloc(rows * dim);
    for (size_t i = 0; i < rows * dim; i++) ternary_vec_set(x, i, (int)(rnd() % 3) - 1);
    float *out = malloc(rows * dim * sizeof(float));
    int32_t *zd_raw = malloc(rows * dim * sizeof(int32_t));
    ternary_swiglu_forward(s, x, out, zd_raw);
    int8_t *Wu = weights(s->wu), *Wd = weights(s->wd);
    const float su = s->wu->scale;
    double maxerr = 0, maxo = 1e-9;
    int8_t *T = malloc(rows * hidden);
    for (size_t p = 0; p < rows; p++) {
        float *h = malloc(hidden * sizeof(float)); int32_t *zu = malloc(hidden * sizeof(int32_t));
        for (size_t j = 0; j < hidden; j++) {
            int32_t a = 0;
            for (size_t i = 0; i < dim; i++) a += Wu[j * dim + i] * ternary_vec_get(x, p * dim + i);
            zu[j] = a; const float av = su * (float)a; h[j] = av > 0.0f ? av * av : 0.0f;
        }
        double sum = 0; for (size_t j = 0; j < hidden; j++) sum += fabs(h[j]);
        const double thr = (double)s->band * sum / (double)hidden;
        for (size_t j = 0; j < hidden; j++) {
            const int t = ternary_vec_get(s->tpos[p], j);
            T[p * hidden + j] = (int8_t)t;
            if (fabs(fabs((double)h[j]) - thr) > 1e-3 * (thr + 1e-12)) {   /* lejos del umbral: debe coincidir */
                const int tr = fabs(h[j]) > thr ? (h[j] > 0 ? 1 : -1) : 0;
                CHECK(t == tr, "t fila %zu j=%zu: %d vs %d (h=%.6g thr=%.6g)", p, j, t, tr, (double)h[j], thr);
            }
            CHECK(h[j] >= 0.0f, "h de relu^2 no puede ser negativa");
        }
        for (size_t d = 0; d < dim; d++) {
            int32_t z = 0; for (size_t j = 0; j < hidden; j++) z += Wd[d * hidden + j] * T[p * hidden + j];
            const double r = (double)s->wd->scale * (double)z;
            maxerr = fmax(maxerr, fabs(r - out[p * dim + d])); maxo = fmax(maxo, fabs(r));
            CHECK(z == zd_raw[p * dim + d], "Z crudo de down fila %zu d=%zu: %d vs %d", p, d, z, zd_raw[p * dim + d]);
        }
        free(h); free(zu);
    }
    CHECK(maxerr <= 1e-5 * maxo + 1e-9, "forward ternario: error %.3g (max %.3g)", maxerr, maxo);

    /* backward */
    float *go = malloc(rows * dim * sizeof(float));
    for (size_t i = 0; i < rows * dim; i++) go[i] = rndf();
    float *gwu = malloc(hidden * dim * 4), *gwd = malloc(dim * hidden * 4);
    float *gwp[2] = { gwu, gwd };
    float *gx = calloc(rows * dim, sizeof(float)); float gsc[2] = { 0, 0 };
    ternary_swiglu_backward(s, x, go, gwp, gx, gsc);
    double *rwu = calloc(hidden * dim, 8), *rwd = calloc(dim * hidden, 8), *rgx = calloc(rows * dim, 8);
    double rs_d = 0;
    for (size_t p = 0; p < rows; p++) {
        double *gt = calloc(hidden, 8);
        for (size_t d = 0; d < dim; d++) {
            const double g = go[p * dim + d];
            rs_d += g * zd_raw[p * dim + d];
            for (size_t j = 0; j < hidden; j++) { rwd[d * hidden + j] += g * T[p * hidden + j]; gt[j] += g * s->wd->scale * Wd[d * hidden + j]; }
        }
        for (size_t j = 0; j < hidden; j++) {
            int32_t zu = 0; for (size_t i = 0; i < dim; i++) zu += Wu[j * dim + i] * ternary_vec_get(x, p * dim + i);
            const double u = (double)su * zu;
            const double gu = u > 0 ? gt[j] * 2.0 * u : 0.0;
            for (size_t i = 0; i < dim; i++) {
                rwu[j * dim + i] += gu * ternary_vec_get(x, p * dim + i);
                rgx[p * dim + i] += gu * su * Wu[j * dim + i];
            }
        }
        free(gt);
    }
    double mg = 1e-9, eu = 0, ed = 0, ex = 0, mx = 1e-9;
    for (size_t i = 0; i < hidden * dim; i++) { mg = fmax(mg, fmax(fabs(rwu[i]), fabs(rwd[i]))); eu = fmax(eu, fabs(rwu[i] - gwu[i])); ed = fmax(ed, fabs(rwd[i] - gwd[i])); }
    for (size_t i = 0; i < rows * dim; i++) { ex = fmax(ex, fabs(rgx[i] - gx[i])); mx = fmax(mx, fabs(rgx[i])); }
    CHECK(mg > 1e-3, "los gradientes de referencia no deben ser ~0 (%.3g)", mg);
    CHECK(eu <= 1e-4 * mg && ed <= 1e-4 * mg, "grad_w: up %.3g down %.3g (max %.3g)", eu, ed, mg);
    CHECK(ex <= 1e-4 * mx, "grad_x: %.3g (max %.3g)", ex, mx);
    CHECK(fabs(rs_d - gsc[1]) <= 1e-4 * (fabs(rs_d) + 1e-6), "escala down: %.6g vs %.6g", rs_d, gsc[1]);
    CHECK(gsc[0] == 0.0f, "la escala de up no recibe gradiente");
    printf("  ternaria rows=%zu dim=%zu hidden=%zu: fwd err %.2g, grad up/down %.2g/%.2g x %.2g\n", rows, dim, hidden, maxerr / maxo, eu / mg, ed / mg, ex / mx);
    free(Wu); free(Wd); free(T); free(out); free(zd_raw); free(go); free(gwu); free(gwd); free(gx); free(rwu); free(rwd); free(rgx);
    ternary_vec_free(x); ternary_swiglu_free(s);
}

/* ---------- 3. ruta uint8 ---------- */
static void test_u8(size_t rows, size_t dim, size_t hidden, unsigned seed) {
    TernarySwiGLU *s = build(rows, dim, hidden, 2, seed, NULL);
    CHECK(s != NULL, "build");
    if (!s) return;
    rs = seed;
    s->wu->scale = 0.05f; s->wd->scale = 0.07f;
    float *xf = malloc(rows * dim * sizeof(float));
    for (size_t i = 0; i < rows * dim; i++) xf[i] = rndf() * 3.0f;
    TernaryU8 *x = ternary_u8_alloc(rows, dim);
    ternary_u8_quantize(x, xf);
    float *out = malloc(rows * dim * sizeof(float));
    ternary_swiglu_forward_u8(s, x, out, NULL);
    int8_t *Wu = weights(s->wu), *Wd = weights(s->wd);
    double maxo = 1e-9, maxe = 0;
    for (size_t p = 0; p < rows; p++) {
        const uint8_t *q = x->q + p * dim; const float sx = x->scale[p]; const int zp = x->zp[p];
        double *h = malloc(hidden * sizeof(double));
        for (size_t j = 0; j < hidden; j++) {
            int32_t zu = ternary_dot_u8_reference(Wu + j * dim, dim, q, zp);
            double a = (double)s->wu->scale * sx * zu;
            h[j] = a > 0 ? a * a : 0.0;
        }
        double hmax = 1e-12; for (size_t j = 0; j < hidden; j++) hmax = fmax(hmax, fabs(h[j]));
        for (size_t j = 0; j < hidden; j++) {
            double dq = (double)s->hq.scale[p] * ((int)s->hq.q[p * hidden + j] - s->hq.zp[p]);
            CHECK(fabs(dq - h[j]) <= 0.5001 * s->hq.scale[p] + 1e-5 * hmax, "h fila %zu j=%zu: %.6g vs %.6g", p, j, dq, h[j]);
            CHECK(s->hq.zp[p] == 0, "con h >= 0 el zero-point debe ser 0 (fila %zu: %d)", p, s->hq.zp[p]);
        }
        for (size_t d = 0; d < dim; d++) {
            int32_t z = ternary_dot_u8_reference(Wd + d * hidden, hidden, s->hq.q + p * hidden, s->hq.zp[p]);
            double r = (double)s->wd->scale * s->hq.scale[p] * z;
            maxo = fmax(maxo, fabs(r)); maxe = fmax(maxe, fabs(r - out[p * dim + d]));
        }
        free(h);
    }
    CHECK(maxe <= 1e-5 * maxo + 1e-9, "forward u8: error %.3g vs max %.3g", maxe, maxo);

    float *go = malloc(rows * dim * sizeof(float));
    for (size_t i = 0; i < rows * dim; i++) go[i] = rndf();
    float *gwu = malloc(hidden * dim * 4), *gwd = malloc(dim * hidden * 4);
    float *gwp[2] = { gwu, gwd };
    float *gx = calloc(rows * dim, sizeof(float)); float gsc[2] = { 0, 0 };
    ternary_swiglu_backward_u8(s, x, go, gwp, gx, gsc);
    double *rwu = calloc(hidden * dim, 8), *rwd = calloc(dim * hidden, 8), *rgx = calloc(rows * dim, 8);
    double rd = 0;
    for (size_t p = 0; p < rows; p++) {
        const uint8_t *q = x->q + p * dim; const double sx = x->scale[p]; const int zp = x->zp[p];
        double *gt = calloc(hidden, 8);
        for (size_t d = 0; d < dim; d++) {
            int32_t z = ternary_dot_u8_reference(Wd + d * hidden, hidden, s->hq.q + p * hidden, s->hq.zp[p]);
            double g = go[p * dim + d];
            rd += g * s->hq.scale[p] * z;
            for (size_t j = 0; j < hidden; j++) {
                double hd = (double)s->hq.scale[p] * ((int)s->hq.q[p * hidden + j] - s->hq.zp[p]);
                rwd[d * hidden + j] += g * hd;
                gt[j] += g * s->wd->scale * Wd[d * hidden + j];
            }
        }
        for (size_t j = 0; j < hidden; j++) {
            int32_t zu = ternary_dot_u8_reference(Wu + j * dim, dim, q, zp);
            double u = (double)s->wu->scale * sx * zu;
            double gu = u > 0 ? gt[j] * 2.0 * u : 0.0;
            for (size_t i = 0; i < dim; i++) {
                double xd = sx * ((int)q[i] - zp);
                rwu[j * dim + i] += gu * xd;
                rgx[p * dim + i] += gu * s->wu->scale * Wu[j * dim + i];
            }
        }
        free(gt);
    }
    double mg = 1e-9, eu = 0, ed = 0, ex = 0, mx = 1e-9;
    for (size_t i = 0; i < hidden * dim; i++) { mg = fmax(mg, fmax(fabs(rwu[i]), fabs(rwd[i]))); eu = fmax(eu, fabs(rwu[i] - gwu[i])); ed = fmax(ed, fabs(rwd[i] - gwd[i])); }
    for (size_t i = 0; i < rows * dim; i++) { ex = fmax(ex, fabs(rgx[i] - gx[i])); mx = fmax(mx, fabs(rgx[i])); }
    CHECK(eu <= 1e-4 * mg && ed <= 1e-4 * mg, "grad_w u8: up %.3g down %.3g (max %.3g)", eu, ed, mg);
    CHECK(ex <= 1e-4 * mx, "grad_x u8: %.3g (max %.3g)", ex, mx);
    CHECK(fabs(rd - gsc[1]) <= 1e-4 * (fabs(rd) + 1e-6), "escala down u8: %.6g vs %.6g", rd, gsc[1]);
    CHECK(gsc[0] == 0.0f, "escala de up u8 deberia quedar en 0");
    printf("  uint8 rows=%zu dim=%zu hidden=%zu: fwd err %.2g, grad up/down %.2g/%.2g x %.2g\n", rows, dim, hidden, maxe / maxo, eu / mg, ed / mg, ex / mx);
    free(Wu); free(Wd); free(xf); free(out); free(go); free(gwu); free(gwd); free(gx); free(rwu); free(rwd); free(rgx);
    ternary_u8_free(x); ternary_swiglu_free(s);
}

/* ---------- 4. dentro del stack ---------- */
#define ROWS 3
#define DIM 24
#define HID 40
#define NIN (ROWS * DIM)

static TernaryStack *new_stack(unsigned seed) {
    TernaryLayer l;
    TernarySwiGLU *s = build(ROWS, DIM, HID, 2, seed, &l);
    if (!s) return NULL;
    TernaryStack *st = ternary_stack_create();
    if (!ternary_stack_add(st, l, TERNARY_ACTIVATION_UINT8)) return NULL;
    return st;
}
static int same_dense(const TernaryDense *a, const TernaryDense *b) {
    if (a->input_dim != b->input_dim || a->output_dim != b->output_dim || a->scale != b->scale) return 0;
    for (size_t o = 0; o < a->output_dim; o++) for (size_t i = 0; i < a->input_dim; i++)
        if (ternary_dense_get_weight(a, o, i) != ternary_dense_get_weight(b, o, i)) return 0;
    return 1;
}

static void test_stack(void) {
    TernaryStack *st = new_stack(5);
    CHECK(st != NULL, "stack");
    if (!st) return;
    TernaryCapabilities cn; memset(&cn, 0, sizeof(cn)); cn.num_cores = 1; cn.dot_impl_name = "scalar";
    TernaryVec *x = ternary_vec_alloc(NIN); size_t xd[1] = { NIN };
    TernaryTensor *xt = ternary_tensor_view_ternary(xd, 1, x);
    for (size_t i = 0; i < NIN; i++) ternary_vec_set(x, i, (int)(rnd() % 3) - 1);
    float a[NIN], b[NIN], go[NIN];
    ternary_stack_forward(st, xt, a, &cn);
    TernarySwiGLU *sw = (TernarySwiGLU *)ternary_stack_layer(st, 0).impl;
    ternary_swiglu_forward(sw, x, b, NULL);
    CHECK(memcmp(a, b, sizeof(a)) == 0, "forward del stack distinto del directo");

    /* entrenamiento: el stack maneja 2 matrices (grad_w, escalas, votos); sin NaN y con cambios de pesos */
    int8_t *wu0 = weights(sw->wu), *wd0 = weights(sw->wd);
    for (int ep = 0; ep < 30; ep++) {
        ternary_stack_zero_gradients(st);
        ternary_stack_forward(st, xt, a, &cn);
        for (size_t o = 0; o < NIN; o++) go[o] = (a[o] - 0.5f * rndf()) / (float)NIN;
        ternary_stack_backward(st, xt, go);
        ternary_stack_apply_gradients(st, 0.05f);
    }
    int nan = 0; for (size_t o = 0; o < NIN; o++) if (!isfinite(a[o])) nan = 1;
    CHECK(!nan, "salida con NaN tras entrenar");
    int8_t *wu1 = weights(sw->wu), *wd1 = weights(sw->wd);
    size_t chg = 0;
    for (size_t i = 0; i < sw->wu->output_dim * sw->wu->input_dim; i++) chg += wu0[i] != wu1[i];
    for (size_t i = 0; i < sw->wd->output_dim * sw->wd->input_dim; i++) chg += wd0[i] != wd1[i];
    CHECK(chg > 0, "ningun peso cambio tras 30 epocas de entrenamiento");
    printf("  stack: %zu pesos cambiaron en 30 epocas, sin NaN\n", chg);
    free(wu0); free(wd0); free(wu1); free(wd1);

    /* checkpoint completo y export empaquetado */
    const char *path = tmp_path("test_relu2_ck.bin");
    CHECK(ternary_stack_save(st, path, 1), "guardar checkpoint");
    TernaryStack *c = ternary_stack_load(path);
    CHECK(c != NULL, "cargar checkpoint");
    if (c) {
        TernarySwiGLU *sc = (TernarySwiGLU *)ternary_stack_layer(c, 0).impl;
        CHECK(sc->gate_mode == 2 && sc->wg == NULL, "el checkpoint debe recordar gate_mode 2 y no tener W_gate");
        CHECK(same_dense(sw->wu, sc->wu) && same_dense(sw->wd, sc->wd), "el checkpoint cargado difiere del guardado");
        ternary_stack_free(c);
    }
    path = tmp_path("test_relu2_ex.bin");
    CHECK(ternary_stack_save(st, path, 0), "guardar export");
    TernaryStack *e = ternary_stack_load(path);
    CHECK(e != NULL, "cargar export");
    if (e) {
        for (int k = 0; k < 5; k++) {
            for (size_t i = 0; i < NIN; i++) ternary_vec_set(x, i, (int)(rnd() % 3) - 1);
            float o1[NIN], o2[NIN];
            ternary_stack_forward(st, xt, o1, &cn); ternary_stack_forward(e, xt, o2, &cn);
            CHECK(memcmp(o1, o2, sizeof(o1)) == 0, "export: inferencia distinta");
        }
        ternary_stack_free(e);
    }
    model_dir_remove(tmp_path("test_relu2_ck.bin")); model_dir_remove(tmp_path("test_relu2_ex.bin"));
    ternary_tensor_free(xt); ternary_vec_free(x); ternary_stack_free(st);
}

int main(void) {
    printf("-- ReLU² sin compuerta (swiglu, gate_mode=2)\n");
    test_structure();
    test_ternary(1, 16, 24, 1); test_ternary(5, 40, 70, 2); test_ternary(7, 64, 130, 3);
    test_u8(1, 16, 24, 1); test_u8(5, 40, 70, 2); test_u8(7, 64, 130, 3);
    test_stack();
    if (fallas) { printf("%d fallas\n", fallas); return 1; }
    printf("OK\n");
    return 0;
}
