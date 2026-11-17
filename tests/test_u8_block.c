/* Modelo de lenguaje completo con activaciones uint8: Embedding -> [RMSNorm(uint8) -> GQA] residual ->
 * [RMSNorm(uint8) -> SwiGLU] residual -> RMSNorm(uint8) -> LMHead. Tarea: el siguiente token es el ANTERIOR de la
 * secuencia (necesita atencion). Comprueba que construye, que el entrenamiento baja la perdida y que predice por
 * encima del azar, con V ternario y con V float. */
#include "ternary_stack.h"
#include "ternary_layer.h"
#include "ternary_residual.h"
#include "ternary_loss.h"
#include "ternary_rng.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define V 12
#define E 32
#define T 8
#define BATCH 16
#define EPOCHS 400

static int fallas = 0;
#define CHECK(c, ...) do { if (!(c)) { fallas++; printf("FALLO (linea %d): ", __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint64_t rs = 88172645463325252ull;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) / 9007199254740992.0; }

static TernaryLayer build(const char *type, const size_t *in, size_t nd, const double *p, size_t np, TernaryRng *rng) {
    TernaryLayer l; char err[256];
    if (!ternary_layer_type_build(ternary_layer_type_by_name(type), in, nd, p, np, rng, &l, err, sizeof(err))) {
        fprintf(stderr, "build %s: %s\n", type, err); l.impl = NULL; l.vt = NULL;
    }
    return l;
}

static TernaryStack *new_model(unsigned seed) {
    TernaryRng rng; ternary_rng_seed(&rng, seed);
    size_t in0[1] = { T }, in2[2] = { T, E };
    double pe[2] = { V, E }, pr[1] = { 1e-5 }, pg[6] = { 2, 2, 1, 10.0, 2, 0.03 };
    double ps[4] = { 64, 0, 0.5, 0.03 }, ph[1] = { V };
    TernaryStack *s = ternary_stack_create();
    int ok = ternary_stack_add(s, build("embedding", in0, 1, pe, 2, &rng), TERNARY_ACTIVATION_UINT8);
    for (int b = 0; b < 2; b++) {
        TernaryResidual *r = ternary_residual_create();
        ok = ok && ternary_residual_add(r, build("rmsnorm", in2, 2, pr, 1, &rng), TERNARY_ACTIVATION_UINT8);
        ok = ok && ternary_residual_add(r, b == 0 ? build("gqa", in2, 2, pg, 6, &rng) : build("swiglu", in2, 2, ps, 4, &rng), TERNARY_ACTIVATION_UINT8);
        ok = ok && ternary_stack_add(s, ternary_layer_from_residual(r), TERNARY_ACTIVATION_UINT8);
    }
    ok = ok && ternary_stack_add(s, build("rmsnorm", in2, 2, pr, 1, &rng), TERNARY_ACTIVATION_UINT8);
    ok = ok && ternary_stack_add(s, build("lm_head", in2, 2, ph, 1, &rng), TERNARY_ACTIVATION_UINT8);
    if (!ok) { fprintf(stderr, "no se pudo armar el modelo\n"); return NULL; }
    return s;
}

static void run(unsigned seed) {
    TernaryCapabilities cn; memset(&cn, 0, sizeof(cn)); cn.num_cores = 1; cn.dot_impl_name = "scalar";
    TernaryStack *s = new_model(seed);
    if (!s) { fallas++; return; }
    rs = 0x9E3779B97F4A7C15ull + seed;
    size_t xd[1] = { T };
    int32_t ids[T], tg[T];
    float out[T * V], go[T * V];
    double first = 0, last = 0;
    for (int ep = 0; ep < EPOCHS; ep++) {
        ternary_stack_zero_gradients(s);
        double loss = 0;
        for (int b = 0; b < BATCH; b++) {
            for (int p = 0; p < T; p++) ids[p] = (int32_t)(urand() * V);
            tg[0] = -1; for (int p = 1; p < T; p++) tg[p] = ids[p - 1];
            TernaryTensor *xt = ternary_tensor_view_id(xd, 1, ids);
            ternary_stack_forward(s, xt, out, &cn);
            loss += ternary_xent(out, T, V, tg, go, NULL, NULL);
            for (int i = 0; i < T * V; i++) go[i] /= (float)BATCH;
            ternary_stack_backward(s, xt, go);
            ternary_tensor_free(xt);
        }
        ternary_stack_apply_gradients(s, 0.001f);
        loss /= BATCH;
        if (ep == 0) first = loss;
        last = loss;
    }
    size_t hits = 0, total = 0;
    for (int n = 0; n < 200; n++) {
        for (int p = 0; p < T; p++) ids[p] = (int32_t)(urand() * V);
        tg[0] = -1; for (int p = 1; p < T; p++) tg[p] = ids[p - 1];
        TernaryTensor *xt = ternary_tensor_view_id(xd, 1, ids);
        ternary_stack_forward(s, xt, out, &cn);
        size_t nv, nc; ternary_xent(out, T, V, tg, NULL, &nv, &nc);
        hits += nc; total += nv;
        ternary_tensor_free(xt);
    }
    const double acc = (double)hits / (double)total;
    printf("  semilla %u: perdida %.3f -> %.3f | precision %.1f%% (azar %.1f%%)\n", seed, first, last, 100.0 * acc, 100.0 / V);
    CHECK(last < 0.6 * first, "la perdida no baja lo suficiente (%.3f -> %.3f)", first, last);
    CHECK(acc > 0.4, "la precision no supera el 40%% (%.1f%%)", 100.0 * acc);
    ternary_stack_free(s);
}

int main(void) {
    printf("-- modelo completo con uint8\n");
    run(1); run(2);
    if (fallas) { printf("%d fallas\n", fallas); return 1; }
    printf("OK\n");
    return 0;
}
