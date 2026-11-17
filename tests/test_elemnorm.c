/* Normalizacion por elemento del gradiente de la cabeza (TernaryElemNorm, estado de 4 bits logaritmicos + escala por fila):
 *  - el escalar coincide con una referencia independiente (codigo con log2f, mismas operaciones float),
 *  - AVX-512 da los mismos bits que el escalar,
 *  - el checkpoint guarda y restaura el estado: seguir entrenando tras cargar da bits identicos,
 *  - head_beta2 = 0 no reserva estado y el export de inferencia no lleva nada de esto. */
#define _DEFAULT_SOURCE
#include "ternary_stack.h"
#include "ternary_layer.h"
#include "ternary_embedding.h"
#include "ternary_rmsnorm.h"
#include "ternary_head.h"
#include "ternary_avx512.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "tmp_path.h"   /* hay plataformas sin /tmp: usa $TMPDIR */

static int fallas = 0;
#define CHECK(c, ...) do { if (!(c)) { fallas++; printf("FALLO (linea %d): ", __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint64_t rs = 88172645463325252ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 32); }
static float rndf(void) { return (float)((int)(rnd() % 20001) - 10000) / 10000.0f; }

static int code_at(const TernaryElemNorm *n, size_t r, size_t k) { const uint8_t b = n->code[r * ((n->cols + 1) / 2) + (k >> 1)]; return (k & 1) ? (b >> 4) : (b & 15); }

static void test_contra_referencia(void) {
    const size_t R = 23, W = 37;
    TernaryElemNorm *n = ternary_elemnorm_alloc(R, W);
    float *g = malloc(R * W * sizeof(float)), *g0 = malloc(R * W * sizeof(float));
    float *vq = calloc(R * W, sizeof(float));   /* v cuantizado de la referencia */
    long mism = 0, tot = 0; double maxrel = 0;
    for (int step = 1; step <= 8; step++) {
        for (size_t r = 0; r < R; r++) { const float sc = powf(10.0f, -(float)(r % 6)); for (size_t k = 0; k < W; k++) g[r * W + k] = sc * rndf() * ((rnd() % 5) ? 1.0f : 0.0f); }
        memcpy(g0, g, R * W * sizeof(float));
        ternary_elemnorm_apply_scalar(n, g, 0.99f);
        const float b2 = 0.99f, omb = 1.0f - b2, bc = (float)(1.0 - pow((double)b2, (double)step));
        /* eps con la formula del escalar (suma canonica de 16 carriles) */
        double d[16] = {0}; for (size_t j = 0; j < R * W; j++) d[j & 15] += (double)g0[j] * (double)g0[j];
        double v8[8], t4[4], t2[2]; for (int l = 0; l < 8; l++) v8[l] = d[l] + d[l + 8]; for (int l = 0; l < 4; l++) t4[l] = v8[l] + v8[l + 4]; for (int l = 0; l < 2; l++) t2[l] = t4[l] + t4[l + 2];
        const float eps = (float)(1e-3 * sqrt((t2[0] + t2[1]) / (double)(R * W)) + 1e-20);
        for (size_t r = 0; r < R; r++) {
            float vn[64], mx = 0;
            for (size_t k = 0; k < W; k++) { vn[k] = b2 * vq[r * W + k] + omb * (g0[r * W + k] * g0[r * W + k]); if (vn[k] > mx) mx = vn[k]; }
            for (size_t k = 0; k < W; k++) {
                int c = 15;
                if (mx > 0) { const float x = vn[k] / mx; if (x >= 0x1p-16f) { long q = lrintf(-log2f(x)); c = q < 0 ? 0 : (q > 15 ? 15 : (int)q); } }
                vq[r * W + k] = ldexpf(mx, -c);
                const float gh = g0[r * W + k] / (sqrtf(vq[r * W + k] / bc) + eps);
                tot++; if (code_at(n, r, k) != c) mism++;
                const double rel = fabs((double)gh - (double)g[r * W + k]) / (fabs((double)gh) + 1e-30);
                if (code_at(n, r, k) == c && rel > maxrel) maxrel = rel;
            }
            CHECK(n->rmax[r] == mx, "rmax de la fila %zu (paso %d)", r, step);
        }
    }
    CHECK(mism * 1000 <= tot, "codigos distintos de la referencia: %ld de %ld", mism, tot);
    CHECK(maxrel < 1e-5, "gradiente normalizado distinto de la referencia (rel %.2e)", maxrel);
    CHECK(n->steps == 8, "pasos");
    ternary_elemnorm_free(n); free(g); free(g0); free(vq);
    printf("  escalar contra referencia (%ld valores, %ld codigos distintos, rel max %.1e): OK\n", tot, mism, maxrel);
}

static void test_avx_igual_escalar(void) {
    if (!ternary_avx512_enabled()) { printf("  AVX-512 no disponible: se omite\n"); return; }
    static const size_t forms[][2] = { {1, 1}, {3, 5}, {9, 16}, {17, 33}, {40, 64}, {64, 100}, {5, 130} };
    long bad = 0;
    for (size_t f = 0; f < sizeof forms / sizeof forms[0]; f++) {
        const size_t R = forms[f][0], W = forms[f][1];
        TernaryElemNorm *a = ternary_elemnorm_alloc(R, W), *b = ternary_elemnorm_alloc(R, W);
        float *ga = malloc(R * W * sizeof(float)), *gb = malloc(R * W * sizeof(float));
        for (int step = 0; step < 12; step++) {
            for (size_t r = 0; r < R; r++) { const float sc = powf(10.0f, -(float)(r % 7)); for (size_t k = 0; k < W; k++) ga[r * W + k] = gb[r * W + k] = sc * rndf() * ((rnd() % 4) ? 1.0f : 0.0f); }
            if (step == 5) for (size_t k = 0; k < W && k < R * W; k++) ga[k] = gb[k] = 0.0f;                   /* una fila sin gradiente */
            if (step == 7 && R * W > 3) { ga[1] = gb[1] = 1e-30f; ga[2] = gb[2] = 3e-39f; }                    /* tiny y denormal */
            ternary_elemnorm_apply_scalar(a, ga, 0.99f);
            ternary_elemnorm_apply(b, gb, 0.99f);
            if (memcmp(ga, gb, R * W * sizeof(float))) bad++;
            if (memcmp(a->code, b->code, R * ((W + 1) / 2))) bad++;
            if (memcmp(a->rmax, b->rmax, R * sizeof(float))) bad++;
            if (a->steps != b->steps) bad++;
        }
        ternary_elemnorm_free(a); ternary_elemnorm_free(b); free(ga); free(gb);
    }
    CHECK(bad == 0, "AVX-512 distinto del escalar en %ld comprobaciones", bad);
    printf("  AVX-512 igual al escalar (7 formas x 12 pasos, incluye filas nulas, valores minimos y denormales): OK\n");
}

static TernaryStack *mk_stack(size_t V, size_t E, size_t T, size_t VOC, unsigned seed) {
    TernaryStack *s = ternary_stack_create();
    TernaryEmbedding *e = ternary_embedding_alloc(V, E, T);
    TernaryRMSNorm *n = ternary_rmsnorm_alloc(T, E, 1e-5f);
    TernaryHead *h = ternary_head_alloc(T, E, VOC);
    srand(seed);
    for (size_t v = 0; v < V; v++) for (size_t d = 0; d < E; d++) ternary_embedding_set_weight(e, v, d, (rand() % 3) - 1);
    for (size_t o = 0; o < VOC; o++) for (size_t i = 0; i < E; i++) ternary_dense_set_weight(h->w, o, i, (rand() % 3) - 1);
    ternary_stack_add(s, ternary_layer_from_embedding(e), TERNARY_ACTIVATION_UINT8);
    ternary_stack_add(s, ternary_layer_from_rmsnorm(n), TERNARY_ACTIVATION_UINT8);
    ternary_stack_add(s, ternary_layer_from_head(h), TERNARY_ACTIVATION_UINT8);
    return s;
}
static TernaryHead *head_of(TernaryStack *s) { return (TernaryHead *)ternary_stack_layer(s, ternary_stack_num_layers(s) - 1).impl; }
static void paso(TernaryStack *s, const int32_t *ids, const int32_t *tg, size_t T) {
    TernaryCapabilities cn; memset(&cn, 0, sizeof cn); cn.num_cores = 1; cn.dot_impl_name = "scalar";
    size_t dims[1] = { T }; int32_t buf[64]; memcpy(buf, ids, T * sizeof(int32_t));
    TernaryTensor *x = ternary_tensor_view_id(dims, 1, buf);
    size_t nv, nc;
    ternary_stack_zero_gradients(s);
    ternary_stack_train_xent(s, x, &cn, tg, 1.0f, &nv, &nc);
    ternary_stack_apply_gradients(s, 0.02f);
    ternary_tensor_free(x);
}
static int head_igual(TernaryHead *a, TernaryHead *b) {
    const size_t n = a->vocab * a->dim;
    if (memcmp(a->w->counter_W, b->w->counter_W, n)) return 0;
    for (size_t o = 0; o < a->vocab; o++) for (size_t i = 0; i < a->dim; i++) if (ternary_dense_get_weight(a->w, o, i) != ternary_dense_get_weight(b->w, o, i)) return 0;
    return a->w->scale == b->w->scale;
}

static void test_checkpoint(void) {
    const size_t V = 40, E = 16, T = 8, VOC = 40;
    int32_t ids[64], tg[64];
    TernaryStack *s = mk_stack(V, E, T, VOC, 3);
    CHECK(ternary_stack_get_update_rule(s).head_beta2 == 0.99f, "head_beta2 por defecto 0.99");
    for (int t = 0; t < 7; t++) {
        for (size_t i = 0; i < T; i++) { ids[i] = rnd() % 8; tg[i] = (rnd() % 4 == 0) ? (int32_t)(rnd() % VOC) : (int32_t)(rnd() % 3); }
        paso(s, ids, tg, T);
    }
    TernaryHead *h = head_of(s);
    CHECK(h->vn && h->vn->steps == 7, "estado reservado tras 7 pasos (steps=%u)", h->vn ? h->vn->steps : 0);
    char dir[1100]; snprintf(dir, sizeof dir, "%s/ternary_elemnorm_XXXXXX", tmp_dir());
    CHECK(mkdtemp(dir) != NULL, "mkdtemp");
    CHECK(ternary_stack_save(s, dir, 1), "guardar checkpoint");
    char err[300] = ""; TernaryStack *r = ternary_stack_load_ex(dir, err, sizeof err);
    CHECK(r != NULL, "cargar checkpoint: %s", err);
    if (r) {
        TernaryHead *hr = head_of(r);
        CHECK(hr->vn && hr->vn->steps == h->vn->steps && !memcmp(hr->vn->rmax, h->vn->rmax, VOC * sizeof(float)) &&
              !memcmp(hr->vn->code, h->vn->code, VOC * ((E + 1) / 2)), "el checkpoint restaura el estado de la normalizacion");
        CHECK(ternary_stack_get_update_rule(r).head_beta2 == 0.99f, "head_beta2 restaurado");
        for (int t = 0; t < 5; t++) {
            for (size_t i = 0; i < T; i++) { ids[i] = rnd() % 8; tg[i] = (rnd() % 4 == 0) ? (int32_t)(rnd() % VOC) : (int32_t)(rnd() % 3); }
            paso(s, ids, tg, T); paso(r, ids, tg, T);
        }
        CHECK(head_igual(h, hr) && h->vn->steps == hr->vn->steps && !memcmp(hr->vn->code, h->vn->code, VOC * ((E + 1) / 2)), "seguir entrenando tras cargar da los mismos bits");
        ternary_stack_free(r);
    }
    /* instantanea en memoria (restore_best) */
    TernaryStackSnapshot snap = { NULL, 0 };
    CHECK(ternary_stack_snapshot(s, &snap), "instantanea");
    const uint32_t st0 = head_of(s)->vn->steps;
    for (int t = 0; t < 3; t++) { for (size_t i = 0; i < T; i++) { ids[i] = rnd() % 8; tg[i] = (int32_t)(rnd() % 3); } paso(s, ids, tg, T); }
    CHECK(head_of(s)->vn->steps == st0 + 3, "pasos tras 3 mas");
    CHECK(ternary_stack_restore(s, &snap), "restaurar instantanea");
    CHECK(head_of(s)->vn && head_of(s)->vn->steps == st0, "la instantanea devuelve el estado de la normalizacion (steps %u, esperado %u)", head_of(s)->vn ? head_of(s)->vn->steps : 0, st0);
    ternary_stack_snapshot_free(&snap);
    /* export de inferencia: sin estado */
    char dir2[1100]; snprintf(dir2, sizeof dir2, "%s/ternary_elemnorm_XXXXXX", tmp_dir());
    CHECK(mkdtemp(dir2) != NULL, "mkdtemp2");
    CHECK(ternary_stack_save(s, dir2, 0), "exportar");
    TernaryStack *e = ternary_stack_load_ex(dir2, err, sizeof err);
    CHECK(e != NULL && head_of(e)->vn == NULL, "el export de inferencia no lleva el estado");
    ternary_stack_free(e);
    ternary_stack_free(s);
    char cmd[2400]; snprintf(cmd, sizeof cmd, "rm -rf %s %s", dir, dir2); if (system(cmd)) {}
    printf("  checkpoint (estado, continuar bit a bit, instantanea, export de inferencia): OK\n");
}

static void test_apagada(void) {
    const size_t V = 40, E = 16, T = 8, VOC = 40; int32_t ids[64], tg[64];
    TernaryStack *s = mk_stack(V, E, T, VOC, 3);
    ternary_stack_set_head_beta2(s, 0.0f);
    for (int t = 0; t < 4; t++) { for (size_t i = 0; i < T; i++) { ids[i] = rnd() % 8; tg[i] = (int32_t)(rnd() % 3); } paso(s, ids, tg, T); }
    CHECK(head_of(s)->vn == NULL, "con head_beta2 = 0 no se reserva estado");
    ternary_stack_free(s);
    printf("  head_beta2 = 0: sin estado, R2 por fila: OK\n");
}

int main(void) {
    printf("-- normalizacion por elemento de la cabeza\n");
    test_contra_referencia();
    test_avx_igual_escalar();
    test_checkpoint();
    test_apagada();
    if (fallas) { printf("%d fallas\n", fallas); return 1; }
    printf("OK\n");
    return 0;
}
