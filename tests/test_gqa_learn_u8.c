/*
 * [Variante uint8 de test_gqa_learn.c: la entrada del bloque llega como filas uint8 (escala + zero-point por posicion)
 * en vez de un TernaryVec; Q/K salen de Z entero >> TERNARY_GQA_U8_SHIFT, la salida de atencion se cuantiza a uint8 antes
 * de Wo. Misma tarea, mismos umbrales.]
 * GQA APRENDE a traves del camino real de la libreria (capa "gqa" del TernaryStack, inicializacion
 * de la libreria, RoPE entero, piso de votos reducido para Wq/Wk, recorte por defecto).
 * Tarea relativa: la salida en la posicion de consulta q debe reproducir (signo por bit) el contenido
 * del token a distancia D=2. Se entrena con q en [2, SEQ/2-1] y se mide con q en [SEQ/2, SEQ-1],
 * posiciones absolutas que el entrenamiento NUNCA vio: solo una atencion relativa (RoPE) lo resuelve.
 * Las primeras POSD dims son un canal constante (+1) para que Q/K tengan componente independiente del
 * contenido; Wv se mantiene en 0 sobre ese canal (sin eso el desplazamiento constante tapa el contenido).
 * Comprobado: con los valores antiguos (-DTERNARY_GQA_QK_LIM_MULT=16 -DTERNARY_GQA_QK_FLOOR_FACTOR=1.0f) este test FALLA en 2 de 3
 * semillas (59% por colapso del recorte; 46% con Wq casi sin cambios); con los valores actuales pasan las 3 (~93%).
 */
#include "ternary_stack.h"
#include "ternary_layer.h"
#include "ternary_gqa.h"
#include "ternary_rng.h"
#include "ternary_u8.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define E 32
#define SEQ 16
#define POSD 16
#define D 2
#define BATCH 16
#define EPOCHS 800
/* n_heads, n_kv, rope(entero), base, shift */
static const double PARAMS[5]  = { 1, 1, 1, 10.0, 1 };   /* 1 cabeza, head_dim 32 */
static const double PARAMS2[5] = { 2, 2, 1, 10.0, 1 };   /* 2 cabezas, K/V sin compartir (kvw=32), head_dim 16 */
static const float SCALE_LR = 0.001f;

static void zero_v_channel(TernaryGQA *g) {
    for (size_t o = 0; o < g->wv->output_dim; o++) for (size_t i = 0; i < POSD; i++) ternary_dense_set_weight(g->wv, o, i, 0);
}
static void make_sample(TernaryU8 *x, int c[SEQ][E]) {
    float f[SEQ * E];
    for (size_t p = 0; p < SEQ; p++) for (size_t i = 0; i < E; i++) {
        int v = (i < POSD) ? 1 : (rand() % 2) * 2 - 1;
        c[p][i] = v; f[p * E + i] = (float)v;
    }
    ternary_u8_quantize(x, f);
}

static int run_seed(const double *PARAMS, unsigned seed, double *train_acc, double *test_acc, long *qk_changes) {
    const TernaryLayerType *t = ternary_layer_type_by_name("gqa");
    TernaryRng rng; ternary_rng_seed(&rng, seed);
    size_t shape[2] = { SEQ, E }; TernaryLayer l; char err[256] = "";
    if (!t || !ternary_layer_type_build(t, shape, 2, PARAMS, 5, &rng, &l, err, sizeof(err))) { fprintf(stderr, "build: %s\n", err); return 0; }
    TernaryStack *s = ternary_stack_create();
    if (!ternary_stack_add(s, l, TERNARY_ACTIVATION_UINT8)) return 0;
    TernaryGQA *g = (TernaryGQA *)ternary_stack_layer(s, 0).impl;
    zero_v_channel(g);
    int *wq0 = malloc(sizeof(int) * g->wq->output_dim * E);
    for (size_t o = 0; o < g->wq->output_dim; o++) for (size_t i = 0; i < E; i++) wq0[o * E + i] = ternary_dense_get_weight(g->wq, o, i);

    TernaryCapabilities cn; memset(&cn, 0, sizeof(cn)); cn.num_cores = 1; cn.dot_impl_name = "scalar";
    TernaryU8 *x = ternary_u8_alloc(SEQ, E); size_t xd[2] = { SEQ, E };
    TernaryTensor *xt = ternary_tensor_view_u8(xd, 2, x);
    float out[SEQ * E], go[SEQ * E]; int c[SEQ][E];
    srand(1000 + seed);
    double tr = 0, te = 0; int trn = 0, ten = 0;
    for (int ep = 0; ep < EPOCHS; ep++) {
        ternary_stack_zero_gradients(s);
        long ok = 0, tot = 0;
        for (int b = 0; b < BATCH; b++) {
            make_sample(x, c);
            int q = D + rand() % (SEQ / 2 - D);
            ternary_stack_forward(s, xt, out, &cn);
            memset(go, 0, sizeof go);
            for (size_t d = POSD; d < E; d++) {
                float o = out[q * E + d];
                go[q * E + d] = (2.0f / E) * (o - (float)c[q - D][d]);
                int sg = o > 0 ? 1 : (o < 0 ? -1 : 0);
                ok += (sg == c[q - D][d]); tot++;
            }
            ternary_stack_backward(s, xt, go);
        }
        ternary_stack_apply_gradients(s, SCALE_LR);
        zero_v_channel(g);
        if (ep >= EPOCHS - 50) {
            tr += 100.0 * ok / tot; trn++;
            long hok = 0, htot = 0;
            for (int b = 0; b < BATCH; b++) {
                make_sample(x, c); ternary_stack_forward(s, xt, out, &cn);
                int q = SEQ / 2 + rand() % (SEQ - SEQ / 2);
                for (size_t d = POSD; d < E; d++) { float o = out[q * E + d]; int sg = o > 0 ? 1 : (o < 0 ? -1 : 0); hok += (sg == c[q - D][d]); htot++; }
            }
            te += 100.0 * hok / htot; ten++;
        }
    }
    long ch = 0;
    for (size_t o = 0; o < g->wq->output_dim; o++) for (size_t i = 0; i < E; i++) ch += (wq0[o * E + i] != ternary_dense_get_weight(g->wq, o, i));
    *train_acc = tr / trn; *test_acc = te / ten; *qk_changes = ch;
    free(wq0); ternary_u8_free(x); ternary_stack_free(s);
    return 1;
}

int main(void) {
    int fails = 0;
    for (int cfg = 0; cfg < 2; cfg++) {
        const double *P = cfg ? PARAMS2 : PARAMS;
        const double min_acc = cfg ? 80.0 : 85.0;   /* varias cabezas: techo menor (medido 85-93%) */
        for (unsigned seed = 1; seed <= (cfg ? 2u : 3u); seed++) {
            double tr, te; long ch;
            if (!run_seed(P, seed, &tr, &te, &ch)) { printf("FALLO: no se pudo construir/entrenar (cfg %d, semilla %u)\n", cfg, seed); return 1; }
            printf("[%s, semilla %u] entrenamiento %.1f%% | posiciones NO vistas %.1f%% (azar ~50%%) | pesos de Wq que cambiaron: %ld\n", cfg ? "2 cabezas" : "1 cabeza", seed, tr, te, ch);
            if (te < min_acc) { fails++; printf("FALLO: no aprendio la atencion relativa (%.1f%% < %.0f%%)\n", te, min_acc); }
            if (ch == 0) { fails++; printf("FALLO: Wq no cambio ningun peso\n"); }
        }
    }
    printf("%s\n", fails ? "test_gqa_learn_u8: FALLO" : "test_gqa_learn_u8: TODO OK");
    return fails ? 1 : 0;
}
