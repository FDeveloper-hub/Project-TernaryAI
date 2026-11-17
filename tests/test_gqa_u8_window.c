/* GQA uint8: la ventana deslizante con window >= seq_len (sinks 0) debe dar los MISMOS bits que la atencion completa,
 * en forward, grad_w, grad_x y escalas; y con una ventana corta el forward ignora lo que queda fuera de ella
 * (cambiar un token fuera de la ventana no altera la salida). Con y sin V ternario. */
#include "ternary_gqa.h"
#include "ternary_u8.h"
#include "ternary_rng.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fallas = 0;
#define CHECK(c, ...) do { if (!(c)) { fallas++; printf("FALLO (linea %d): ", __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

#define E 32
#define T 20

static TernaryGQA *mk(size_t window, unsigned seed) {
    TernaryGQA *g = ternary_gqa_alloc_window(E, T, 4, 2, window, 0);
    if (!g || ternary_gqa_enable_rope_int(g, 10.0f, 2) != 0) { printf("no se pudo crear\n"); exit(1); }
    TernaryRng rng; ternary_rng_seed(&rng, seed);
    TernaryDense *m[4] = { g->wq, g->wk, g->wv, g->wo };
    for (int k = 0; k < 4; k++) for (size_t o = 0; o < m[k]->output_dim; o++) for (size_t i = 0; i < m[k]->input_dim; i++)
        ternary_dense_set_weight(m[k], o, i, ternary_rng_sample_faninit(&rng, m[k]->input_dim));
    g->wo->scale = 0.05f; g->wv->scale = 0.04f;
    return g;
}

static void run(TernaryGQA *g, const TernaryU8 *x, const float *go, float *out, float **gw, float *gx, float *gs) {
    ternary_gqa_forward_u8(g, x, out, NULL);
    memset(gx, 0, T * E * sizeof(float)); memset(gs, 0, 4 * sizeof(float));
    ternary_gqa_backward_u8(g, x, go, gw, gx, gs);
}

int main(void) {
    printf("-- GQA uint8: ventana\n");
    srand(5);
    float xf[T * E], go[T * E];
    for (int i = 0; i < T * E; i++) { xf[i] = ((float)(rand() % 2001) / 1000.0f - 1.0f) * 2.0f; go[i] = (float)(rand() % 2001) / 1000.0f - 1.0f; }
    TernaryU8 *x = ternary_u8_alloc(T, E); ternary_u8_quantize(x, xf);
    {
        TernaryGQA *a = mk(0, 3), *b = mk(T, 3);
        size_t sz[4] = { a->wq->output_dim * E, a->wk->output_dim * E, a->wv->output_dim * E, E * a->wo->input_dim };
        float *ga[4], *gb[4];
        for (int k = 0; k < 4; k++) { ga[k] = malloc(sz[k] * 4); gb[k] = malloc(sz[k] * 4); }
        float oa[T * E], ob[T * E], xa[T * E], xb[T * E], sa[4], sb[4];
        run(a, x, go, oa, ga, xa, sa);
        run(b, x, go, ob, gb, xb, sb);
        CHECK(memcmp(oa, ob, sizeof oa) == 0, "forward: completa vs ventana=T distinto");
        for (int k = 0; k < 4; k++) CHECK(memcmp(ga[k], gb[k], sz[k] * 4) == 0, "grad_w[%d] distinto", k);
        CHECK(memcmp(xa, xb, sizeof xa) == 0, "grad_x distinto");
        CHECK(memcmp(sa, sb, sizeof sa) == 0, "grad_scale distinto");
        double nz = 0; for (int i = 0; i < T * E; i++) nz += fabs(oa[i]);
        CHECK(nz > 0, "salida toda en cero");
        printf("  completa == ventana(T) bit a bit (|salida|=%.3g)\n", nz);
        /* ventana corta: un token fuera de la ventana no cambia la salida de las posiciones que no lo ven */
        const size_t W = 4;
        TernaryGQA *c = mk(W, 3);
        float oc1[T * E], oc2[T * E], gxc[T * E], gsc[4]; float *gc[4];
        for (int k = 0; k < 4; k++) gc[k] = malloc(sz[k] * 4);
        run(c, x, go, oc1, gc, gxc, gsc);
        float xf2[T * E]; memcpy(xf2, xf, sizeof xf);
        for (int i = 0; i < E; i++) xf2[0 * E + i] = -xf2[0 * E + i] * 3.0f;   /* cambia el token 0 */
        TernaryU8 *x2 = ternary_u8_alloc(T, E); ternary_u8_quantize(x2, xf2);
        run(c, x2, go, oc2, gc, gxc, gsc);
        int same_far = 1;
        for (size_t p = W; p < T; p++) for (int d = 0; d < E; d++) if (oc1[p * E + d] != oc2[p * E + d]) same_far = 0;
        int diff_near = 0;
        for (int d = 0; d < E; d++) if (oc1[0 * E + d] != oc2[0 * E + d]) diff_near = 1;
        CHECK(same_far, "la ventana W=%zu deja pasar un token lejano", W);
        CHECK(diff_near, "cambiar el token 0 no cambia la salida de la posicion 0");
        for (int k = 0; k < 4; k++) { free(ga[k]); free(gb[k]); free(gc[k]); }
        ternary_gqa_free(a); ternary_gqa_free(b); ternary_gqa_free(c); ternary_u8_free(x2);
    }
    ternary_u8_free(x);
    if (fallas) { printf("%d fallas\n", fallas); return 1; }
    printf("OK\n");
    return 0;
}
