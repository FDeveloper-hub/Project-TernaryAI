/*
 * Atencion GQA con ventana deslizante (ternary_gqa_alloc_window). Verifica:
 *  1. W >= T y sinks = 0: forward y backward IDENTICOS bit a bit a la atencion completa (ternary_gqa_alloc).
 *  2. Localidad del forward: con ventana W la salida de la posicion i no depende de x[j] para j <= i-W (salvo sinks).
 *  3. Localidad del backward: con gradiente solo en la fila i, grad_x solo es no nulo en la ventana (y los sinks).
 *  4. Linealidad: backward con gradiente en TODAS las filas = suma de los backward de cada fila por separado
 *     (comprueba la finalizacion diferida de bloques, con T > bloque y ventanas chicas, con y sin sinks).
 *  5. Que no cuelgue ni se salga de memoria con T no multiplo del bloque (correr tambien con ASan).
 */
#include "ternary_gqa.h"
#include "ternary_avx512.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("FALLA: " __VA_ARGS__); printf("\n"); } } while (0)

static void randomize(TernaryGQA *g, unsigned seed) {
    srand(seed);
    TernaryDense *ws[4] = {g->wq, g->wk, g->wv, g->wo};
    for (int m = 0; m < 4; m++)
        for (size_t o = 0; o < ws[m]->output_dim; o++)
            for (size_t i = 0; i < ws[m]->input_dim; i++)
                ternary_dense_set_weight(ws[m], o, i, (rand() % 3) - 1);
    ws[3]->scale = 0.5f;
}

static TernaryGQA *make(size_t E, size_t T, size_t nh, size_t nkv, size_t W, size_t S, unsigned seed) {
    TernaryGQA *g = W ? ternary_gqa_alloc_window(E, T, nh, nkv, W, S) : ternary_gqa_alloc(E, T, nh, nkv);
    if (!g || ternary_gqa_enable_rope_int(g, 10.0f, 2) != 0) { printf("alloc fallo\n"); exit(1); }
    randomize(g, seed);
    return g;
}

static TernaryVec *rand_x(size_t n, unsigned seed) {
    srand(seed);
    TernaryVec *x = ternary_vec_alloc(n);
    for (size_t i = 0; i < n; i++) ternary_vec_set(x, i, (rand() % 3) - 1);
    return x;
}

typedef struct { float *gw[4]; float *gx; float gs[4]; } Grads;
static Grads grads_new(const TernaryGQA *g) {
    Grads r; TernaryDense *ws[4] = {g->wq, g->wk, g->wv, g->wo};
    for (int m = 0; m < 4; m++) r.gw[m] = (float *)calloc(ws[m]->output_dim * ws[m]->input_dim, sizeof(float));
    r.gx = (float *)calloc(g->seq_len * g->embed_dim, sizeof(float));
    memset(r.gs, 0, sizeof(r.gs));
    return r;
}
static void grads_free(Grads *r) { for (int m = 0; m < 4; m++) free(r->gw[m]); free(r->gx); }

int main(void) {
    const size_t E = 16, NH = 4, NKV = 2;
    /* 1. W >= T equivale a la atencion completa */
    for (size_t T = 5; T <= 130; T += (T < 20 ? 7 : 55)) {
            TernaryGQA *a = make(E, T, NH, NKV, 0, 0, 11 + (unsigned)T), *b = make(E, T, NH, NKV, T + 3, 0, 11 + (unsigned)T);
            TernaryVec *x = rand_x(T * E, 5);
            float *oa = (float *)malloc(T * E * 4), *ob = (float *)malloc(T * E * 4), *go = (float *)malloc(T * E * 4);
            int32_t *ra = (int32_t *)malloc(T * E * 4), *rb = (int32_t *)malloc(T * E * 4);
            for (size_t i = 0; i < T * E; i++) go[i] = (float)((int)(i * 7 % 13) - 6) * 0.1f;
            ternary_gqa_forward(a, x, oa, ra); ternary_gqa_forward(b, x, ob, rb);
            CHECK(memcmp(oa, ob, T * E * 4) == 0 && memcmp(ra, rb, T * E * 4) == 0, "forward W>=T distinto (T=%zu)", T);
            Grads ga = grads_new(a), gb = grads_new(b);
            ternary_gqa_backward(a, x, go, ga.gw, ga.gx, ga.gs, ra); ternary_gqa_backward(b, x, go, gb.gw, gb.gx, gb.gs, rb);
            TernaryDense *ws[4] = {a->wq, a->wk, a->wv, a->wo};
            int same = memcmp(ga.gx, gb.gx, T * E * 4) == 0 && memcmp(ga.gs, gb.gs, sizeof(ga.gs)) == 0;
            for (int m = 0; m < 4; m++) same &= memcmp(ga.gw[m], gb.gw[m], ws[m]->output_dim * ws[m]->input_dim * 4) == 0;
            CHECK(same, "backward W>=T distinto (T=%zu)", T);
            grads_free(&ga); grads_free(&gb); free(oa); free(ob); free(go); free(ra); free(rb);
            ternary_vec_free(x); ternary_gqa_free(a); ternary_gqa_free(b);
        }
    printf("[W>=T] comparado con atencion completa\n");

    /* 2-4. localidad y linealidad */
    const size_t T = 520;   /* > 2*(W+2*bloque) para que el buffer deslizante se desplace varias veces */
    const size_t cfg[][2] = { {7, 0}, {20, 0}, {9, 4}, {60, 3}, {1, 0}, {1, 2}, {130, 4} };
    for (size_t c = 0; c < sizeof(cfg) / sizeof(cfg[0]); c++) {
        const size_t W = cfg[c][0], S = cfg[c][1];
        TernaryGQA *g = make(E, T, NH, NKV, W, S, 99);
        TernaryVec *x = rand_x(T * E, 17);
        float *o1 = (float *)malloc(T * E * 4), *o2 = (float *)malloc(T * E * 4);
        ternary_gqa_forward(g, x, o1, NULL);
        for (size_t k = 0; k < 40; k++) {       /* perturba x[j] y mira que i >= j+W (fuera de sinks) no cambie */
            const size_t j = 5 + (size_t)(rand() % 400);
            TernaryVec *y = ternary_vec_alloc(T * E);
            for (size_t i = 0; i < T * E; i++) ternary_vec_set(y, i, ternary_vec_get(x, i));
            for (size_t d = 0; d < E; d++) ternary_vec_set(y, j * E + d, -ternary_vec_get(x, j * E + d) + ((d & 1) ? 1 : -1));
            ternary_gqa_forward(g, y, o2, NULL);
            int ok = 1;
            for (size_t i = j + W; i < T; i++) if (memcmp(o1 + i * E, o2 + i * E, E * 4) != 0) ok = 0;
            CHECK(ok, "forward no es local (W=%zu S=%zu j=%zu)", W, S, j);
            for (size_t i = 0; i < j; i++) if (memcmp(o1 + i * E, o2 + i * E, E * 4) != 0) ok = 0;
            CHECK(ok, "forward causal roto (W=%zu j=%zu)", W, j);
            ternary_vec_free(y);
        }
        if (S > 0) {   /* los sinks si llegan a todas las filas: cambiar x[0] altera filas lejanas */
            TernaryVec *y = ternary_vec_alloc(T * E);
            for (size_t i = 0; i < T * E; i++) ternary_vec_set(y, i, ternary_vec_get(x, i));
            for (size_t d = 0; d < E; d++) ternary_vec_set(y, d, 1 - 2 * (int)(d & 1));
            ternary_gqa_forward(g, y, o2, NULL);
            int changed = 0;
            for (size_t i = W + 1; i < T; i++) if (memcmp(o1 + i * E, o2 + i * E, E * 4) != 0) changed = 1;
            CHECK(changed, "los sinks no influyen en filas lejanas (W=%zu S=%zu)", W, S);
            ternary_vec_free(y);
        }
        /* backward: todas las filas vs suma de una por una */
        float *go = (float *)calloc(T * E, 4), *g1 = (float *)calloc(T * E, 4);
        for (size_t i = 0; i < T * E; i++) go[i] = (float)((int)(i * 7 % 13) - 6) * 0.1f;
        ternary_gqa_forward(g, x, o1, NULL);
        Grads all = grads_new(g), sum = grads_new(g), one = grads_new(g);
        ternary_gqa_backward(g, x, go, all.gw, all.gx, all.gs, NULL);
        TernaryDense *ws[4] = {g->wq, g->wk, g->wv, g->wo};
        for (size_t i = 0; i < T; i++) {
            memset(g1, 0, T * E * 4); memcpy(g1 + i * E, go + i * E, E * 4);
            memset(one.gx, 0, T * E * 4); memset(one.gs, 0, sizeof(one.gs));
            ternary_gqa_backward(g, x, g1, one.gw, one.gx, one.gs, NULL);
            for (size_t k = 0; k < T * E; k++) sum.gx[k] += one.gx[k];
            for (int m = 0; m < 4; m++) for (size_t k = 0; k < ws[m]->output_dim * ws[m]->input_dim; k++) sum.gw[m][k] += one.gw[m][k];
            for (int m = 0; m < 4; m++) sum.gs[m] += one.gs[m];
            /* localidad: solo ventana + sinks */
            const size_t j0 = (i + 1 > W) ? i + 1 - W : 0;
            int ok = 1;
            for (size_t j = 0; j < i + 1; j++) {
                const int visible = (j >= j0) || (j < S);
                if (visible) continue;
                for (size_t d = 0; d < E; d++) if (one.gx[j * E + d] != 0.0f) ok = 0;
            }
            for (size_t j = i + 1; j < T; j++) for (size_t d = 0; d < E; d++) if (one.gx[j * E + d] != 0.0f) ok = 0;
            CHECK(ok, "backward no es local (W=%zu S=%zu i=%zu)", W, S, i);
        }
        double maxrel = 0.0;
        /* error relativo a la magnitud maxima de cada arreglo (las sumas en otro orden cancelan terminos grandes) */
        #define CMP(A, B, N) do { double mx = 1e-6; for (size_t k = 0; k < (N); k++) if (fabs((double)(B)[k]) > mx) mx = fabs((double)(B)[k]); \
            for (size_t k = 0; k < (N); k++) { double d = fabs((double)(A)[k] - (B)[k]) / mx; if (d > maxrel) maxrel = d; } } while (0)
        CMP(all.gx, sum.gx, T * E);
        for (int m = 0; m < 4; m++) CMP(all.gw[m], sum.gw[m], ws[m]->output_dim * ws[m]->input_dim);
        CMP(all.gs, sum.gs, 4);
        printf("[ventana W=%zu sinks=%zu] error relativo max (todas las filas vs suma por fila) = %.2e\n", W, S, maxrel);
        CHECK(maxrel < 1e-4, "backward no lineal (W=%zu S=%zu): %.3e", W, S, maxrel);
        grads_free(&all); grads_free(&sum); grads_free(&one); free(go); free(g1); free(o1); free(o2);
        ternary_vec_free(x); ternary_gqa_free(g);
    }

    /* 6. AVX-512 == escalar, bit a bit (misma capa, una con los nucleos y otra con el buffer de nucleos quitado) */
    if (ternary_avx512_enabled()) {
        const struct { size_t E, NH, NKV; } shp[] = { {16, 4, 2}, {64, 2, 1}, {192, 4, 2}, {320, 4, 1}, {96, 8, 2} };
        const size_t wc[][2] = { {7, 0}, {20, 4}, {50, 3}, {1, 0}, {200, 5} };
        for (size_t a = 0; a < sizeof(shp) / sizeof(shp[0]); a++)
            for (size_t c = 0; c < sizeof(wc) / sizeof(wc[0]); c++)
                {
                    const size_t Tn = 530, Wn = wc[c][0], Sn = wc[c][1], En = shp[a].E;
                    TernaryGQA *ga = make(En, Tn, shp[a].NH, shp[a].NKV, Wn, Sn, 7 + (unsigned)a), *gb = make(En, Tn, shp[a].NH, shp[a].NKV, Wn, Sn, 7 + (unsigned)a);
                    CHECK(ga->w_kT != NULL, "la capa deberia tener la disposicion AVX-512");
                    free(gb->w_kT); free(gb->w_vT); gb->w_kT = NULL; gb->w_vT = NULL;
                    TernaryVec *x = rand_x(Tn * En, 3 + (unsigned)c);
                    float *oa = (float *)malloc(Tn * En * 4), *ob = (float *)malloc(Tn * En * 4), *go = (float *)malloc(Tn * En * 4);
                    int32_t *ra = (int32_t *)malloc(Tn * En * 4), *rb = (int32_t *)malloc(Tn * En * 4);
                    for (size_t i = 0; i < Tn * En; i++) go[i] = (float)((int)(i * 7 % 13) - 6) * 0.1f;
                    ternary_gqa_forward(ga, x, oa, ra); ternary_gqa_forward(gb, x, ob, rb);
                    CHECK(memcmp(oa, ob, Tn * En * 4) == 0 && memcmp(ra, rb, Tn * En * 4) == 0, "forward AVX != escalar (E=%zu W=%zu S=%zu)", En, Wn, Sn);
                    Grads g1 = grads_new(ga), g2 = grads_new(gb);
                    ternary_gqa_backward(ga, x, go, g1.gw, g1.gx, g1.gs, ra); ternary_gqa_backward(gb, x, go, g2.gw, g2.gx, g2.gs, rb);
                    TernaryDense *w4[4] = {ga->wq, ga->wk, ga->wv, ga->wo};
                    int same = memcmp(g1.gx, g2.gx, Tn * En * 4) == 0 && memcmp(g1.gs, g2.gs, sizeof(g1.gs)) == 0;
                    for (int m = 0; m < 4; m++) same &= memcmp(g1.gw[m], g2.gw[m], w4[m]->output_dim * w4[m]->input_dim * 4) == 0;
                    CHECK(same, "backward AVX != escalar (E=%zu W=%zu S=%zu)", En, Wn, Sn);
                    grads_free(&g1); grads_free(&g2); free(oa); free(ob); free(go); free(ra); free(rb);
                    ternary_vec_free(x); ternary_gqa_free(ga); ternary_gqa_free(gb);
                }
        printf("[AVX-512] ventana: forward y backward identicos al escalar bit a bit\n");
    } else printf("[AVX-512] no disponible: se omite la comparacion\n");
    if (fails) { printf("%d FALLAS\n", fails); return 1; }
    printf("OK test_gqa_window\n");
    return 0;
}
