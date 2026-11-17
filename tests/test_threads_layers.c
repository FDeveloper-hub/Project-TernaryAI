/* Capas sueltas con 1 hilo contra varios: forward y backward deben dar EXACTAMENTE los mismos bits (Conv2D, Embedding, RMSNorm, cuantizador uint8). */
#include "ternary_conv2d.h"
#include "ternary_gqa.h"
#include "ternary_embedding.h"
#include "ternary_rmsnorm.h"
#include "ternary_u8.h"
#include "ternary_threadpool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
static void check(const char *what, const void *a, const void *b, size_t n, int th) {
    if (memcmp(a, b, n)) { printf("FALLA %s con %d hilos\n", what, th); fails++; }
}
static float frand(void) { return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }

#define NB 6
int main(void) {
    const int ths[] = { 2, 3, 8, 64 };
    srand(7);
    /* ---- Conv2D ---- */
    {
        const size_t H = 20, W = 18, C = 5, KH = 3, KW = 3, CO = 11;
        TernaryConv2D *l = ternary_conv2d_alloc(H, W, C, KH, KW, CO, 1, 1);
        for (size_t o = 0; o < CO; o++) for (size_t k = 0; k < KH * KW * C; k++) ternary_conv2d_set_weight(l, o, k, rand() % 3 - 1);
        const size_t n_img = H * W * C, n_out = l->out_height * l->out_width * CO, KS = l->kernel_size;
        TernaryVec *img = ternary_vec_alloc(n_img);
        for (size_t i = 0; i < n_img; i++) ternary_vec_set(img, i, rand() % 3 - 1);
        float *go = malloc(n_out * 4); for (size_t i = 0; i < n_out; i++) go[i] = frand();
        float *res[2][4]; int32_t *raw[2];
        for (int pass = 0; pass < 2; pass++) {
            for (int t = 0; t < (pass ? 4 : 1); t++) {
                ternary_set_num_threads(pass ? ths[t] : 1);
                float *out = malloc(n_out * 4), *gk = calloc(CO * KS, 4), *gb = calloc(CO, 4), gs = 0.5f;
                int32_t *rw = malloc(n_out * 4);
                ternary_conv2d_forward(l, img, out, rw);
                ternary_conv2d_backward(l, img, go, gk, gb, &gs, rw);
                { /* sin Z crudo (NULL): mismos bits que con Z crudo, con cualquier numero de hilos */
                    float *gk2 = calloc(CO * KS, 4), *gb2 = calloc(CO, 4), gs2 = 0.5f;
                    ternary_conv2d_backward(l, img, go, gk2, gb2, &gs2, NULL);
                    check("conv2d gk sin Z", gk2, gk, CO * KS * 4, pass ? ths[t] : 1); check("conv2d gb sin Z", gb2, gb, CO * 4, pass ? ths[t] : 1);
                    check("conv2d gs sin Z", &gs2, &gs, 4, pass ? ths[t] : 1); free(gk2); free(gb2);
                }
                if (!pass) { res[0][0] = out; res[0][1] = gk; res[0][2] = gb; res[0][3] = malloc(4); *res[0][3] = gs; raw[0] = rw; }
                else {
                    check("conv2d out", out, res[0][0], n_out * 4, ths[t]); check("conv2d gk", gk, res[0][1], CO * KS * 4, ths[t]);
                    check("conv2d gb", gb, res[0][2], CO * 4, ths[t]); check("conv2d gs", &gs, res[0][3], 4, ths[t]); check("conv2d z", rw, raw[0], n_out * 4, ths[t]);
                    free(out); free(gk); free(gb); free(rw);
                }
            }
        }
        ternary_conv2d_free(l); ternary_vec_free(img);
    }
    /* ---- Embedding (ids repetidos) ---- */
    {
        const size_t V = 50, E = 70, R = 300;
        TernaryEmbedding *l = ternary_embedding_alloc(V, E, R);
        for (size_t v = 0; v < V; v++) for (size_t d = 0; d < E; d++) ternary_embedding_set_weight(l, v, d, rand() % 3 - 1);
        int32_t *ids = malloc(R * 4); for (size_t p = 0; p < R; p++) ids[p] = (p % 37 == 0) ? -1 : rand() % 12;
        float *go = malloc(R * E * 4); for (size_t i = 0; i < R * E; i++) go[i] = frand();
        float *ref_out = NULL, *ref_gw = NULL; float ref_gs = 0;
        for (int t = -1; t < 4; t++) {
            ternary_set_num_threads(t < 0 ? 1 : ths[t]);
            float *out = malloc(R * E * 4), *gw = malloc(V * E * 4), gs = 0.25f; int32_t *rw = malloc(R * E * 4);
            ternary_embedding_forward(l, ids, out, rw);
            ternary_embedding_backward(l, ids, go, gw, &gs, rw);
            if (t < 0) { ref_out = out; ref_gw = gw; ref_gs = gs; free(rw); continue; }
            check("embedding out", out, ref_out, R * E * 4, ths[t]); check("embedding gw", gw, ref_gw, V * E * 4, ths[t]); check("embedding gs", &gs, &ref_gs, 4, ths[t]);
            free(out); free(gw); free(rw);
        }
        ternary_embedding_free(l);
    }
    /* ---- RMSNorm ---- */
    {
        const size_t R = 700, D = 90;
        TernaryRMSNorm *l = ternary_rmsnorm_alloc(R, D, 1e-5f);
        for (size_t j = 0; j < D; j++) ternary_dense_set_weight(l->gain, 0, j, rand() % 3 - 1);
        l->gain->scale = 0.8f;
        float *x = malloc(R * D * 4), *go = malloc(R * D * 4);
        for (size_t i = 0; i < R * D; i++) { x[i] = frand() * 3; go[i] = frand(); }
        float *ro = NULL, *rgg = NULL, *rgx = NULL; float rgs = 0;
        for (int t = -1; t < 4; t++) {
            ternary_set_num_threads(t < 0 ? 1 : ths[t]);
            float *out = malloc(R * D * 4), *gg = malloc(D * 4), *gx = calloc(R * D, 4), gs = 0.1f;
            ternary_rmsnorm_forward(l, x, out);
            ternary_rmsnorm_backward(l, x, go, gg, gx, &gs);
            if (t < 0) { ro = out; rgg = gg; rgx = gx; rgs = gs; continue; }
            check("rmsnorm out", out, ro, R * D * 4, ths[t]); check("rmsnorm gg", gg, rgg, D * 4, ths[t]);
            check("rmsnorm gx", gx, rgx, R * D * 4, ths[t]); check("rmsnorm gs", &gs, &rgs, 4, ths[t]);
            free(out); free(gg); free(gx);
        }
        ternary_rmsnorm_free(l);
    }
    /* ---- cuantizador uint8 ---- */
    {
        const size_t R = 500, N = 100;
        float *x = malloc(R * N * 4); for (size_t i = 0; i < R * N; i++) x[i] = (rand() % 9 == 0) ? 0.0f : frand() * 5;
        TernaryU8 *ref = NULL;
        for (int t = -1; t < 4; t++) {
            ternary_set_num_threads(t < 0 ? 1 : ths[t]);
            TernaryU8 *q = ternary_u8_alloc(R, N);
            ternary_u8_quantize(q, x);
            if (t < 0) { ref = q; continue; }
            check("u8 q", q->q, ref->q, R * N, ths[t]); check("u8 scale", q->scale, ref->scale, R * 4, ths[t]); check("u8 zp", q->zp, ref->zp, R * 4, ths[t]);
            ternary_u8_free(q);
        }
    }
    /* ---- GQA con entrada ternaria (ruta antigua no uint8): RoPE entero + smear, y la variante con puntaje en floats (sin RoPE) ---- */
    for (int variant = 0; variant < 2; variant++) {
        const size_t E = 64, SEQ = 300, NH = 4, NKV = 2, qw = E, kvw = E / NH * NKV;
        float *res[2][7]; size_t len[7] = { SEQ * E, SEQ * E, qw * E, kvw * E, kvw * E, E * qw, 4 };
        for (int pass = 0; pass < 5; pass++) {
            ternary_set_num_threads(pass ? ths[pass - 1] : 1);
            TernaryGQA *g = ternary_gqa_alloc(E, SEQ, NH, NKV);
            if (variant == 0) { ternary_gqa_enable_rope_int(g, 10.0f, 2); ternary_gqa_set_smear(g, 1); }
            srand(21);
            TernaryDense *ws[4] = { g->wq, g->wk, g->wv, g->wo };
            for (int m = 0; m < 4; m++) for (size_t o = 0; o < ws[m]->output_dim; o++) for (size_t i = 0; i < ws[m]->input_dim; i++)
                ternary_dense_set_weight(ws[m], o, i, (rand() % 3) - 1);
            g->wo->scale = 0.5f;
            TernaryVec *x = ternary_vec_alloc(SEQ * E);
            for (size_t i = 0; i < SEQ * E; i++) ternary_vec_set(x, i, rand() % 3 - 1);
            float *out = calloc(SEQ * E, 4), *go = malloc(SEQ * E * 4), *gx = calloc(SEQ * E, 4);
            for (size_t i = 0; i < SEQ * E; i++) go[i] = frand();
            float *gw[4] = { calloc(qw * E, 4), calloc(kvw * E, 4), calloc(kvw * E, 4), calloc(E * qw, 4) }, gs[4] = { 0 };
            ternary_gqa_forward(g, x, out, NULL);
            ternary_gqa_backward(g, x, go, gw, gx, gs, NULL);
            float *cur[7] = { out, gx, gw[0], gw[1], gw[2], gw[3], gs };
            if (pass == 0) { for (int r = 0; r < 7; r++) { res[0][r] = malloc(len[r] * 4); memcpy(res[0][r], cur[r], len[r] * 4); } }
            else for (int r = 0; r < 7; r++) check("gqa ternaria", cur[r], res[0][r], len[r] * 4, ths[pass - 1]);
            free(out); free(go); free(gx); for (int m = 0; m < 4; m++) free(gw[m]);
            ternary_vec_free(x); ternary_gqa_free(g);
        }
        (void)res[1];
    }
    ternary_set_num_threads(1);
    if (fails) { printf("test_threads_layers: %d FALLAS\n", fails); return 1; }
    printf("test_threads_layers: OK\n");
    return 0;
}
