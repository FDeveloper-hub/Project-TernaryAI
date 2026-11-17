/*
 * Escala float de GQA. Verifica que la salida final de GQA es
 * scale_Wo * (z entero de Wo) igual que en Dense (lineal exacto en la escala),
 * y que las escalas de Wq, Wk y Wv NO alteran la salida (Q/K/V se usan por
 * su signo, y despues attn_cat se binariza otra vez), por lo que solo el
 * gradiente de la escala de Wo (grad_scale_out[3]) debe aplicarse.
 */
#include "ternary_gqa.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FALLA: " __VA_ARGS__); printf("\n"); } } while (0)

static void fill(TernaryGQA *g) {
    TernaryDense *w[4] = {g->wq, g->wk, g->wv, g->wo};
    for (int m = 0; m < 4; m++) for (size_t o = 0; o < w[m]->output_dim; o++) for (size_t i = 0; i < w[m]->input_dim; i++) ternary_dense_set_weight(w[m], o, i, (rand()%3)-1);
}

int main(void) {
    srand(3);
    TernaryGQA *g = ternary_gqa_alloc(16, 6, 2, 1);
    fill(g);
    TernaryVec *x = ternary_vec_alloc(6*16);
    for (size_t i = 0; i < 6*16; i++) ternary_vec_set(x, i, (rand()%3)-1);
    float o1[96], o2[96], o3[96]; int32_t r1[96], r2[96];
    CHECK(g->wo->scale == 1.0f, "la escala inicial de Wo deberia ser 1.0");
    ternary_gqa_forward(g, x, o1, r1);
    g->wo->scale = 0.25f; ternary_gqa_forward(g, x, o2, r2);
    double worst = 0; int raw_same = 1;
    for (int i = 0; i < 96; i++) { worst = fmax(worst, fabs(o2[i] - 0.25f*o1[i])); if (r1[i] != r2[i]) raw_same = 0; }
    printf("[Wo.scale] error max out(0.25) vs 0.25*out(1.0): %.2e, z crudo identico: %s\n", worst, raw_same ? "si" : "NO");
    CHECK(worst == 0.0 && raw_same, "la salida no es scale_Wo * z");

    g->wo->scale = 1.0f; g->wq->scale = 3.0f; g->wk->scale = 0.1f; g->wv->scale = 5.0f;
    ternary_gqa_forward(g, x, o3, NULL);
    worst = 0; for (int i = 0; i < 96; i++) worst = fmax(worst, fabs(o3[i] - o1[i]));
    printf("[Wq/Wk/Wv.scale] diferencia en la salida al cambiarlas: %.2e (esperado 0)\n", worst);
    CHECK(worst == 0.0, "la escala de Wq/Wk/Wv altero la salida");

    TernaryGQA *g2 = ternary_gqa_alloc(16, 6, 2, 1);
    srand(3); fill(g2);
    float a1[96], a2[96]; ternary_gqa_forward(g2, x, a1, NULL); g2->wv->scale = 7.0f; ternary_gqa_forward(g2, x, a2, NULL);
    worst = 0; for (int i = 0; i < 96; i++) worst = fmax(worst, fabs(a1[i]-a2[i]));
    printf("[Wv.scale, V float] diferencia en la salida: %.2e (esperado 0: attn_cat se binariza)\n", worst);
    CHECK(worst == 0.0, "la escala de Wv (V float) altero la salida");

    ternary_vec_free(x); ternary_gqa_free(g); ternary_gqa_free(g2);
    printf("fallas: %d\n%s\n", fails, fails ? "test_gqa_scale: FALLO" : "test_gqa_scale: OK");
    return fails ? 1 : 0;
}
