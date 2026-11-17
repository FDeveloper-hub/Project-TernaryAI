/*
 * RoPE entero de GQA (ternary_gqa_enable_rope). Verifica:
 *  1. la tabla en punto fijo contra cos/sin en double (error <= 0.5/256) y
 *     que cada par sea una rotacion (cos^2+sin^2 ~ 1),
 *  2. que los bits de Q/K salgan con el MISMO signo que la rotacion exacta
 *     en double del Z crudo (solo puede discrepar si el valor exacto es
 *     casi cero, dentro del error de la tabla),
 *  3. que RoPE realmente aporta posicion: con la misma entrada en todas las
 *     posiciones, sin RoPE la atencion es uniforme y con RoPE no,
 *  4. que head_dim impar se rechaza y que forward+backward con RoPE dan
 *     valores finitos,
 *  5. puntaje ENTERO (ternary_gqa_enable_rope_int): con la misma entrada en
 *     todas las posiciones, la varianza del puntaje q.k explicada por la
 *     distancia relativa p-r debe ser ~1 con q_rot.k_rot enteros (RoPE
 *     relativo de verdad) y mucho menor con los signos binarizados; y el
 *     forward+backward con puntaje entero da valores finitos.
 */
#include "ternary_gqa.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define E 16
#define SEQ 6
#define NH 4
#define NKV 2

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("FALLA: " __VA_ARGS__); printf("\n"); } } while (0)

static void randomize(TernaryGQA *g) {
    TernaryDense *ws[4] = {g->wq, g->wk, g->wv, g->wo};
    for (int m = 0; m < 4; m++)
        for (size_t o = 0; o < ws[m]->output_dim; o++)
            for (size_t i = 0; i < ws[m]->input_dim; i++)
                ternary_dense_set_weight(ws[m], o, i, (rand() % 3) - 1);
}

int main(void) {
    srand(7);
    const float base = 100.0f;

    /* --- 1. tabla --- */
    TernaryGQA *g = ternary_gqa_alloc(E, SEQ, NH, NKV);
    CHECK(ternary_gqa_enable_rope(g, base) == 0, "enable_rope devolvio error");
    double worst = 0.0, worst_norm = 0.0;
    for (size_t p = 0; p < SEQ; p++)
        for (size_t i = 0; i < g->rope_pairs; i++) {
            double th = pow((double)base, -2.0 * (double)i / (double)g->head_dim), a = p * th;
            double c = g->rope_cos[p * g->rope_pairs + i] / (double)TERNARY_ROPE_SCALE;
            double s = g->rope_sin[p * g->rope_pairs + i] / (double)TERNARY_ROPE_SCALE;
            worst = fmax(worst, fmax(fabs(c - cos(a)), fabs(s - sin(a))));
            worst_norm = fmax(worst_norm, fabs(c * c + s * s - 1.0));
        }
    printf("[tabla] error max cos/sin = %.5f (limite %.5f) | |c^2+s^2-1| max = %.5f\n", worst, 0.5 / TERNARY_ROPE_SCALE + 1e-9, worst_norm);
    CHECK(worst <= 0.5 / TERNARY_ROPE_SCALE + 1e-9, "tabla fuera de tolerancia");
    CHECK(worst_norm < 0.01, "la tabla no es una rotacion");

    /* --- 2. signos de Q/K vs rotacion exacta en double --- */
    long checked = 0, mismatch_ok = 0;
    for (int trial = 0; trial < 200; trial++) {
        randomize(g);
        TernaryVec *x = ternary_vec_alloc(SEQ * E);
        for (size_t i = 0; i < SEQ * E; i++) ternary_vec_set(x, i, (rand() % 3) - 1);
        float out[SEQ * E]; ternary_gqa_forward(g, x, out, NULL);
        size_t H = g->head_dim, qw = NH * H, kvw = NKV * H;
        for (size_t p = 0; p < SEQ; p++) {
            for (int which = 0; which < 2; which++) {
                size_t heads = which ? NKV : NH, w = which ? kvw : qw;
                const int32_t *z = (which ? g->k_z : g->q_z) + p * w;
                const TernaryVec *bits = which ? g->k_bits : g->q_bits;
                for (size_t h = 0; h < heads; h++)
                    for (size_t i = 0; i < g->rope_pairs; i++) {
                        double th = pow((double)base, -2.0 * (double)i / (double)H), an = p * th;
                        double a = z[h * H + 2 * i], b = z[h * H + 2 * i + 1];
                        double ex[2] = { a * cos(an) - b * sin(an), a * sin(an) + b * cos(an) };
                        for (int t = 0; t < 2; t++) {
                            int want = ex[t] > 0 ? 1 : (ex[t] < 0 ? -1 : 0);
                            int got = ternary_vec_get(bits, p * w + h * H + 2 * i + t);
                            checked++;
                            if (want != got) {
                                /* solo tolerable si el valor exacto es casi 0 (error de la tabla) */
                                if (fabs(ex[t]) <= (fabs(a) + fabs(b)) * (0.5 / TERNARY_ROPE_SCALE) + 1e-9) mismatch_ok++;
                                else CHECK(0, "signo distinto con valor exacto %.4f (a=%.0f b=%.0f)", ex[t], a, b);
                            }
                        }
                    }
            }
        }
        ternary_vec_free(x);
    }
    printf("[signos] %ld valores comparados, %ld discrepancias dentro del error de la tabla\n", checked, mismatch_ok);

    /* --- 3. RoPE aporta posicion --- */
    double spread_rope = 0.0, spread_plain = 0.0;
    for (int trial = 0; trial < 50; trial++) {
        TernaryGQA *a = ternary_gqa_alloc(E, SEQ, NH, NKV);
        TernaryGQA *b = ternary_gqa_alloc(E, SEQ, NH, NKV);
        randomize(a);
        TernaryDense *wa[4] = {a->wq, a->wk, a->wv, a->wo}, *wb[4] = {b->wq, b->wk, b->wv, b->wo};
        for (int m = 0; m < 4; m++)
            for (size_t o = 0; o < wa[m]->output_dim; o++)
                for (size_t i = 0; i < wa[m]->input_dim; i++)
                    ternary_dense_set_weight(wb[m], o, i, ternary_dense_get_weight(wa[m], o, i));
        ternary_gqa_enable_rope(b, base);
        TernaryVec *x = ternary_vec_alloc(SEQ * E);
        int row[E]; for (size_t i = 0; i < E; i++) row[i] = (rand() % 3) - 1;
        for (size_t p = 0; p < SEQ; p++) for (size_t i = 0; i < E; i++) ternary_vec_set(x, p * E + i, row[i]);
        float out[SEQ * E];
        ternary_gqa_forward(a, x, out, NULL);
        for (size_t j = 0; j < SEQ; j++) spread_plain = fmax(spread_plain, fabs(a->probs[j] - 1.0 / SEQ));
        ternary_gqa_forward(b, x, out, NULL);
        double mx = 0, mn = 1; for (size_t j = 0; j < SEQ; j++) { mx = fmax(mx, b->probs[j]); mn = fmin(mn, b->probs[j]); }
        spread_rope = fmax(spread_rope, mx - mn);
        ternary_vec_free(x); ternary_gqa_free(a); ternary_gqa_free(b);
    }
    printf("[posicion] misma entrada en todas las posiciones: desvio max de la atencion uniforme sin RoPE = %.6f, dispersion con RoPE = %.4f\n", spread_plain, spread_rope);
    CHECK(spread_plain < 1e-6, "sin RoPE la atencion deberia ser uniforme");
    CHECK(spread_rope > 0.01, "con RoPE la atencion deberia dejar de ser uniforme");

    /* --- 4. head_dim impar y backward finito --- */
    TernaryGQA *odd = ternary_gqa_alloc(15, SEQ, 5, 5);
    CHECK(odd && ternary_gqa_enable_rope(odd, base) == -1, "head_dim impar deberia rechazarse");
    ternary_gqa_free(odd);

    randomize(g);
    TernaryVec *x = ternary_vec_alloc(SEQ * E);
    for (size_t i = 0; i < SEQ * E; i++) ternary_vec_set(x, i, (rand() % 3) - 1);
    float out[SEQ * E]; int32_t raw[SEQ * E]; ternary_gqa_forward(g, x, out, raw);
    size_t H = g->head_dim, qw = NH * H, kvw = NKV * H;
    float *gwq = malloc(qw * E * 4), *gwk = malloc(kvw * E * 4), *gwv = malloc(kvw * E * 4), *gwo = malloc(E * qw * 4);
    float *gw[4] = {gwq, gwk, gwv, gwo};
    float go[SEQ * E], gx[SEQ * E] = {0}, gs[4] = {0};
    for (size_t i = 0; i < SEQ * E; i++) go[i] = (float)((rand() % 2001) - 1000) / 1000.0f;
    ternary_gqa_backward(g, x, go, gw, gx, gs, raw);
    int finite = 1, nonzero_q = 0;
    for (size_t i = 0; i < qw * E; i++) { if (!isfinite(gwq[i])) finite = 0; if (gwq[i] != 0) nonzero_q = 1; }
    for (size_t i = 0; i < kvw * E; i++) { if (!isfinite(gwk[i]) || !isfinite(gwv[i])) finite = 0; }
    for (size_t i = 0; i < SEQ * E; i++) if (!isfinite(gx[i])) finite = 0;
    CHECK(finite, "gradientes no finitos con RoPE");
    CHECK(nonzero_q, "el gradiente de Wq es todo cero con RoPE");
    printf("[backward] gradientes finitos, Wq recibe gradiente: %s\n", (finite && nonzero_q) ? "si" : "NO");


    /* --- 5. puntaje entero: propiedad relativa --- */
    {
        const size_t E2 = 24, S2 = 8, H2 = 24;
        double ex_int = 0, ex_bin = 0; int used = 0;
        for (int trial = 0; trial < 200; trial++) {
            TernaryGQA *r = ternary_gqa_alloc(E2, S2, 1, 1);
            if (ternary_gqa_enable_rope_int(r, 10.0f, 3) != 0) { CHECK(0, "enable_rope_int fallo"); break; }
            randomize(r);
            TernaryVec *xx = ternary_vec_alloc(S2 * E2);
            int row[24]; for (size_t i = 0; i < E2; i++) row[i] = (rand() % 3) - 1;
            for (size_t p = 0; p < S2; p++) for (size_t i = 0; i < E2; i++) ternary_vec_set(xx, p * E2 + i, row[i]);
            float o[8 * 24]; ternary_gqa_forward(r, xx, o, NULL);
            double si[8][8], sb[8][8];
            for (size_t p = 0; p < S2; p++) for (size_t j = 0; j <= p; j++) {
                double a = 0; for (size_t d = 0; d < H2; d++) a += (double)r->q_rot[p * H2 + d] * (double)r->k_rot[j * H2 + d];
                si[p][j] = a / (double)(TERNARY_ROPE_SCALE * TERNARY_ROPE_SCALE);
                double b = 0; for (size_t d = 0; d < H2; d++) b += ternary_vec_get(r->q_bits, p * H2 + d) * ternary_vec_get(r->k_bits, j * H2 + d);
                sb[p][j] = b;
            }
            for (int w = 0; w < 2; w++) {
                double (*s)[8] = w ? sb : si; double sum = 0, sum2 = 0; int n = 0;
                double dsum[8] = {0}; int dn[8] = {0};
                for (size_t p = 0; p < S2; p++) for (size_t j = 0; j <= p; j++) { double v = s[p][j]; sum += v; sum2 += v * v; n++; dsum[p - j] += v; dn[p - j]++; }
                double var = sum2 / n - (sum / n) * (sum / n);
                if (var < 1e-9) { if (w == 0) used--; continue; }
                double within = 0;
                for (size_t p = 0; p < S2; p++) for (size_t j = 0; j <= p; j++) { double m = dsum[p - j] / dn[p - j]; within += (s[p][j] - m) * (s[p][j] - m); }
                within /= n;
                if (w == 0) ex_int += 1.0 - within / var; else ex_bin += 1.0 - within / var;
            }
            used++;
            ternary_vec_free(xx); ternary_gqa_free(r);
        }
        if (used > 0) { ex_int /= used; ex_bin /= used; }
        printf("[puntaje entero] varianza explicada por la distancia relativa: producto entero %.3f | signos binarizados %.3f (%d pruebas)\n", ex_int, ex_bin, used);
        CHECK(ex_int > 0.97, "el puntaje entero deberia depender casi solo de p-r");
        CHECK(ex_int > ex_bin + 0.2, "el puntaje entero deberia ser mucho mas relativo que el binarizado");

        TernaryGQA *r = ternary_gqa_alloc(E2, S2, 2, 1);
        CHECK(ternary_gqa_enable_rope_int(r, 10.0f, 3) == 0, "enable_rope_int (2 cabezas)");
        randomize(r);
        TernaryVec *xx = ternary_vec_alloc(S2 * E2);
        for (size_t i = 0; i < S2 * E2; i++) ternary_vec_set(xx, i, (rand() % 3) - 1);
        float o[8 * 24], gox[8 * 24], gxx[8 * 24] = {0}, gss[4] = {0}; int32_t rawv[8 * 24];
        ternary_gqa_forward(r, xx, o, rawv);
        for (size_t i = 0; i < S2 * E2; i++) gox[i] = (float)((rand() % 2001) - 1000) / 1000.0f;
        size_t qw2 = 2 * 12, kv2 = 1 * 12;
        float *g0 = malloc(qw2 * E2 * 4), *g1 = malloc(kv2 * E2 * 4), *g2 = malloc(kv2 * E2 * 4), *g3 = malloc(E2 * qw2 * 4);
        float *gws[4] = {g0, g1, g2, g3};
        ternary_gqa_backward(r, xx, gox, gws, gxx, gss, rawv);
        int fin = 1, nz = 0;
        for (size_t i = 0; i < qw2 * E2; i++) { if (!isfinite(g0[i])) fin = 0; if (g0[i] != 0) nz = 1; }
        for (size_t i = 0; i < kv2 * E2; i++) if (!isfinite(g1[i]) || !isfinite(g2[i])) fin = 0;
        for (size_t i = 0; i < S2 * E2; i++) if (!isfinite(gxx[i])) fin = 0;
        CHECK(fin && nz, "backward con puntaje entero: finito=%d, Wq con gradiente=%d", fin, nz);
        printf("[puntaje entero] backward finito: %s, Wq recibe gradiente: %s\n", fin ? "si" : "NO", nz ? "si" : "NO");
        free(g0); free(g1); free(g2); free(g3); ternary_vec_free(xx); ternary_gqa_free(r);
    }

    free(gwq); free(gwk); free(gwv); free(gwo); ternary_vec_free(x); ternary_gqa_free(g);
    printf("fallas: %d\n%s\n", fails, fails ? "test_gqa_rope: FALLO" : "test_gqa_rope: OK");
    return fails ? 1 : 0;
}
