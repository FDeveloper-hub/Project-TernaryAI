/*
 * Verifica la Softmax con puntajes enteros (src/ternary_softmax.c) contra
 * una referencia en double con exp() real:
 *  1) forward == referencia (tolerancia de float), suma 1, todo >= 0
 *  2) invariante a desplazar todos los puntajes (bit a bit, porque solo
 *     importan las diferencias enteras)
 *  3) mascara causal: posiciones fuera de n_valid en 0 exacto, el resto
 *     igual a un softmax sobre el prefijo
 *  4) puntajes en los extremos del rango no desbordan ni dan NaN
 *  5) backward == diferencias finitas sobre la referencia, y suma ~0
 *  6) puntaje fuera de rango no rompe (se recorta), n_valid=0 -> ceros
 */
#include "ternary_softmax.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FALLO: %s (linea %d)\n", msg, __LINE__); return 1; } } while (0)

static void ref_softmax(const int32_t *sc, size_t n_valid, double t, double *out) {
    double m = -1e300, sum = 0;
    for (size_t i = 0; i < n_valid; i++) if (sc[i] * t > m) m = sc[i] * t;
    for (size_t i = 0; i < n_valid; i++) { out[i] = exp(sc[i] * t - m); sum += out[i]; }
    for (size_t i = 0; i < n_valid; i++) out[i] /= sum;
}

static int test_vs_referencia(void) {
    const int D = 16; const double t = 1.0 / sqrt((double)D);
    TernarySoftmax *s = ternary_softmax_alloc(D, (float)t);
    srand(7);
    double worst = 0;
    for (int rep = 0; rep < 500; rep++) {
        size_t n = 1 + (size_t)(rand() % 20);
        int32_t sc[20]; float y[20]; double r[20];
        for (size_t i = 0; i < n; i++) sc[i] = (rand() % (2 * D + 1)) - D;
        ternary_softmax_forward(s, sc, n, n, y);
        ref_softmax(sc, n, t, r);
        float tot = 0;
        for (size_t i = 0; i < n; i++) {
            double d = fabs((double)y[i] - r[i]); if (d > worst) worst = d;
            CHECK(y[i] >= 0.0f, "probabilidad negativa");
            tot += y[i];
        }
        CHECK(fabsf(tot - 1.0f) < 1e-5f, "no suma 1");
    }
    CHECK(worst < 1e-6, "forward difiere de la referencia en double");
    printf("[softmax forward] 500 casos vs referencia en double: error maximo %.2e, suma 1 OK\n", worst);

    /* 2) invariancia al desplazamiento */
    int32_t a[6] = { 3, -2, 5, 5, 0, -7 }, b[6];
    float ya[6], yb[6];
    for (int i = 0; i < 6; i++) b[i] = a[i] - 4;   /* sigue dentro de [-D, D] */
    ternary_softmax_forward(s, a, 6, 6, ya);
    ternary_softmax_forward(s, b, 6, 6, yb);
    CHECK(memcmp(ya, yb, sizeof(ya)) == 0, "no es invariante al desplazamiento (bit a bit)");
    printf("[softmax] invariante a sumar una constante a todos los puntajes, bit a bit\n");

    /* 3) mascara causal */
    int32_t c[6] = { 1, 4, -3, 2, 9, 9 };
    float yc[6], yp[3];
    ternary_softmax_forward(s, c, 6, 3, yc);
    ternary_softmax_forward(s, c, 3, 3, yp);
    for (int i = 3; i < 6; i++) CHECK(yc[i] == 0.0f, "posicion enmascarada no quedo en 0 exacto");
    CHECK(memcmp(yc, yp, sizeof(yp)) == 0, "con mascara difiere del softmax sobre el prefijo");
    printf("[softmax] mascara causal: futuras en 0 exacto, prefijo == softmax del prefijo\n");

    /* 4) extremos */
    int32_t e[4] = { D, -D, D, -D }; float ye[4];
    ternary_softmax_forward(s, e, 4, 4, ye);
    for (int i = 0; i < 4; i++) CHECK(isfinite(ye[i]) && ye[i] >= 0, "extremos dan NaN/inf/negativo");
    printf("[softmax] puntajes en los extremos [-%d,%d]: finitos y >= 0\n", D, D);

    /* 5) backward vs diferencias finitas */
    int32_t sc[5] = { 2, -1, 4, 0, 4 }; float y[5];
    ternary_softmax_forward(s, sc, 5, 5, y);
    float g[5] = { 0.3f, -1.1f, 0.7f, 0.2f, -0.4f }, ds[5];
    ternary_softmax_backward(s, y, g, 5, ds);
    float sum_ds = 0;
    for (int i = 0; i < 5; i++) {
        double eps = 1e-5, base[5], p[5], q[5];
        double sd[5]; for (int j = 0; j < 5; j++) sd[j] = sc[j];
        (void)base;
        double tt = t;
        /* softmax de referencia con puntaje continuo */
        for (int dir = 0; dir < 2; dir++) {
            double sv[5]; memcpy(sv, sd, sizeof(sv)); sv[i] += dir ? eps : -eps;
            double m = -1e300, sm = 0, o[5];
            for (int j = 0; j < 5; j++) if (sv[j] * tt > m) m = sv[j] * tt;
            for (int j = 0; j < 5; j++) { o[j] = exp(sv[j] * tt - m); sm += o[j]; }
            for (int j = 0; j < 5; j++) o[j] /= sm;
            memcpy(dir ? q : p, o, sizeof(o));
        }
        double num = 0;
        for (int j = 0; j < 5; j++) num += (double)g[j] * (q[j] - p[j]) / (2 * eps);
        CHECK(fabs(num - (double)ds[i]) < 1e-4, "backward difiere de las diferencias finitas");
        sum_ds += ds[i];
    }
    CHECK(fabsf(sum_ds) < 1e-5f, "el gradiente de softmax deberia sumar ~0");
    printf("[softmax backward] igual a diferencias finitas, suma del gradiente ~0\n");

    /* 6) fuera de rango y n_valid=0 */
    int32_t oor[3] = { 1000, 0, -1000 }; float yo[3];
    ternary_softmax_forward(s, oor, 3, 3, yo);
    for (int i = 0; i < 3; i++) CHECK(isfinite(yo[i]), "puntaje fuera de rango dio no finito");
    float z[3] = { 9, 9, 9 };
    ternary_softmax_forward(s, oor, 3, 0, z);
    CHECK(z[0] == 0 && z[1] == 0 && z[2] == 0, "n_valid=0 deberia dar ceros");
    printf("[softmax] puntaje fuera de rango se recorta sin romper; n_valid=0 da ceros\n");

    ternary_softmax_free(s);
    return 0;
}

int main(void) {
    if (test_vs_referencia()) return 1;
    printf("\ntest_softmax: TODO OK\n");
    return 0;
}
