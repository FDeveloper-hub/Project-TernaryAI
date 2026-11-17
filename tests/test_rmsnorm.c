/*
 * RMSNorm (ternary_rmsnorm.h): matematica, propiedades y gradientes, contra una referencia
 * INDEPENDIENTE en double (otra implementacion escrita aqui, no la de la libreria).
 *  1. forward vs la referencia, en varias formas y con magnitudes de 1e-3 a 1e3 y ganancias -1/0/+1;
 *  2. propiedades: invariancia de escala, RMS de la salida == escala (con g=+1), g=-1 invierte, g=0 anula,
 *     filas independientes, fila de ceros finita, forward in situ;
 *  3. backward vs diferencias finitas CENTRALES en double sobre la referencia (x, ganancia tratada como real, escala);
 *  4. contrato de acumulacion del backward (grad_x y grad_scale acumulan, grad_gain se sobreescribe, NULL permitido);
 *  5. validacion de alloc.
 */
#include "ternary_rmsnorm.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FALLA: " __VA_ARGS__); printf("\n"); } } while (0)

static uint64_t rs = 88172645463325252ull;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) / 9007199254740992.0; }
static double nrand(void) { double u = urand() + 1e-12, v = urand(); return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v); }

/* Referencia independiente en double (g real, s real). */
static void ref_forward(const double *x, const double *g, double s, size_t rows, size_t dim, double eps, double *y) {
    for (size_t i = 0; i < rows; i++) {
        double ms = 0; for (size_t j = 0; j < dim; j++) ms += x[i*dim+j] * x[i*dim+j];
        ms /= (double)dim;
        double r = 1.0 / sqrt(ms + eps);
        for (size_t j = 0; j < dim; j++) y[i*dim+j] = s * g[j] * x[i*dim+j] * r;
    }
}
static double ref_loss(const double *x, const double *g, double s, size_t rows, size_t dim, double eps, const double *go) {
    double *y = malloc(rows * dim * sizeof(double)); ref_forward(x, g, s, rows, dim, eps, y);
    double L = 0; for (size_t k = 0; k < rows * dim; k++) L += go[k] * y[k];
    free(y); return L;
}

static void set_gain(TernaryRMSNorm *n, const double *g) { for (size_t j = 0; j < n->dim; j++) ternary_dense_set_weight(n->gain, 0, j, (int)g[j]); }
static void random_gain(double *g, size_t dim) { for (size_t j = 0; j < dim; j++) g[j] = (double)((int)(urand() * 3.0) - 1); }

int main(void) {
    /* ---- 5. alloc ---- */
    CHECK(!ternary_rmsnorm_alloc(0, 8, 1e-5f), "rows=0 debia rechazarse");
    CHECK(!ternary_rmsnorm_alloc(2, 0, 1e-5f), "dim=0 debia rechazarse");
    CHECK(!ternary_rmsnorm_alloc(2, 8, 0.0f), "eps=0 debia rechazarse");
    CHECK(!ternary_rmsnorm_alloc(2, 8, -1e-3f), "eps<0 debia rechazarse");
    { TernaryRMSNorm *n = ternary_rmsnorm_alloc(2, 9, 1e-5f); int ok = n && n->gain->scale == 1.0f;
      for (size_t j = 0; ok && j < 9; j++) ok = ternary_dense_get_weight(n->gain, 0, j) == 1;
      CHECK(ok, "estado inicial: ganancia +1 en todo y escala 1"); ternary_rmsnorm_free(n); }
    printf("[alloc] rows/dim/eps invalidos rechazados; inicio = ganancia +1, escala 1\n");

    /* ---- 1. forward vs referencia ---- */
    static const size_t shapes[][2] = { {1,1}, {1,7}, {3,64}, {8,257}, {2,4096}, {5,33} };
    double worst_rel = 0; long nvals = 0;
    for (size_t si = 0; si < sizeof(shapes)/sizeof(shapes[0]); si++) {
        size_t rows = shapes[si][0], dim = shapes[si][1];
        for (int trial = 0; trial < 60; trial++) {
            TernaryRMSNorm *n = ternary_rmsnorm_alloc(rows, dim, 1e-5f);
            double *g = malloc(dim * sizeof(double)), *x = malloc(rows*dim*sizeof(double)), *y = malloc(rows*dim*sizeof(double));
            float *xf = malloc(rows*dim*sizeof(float)), *of = malloc(rows*dim*sizeof(float));
            random_gain(g, dim); set_gain(n, g);
            n->gain->scale = (float)(0.05 + urand() * 3.0);
            for (size_t i = 0; i < rows; i++) {
                double mag = pow(10.0, -3.0 + 6.0 * urand());
                for (size_t j = 0; j < dim; j++) { xf[i*dim+j] = (float)(nrand() * mag); x[i*dim+j] = xf[i*dim+j]; }
            }
            ternary_rmsnorm_forward(n, xf, of);
            ref_forward(x, g, (double)n->gain->scale, rows, dim, (double)n->eps, y);
            for (size_t i = 0; i < rows; i++) {
                double mx = 0; for (size_t j = 0; j < dim; j++) mx = fmax(mx, fabs(y[i*dim+j]));
                for (size_t j = 0; j < dim; j++) { double e = fabs((double)of[i*dim+j] - y[i*dim+j]) / (mx + 1e-30); if (e > worst_rel) worst_rel = e; nvals++; }
            }
            free(g); free(x); free(y); free(xf); free(of); ternary_rmsnorm_free(n);
        }
    }
    printf("[forward] %ld valores vs referencia double (formas hasta 2x4096, magnitudes 1e-3..1e3): error relativo max %.2e\n", nvals, worst_rel);
    CHECK(worst_rel < 5e-6, "forward se aparta de la referencia: %.3e", worst_rel);

    /* ---- 2. propiedades ---- */
    {
        const size_t rows = 4, dim = 48;
        TernaryRMSNorm *n = ternary_rmsnorm_alloc(rows, dim, 1e-12f);
        n->gain->scale = 1.7f;
        float x[4*48], cx[4*48], o1[4*48], o2[4*48];
        for (size_t k = 0; k < rows*dim; k++) x[k] = (float)(nrand() * 0.8 + 0.3);
        ternary_rmsnorm_forward(n, x, o1);
        double worst_rms = 0;
        for (size_t i = 0; i < rows; i++) { double ms = 0; for (size_t j = 0; j < dim; j++) ms += (double)o1[i*dim+j]*o1[i*dim+j]; worst_rms = fmax(worst_rms, fabs(sqrt(ms/dim) - 1.7)); }
        printf("[propiedad] RMS de cada fila de salida con g=+1 vs escala 1.7: desvio max %.2e\n", worst_rms);
        CHECK(worst_rms < 2e-5, "RMS de la salida != escala");
        double worst_inv = 0;
        static const double cs[] = { 1e-2, 0.5, 7.0, 300.0 };
        for (int c = 0; c < 4; c++) {
            for (size_t k = 0; k < rows*dim; k++) cx[k] = (float)(x[k] * cs[c]);
            ternary_rmsnorm_forward(n, cx, o2);
            for (size_t k = 0; k < rows*dim; k++) worst_inv = fmax(worst_inv, fabs((double)o2[k] - (double)o1[k]));
        }
        printf("[propiedad] invariancia de escala out(c*x) == out(x), c en {1e-2, 0.5, 7, 300}: diferencia max %.2e\n", worst_inv);
        CHECK(worst_inv < 5e-5, "no es invariante a la escala: %.3e", worst_inv);
        /* gain -1 invierte, 0 anula */
        for (size_t j = 0; j < dim; j++) ternary_dense_set_weight(n->gain, 0, j, (j % 3 == 0) ? -1 : ((j % 3 == 1) ? 0 : 1));
        ternary_rmsnorm_forward(n, x, o2);
        int ok = 1;
        for (size_t i = 0; i < rows; i++) for (size_t j = 0; j < dim; j++) {
            float a = o1[i*dim+j], b = o2[i*dim+j];
            if (j % 3 == 0) ok &= (b == -a); else if (j % 3 == 1) ok &= (b == 0.0f); else ok &= (b == a);
        }
        CHECK(ok, "g=-1 no invierte exactamente / g=0 no anula / g=+1 cambia");
        printf("[propiedad] g=-1 invierte el signo exacto, g=0 anula, g=+1 intacto: %s\n", ok ? "OK" : "FALLA");
        for (size_t j = 0; j < dim; j++) ternary_dense_set_weight(n->gain, 0, j, 1);
        /* filas independientes */
        float xb[4*48]; memcpy(xb, x, sizeof(x));
        for (size_t j = 0; j < dim; j++) xb[1*dim+j] *= 9.0f;
        ternary_rmsnorm_forward(n, xb, o2);
        int indep = (memcmp(o1, o2, dim * sizeof(float)) == 0) && (memcmp(o1 + 2*dim, o2 + 2*dim, 2*dim*sizeof(float)) == 0);
        CHECK(indep, "cambiar la fila 1 altero otras filas");
        printf("[propiedad] filas independientes (cambiar la fila 1 no toca las demas): %s\n", indep ? "OK" : "FALLA");
        /* fila de ceros: finita y cero */
        float z[4*48] = {0}; ternary_rmsnorm_forward(n, z, o2);
        int zfin = 1; for (size_t k = 0; k < rows*dim; k++) zfin &= (o2[k] == 0.0f);
        CHECK(zfin, "fila de ceros debia dar ceros finitos");
        { TernaryRMSNorm *n5 = ternary_rmsnorm_alloc(1, 8, 1e-5f); float z8[8] = {0}, o8[8]; ternary_rmsnorm_forward(n5, z8, o8);
          int f8 = 1; for (int j = 0; j < 8; j++) f8 &= isfinite(o8[j]) && o8[j] == 0.0f; CHECK(f8, "ceros con eps=1e-5"); ternary_rmsnorm_free(n5); }
        /* in situ */
        float inpl[4*48]; memcpy(inpl, x, sizeof(x)); ternary_rmsnorm_forward(n, inpl, inpl);
        ternary_rmsnorm_forward(n, x, o2);
        CHECK(memcmp(inpl, o2, sizeof(inpl)) == 0, "forward in situ distinto del fuera de lugar");
        printf("[propiedad] fila de ceros -> ceros finitos; forward in situ identico: OK\n");
        ternary_rmsnorm_free(n);
    }

    /* ---- 3. backward vs diferencias finitas centrales (referencia double) ---- */
    {
        double worst_x = 0, worst_g = 0, worst_s = 0; int n_checked = 0;
        static const size_t sh[][2] = { {1,5}, {3,16}, {2,33} };
        for (size_t si = 0; si < 3; si++) {
            size_t rows = sh[si][0], dim = sh[si][1];
            for (int trial = 0; trial < 20; trial++) {
                TernaryRMSNorm *n = ternary_rmsnorm_alloc(rows, dim, 1e-5f);
                double *g = malloc(dim*sizeof(double)), *x = malloc(rows*dim*sizeof(double)), *go = malloc(rows*dim*sizeof(double));
                float *xf = malloc(rows*dim*sizeof(float)), *gof = malloc(rows*dim*sizeof(float));
                float *gx = calloc(rows*dim, sizeof(float)), *gg = malloc(dim*sizeof(float)), gs = 0;
                random_gain(g, dim); set_gain(n, g); n->gain->scale = (float)(0.2 + urand() * 2.0);
                double s = (double)n->gain->scale;
                for (size_t k = 0; k < rows*dim; k++) { xf[k] = (float)(nrand() * (0.3 + 3.0 * urand())); x[k] = xf[k]; go[k] = nrand(); gof[k] = (float)go[k]; go[k] = gof[k]; }
                ternary_rmsnorm_backward(n, xf, gof, gg, gx, &gs);
                double eps_ = (double)n->eps, h = 1e-6;
                double scale_ref = 0;
                for (size_t k = 0; k < rows*dim; k++) {
                    double xp = x[k], p, m; x[k] = xp + h; p = ref_loss(x, g, s, rows, dim, eps_, go); x[k] = xp - h; m = ref_loss(x, g, s, rows, dim, eps_, go); x[k] = xp;
                    double num = (p - m) / (2*h); double den = fabs(num) + 1e-3;
                    worst_x = fmax(worst_x, fabs(num - (double)gx[k]) / den); n_checked++;
                }
                for (size_t j = 0; j < dim; j++) {
                    double gp = g[j], p, m; g[j] = gp + h; p = ref_loss(x, g, s, rows, dim, eps_, go); g[j] = gp - h; m = ref_loss(x, g, s, rows, dim, eps_, go); g[j] = gp;
                    double num = (p - m) / (2*h); worst_g = fmax(worst_g, fabs(num - (double)gg[j]) / (fabs(num) + 1e-3)); n_checked++;
                }
                { double p = ref_loss(x, g, s + h, rows, dim, eps_, go), m = ref_loss(x, g, s - h, rows, dim, eps_, go); scale_ref = (p - m) / (2*h);
                  worst_s = fmax(worst_s, fabs(scale_ref - (double)gs) / (fabs(scale_ref) + 1e-3)); n_checked++; }
                free(g); free(x); free(go); free(xf); free(gof); free(gx); free(gg); ternary_rmsnorm_free(n);
            }
        }
        printf("[backward] %d derivadas vs diferencias finitas centrales (referencia double): error relativo max  x %.2e | ganancia %.2e | escala %.2e\n", n_checked, worst_x, worst_g, worst_s);
        CHECK(worst_x < 2e-3 && worst_g < 2e-3 && worst_s < 2e-3, "gradientes no coinciden (x %.3e, g %.3e, s %.3e)", worst_x, worst_g, worst_s);
    }

    /* ---- 4. contrato de acumulacion ---- */
    {
        const size_t rows = 3, dim = 11;
        TernaryRMSNorm *n = ternary_rmsnorm_alloc(rows, dim, 1e-5f);
        double g[11]; random_gain(g, dim); set_gain(n, g); n->gain->scale = 0.8f;
        float x[33], go[33], gx0[33] = {0}, gg0[11], gs0 = 0;
        for (int k = 0; k < 33; k++) { x[k] = (float)nrand(); go[k] = (float)nrand(); }
        ternary_rmsnorm_backward(n, x, go, gg0, gx0, &gs0);
        float gx1[33], gg1[11], gs1 = 5.0f;
        for (int k = 0; k < 33; k++) gx1[k] = 1.0f;
        for (int j = 0; j < 11; j++) gg1[j] = 777.0f;   /* basura: debe sobreescribirse */
        ternary_rmsnorm_backward(n, x, go, gg1, gx1, &gs1);
        double ex = 0, eg = 0;
        for (int k = 0; k < 33; k++) ex = fmax(ex, fabs((double)gx1[k] - 1.0 - (double)gx0[k]));
        for (int j = 0; j < 11; j++) eg = fmax(eg, fabs((double)gg1[j] - (double)gg0[j]));
        CHECK(ex < 1e-5, "grad_x no acumula (dif %.2e)", ex);
        CHECK(eg == 0.0, "grad_gain no se sobreescribe (dif %.2e)", eg);
        CHECK(fabs((double)gs1 - 5.0 - (double)gs0) < 1e-5, "grad_scale no acumula");
        float gg2[11]; ternary_rmsnorm_backward(n, x, go, gg2, NULL, NULL);   /* NULL permitido */
        CHECK(memcmp(gg2, gg0, sizeof(gg0)) == 0, "con grad_x y grad_scale NULL cambia la ganancia");
        printf("[contrato] grad_x y grad_scale acumulan, grad_gain se sobreescribe, NULL permitido: OK\n");
        ternary_rmsnorm_free(n);
    }

    printf("fallas: %d\n%s\n", fails, fails ? "test_rmsnorm: FALLO" : "test_rmsnorm: TODO OK");
    return fails ? 1 : 0;
}
