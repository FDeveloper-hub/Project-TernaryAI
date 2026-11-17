/* SwiGLU con entrada uint8: forward contra una referencia en double calculada desde las definiciones (Z enteros
 * por el producto de referencia, silu con exp de libm, cuantizacion compartida) y backward contra la referencia
 * del estimador directo escrita aparte en double (grad_w, grad_x, escalas). Las dos compuertas. */
#include "ternary_swiglu.h"
#include "ternary_u8.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fallas = 0;
#define CHECK(c, ...) do { if (!(c)) { fallas++; printf("FALLO (linea %d): ", __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)
static uint32_t rs = 777u;
static uint32_t rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }
static float rndf(void) { return ((float)(rnd() % 20001) / 10000.0f) - 1.0f; }

static void weights(const TernaryDense *d, int8_t *w) {
    for (size_t o = 0; o < d->output_dim; o++) for (size_t i = 0; i < d->input_dim; i++) w[o * d->input_dim + i] = (int8_t)ternary_dense_get_weight(d, o, i);
}
static double silu_d(double a) { return a / (1.0 + exp(-a)); }
static double dsilu_d(double a) { double s = 1.0 / (1.0 + exp(-a)); return s * (1.0 + a * (1.0 - s)); }

static void caso(size_t rows, size_t dim, size_t hidden, int mode, unsigned seed) {
    TernarySwiGLU *s = ternary_swiglu_alloc(rows, dim, hidden, mode, 0.5f);
    rs = seed;
    TernaryDense *m[3] = { s->wg, s->wu, s->wd };
    for (int k = 0; k < 3; k++) for (size_t o = 0; o < m[k]->output_dim; o++) for (size_t i = 0; i < m[k]->input_dim; i++) ternary_dense_set_weight(m[k], o, i, (int)(rnd() % 3) - 1);
    s->wg->scale = 0.03f + 0.01f * (float)(rnd() % 5); s->wu->scale = 0.05f; s->wd->scale = 0.07f;
    float *xf = malloc(rows * dim * sizeof(float));
    for (size_t i = 0; i < rows * dim; i++) xf[i] = rndf() * 3.0f;
    TernaryU8 *x = ternary_u8_alloc(rows, dim);
    ternary_u8_quantize(x, xf);
    float *out = malloc(rows * dim * sizeof(float));
    ternary_swiglu_forward_u8(s, x, out, NULL);

    int8_t *W[3];
    for (int k = 0; k < 3; k++) { W[k] = malloc(m[k]->output_dim * m[k]->input_dim); weights(m[k], W[k]); }
    double maxo = 1e-9, maxe = 0;
    double *ref = malloc(rows * dim * sizeof(double));
    for (size_t p = 0; p < rows; p++) {
        const uint8_t *q = x->q + p * dim; const float sx = x->scale[p]; const int zp = x->zp[p];
        double *h = malloc(hidden * sizeof(double));
        for (size_t j = 0; j < hidden; j++) {
            int32_t zg = ternary_dot_u8_reference(W[0] + j * dim, dim, q, zp), zu = ternary_dot_u8_reference(W[1] + j * dim, dim, q, zp);
            double a = (double)s->wg->scale * sx * zg, u = (double)s->wu->scale * sx * zu;
            h[j] = mode == 0 ? silu_d(a) * u : (zg > 0 ? 1.0 : zg < 0 ? -1.0 : 0.0) * u;
        }
        double hmax = 1e-12;
        for (size_t j = 0; j < hidden; j++) if (fabs(h[j]) > hmax) hmax = fabs(h[j]);
        for (size_t j = 0; j < hidden; j++) {
            double dq = (double)s->hq.scale[p] * ((int)s->hq.q[p * hidden + j] - s->hq.zp[p]);
            CHECK(fabs(dq - h[j]) <= 0.5001 * s->hq.scale[p] + 1e-5 * hmax, "h fila %zu j=%zu: %.6g vs %.6g", p, j, dq, h[j]);
        }
        for (size_t d = 0; d < dim; d++) {
            int32_t z = ternary_dot_u8_reference(W[2] + d * hidden, hidden, s->hq.q + p * hidden, s->hq.zp[p]);
            ref[p * dim + d] = (double)s->wd->scale * s->hq.scale[p] * z;
        }
        free(h);
    }
    for (size_t i = 0; i < rows * dim; i++) { if (fabs(ref[i]) > maxo) maxo = fabs(ref[i]); double e = fabs(ref[i] - out[i]); if (e > maxe) maxe = e; }
    CHECK(maxe <= 1e-5 * maxo + 1e-9, "forward: error %.3g vs max %.3g", maxe, maxo);

    float *go = malloc(rows * dim * sizeof(float));
    for (size_t i = 0; i < rows * dim; i++) go[i] = rndf();
    float *gw0 = malloc(hidden * dim * 4), *gw1 = malloc(hidden * dim * 4), *gw2 = malloc(dim * hidden * 4);
    float *gwp[3] = { gw0, gw1, gw2 };
    float *gx = calloc(rows * dim, sizeof(float)); float gsc[3] = { 0, 0, 0 };
    ternary_swiglu_backward_u8(s, x, go, gwp, gx, gsc);
    double *rw0 = calloc(hidden * dim, 8), *rw1 = calloc(hidden * dim, 8), *rw2 = calloc(dim * hidden, 8), *rgx = calloc(rows * dim, 8);
    double rg = 0, rd = 0;
    for (size_t p = 0; p < rows; p++) {
        const uint8_t *q = x->q + p * dim; const double sx = x->scale[p]; const int zp = x->zp[p];
        double *gt = calloc(hidden, 8);
        for (size_t d = 0; d < dim; d++) {
            int32_t z = ternary_dot_u8_reference(W[2] + d * hidden, hidden, s->hq.q + p * hidden, s->hq.zp[p]);
            double g = go[p * dim + d];
            rd += g * s->hq.scale[p] * z;
            for (size_t j = 0; j < hidden; j++) {
                double hd = (double)s->hq.scale[p] * ((int)s->hq.q[p * hidden + j] - s->hq.zp[p]);
                rw2[d * hidden + j] += g * hd;
                gt[j] += g * s->wd->scale * W[2][d * hidden + j];
            }
        }
        for (size_t j = 0; j < hidden; j++) {
            int32_t zg = ternary_dot_u8_reference(W[0] + j * dim, dim, q, zp), zu = ternary_dot_u8_reference(W[1] + j * dim, dim, q, zp);
            double a = (double)s->wg->scale * sx * zg, u = (double)s->wu->scale * sx * zu;
            double ga, gu;
            if (mode == 0) { gu = gt[j] * silu_d(a); ga = gt[j] * u * dsilu_d(a); }
            else { gu = gt[j] * (zg > 0 ? 1 : zg < 0 ? -1 : 0); ga = gt[j] * u; }
            if (mode == 0) rg += ga * sx * zg;
            for (size_t i = 0; i < dim; i++) {
                double xd = sx * ((int)q[i] - zp);
                rw0[j * dim + i] += ga * xd; rw1[j * dim + i] += gu * xd;
                rgx[p * dim + i] += (ga * s->wg->scale * W[0][j * dim + i]) + (gu * s->wu->scale * W[1][j * dim + i]);
            }
        }
        free(gt);
    }
    double mg = 1e-9;
    for (size_t i = 0; i < hidden * dim; i++) { if (fabs(rw0[i]) > mg) mg = fabs(rw0[i]); if (fabs(rw1[i]) > mg) mg = fabs(rw1[i]); if (fabs(rw2[i]) > mg) mg = fabs(rw2[i]); }
    double e0 = 0, e1 = 0, e2 = 0, ex = 0, mx = 1e-9;
    for (size_t i = 0; i < hidden * dim; i++) { e0 = fmax(e0, fabs(rw0[i] - gw0[i])); e1 = fmax(e1, fabs(rw1[i] - gw1[i])); e2 = fmax(e2, fabs(rw2[i] - gw2[i])); }
    for (size_t i = 0; i < rows * dim; i++) { ex = fmax(ex, fabs(rgx[i] - gx[i])); mx = fmax(mx, fabs(rgx[i])); }
    CHECK(e0 <= 1e-4 * mg && e1 <= 1e-4 * mg && e2 <= 1e-4 * mg, "grad_w: %.3g %.3g %.3g (max %.3g)", e0, e1, e2, mg);
    CHECK(ex <= 1e-4 * mx, "grad_x: %.3g (max %.3g)", ex, mx);
    CHECK(fabs(rd - gsc[2]) <= 1e-4 * (fabs(rd) + 1e-6), "escala down: %.6g vs %.6g", rd, gsc[2]);
    CHECK(fabs((mode == 0 ? rg : 0.0) - gsc[0]) <= 1e-4 * (fabs(rg) + 1e-6), "escala gate: %.6g vs %.6g", rg, gsc[0]);
    CHECK(gsc[1] == 0.0f, "escala up deberia quedar en 0");
    printf("  modo %d rows=%zu dim=%zu hidden=%zu: fwd err %.2g, grad err %.2g/%.2g/%.2g x %.2g\n", mode, rows, dim, hidden, maxe / maxo, e0 / mg, e1 / mg, e2 / mg, ex / mx);
    for (int k = 0; k < 3; k++) free(W[k]);
    free(xf); free(out); free(ref); free(go); free(gw0); free(gw1); free(gw2); free(gx); free(rw0); free(rw1); free(rw2); free(rgx);
    ternary_u8_free(x); ternary_swiglu_free(s);
}

int main(void) {
    printf("-- SwiGLU uint8\n");
    for (int mode = 0; mode < 2; mode++) {
        caso(1, 16, 24, mode, 1);
        caso(5, 40, 70, mode, 2);
        caso(7, 64, 130, mode, 3);
    }
    if (fallas) { printf("%d fallas\n", fallas); return 1; }
    printf("OK\n");
    return 0;
}
