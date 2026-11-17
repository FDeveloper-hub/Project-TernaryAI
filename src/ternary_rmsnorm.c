#include "ternary_mem.h"
#include "ternary_rmsnorm.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "ternary_threadpool.h"
#include "ternary_avx512.h"
#include "ternary_neon_policy.h"
#include "ternary_fexp.h"

TernaryRMSNorm *ternary_rmsnorm_alloc(size_t rows, size_t dim, float eps) {
    if (rows == 0 || dim == 0 || !(eps > 0.0f)) return NULL;
    TernaryRMSNorm *n = (TernaryRMSNorm *)ternary_calloc64(1, sizeof(*n));
    if (!n) return NULL;
    n->rows = rows; n->dim = dim; n->eps = eps;
    n->gain = ternary_dense_alloc(dim, 1, 0);
    if (!n->gain) { free(n); return NULL; }
    for (size_t j = 0; j < dim; j++) ternary_dense_set_weight(n->gain, 0, j, 1);
    return n;
}

void ternary_rmsnorm_free(TernaryRMSNorm *n) {
    if (!n) return;
    ternary_dense_free(n->gain);
    free(n);
}

/* 1/sqrt(mean(x^2)+eps) de una fila, con la suma en double. */
static float row_inv_rms(const float *x, size_t dim, float eps) {
    const double ss = ternary_canon_sumsq(x, dim);   /* orden canonico: mismos bits que AVX-512 */
    return (float)(1.0 / sqrt(ss / (double)dim + (double)eps));
}

/* ---- repartido en hilos ----
 * Forward: las filas son independientes. Backward: por bloques de RMS_BLK filas; fase 1 por fila (grad_x, termino de escala, y el aporte
 * de la fila a la ganancia en un buffer del bloque) y fase 2 por COLUMNAS (suma del aporte de las filas en orden ascendente), asi
 * el orden de las sumas es el de siempre y el resultado no depende del reparto. */
#define RMS_BLK 128
#define RMS_ROW_GRAIN 8

typedef struct { const TernaryRMSNorm *n; const float *x; float *out; const uint16_t *pm; size_t nt; } RmsF;
static void rms_fwd_range(size_t i0, size_t i1, void *c_) {
    RmsF *c = (RmsF *)c_;
    const TernaryRMSNorm *n = c->n; const size_t D = n->dim;
    if (c->pm) {
        if (ternary_avx512_enabled()) ternary_avx512_rmsnorm_forward(i1 - i0, D, n->eps, n->gain->scale, c->pm, c->pm + c->nt, c->x + i0 * D, c->out + i0 * D);
        else ternary_neon_rmsnorm_forward(i1 - i0, D, n->eps, n->gain->scale, c->pm, c->pm + c->nt, c->x + i0 * D, c->out + i0 * D);
        return;
    }
    for (size_t i = i0; i < i1; i++) {
        const float *xi = c->x + i * D;
        float *oi = c->out + i * D;
        const float f = n->gain->scale * row_inv_rms(xi, D, n->eps);
        for (size_t j = 0; j < D; j++) {
            int g = ternary_vec_get(n->gain->W, j);   /* fila 0 de la matriz 1 x dim */
            oi[j] = (g == 0) ? 0.0f : (g > 0 ? f * xi[j] : -f * xi[j]);
        }
    }
}

void ternary_rmsnorm_forward(const TernaryRMSNorm *n, const float *x, float *out) {
    const size_t D = n->dim;
    uint16_t *pm = NULL;
    const size_t nt = (D + 15) / 16;
    if (TN_VEC(RMS)) {
        pm = (uint16_t *)ternary_malloc64(2 * nt * sizeof(uint16_t));
        if (pm) ternary_avx512_split_planes(pm, pm + nt, D, n->gain->W->mask, n->gain->W->sign);
    }
    RmsF c = { n, x, out, pm, nt };
    size_t grain = 16384 / D; if (grain < 1) grain = 1;
    ternary_parallel_ranges(ternary_get_compute_pool(), 0, n->rows, grain, rms_fwd_range, &c);
    free(pm);
}

typedef struct {
    const TernaryRMSNorm *n; const float *x, *go; float *gx, *gg; double *gs; size_t b0; const uint16_t *pm; size_t nt;
} RmsB;
static void rms_bwd_rows(size_t i0, size_t i1, void *c_) {
    RmsB *c = (RmsB *)c_;
    const TernaryRMSNorm *n = c->n; const size_t D = n->dim; const float s = n->gain->scale;
    float *tmp = c->pm ? (float *)ternary_calloc64(D, sizeof(float)) : NULL;
    for (size_t k = i0; k < i1; k++) {
        const size_t i = c->b0 + k;
        const float *xi = c->x + i * D, *go = c->go + i * D;
        float *gg = c->gg + k * D;
        if (c->pm) {
            memset(tmp, 0, D * sizeof(float));
            double g1 = 0.0;
            if (ternary_avx512_enabled()) ternary_avx512_rmsnorm_backward(1, D, n->eps, s, c->pm, c->pm + c->nt, xi, go, tmp, c->gx ? c->gx + i * D : NULL, &g1);
            else ternary_neon_rmsnorm_backward(1, D, n->eps, s, c->pm, c->pm + c->nt, xi, go, tmp, c->gx ? c->gx + i * D : NULL, &g1);
            memcpy(gg, tmp, D * sizeof(float));
            c->gs[k] = g1;
            continue;
        }
        const float r = row_inv_rms(xi, D, n->eps);
        double dl[16] = {0};
        for (size_t j = 0; j < D; j++) {
            int g = ternary_vec_get(n->gain->W, j);
            dl[j & 15] += (double)g * (double)go[j] * (double)xi[j];
            gg[j] = go[j] * s * xi[j] * r;
        }
        const double dot = ternary_canon_reduce_d16(dl);
        c->gs[k] = dot * (double)r;                      /* dL/ds = sum_j go_j*g_j*x_j*r */
        if (c->gx) {
            float *gx = c->gx + i * D;
            const float cc = (float)(dot * (double)r * (double)r / (double)D);   /* r^2*dot/D */
            for (size_t kk = 0; kk < D; kk++) {
                int g = ternary_vec_get(n->gain->W, kk);
                gx[kk] += s * r * ((float)g * go[kk] - xi[kk] * cc);
            }
        }
    }
    free(tmp);
}
typedef struct { float *gain; const float *gg; size_t P, D; } RmsCols;
static void rms_bwd_cols(size_t j0, size_t j1, void *c_) {
    RmsCols *c = (RmsCols *)c_;
    for (size_t p = 0; p < c->P; p++) {
        const float *row = c->gg + p * c->D;
        for (size_t j = j0; j < j1; j++) c->gain[j] += row[j];
    }
}

static void rms_cols_blocks(size_t b0, size_t b1, void *c_) {
    RmsCols *c = (RmsCols *)c_;
    size_t j1 = b1 * 16; if (j1 > c->D) j1 = c->D;
    rms_bwd_cols(b0 * 16, j1, c_);
}

void ternary_rmsnorm_backward(const TernaryRMSNorm *n, const float *x, const float *grad_out,
                              float *grad_gain_out, float *grad_x_out, float *grad_scale_out) {
    const size_t D = n->dim, R = n->rows;
    for (size_t j = 0; j < D; j++) grad_gain_out[j] = 0.0f;
    uint16_t *pm = NULL;
    const size_t nt = (D + 15) / 16;
    if (TN_VEC(RMS)) {
        pm = (uint16_t *)ternary_malloc64(2 * nt * sizeof(uint16_t));
        if (pm) ternary_avx512_split_planes(pm, pm + nt, D, n->gain->W->mask, n->gain->W->sign);
    }
    const size_t K = R < RMS_BLK ? R : RMS_BLK;
    float *gg = (float *)ternary_malloc64(K * D * sizeof(float));
    double *gs = (double *)ternary_malloc64(K * sizeof(double));
    if (!gg || !gs) abort();
    TernaryThreadPool *pool = ternary_get_compute_pool();
    RmsB B = { n, x, grad_out, grad_x_out, gg, gs, 0, pm, nt };
    RmsCols C = { grad_gain_out, gg, 0, D };
    double gscale = 0.0;
    for (size_t b0 = 0; b0 < R; b0 += RMS_BLK) {
        const size_t P = (R - b0 < RMS_BLK) ? (R - b0) : RMS_BLK;
        B.b0 = b0; C.P = P;
        size_t grain = 16384 / D; if (grain < 1) grain = 1;
        ternary_parallel_ranges(pool, 0, P, grain, rms_bwd_rows, &B);
        ternary_parallel_ranges(pool, 0, (D + 15) / 16, 4, rms_cols_blocks, &C);
        for (size_t k = 0; k < P; k++) gscale += gs[k];
    }
    free(gg); free(gs); free(pm);
    if (grad_scale_out) *grad_scale_out += (float)gscale;
}
