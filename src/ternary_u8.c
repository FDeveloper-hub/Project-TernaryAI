#include "ternary_mem.h"
#include "ternary_threadpool.h"
#include "ternary_avx512.h"
#include "ternary_neon_policy.h"
#include "ternary_u8.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

TernaryU8 *ternary_u8_alloc(size_t rows, size_t row_len) {
    if (rows == 0 || row_len == 0) return NULL;
    TernaryU8 *t = (TernaryU8 *)ternary_calloc64(1, sizeof(TernaryU8));
    if (!t) return NULL;
    t->rows = rows; t->row_len = row_len;
    t->q = (uint8_t *)ternary_calloc64(rows * row_len, 1);
    t->scale = (float *)ternary_malloc64(rows * sizeof(float));
    t->zp = (int32_t *)ternary_calloc64(rows, sizeof(int32_t));
    if (!t->q || !t->scale || !t->zp) { ternary_u8_free(t); return NULL; }
    for (size_t r = 0; r < rows; r++) t->scale[r] = 1.0f;
    return t;
}

void ternary_u8_free(TernaryU8 *t) {
    if (!t) return;
    free(t->q); free(t->scale); free(t->zp); free(t);
}

typedef struct { TernaryU8 *dst; const float *x; } U8Q;
static void u8q_range(size_t r0, size_t r1, void *c_) {
    TernaryU8 *dst = ((U8Q *)c_)->dst; const float *x = ((U8Q *)c_)->x;
    const size_t n = dst->row_len;
    if (ternary_avx512_enabled()) {
        for (size_t r = r0; r < r1; r++) ternary_avx512_u8_quantize_row(x + r * n, n, dst->q + r * n, dst->scale + r, dst->zp + r);
        return;
    }
    if (TN_ON(QUANT)) {
        for (size_t r = r0; r < r1; r++) ternary_neon_u8_quantize_row(x + r * n, n, dst->q + r * n, dst->scale + r, dst->zp + r);
        return;
    }
    for (size_t r = r0; r < r1; r++) {
        const float *xr = x + r * n;
        uint8_t *qr = dst->q + r * n;
        float lo = 0.0f, hi = 0.0f;                 /* el rango incluye el 0 */
        for (size_t i = 0; i < n; i++) {
            const float v = xr[i];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        float sc = (hi - lo) / 255.0f;
        if (!(sc > 0.0f) || !isfinite(sc)) {         /* fila de ceros (o no finita): todo en el zero-point 0 */
            dst->scale[r] = 1.0f; dst->zp[r] = 0;
            memset(qr, 0, n);
            continue;
        }
        int zp = (int)rintf(-lo / sc);
        if (zp < 0) zp = 0;
        if (zp > 255) zp = 255;
        dst->scale[r] = sc; dst->zp[r] = zp;
        const float inv = 1.0f / sc;
        for (size_t i = 0; i < n; i++) {
            float qf = rintf(xr[i] * inv) + (float)zp;
            if (qf < 0.0f) qf = 0.0f;
            if (qf > 255.0f) qf = 255.0f;
            qr[i] = (uint8_t)qf;
        }
    }
}

/* Cada fila se cuantiza por su cuenta: se reparten filas en hilos (mismo resultado con cualquier reparto). */
void ternary_u8_quantize(TernaryU8 *dst, const float *x) {
    U8Q c = { dst, x };
    size_t grain = dst->row_len ? 16384 / dst->row_len : 0;
    if (grain < 1) grain = 1;
    ternary_parallel_ranges(ternary_get_compute_pool(), 0, dst->rows, grain, u8q_range, &c);
}

void ternary_u8_dequant(const TernaryU8 *t, float *out) {
    const size_t n = t->row_len;
    if (ternary_avx512_enabled()) {
        for (size_t r = 0; r < t->rows; r++) ternary_avx512_u8_to_f32(t->q + r * n, n, t->scale[r], t->zp[r], out + r * n);
        return;
    }
    if (ternary_neon_enabled()) {
        for (size_t r = 0; r < t->rows; r++) ternary_neon_u8_to_f32(t->q + r * n, n, t->scale[r], t->zp[r], out + r * n);
        return;
    }
    for (size_t r = 0; r < t->rows; r++) {
        const float sc = t->scale[r];
        const int32_t zp = t->zp[r];
        for (size_t i = 0; i < n; i++) out[r * n + i] = sc * (float)((int32_t)t->q[r * n + i] - zp);
    }
}

int32_t ternary_dot_u8_scalar(const uint64_t *wm, const uint64_t *ws, size_t n, const uint8_t *q, int32_t zp) {
    const size_t nw = (n + 63) / 64;
    int32_t sum_pos = 0, sum_neg = 0, npos = 0, nneg = 0;
    for (size_t k = 0; k < nw; k++) {
        uint64_t pos = wm[k] & ~ws[k], neg = wm[k] & ws[k];
        npos += __builtin_popcountll(pos);
        nneg += __builtin_popcountll(neg);
        const uint8_t *qk = q + 64 * k;
        while (pos) { sum_pos += qk[__builtin_ctzll(pos)]; pos &= pos - 1; }
        while (neg) { sum_neg += qk[__builtin_ctzll(neg)]; neg &= neg - 1; }
    }
    return sum_pos - sum_neg - zp * (npos - nneg);
}

int32_t ternary_dot_u8_reference(const int8_t *w, size_t n, const uint8_t *q, int32_t zp) {
    int32_t acc = 0;
    for (size_t i = 0; i < n; i++) acc += (int32_t)w[i] * ((int32_t)q[i] - zp);
    return acc;
}
