#include "ternary_mem.h"
#include "ternary_softmax.h"
#include <math.h>
#include <stdlib.h>
#include "ternary_avx512.h"
#include "ternary_neon_policy.h"
#include "ternary_fexp.h"

TernarySoftmax *ternary_softmax_alloc(int32_t max_abs_score, float inv_temp) {
    if (max_abs_score < 0) return NULL;
    TernarySoftmax *s = (TernarySoftmax *)ternary_malloc64(sizeof(TernarySoftmax));
    if (!s) return NULL;
    s->max_abs_score = max_abs_score;
    s->inv_temp = inv_temp;
    s->table_len = 2 * (size_t)max_abs_score + 1;
    s->table = (float *)ternary_malloc64(s->table_len * sizeof(float));
    if (!s->table) { free(s); return NULL; }
    for (size_t k = 0; k < s->table_len; k++)
    {
        /* Entradas < 1e-20 se ponen en 0: su aporte a una suma >= 1 no cambia ni un bit de float, y como
         * float estarian cerca de los denormales (exp(-87..-103)), que en CPU cuestan ~100x por operacion en
         * todo el forward/backward de atencion. */
        float e = (float)exp(-(double)k * (double)inv_temp);
        s->table[k] = (e < 1e-20f) ? 0.0f : e;
    }
    return s;
}

void ternary_softmax_free(TernarySoftmax *s) {
    if (!s) return;
    free(s->table);
    free(s);
}

void ternary_softmax_forward(const TernarySoftmax *s, const int32_t *scores, size_t n,
                              size_t n_valid, float *out) {
    if (n_valid > n) n_valid = n;
    for (size_t i = n_valid; i < n; i++) out[i] = 0.0f;
    if (n_valid == 0) return;

    if (ternary_avx512_enabled()) { ternary_avx512_softmax_fwd(scores, n_valid, s->table, s->table_len, out); return; }
    if (TN_ON(ATTN)) { ternary_neon_softmax_fwd(scores, n_valid, s->table, s->table_len, out); return; }
    int32_t m = scores[0];
    for (size_t i = 1; i < n_valid; i++) if (scores[i] > m) m = scores[i];

    float part[16] = {0};                                  /* suma en 16 carriles (orden canonico, igual que AVX-512) */
    for (size_t i = 0; i < n_valid; i++) {
        int64_t k = (int64_t)m - (int64_t)scores[i];      /* >= 0 */
        if (k >= (int64_t)s->table_len) k = (int64_t)s->table_len - 1;
        float e = s->table[k];
        out[i] = e;
        part[i & 15] += e;
    }
    const float sum = ternary_sum16(part);
    float inv = 1.0f / sum;    /* sum >= 1: el maximo aporta table[0] = 1 */
    for (size_t i = 0; i < n_valid; i++) out[i] *= inv;
}

void ternary_softmax_backward(const TernarySoftmax *s, const float *y, const float *g, size_t n,
                               float *out_ds) {
    if (ternary_avx512_enabled()) { ternary_avx512_softmax_bwd(y, g, n, s->inv_temp, out_ds); return; }
    if (TN_ON(ATTN)) { ternary_neon_softmax_bwd(y, g, n, s->inv_temp, out_ds); return; }
    float part[16] = {0};
    for (size_t j = 0; j < n; j++) part[j & 15] += g[j] * y[j];
    const float dot = ternary_sum16(part);
    for (size_t i = 0; i < n; i++) out_ds[i] = s->inv_temp * y[i] * (g[i] - dot);
}
