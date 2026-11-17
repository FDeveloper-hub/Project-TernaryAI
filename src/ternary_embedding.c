#include "ternary_mem.h"
#include "ternary_avx512.h"
#include "ternary_fexp.h"
#include "ternary_embedding.h"
#include "ternary_dense.h"   /* ternary_r2_update_row */
#include <math.h>
#include <stdlib.h>
#include "ternary_threadpool.h"
#include <string.h>

TernaryEmbedding *ternary_embedding_alloc(size_t vocab_size, size_t embed_dim, size_t seq_len) {
    if (vocab_size == 0 || embed_dim == 0 || seq_len == 0) return NULL;
    TernaryEmbedding *l = (TernaryEmbedding *)ternary_malloc64(sizeof(TernaryEmbedding));
    if (!l) return NULL;
    l->vocab_size = vocab_size;
    l->embed_dim = embed_dim;
    l->seq_len = seq_len;
    l->scale = 1.0f;
    l->W = ternary_vec_alloc(vocab_size * embed_dim);
    l->counter_W = (int8_t *)ternary_calloc64(vocab_size * embed_dim, sizeof(int8_t));
    if (!l->W || !l->counter_W) { ternary_embedding_free(l); return NULL; }
    return l;
}

void ternary_embedding_free(TernaryEmbedding *l) {
    if (!l) return;
    if (l->W) ternary_vec_free(l->W);
    free(l->counter_W);
    free(l);
}

int ternary_embedding_get_weight(const TernaryEmbedding *l, size_t v, size_t d) {
    return ternary_vec_get(l->W, v * l->embed_dim + d);
}
void ternary_embedding_set_weight(TernaryEmbedding *l, size_t v, size_t d, int value) {
    ternary_vec_set(l->W, v * l->embed_dim + d, value);
}

/* Repartido en hilos: forward por posicion; backward por bloques de posiciones con (A) terminos de la escala por posicion y (B) suma en gw
 * por FILAS del vocabulario (cada tarea recorre las posiciones en orden y toma las suyas): mismo orden de sumas que el calculo serial. */
#define EMB_BLK 64
#define EMB_V_GRAIN 32

typedef struct { const TernaryEmbedding *l; const int32_t *ids; float *out; int32_t *raw; } EmbF;
static void emb_fwd_range(size_t p0, size_t p1, void *c_) {
    EmbF *c = (EmbF *)c_;
    const TernaryEmbedding *l = c->l;
    const size_t E = l->embed_dim;
    for (size_t p = p0; p < p1; p++) {
        int32_t id = c->ids[p];
        int valid = id >= 0 && (size_t)id < l->vocab_size;
        if (valid && ternary_avx512_enabled()) {
            ternary_avx512_unpack_row(l->W->mask, l->W->sign, l->vocab_size * E, (size_t)id * E, E, l->scale,
                                      c->out + p * E, c->raw ? c->raw + p * E : NULL);
            continue;
        }
        for (size_t d = 0; d < E; d++) {
            int w = valid ? ternary_vec_get(l->W, (size_t)id * E + d) : 0;
            c->out[p * E + d] = l->scale * (float)w;
            if (c->raw) c->raw[p * E + d] = w;
        }
    }
}

void ternary_embedding_forward(const TernaryEmbedding *l, const int32_t *ids,
                                float *out, int32_t *out_raw_opt) {
    EmbF c = { l, ids, out, out_raw_opt };
    size_t grain = 16384 / l->embed_dim; if (grain < 1) grain = 1;
    ternary_parallel_ranges(ternary_get_compute_pool(), 0, l->seq_len, grain, emb_fwd_range, &c);
}

typedef struct {
    const TernaryEmbedding *l; const int32_t *ids; const float *go; const int32_t *z; float *gw; float *terms; size_t b0, P;
} EmbB;
static void emb_bwd_terms(size_t i0, size_t i1, void *c_) {
    EmbB *c = (EmbB *)c_;
    const TernaryEmbedding *l = c->l;
    const size_t E = l->embed_dim;
    for (size_t k = i0; k < i1; k++) {
        const size_t p = c->b0 + k;
        const int32_t id = c->ids[p];
        const int valid = id >= 0 && (size_t)id < l->vocab_size;
        for (size_t d = 0; d < E; d++) {
            const float g = c->go[p * E + d];
            const int32_t z = c->z ? c->z[p * E + d] : (valid ? ternary_vec_get(l->W, (size_t)id * E + d) : 0);
            c->terms[k * E + d] = g * (float)z;
        }
    }
}
static void emb_bwd_rows(size_t v0, size_t v1, void *c_) {
    EmbB *c = (EmbB *)c_;
    const size_t E = c->l->embed_dim;
    for (size_t k = 0; k < c->P; k++) {
        const int32_t id = c->ids[c->b0 + k];
        if (id < 0 || (size_t)id < v0 || (size_t)id >= v1) continue;
        float *row = c->gw + (size_t)id * E; const float *g = c->go + (c->b0 + k) * E;
        for (size_t d = 0; d < E; d++) row[d] += g[d];
    }
}

void ternary_embedding_backward(const TernaryEmbedding *l, const int32_t *ids,
                                 const float *grad_out, float *grad_w_out, float *grad_scale_out,
                                 const int32_t *z_raw_opt) {
    const size_t E = l->embed_dim, R = l->seq_len;
    ternary_par_zero(grad_w_out, l->vocab_size * E * sizeof(float));
    const size_t K = R < EMB_BLK ? R : EMB_BLK;
    float *terms = (float *)ternary_malloc64(K * E * sizeof(float));
    if (!terms) abort();
    TernaryThreadPool *pool = ternary_get_compute_pool();
    EmbB B = { l, ids, grad_out, z_raw_opt, grad_w_out, terms, 0, 0 };
    float gs = *grad_scale_out;
    for (size_t b0 = 0; b0 < R; b0 += EMB_BLK) {
        const size_t P = (R - b0 < EMB_BLK) ? (R - b0) : EMB_BLK;
        B.b0 = b0; B.P = P;
        size_t grain = 16384 / E; if (grain < 1) grain = 1;
        ternary_parallel_ranges(pool, 0, P, grain, emb_bwd_terms, &B);
        ternary_parallel_ranges(pool, 0, l->vocab_size, EMB_V_GRAIN, emb_bwd_rows, &B);
        for (size_t i = 0; i < P * E; i++) gs += terms[i];     /* orden de siempre: posicion, dimension */
    }
    *grad_scale_out = gs;
    free(terms);
}

void ternary_embedding_apply_row_gradients(TernaryEmbedding *l, size_t row, const float *grad_row) {
    const TernaryUpdateRule rule = ternary_update_rule_get();
    ternary_r2_update_row(l->W, l->counter_W, row * l->embed_dim, grad_row, l->embed_dim, &rule);   /* regla R2 */
}

void ternary_embedding_apply_scale_gradient(TernaryEmbedding *l, float grad_scale, float scale_lr) {
    float s = l->scale - scale_lr * grad_scale;
    if (s < 0.02f) s = 0.02f;   /* mismo piso que Dense/Conv2D */
    l->scale = s;
}
