#include "ternary_mem.h"
#include "ternary_prof.h"
#include "ternary_head.h"
#include "ternary_neon_policy.h"
#include "ternary_threadpool.h"
#include <stdlib.h>
#include <string.h>
#include "ternary_loss.h"
#include "ternary_avx512.h"
#include "ternary_fexp.h"
#include <math.h>

TernaryHead *ternary_head_alloc(size_t rows, size_t dim, size_t vocab) {
    if (rows == 0 || dim == 0 || vocab == 0 || dim > TERNARY_HEAD_MAX_DIM || vocab > TERNARY_HEAD_MAX_VOCAB) return NULL;
    TernaryHead *h = (TernaryHead *)ternary_calloc64(1, sizeof(*h));
    if (!h) return NULL;
    h->rows = rows; h->dim = dim; h->vocab = vocab;
    h->w = ternary_dense_alloc(dim, vocab, 0);
    h->x_row = ternary_vec_alloc(dim);
    h->xpos = (TernaryVec **)ternary_calloc64(rows, sizeof(TernaryVec *));
    h->gview = (const float **)ternary_calloc64(rows, sizeof(float *));
    h->gxview = (float **)ternary_calloc64(rows, sizeof(float *));
    h->zview = (const int32_t **)ternary_calloc64(rows, sizeof(int32_t *));
    h->oview = (float **)ternary_calloc64(rows, sizeof(float *));
    h->rview = (int32_t **)ternary_calloc64(rows, sizeof(int32_t *));
    h->xcv = (const TernaryVec **)ternary_calloc64(rows, sizeof(TernaryVec *));
    if (!h->w || !h->x_row || !h->xpos || !h->gview || !h->gxview || !h->zview || !h->oview || !h->rview || !h->xcv) {
        ternary_head_free(h); return NULL;
    }
    for (size_t p = 0; p < rows; p++) {
        h->xpos[p] = ternary_vec_alloc(dim);
        if (!h->xpos[p]) { ternary_head_free(h); return NULL; }
        h->xcv[p] = h->xpos[p];
    }
    return h;
}

void ternary_head_free(TernaryHead *h) {
    if (!h) return;
    ternary_dense_free(h->w);
    ternary_vec_free(h->x_row);
    if (h->xpos) for (size_t p = 0; p < h->rows; p++) ternary_vec_free(h->xpos[p]);
    free(h->xpos); free(h->gview); free(h->gxview); free(h->zview); free(h->oview); free(h->rview); free(h->xcv);
    free(h->z); free(h->c_logits); free(h->c_grad); free(h->c_z);
    ternary_elemnorm_free(h->vn);
    free(h);
}

/* Copia las posiciones de x (rows*dim concatenadas) a un TernaryVec por posicion. Por palabras de 64 bits: cada
 * posicion empieza en el bit p*dim, que en general no esta alineado. */
static void extract_words(const TernaryVec *src, size_t bit0, size_t n, TernaryVec *dst) {
    const size_t nw = dst->words;
    const size_t w0 = bit0 >> 6, sh = bit0 & 63;
    for (size_t j = 0; j < nw; j++) {
        uint64_t m = src->mask[w0 + j] >> sh, sg = src->sign[w0 + j] >> sh;
        if (sh && (w0 + j + 1) < src->words) {
            m |= src->mask[w0 + j + 1] << (64 - sh);
            sg |= src->sign[w0 + j + 1] << (64 - sh);
        }
        dst->mask[j] = m; dst->sign[j] = sg;
    }
    const size_t rem = n & 63;   /* bits sobrantes de la ultima palabra: siempre en 0 */
    if (rem) { const uint64_t keep = (1ull << rem) - 1; dst->mask[nw - 1] &= keep; dst->sign[nw - 1] &= keep; }
}

static void extract_all(const TernaryHead *h, const TernaryVec *x) {
    for (size_t p = 0; p < h->rows; p++) extract_words(x, p * h->dim, h->dim, h->xpos[p]);
}

/* Posiciones por bloque. Multiplo de 6 (grupo de las tablas de suma de Dense). */
#ifndef HEAD_BLOCK
#define HEAD_BLOCK 48
#endif

void ternary_head_forward(TernaryHead *h, const TernaryVec *x, float *out, int32_t *out_raw_opt) {
    TernaryCapabilities caps; memset(&caps, 0, sizeof(caps)); caps.num_cores = 1; caps.dot_impl_name = "scalar";
    extract_all(h, x);
    int32_t *zd = out_raw_opt;
    if (!zd) {
        if (!h->z) h->z = (int32_t *)ternary_calloc64(h->rows * h->vocab, sizeof(int32_t));
        zd = h->z;
    }
    h->zcur = zd;
    TernaryThreadPool *pool = ternary_get_compute_pool();
    if (pool && ternary_threadpool_num_threads(pool) > 1) {
        /* Con hilos: bloques de posiciones, una sola reparticion de filas de W por bloque (forward_block). */
        caps.cpu_pool = pool;
        for (size_t p = 0; p < h->rows; p++) { h->oview[p] = out + p * h->vocab; h->rview[p] = zd + p * h->vocab; }
        for (size_t p0 = 0; p0 < h->rows; p0 += HEAD_BLOCK) {
            size_t S = h->rows - p0 < HEAD_BLOCK ? h->rows - p0 : HEAD_BLOCK;
            ternary_dense_forward_block(h->w, S, h->xcv + p0, h->oview + p0, h->rview + p0, &caps);
        }
    } else {
        /* Un hilo: posicion por posicion (medido: el forward por bloques es ~2x mas lento sin hilos). */
        for (size_t p = 0; p < h->rows; p++)
            ternary_dense_forward_auto(h->w, h->xpos[p], out + p * h->vocab, zd + p * h->vocab, &caps);
    }
}

void ternary_head_backward(TernaryHead *h, const TernaryVec *x, const float *grad_out,
                            float *grad_w_out, float *grad_x_out, float *grad_scale_out) {
    const size_t D = h->dim, V = h->vocab;
    ternary_par_zero(grad_w_out, V * D * sizeof(float));
    /* dL/descala en double y en orden (posicion, vocab), como siempre. */
    double gscale = 0.0;
    for (size_t p = 0; p < h->rows; p++) {
        const float *g = grad_out + p * V;
        const int32_t *z = h->zcur + p * V;
        gscale += ternary_avx512_enabled() ? ternary_avx512_dot_gz(g, z, V, 0, 0.0) : (TN_ON(HEAD) ? ternary_neon_dot_gz(g, z, V, 0, 0.0) : ternary_canon_dot_gz(g, z, V));
    }
    *grad_scale_out += (float)gscale;

    /* W y X: cada posicion es una "muestra" de un bloque de Dense (backward_block). Sin bias; la escala de
     * Dense (en float) se descarta: la de arriba es la que vale. */
    extract_all(h, x);
    for (size_t p = 0; p < h->rows; p++) {
        h->gview[p] = grad_out + p * V; h->zview[p] = h->zcur + p * V; h->gxview[p] = grad_x_out + p * D;
    }
    float dummy = 0.0f;
    for (size_t p0 = 0; p0 < h->rows; p0 += HEAD_BLOCK) {
        size_t S = h->rows - p0 < HEAD_BLOCK ? h->rows - p0 : HEAD_BLOCK;
        ternary_dense_backward_block(h->w, S, h->xcv + p0, h->gview + p0, h->zview + p0,
                                     grad_w_out, h->gxview + p0, &dummy, NULL);
    }
}

/* ---- ruta uint8 repartida en hilos (mismo esquema que SwiGLU/GQA): fase por posicion y fase por FILAS de la matriz de pesos
 * (gw con p ascendente y bloques ascendentes). Los terminos escalares (perdida, aciertos, dL/descala, en double) se guardan por
 * posicion y se suman en serie en el orden de siempre: bit a bit igual con cualquier numero de hilos. ---- */
#define HEAD_ROW_GRAIN 16

typedef struct { TernaryHead *h; const TernaryU8 *x; float *out; int32_t *zd; const TernaryFwdW *fw; } HdFwd;
static void hd_fwd_range(size_t p0, size_t p1, void *c_) {
    HdFwd *c = (HdFwd *)c_;
    TernaryHead *h = c->h;
    ternary_dense_forward_u8_batch(h->w, c->fw, p1 - p0, c->x->q + p0 * h->dim, h->dim, c->x->scale + p0, c->x->zp + p0,
                                   c->out + p0 * h->vocab, h->vocab, c->zd + p0 * h->vocab, h->vocab);
}

void ternary_head_forward_u8(TernaryHead *h, const TernaryU8 *x, float *out, int32_t *out_raw_opt) {
    int32_t *zd = out_raw_opt;
    if (!zd) {
        if (!h->z) h->z = (int32_t *)ternary_calloc64(h->rows * h->vocab, sizeof(int32_t));
        zd = h->z;
    }
    h->zcur = zd;
    TernaryFwdW *fw = ternary_dense_fwdw_create(h->w);
    HdFwd c = { h, x, out, zd, fw };
    ternary_parallel_ranges(ternary_get_compute_pool(), 0, h->rows, 0, hd_fwd_range, &c);
    ternary_dense_fwdw_free(fw);
}

/* Filas de pesos [o0, o1) sobre P posiciones (la escala de Dense se descarta: sin terminos). */
typedef struct { TernaryHead *h; size_t P; const float *gout; const int32_t *z; const float *xf; const float *sx; float *gw; } HdW;
static void hd_w_range(size_t t0, size_t t1, void *c_) {
    HdW *c = (HdW *)c_;
    const size_t D = c->h->dim, V = c->h->vocab;
    for (size_t t = t0; t < t1; t++) {
        const size_t o0 = t * HEAD_ROW_GRAIN, o1 = (o0 + HEAD_ROW_GRAIN < V) ? o0 + HEAD_ROW_GRAIN : V;
        ternary_dense_u8_gradw(c->h->w, o0, o1, c->P, c->gout, V, c->xf, D, c->sx, c->z, V, c->gw, NULL, 0);
    }
}

typedef struct { TernaryHead *h; const TernaryU8 *x; const float *grad_out; float *grad_x_out; const int32_t *z; size_t b0; float *xf; double *gsc; const TernaryPlanes *pl; } HdBwd;
#define HD_SUB 16   /* posiciones por tanda del grad_x por lotes */
static void hd_bwd_range(size_t i0, size_t i1, void *c_) {
    HdBwd *c = (HdBwd *)c_;
    const size_t D = c->h->dim, V = c->h->vocab;
    for (size_t ia = i0; ia < i1; ia += HD_SUB) {
        const size_t ib = (ia + HD_SUB < i1) ? ia + HD_SUB : i1;
        for (size_t pi = ia; pi < ib; pi++) {
            const size_t p = c->b0 + pi;
            const float *g = c->grad_out + p * V;
            const int32_t *z = c->z + p * V;
            const double sx = (double)c->x->scale[p];
            c->gsc[pi] = ternary_avx512_enabled() ? ternary_avx512_dot_gz(g, z, V, 1, sx) : (TN_ON(HEAD) ? ternary_neon_dot_gz(g, z, V, 1, sx) : ternary_canon_dot_gsz(g, z, V, sx));
        }
        ternary_dense_u8_gradx_batch(c->h->w, c->pl, ib - ia, c->grad_out + (c->b0 + ia) * V, V, c->grad_x_out + (c->b0 + ia) * D, D);
        for (size_t pi = ia; pi < ib; pi++) {
            const size_t p = c->b0 + pi;
            ternary_dense_u8_to_f32(c->x->q + p * D, D, c->x->scale[p], c->x->zp[p], c->xf + pi * D);
        }
    }
}

void ternary_head_backward_u8(TernaryHead *h, const TernaryU8 *x, const float *grad_out,
                              float *grad_w_out, float *grad_x_out, float *grad_scale_out) {
    const size_t D = h->dim, V = h->vocab, R = h->rows;
    ternary_par_zero(grad_w_out, V * D * sizeof(float));
    TernaryThreadPool *pool = ternary_get_compute_pool();
    float *xf = (float *)ternary_malloc64(HEAD_BLOCK * D * sizeof(float));
    if (!xf) abort();
    double gsc[HEAD_BLOCK];
    double gscale = 0.0;                      /* dL/descala en double y en orden (posicion, vocab) */
    const size_t nch = (V + HEAD_ROW_GRAIN - 1) / HEAD_ROW_GRAIN;
    TernaryPlanes *pl = ternary_dense_planes_create(h->w);
    HdBwd A = { h, x, grad_out, grad_x_out, h->zcur, 0, xf, gsc, pl };
    HdW B = { h, 0, NULL, h->zcur, xf, NULL, grad_w_out };
    for (size_t b0 = 0; b0 < R; b0 += HEAD_BLOCK) {
        const size_t P = (R - b0 < HEAD_BLOCK) ? (R - b0) : HEAD_BLOCK;
        A.b0 = b0;
        ternary_parallel_ranges(pool, 0, P, 0, hd_bwd_range, &A);
        B.P = P; B.gout = grad_out + b0 * V; B.z = h->zcur + b0 * V; B.sx = x->scale + b0;
        ternary_parallel_ranges(pool, 0, nch, 1, hd_w_range, &B);
        for (size_t pi = 0; pi < P; pi++) gscale += gsc[pi];
    }
    free(xf);
    ternary_dense_planes_free(pl);
    *grad_scale_out += (float)gscale;
}

double ternary_head_train_xent(TernaryHead *h, const TernaryVec *x, const int32_t *targets, float inv_n,
                               float *grad_w_out, float *grad_x_out, float *grad_scale_out,
                               size_t *n_valid_out, size_t *n_correct_out) {
    const size_t R = h->rows, D = h->dim, V = h->vocab;
    if (n_valid_out) *n_valid_out = 0;
    if (n_correct_out) *n_correct_out = 0;
    size_t n = 0;
    for (size_t p = 0; p < R; p++) {
        if (targets[p] >= 0 && (size_t)targets[p] >= V) return (double)NAN;
        if (targets[p] >= 0) n++;
    }
    if (!h->c_logits) {
        h->c_logits = (float *)ternary_malloc64(HEAD_BLOCK * V * sizeof(float));
        h->c_grad = (float *)ternary_malloc64(HEAD_BLOCK * V * sizeof(float));
        h->c_z = (int32_t *)ternary_malloc64(HEAD_BLOCK * V * sizeof(int32_t));
        if (!h->c_logits || !h->c_grad || !h->c_z) {
            free(h->c_logits); free(h->c_grad); free(h->c_z);
            h->c_logits = NULL; h->c_grad = NULL; h->c_z = NULL;
            return (double)NAN;
        }
    }
    ternary_par_zero(grad_w_out, V * D * sizeof(float));
    extract_all(h, x);
    TernaryCapabilities caps; memset(&caps, 0, sizeof(caps)); caps.num_cores = 1; caps.dot_impl_name = "scalar";
    TernaryThreadPool *pool = ternary_get_compute_pool();
    const int threaded = pool && ternary_threadpool_num_threads(pool) > 1;
    if (threaded) caps.cpu_pool = pool;
    double gscale = 0.0, total = 0.0;
    size_t correct = 0;
    float dummy = 0.0f;
    for (size_t p0 = 0; p0 < R; p0 += HEAD_BLOCK) {
        const size_t S = R - p0 < HEAD_BLOCK ? R - p0 : HEAD_BLOCK;
        if (threaded) {
            for (size_t j = 0; j < S; j++) { h->oview[j] = h->c_logits + j * V; h->rview[j] = h->c_z + j * V; }
            ternary_dense_forward_block(h->w, S, h->xcv + p0, h->oview, h->rview, &caps);
        } else {
            for (size_t j = 0; j < S; j++)
                ternary_dense_forward_auto(h->w, h->xpos[p0 + j], h->c_logits + j * V, h->c_z + j * V, &caps);
        }
        /* Perdida y gradiente por fila (ternary_xent) y luego grad *= inv_n, igual que la ruta sin fusionar. */
        for (size_t j = 0; j < S; j++) {
            float *g = h->c_grad + j * V;
            const int32_t t = targets[p0 + j];
            if (t < 0) { memset(g, 0, V * sizeof(float)); }
            else {
                int ok = 0;
                total += ternary_xent_row(h->c_logits + j * V, V, (size_t)t, n, g, &ok);
                correct += (size_t)ok;
                if (ternary_avx512_enabled()) ternary_avx512_scale_f32(g, V, inv_n); else if (TN_ON(ELEM)) ternary_neon_scale_f32(g, V, inv_n); else for (size_t v = 0; v < V; v++) g[v] *= inv_n;
            }
            const int32_t *z = h->c_z + j * V;
            gscale += ternary_avx512_enabled() ? ternary_avx512_dot_gz(g, z, V, 0, 0.0) : (TN_ON(HEAD) ? ternary_neon_dot_gz(g, z, V, 0, 0.0) : ternary_canon_dot_gz(g, z, V));
            h->gview[j] = g; h->zview[j] = z; h->gxview[j] = grad_x_out + (p0 + j) * D;
        }
        ternary_dense_backward_block(h->w, S, h->xcv + p0, h->gview, h->zview, grad_w_out, h->gxview, &dummy, NULL);
    }
    *grad_scale_out += (float)gscale;
    if (n_valid_out) *n_valid_out = n;
    if (n_correct_out) *n_correct_out = correct;
    return n ? total / (double)n : 0.0;
}

typedef struct {
    TernaryHead *h; const TernaryU8 *x; const int32_t *targets; float inv_n; size_t n, p0; float *grad_x_out; float *xf;
    TernaryPlanes *pl; TernaryFwdW *fw;
    double tot[HEAD_BLOCK], gs[HEAD_BLOCK]; int ok[HEAD_BLOCK];
} HdXent;
static void hd_xent_range(size_t j0, size_t j1, void *c_) {
    HdXent *c = (HdXent *)c_;
    TernaryHead *h = c->h;
    const size_t D = h->dim, V = h->vocab;
    for (size_t ja = j0; ja < j1; ja += HD_SUB) {
        const size_t jb = (ja + HD_SUB < j1) ? ja + HD_SUB : j1;
        ternary_dense_forward_u8_batch(h->w, c->fw, jb - ja, c->x->q + (c->p0 + ja) * D, D, c->x->scale + c->p0 + ja, c->x->zp + c->p0 + ja,
                                       h->c_logits + ja * V, V, h->c_z + ja * V, V);
        for (size_t j = ja; j < jb; j++) {
            const size_t p = c->p0 + j;
            float *logits = h->c_logits + j * V, *g = h->c_grad + j * V;
            int32_t *z = h->c_z + j * V;
            const int32_t t = c->targets[p];
            c->tot[j] = 0.0; c->ok[j] = 0;
            if (t < 0) memset(g, 0, V * sizeof(float));
            else {
                int ok = 0;
                c->tot[j] = ternary_xent_row(logits, V, (size_t)t, c->n, g, &ok);
                c->ok[j] = ok;
                if (ternary_avx512_enabled()) ternary_avx512_scale_f32(g, V, c->inv_n); else if (TN_ON(ELEM)) ternary_neon_scale_f32(g, V, c->inv_n); else for (size_t v = 0; v < V; v++) g[v] *= c->inv_n;
            }
            const double sx = (double)c->x->scale[p];
            c->gs[j] = ternary_avx512_enabled() ? ternary_avx512_dot_gz(g, z, V, 1, sx) : (TN_ON(HEAD) ? ternary_neon_dot_gz(g, z, V, 1, sx) : ternary_canon_dot_gsz(g, z, V, sx));
        }
        ternary_dense_u8_gradx_batch(h->w, c->pl, jb - ja, h->c_grad + ja * V, V, c->grad_x_out + (c->p0 + ja) * D, D);
        for (size_t j = ja; j < jb; j++) {
            const size_t p = c->p0 + j;
            ternary_dense_u8_to_f32(c->x->q + p * D, D, c->x->scale[p], c->x->zp[p], c->xf + j * D);
        }
    }
}

/* Igual que ternary_head_train_xent pero con la entrada uint8 (una fila por posicion, escala y zero-point propios).
 * Por bloques de HEAD_BLOCK posiciones: solo se materializan HEAD_BLOCK x V logits/enteros/gradientes (nunca rows*vocab).
 * Repartido en hilos (ver arriba). */
double ternary_head_train_xent_u8(TernaryHead *h, const TernaryU8 *x, const int32_t *targets, float inv_n,
                                  float *grad_w_out, float *grad_x_out, float *grad_scale_out,
                                  size_t *n_valid_out, size_t *n_correct_out) {
    const size_t R = h->rows, D = h->dim, V = h->vocab;
    if (n_valid_out) *n_valid_out = 0;
    if (n_correct_out) *n_correct_out = 0;
    size_t n = 0;
    for (size_t p = 0; p < R; p++) {
        if (targets[p] >= 0 && (size_t)targets[p] >= V) return (double)NAN;
        if (targets[p] >= 0) n++;
    }
    if (!h->c_logits) {
        h->c_logits = (float *)ternary_malloc64(HEAD_BLOCK * V * sizeof(float));
        h->c_grad = (float *)ternary_malloc64(HEAD_BLOCK * V * sizeof(float));
        h->c_z = (int32_t *)ternary_malloc64(HEAD_BLOCK * V * sizeof(int32_t));
        if (!h->c_logits || !h->c_grad || !h->c_z) {
            free(h->c_logits); free(h->c_grad); free(h->c_z);
            h->c_logits = NULL; h->c_grad = NULL; h->c_z = NULL;
            return (double)NAN;
        }
    }
    ternary_par_zero(grad_w_out, V * D * sizeof(float));
    TernaryThreadPool *pool = ternary_get_compute_pool();
    HdXent *A = (HdXent *)ternary_calloc64(1, sizeof(HdXent));
    float *xf = (float *)ternary_malloc64(HEAD_BLOCK * D * sizeof(float));
    if (!A || !xf) abort();
    A->h = h; A->x = x; A->targets = targets; A->inv_n = inv_n; A->n = n; A->grad_x_out = grad_x_out; A->xf = xf;
    A->pl = ternary_dense_planes_create(h->w);
    A->fw = ternary_dense_fwdw_create(h->w);
    HdW B = { h, 0, h->c_grad, h->c_z, xf, NULL, grad_w_out };
    const size_t nch = (V + HEAD_ROW_GRAIN - 1) / HEAD_ROW_GRAIN;
    double gscale = 0.0, total = 0.0;
    size_t correct = 0;
    for (size_t p0 = 0; p0 < R; p0 += HEAD_BLOCK) {
        const size_t S = R - p0 < HEAD_BLOCK ? R - p0 : HEAD_BLOCK;
        A->p0 = p0;
        TPROF_BEGIN("logits_xent_gradx");
        ternary_parallel_ranges(pool, 0, S, 0, hd_xent_range, A);
        TPROF_END();
        B.P = S; B.sx = x->scale + p0;
        TPROF_BEGIN("bwd_weight_rows");
        ternary_parallel_ranges(pool, 0, nch, 1, hd_w_range, &B);
        TPROF_END();
        for (size_t j = 0; j < S; j++) {                   /* en el orden de siempre */
            if (targets[p0 + j] >= 0) { total += A->tot[j]; correct += (size_t)A->ok[j]; }
            gscale += A->gs[j];
        }
    }
    ternary_dense_planes_free(A->pl); ternary_dense_fwdw_free(A->fw);
    free(A); free(xf);
    *grad_scale_out += (float)gscale;
    if (n_valid_out) *n_valid_out = n;
    if (n_correct_out) *n_correct_out = correct;
    return n ? total / (double)n : 0.0;
}
