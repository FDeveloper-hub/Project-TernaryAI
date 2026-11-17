#include "ternary_mem.h"
#include "ternary_avx512.h"
#include "ternary_neon_policy.h"
#include "ternary_prof.h"
#include "ternary_fexp.h"
#include "ternary_swiglu.h"
#include <stdint.h>
#include "ternary_threadpool.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int sign_i32(int32_t v) { return v > 0 ? 1 : (v < 0 ? -1 : 0); }

size_t ternary_swiglu_num_matrices(const TernarySwiGLU *s) { return s->gate_mode == TERNARY_SWIGLU_GATE_RELU2 ? 2 : 3; }

TernaryDense *ternary_swiglu_matrix(const TernarySwiGLU *s, size_t m) {
    if (s->gate_mode == TERNARY_SWIGLU_GATE_RELU2) return m == 0 ? s->wu : m == 1 ? s->wd : NULL;
    return m == 0 ? s->wg : m == 1 ? s->wu : m == 2 ? s->wd : NULL;
}

void ternary_swiglu_refresh_tables(TernarySwiGLU *s) {
    if (s->gate_mode == TERNARY_SWIGLU_GATE_RELU2) return;
    const double sg = (double)s->wg->scale;
    const int64_t D = (int64_t)s->dim;
    for (int64_t z = -D; z <= D; z++) {
        const double a = sg * (double)z;
        const double sig = 1.0 / (1.0 + exp(-a));
        s->silu_tab[z + D] = (float)(a * sig);
        s->dsilu_tab[z + D] = (float)(sig * (1.0 + a * (1.0 - sig)));
    }
}

/* Posiciones por bloque de Dense (multiplo de 6: grupo de las tablas de suma de Dense). */
#ifndef SWIGLU_BLOCK
#define SWIGLU_BLOCK 48
#endif

/* Reparte el estado que crece con rows. Con base == NULL solo cuenta. */
static size_t swiglu_carve(TernarySwiGLU *s, unsigned char *base) {
    TernaryCarve c = { base, 0 };
    const size_t R = s->rows, H = s->hidden, E = s->dim, words = (H + 63) / 64;
    /* Sin compuerta (RELU2) no hay Z de gate: no se reserva. */
    int32_t *zg = (int32_t *)ternary_carve(&c, (s->gate_mode == TERNARY_SWIGLU_GATE_RELU2 ? 0 : R * H) * sizeof(int32_t));
    int32_t *zu = (int32_t *)ternary_carve(&c, R * H * sizeof(int32_t));
    int32_t *zd = (int32_t *)ternary_carve(&c, R * E * sizeof(int32_t));
    /* Los punteros (tpos y los TernaryVec) NO viven en el espacio compartido: otra capa lo pisaria entre el forward y
     * el backward y estos punteros no se recalculan. Solo las palabras de bits (se reescriben en cada forward). */
    uint64_t *w = (uint64_t *)ternary_carve(&c, R * 2 * words * sizeof(uint64_t));
    /* ruta uint8: h cuantizado por posicion (se reescribe entero en cada forward) */
    uint8_t *hq_q = (uint8_t *)ternary_carve(&c, R * H);
    float *hq_scale = (float *)ternary_carve(&c, R * sizeof(float));
    int32_t *hq_zp = (int32_t *)ternary_carve(&c, R * sizeof(int32_t));
    if (base) {
        if (!s->tp_raw) s->tp_raw = ternary_malloc64(R * (sizeof(TernaryVec *) + sizeof(TernaryVec)));
        if (!s->tp_raw) return c.off;
        TernaryVec **tp = (TernaryVec **)s->tp_raw;
        TernaryVec *tv = (TernaryVec *)(tp + R);
        s->zg = zg; s->zu = zu; s->zd = zd; s->tpos = tp;
        s->hq.q = hq_q; s->hq.scale = hq_scale; s->hq.zp = hq_zp; s->hq.rows = R; s->hq.row_len = H;
        for (size_t p = 0; p < R; p++) {
            tv[p].n = H; tv[p].words = words;
            tv[p].mask = w + p * 2 * words; tv[p].sign = tv[p].mask + words;
            tp[p] = &tv[p];
        }
    }
    return c.off;
}

size_t ternary_swiglu_state_bytes(const TernarySwiGLU *s) { return swiglu_carve((TernarySwiGLU *)s, NULL); }

void ternary_swiglu_bind_state(TernarySwiGLU *s, void *base) {
    if (!s || !base) return;
    swiglu_carve(s, (unsigned char *)base);
    free(s->state_raw); s->state_raw = NULL;
}

TernarySwiGLU *ternary_swiglu_alloc(size_t rows, size_t dim, size_t hidden, int gate_mode, float band) {
    if (rows == 0 || dim == 0 || hidden == 0 || dim > TERNARY_SWIGLU_MAX_DIM || hidden > TERNARY_SWIGLU_MAX_DIM) return NULL;
    if (gate_mode != TERNARY_SWIGLU_GATE_SILU && gate_mode != TERNARY_SWIGLU_GATE_SIGN && gate_mode != TERNARY_SWIGLU_GATE_RELU2) return NULL;
    const int ng = (gate_mode == TERNARY_SWIGLU_GATE_RELU2);   /* sin compuerta: sin W_gate, sin tablas de silu, sin buffers de gate */
    if (!(band >= 0.0f)) return NULL;
    TernarySwiGLU *s = (TernarySwiGLU *)ternary_calloc64(1, sizeof(*s));
    if (!s) return NULL;
    s->rows = rows; s->dim = dim; s->hidden = hidden; s->gate_mode = gate_mode; s->band = band;
    const size_t mx = hidden > dim ? hidden : dim;
    s->wg = ng ? NULL : ternary_dense_alloc(dim, hidden, 0);
    s->wu = ternary_dense_alloc(dim, hidden, 0);
    s->wd = ternary_dense_alloc(hidden, dim, 0);
    s->silu_tab = (float *)ternary_malloc64((ng ? 1 : 2 * dim + 1) * sizeof(float));
    s->dsilu_tab = (float *)ternary_malloc64((ng ? 1 : 2 * dim + 1) * sizeof(float));
    s->chunk = rows < SWIGLU_BLOCK ? rows : SWIGLU_BLOCK;
    const size_t C = s->chunk;
    s->xc = (TernaryVec **)ternary_calloc64(C, sizeof(TernaryVec *));
    { const size_t need = swiglu_carve(s, NULL);
      s->state_raw = (unsigned char *)ternary_calloc64(need + 64, 1);       /* calloc: las paginas no se tocan hasta usarlas */
      if (s->state_raw) swiglu_carve(s, (unsigned char *)(((uintptr_t)s->state_raw + 63u) & ~(uintptr_t)63u)); }
    s->tmp = (float *)ternary_calloc64(mx * s->chunk, sizeof(float));   /* una salida descartable por muestra del bloque */
    s->h = (float *)ternary_calloc64(hidden, sizeof(float));
    s->f_a = (float *)ternary_calloc64(ng ? 1 : hidden, sizeof(float));
    s->f_u = (float *)ternary_calloc64(hidden, sizeof(float));
    s->grad_t = (float *)ternary_calloc64(C * hidden, sizeof(float));
    s->grad_a = (float *)ternary_calloc64(ng ? 1 : C * hidden, sizeof(float));
    s->grad_u = (float *)ternary_calloc64(C * hidden, sizeof(float));
    s->grad_xc = (float *)ternary_calloc64(C * dim, sizeof(float));
    s->xcv = (const TernaryVec **)ternary_calloc64(C, sizeof(void *)); s->tcv = (const TernaryVec **)ternary_calloc64(C, sizeof(void *));
    s->gv = (const float **)ternary_calloc64(C, sizeof(void *)); s->gav = (const float **)ternary_calloc64(C, sizeof(void *));
    s->guv = (const float **)ternary_calloc64(C, sizeof(void *));
    s->zgv = (const int32_t **)ternary_calloc64(C, sizeof(void *)); s->zuv = (const int32_t **)ternary_calloc64(C, sizeof(void *));
    s->zdv = (const int32_t **)ternary_calloc64(C, sizeof(void *));
    s->gtv = (float **)ternary_calloc64(C, sizeof(void *)); s->gxv = (float **)ternary_calloc64(C, sizeof(void *));
    s->ov = (float **)ternary_calloc64(C, sizeof(void *)); s->rv = (int32_t **)ternary_calloc64(C, sizeof(void *));
    if ((!ng && !s->wg) || !s->wu || !s->wd || !s->silu_tab || !s->dsilu_tab || !s->xc || !s->state_raw || !s->tpos ||
        !s->zg || !s->zu || !s->zd || !s->tmp || !s->h || !s->f_a || !s->f_u || !s->grad_t || !s->grad_a || !s->grad_u ||
        !s->grad_xc || !s->xcv || !s->tcv || !s->gv || !s->gav || !s->guv || !s->zgv || !s->zuv || !s->zdv ||
        !s->gtv || !s->gxv || !s->ov || !s->rv) {
        ternary_swiglu_free(s);
        return NULL;
    }
    for (size_t c = 0; c < C; c++) { s->xc[c] = ternary_vec_alloc(dim); s->xcv[c] = s->xc[c]; if (!s->xc[c]) { ternary_swiglu_free(s); return NULL; } }
    ternary_swiglu_refresh_tables(s);
    return s;
}

void ternary_swiglu_free(TernarySwiGLU *s) {
    if (!s) return;
    ternary_dense_free(s->wg); ternary_dense_free(s->wu); ternary_dense_free(s->wd);
    free(s->silu_tab); free(s->dsilu_tab);
    if (s->xc) for (size_t c = 0; c < s->chunk; c++) ternary_vec_free(s->xc[c]);
    free(s->xc); free(s->state_raw); free(s->tp_raw);
    free(s->tmp); free(s->h); free(s->f_a); free(s->f_u);
    free(s->grad_t); free(s->grad_a); free(s->grad_u); free(s->grad_xc);
    free(s->xcv); free(s->tcv); free(s->gv); free(s->gav); free(s->guv); free(s->zgv); free(s->zuv); free(s->zdv);
    free(s->gtv); free(s->gxv); free(s->ov); free(s->rv);
    free(s);
}

/* Copia 'n' trits que empiezan en el bit bit0 de src a un TernaryVec propio (por palabras de 64 bits). */
static void extract_words(const TernaryVec *src, size_t bit0, size_t n, TernaryVec *dst) {
    const size_t nw = dst->words, w0 = bit0 >> 6, sh = bit0 & 63;
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

/* h de una posicion a partir de sus Z crudos. El unico lugar donde se define la pieza intercambiable:
 * forward, backward (via forward) y ternary_swiglu_product pasan todos por aqui. */
static void product_from_z(const TernarySwiGLU *s, const int32_t *zg, const int32_t *zu, float *h) {
    const size_t H = s->hidden;
    if (ternary_avx512_enabled()) {
        ternary_avx512_swiglu_prod_z(s->gate_mode, H, zg, zu, s->wu->scale, s->silu_tab, (int32_t)s->dim, h);
        return;
    }
    if (TN_ON(SWIGLU)) {
        ternary_neon_swiglu_prod_z(s->gate_mode, H, zg, zu, s->wu->scale, s->silu_tab, (int32_t)s->dim, h);
        return;
    }
    if (s->gate_mode == TERNARY_SWIGLU_GATE_SILU) {
        const float su = s->wu->scale;
        const int64_t D = (int64_t)s->dim;
        for (size_t j = 0; j < H; j++) {
            int64_t k = (int64_t)zg[j] + D;
            if (k < 0) k = 0; else if (k > 2 * D) k = 2 * D;
            h[j] = s->silu_tab[k] * (su * (float)zu[j]);
        }
    } else if (s->gate_mode == TERNARY_SWIGLU_GATE_RELU2) {
        const float su = s->wu->scale;          /* sin compuerta: zg no se usa (puede ser NULL) */
        for (size_t j = 0; j < H; j++) { const float a = su * (float)zu[j]; h[j] = a > 0.0f ? a * a : 0.0f; }
    } else {
        for (size_t j = 0; j < H; j++) h[j] = (float)(sign_i32(zg[j]) * zu[j]);
    }
}

void ternary_swiglu_product(const TernarySwiGLU *s, const TernaryVec *x, size_t row, float *h_out) {
    TernaryCapabilities caps; memset(&caps, 0, sizeof(caps)); caps.num_cores = 1; caps.dot_impl_name = "scalar";
    const size_t H = s->hidden;
    extract_words(x, row * s->dim, s->dim, s->xc[0]);
    const int ng = (s->gate_mode == TERNARY_SWIGLU_GATE_RELU2);
    int32_t *zg = ng ? NULL : s->zg + row * H, *zu = s->zu + row * H;
    if (!ng) ternary_dense_forward_auto(s->wg, s->xc[0], s->tmp, zg, &caps);
    ternary_dense_forward_auto(s->wu, s->xc[0], s->tmp, zu, &caps);
    product_from_z(s, zg, zu, s->h);
    memcpy(h_out, s->h, H * sizeof(float));
}

/* h -> t (cuantizacion con banda) de la posicion p, escrito en s->tpos[p]. */
static void quantize_row(const TernarySwiGLU *s, size_t p) {
    const size_t H = s->hidden;
    if (ternary_avx512_enabled()) {
        const float thr = s->band * (ternary_avx512_sum_abs(s->h, H) / (float)H);
        TernaryVec *t = s->tpos[p];
        memset(t->mask, 0, t->words * sizeof(uint64_t)); memset(t->sign, 0, t->words * sizeof(uint64_t));
        ternary_avx512_pack_thr(s->h, H, thr, t->mask, t->sign);
        return;
    }
    const float sum = ternary_canon_sum_abs(s->h, H);   /* orden canonico: mismos bits que AVX-512 */
    const float thr = s->band * (sum / (float)H);
    TernaryVec *t = s->tpos[p];
    memset(t->mask, 0, t->words * sizeof(uint64_t)); memset(t->sign, 0, t->words * sizeof(uint64_t));
    for (size_t j = 0; j < H; j++) {
        const float v = s->h[j];
        if (fabsf(v) > thr && v != 0.0f) {
            const uint64_t b = (uint64_t)1 << (j & 63);
            t->mask[j >> 6] |= b;
            if (v < 0.0f) t->sign[j >> 6] |= b;
        }
    }
}

void ternary_swiglu_forward(const TernarySwiGLU *s, const TernaryVec *x, float *out, int32_t *out_raw_opt) {
    const size_t H = s->hidden, E = s->dim, C = s->chunk, MX = H > E ? H : E;
    const int ng = (s->gate_mode == TERNARY_SWIGLU_GATE_RELU2);
    TernaryCapabilities caps; memset(&caps, 0, sizeof(caps)); caps.num_cores = 1; caps.dot_impl_name = "scalar";
    TernaryThreadPool *pool = ternary_get_compute_pool();
    if (pool && ternary_threadpool_num_threads(pool) > 1) {
        /* Con hilos: bloques de posiciones; cada matriz reparte sus filas entre los hilos una vez por bloque. */
        caps.cpu_pool = pool;
        for (size_t p0 = 0; p0 < s->rows; p0 += C) {
            const size_t S = (s->rows - p0 < C) ? (s->rows - p0) : C;
            for (size_t c = 0; c < S; c++) {
                extract_words(x, (p0 + c) * E, E, s->xc[c]);
                s->xcv[c] = s->xc[c];
                s->ov[c] = s->tmp + c * MX;
            }
            if (!ng) {
                for (size_t c = 0; c < S; c++) s->rv[c] = s->zg + (p0 + c) * H;
                ternary_dense_forward_block(s->wg, S, s->xcv, s->ov, s->rv, &caps);
            }
            for (size_t c = 0; c < S; c++) s->rv[c] = s->zu + (p0 + c) * H;
            ternary_dense_forward_block(s->wu, S, s->xcv, s->ov, s->rv, &caps);
            for (size_t c = 0; c < S; c++) {
                const size_t p = p0 + c;
                product_from_z(s, ng ? NULL : s->zg + p * H, s->zu + p * H, s->h);
                quantize_row(s, p);
                s->tcv[c] = s->tpos[p];
                s->ov[c] = out + p * E;
                s->rv[c] = s->zd + p * E;
            }
            ternary_dense_forward_block(s->wd, S, s->tcv, s->ov, s->rv, &caps);
        }
    } else {
        for (size_t p = 0; p < s->rows; p++) {
            extract_words(x, p * E, E, s->xc[0]);
            int32_t *zg = ng ? NULL : s->zg + p * H, *zu = s->zu + p * H;
            if (!ng) ternary_dense_forward_auto(s->wg, s->xc[0], s->tmp, zg, &caps);
            ternary_dense_forward_auto(s->wu, s->xc[0], s->tmp, zu, &caps);
            product_from_z(s, zg, zu, s->h);
            quantize_row(s, p);
            ternary_dense_forward_auto(s->wd, s->tpos[p], out + p * E, s->zd + p * E, &caps);
        }
    }
    if (out_raw_opt) memcpy(out_raw_opt, s->zd, s->rows * E * sizeof(int32_t));
}

void ternary_swiglu_backward(TernarySwiGLU *s, const TernaryVec *x, const float *grad_out,
                              float *const *grad_w_out, float *grad_x_out, float *grad_scale_out) {
    const size_t H = s->hidden, E = s->dim, C = s->chunk;
    const int silu_mode = (s->gate_mode == TERNARY_SWIGLU_GATE_SILU);
    const int ng = (s->gate_mode == TERNARY_SWIGLU_GATE_RELU2);   /* sin compuerta: grad_w_out[0] = W_up, [1] = W_down */
    float *const gw_u = grad_w_out[ng ? 0 : 1], *const gw_d = grad_w_out[ng ? 1 : 2];
    const float su = s->wu->scale;
    const int64_t D = (int64_t)E;
    if (!ng) memset(grad_w_out[0], 0, H * E * sizeof(float));
    memset(gw_u, 0, H * E * sizeof(float));
    memset(gw_d, 0, E * H * sizeof(float));
    float gs_g = 0.0f, gs_u_dead = 0.0f, gs_d = 0.0f, gs_g_dead = 0.0f;

    /* Bloques de posiciones: cada posicion es una "muestra" de backward_block de Dense. Los acumuladores de
     * cada elemento siguen en orden ascendente de posicion, asi que con las tablas de Dense apagadas el
     * resultado es identico bit a bit al de ir posicion por posicion. */
    for (size_t p0 = 0; p0 < s->rows; p0 += C) {
        const size_t S = (s->rows - p0 < C) ? (s->rows - p0) : C;
        for (size_t c = 0; c < S; c++) {
            const size_t p = p0 + c;
            extract_words(x, p * E, E, s->xc[c]);
            s->tcv[c] = s->tpos[p];
            s->gv[c] = grad_out + p * E;
            s->zdv[c] = s->zd + p * E;
            s->gtv[c] = s->grad_t + c * H;
            s->gav[c] = s->grad_a + c * H;
            s->guv[c] = s->grad_u + c * H;
            s->zgv[c] = s->zg + p * H;
            s->zuv[c] = s->zu + p * H;
            s->gxv[c] = s->grad_xc + c * E;
        }
        /* 1) W_down: gradiente exacto; grad_t es dL/dt, que la cuantizacion deja pasar tal cual (estimador directo) = dL/dh */
        memset(s->grad_t, 0, S * H * sizeof(float));
        ternary_dense_backward_block(s->wd, S, s->tcv, s->gv, s->zdv, gw_d, s->gtv, &gs_d, NULL);

        /* 2) producto y compuerta */
        for (size_t c = 0; c < S; c++) {
            const int32_t *zg = s->zgv[c], *zu = s->zuv[c];
            const float *gtc = s->grad_t + c * H;
            float *ga = s->grad_a + c * H, *gu = s->grad_u + c * H;
            if (ternary_avx512_enabled()) {
                ternary_avx512_swiglu_bwd_z(s->gate_mode, H, zg, zu, gtc, su, s->silu_tab, s->dsilu_tab, (int32_t)D, gu, ga);
                continue;
            }
            if (TN_ON(SWIGLU)) {
                ternary_neon_swiglu_bwd_z(s->gate_mode, H, zg, zu, gtc, su, s->silu_tab, s->dsilu_tab, (int32_t)D, gu, ga);
                continue;
            }
            for (size_t j = 0; j < H; j++) {
                const float gt = gtc[j];
                const float u = su * (float)zu[j];
                if (ng) {
                    gu[j] = u > 0.0f ? gt * 2.0f * u : 0.0f;   /* d relu(a)^2 / da = 2 relu(a), con a = escala_up * z_up */
                } else if (silu_mode) {
                    int64_t k = (int64_t)zg[j] + D;
                    if (k < 0) k = 0; else if (k > 2 * D) k = 2 * D;
                    gu[j] = gt * s->silu_tab[k];
                    ga[j] = gt * u * s->dsilu_tab[k];     /* dh/da = silu'(a) * u, con a = escala_gate * z_gate */
                } else {
                    gu[j] = gt * (float)sign_i32(zg[j]);
                    ga[j] = gt * u;                       /* sign() de la compuerta: estimador directo */
                }
            }
        }

        /* 3) W_gate y W_up hacia atras; la entrada de ambas es la misma fila x */
        memset(s->grad_xc, 0, S * E * sizeof(float));
        if (!ng) ternary_dense_backward_block(s->wg, S, s->xcv, s->gav, s->zgv, grad_w_out[0], s->gxv,
                                              silu_mode ? &gs_g : &gs_g_dead, NULL);
        ternary_dense_backward_block(s->wu, S, s->xcv, s->guv, s->zuv, gw_u, s->gxv, &gs_u_dead, NULL);
        for (size_t c = 0; c < S; c++)
            for (size_t d = 0; d < E; d++) grad_x_out[(p0 + c) * E + d] += s->grad_xc[c * E + d];
    }
    if (ng) { grad_scale_out[1] += gs_d; return; }     /* [0] = up: no se acumula */
    grad_scale_out[0] += gs_g;      /* 0 en la opcion 2 */
    grad_scale_out[2] += gs_d;
    /* grad_scale_out[1] (escala de up): sin efecto en la salida, no se acumula */
}


/* ======================= ruta uint8 ======================= */
/* silu(a) y su derivada con la exponencial rapida compartida: e = exp(-|a|) (2^y, y = -|a| log2 e), sigmoide
 * sg = 1/(1+e) si a >= 0 y e/(1+e) si a < 0; silu = a*sg; dsilu = sg*(1 + a*(1-sg)). */
static inline void silu_f(float a, float *sv, float *dsv) {
    const float y = -fabsf(a) * 1.44269504088896341f;
    const float e = (y < TERNARY_FEXP_CUTOFF) ? 0.0f : ternary_fexp2_core(y);
    const float sg = (a >= 0.0f) ? 1.0f / (1.0f + e) : e / (1.0f + e);
    *sv = a * sg;
    *dsv = sg * (1.0f + a * (1.0f - sg));
}

void ternary_swiglu_zero_stats(const TernarySwiGLU *s, uint64_t *zeros, uint64_t *total, int reset) {
    if (zeros) *zeros = s->zq_zeros;
    if (total) *total = s->zq_total;
    if (reset) { TernarySwiGLU *sm = (TernarySwiGLU *)(uintptr_t)s; sm->zq_zeros = 0; sm->zq_total = 0; }
}

/* ===================== ruta uint8, repartida en hilos =====================
 * Forward: cada posicion es independiente (escribe sus propias filas de zg/zu/zd/hq/out): se reparten posiciones.
 * Backward: por bloques de s->chunk posiciones, en dos fases con sincronizacion entre ellas:
 *   A) por posicion: grad_x de W_down (dL/dh), derivada de la activacion, grad_x de gate/up y la fila de grad_x_out;
 *   B) por FILAS de cada matriz de pesos: gw[o] += sum_p g[p][o]*x[p] con p ascendente y bloques ascendentes; los terminos de la
 *      escala del gradiente se guardan y el hilo que llama los suma despues en el orden de siempre (p, o ascendentes).
 * Resultado: bit a bit igual al calculo de un solo hilo de antes, con cualquier numero de hilos.
 * Ningun hilo escribe datos de otro, y el reparto no cambia ningun resultado. */
#define SWIGLU_ROW_GRAIN 16     /* filas por tarea en la fase B */

typedef struct { const TernarySwiGLU *s; const TernaryU8 *x; float *out; uint64_t zeros; const TernaryFwdW *fwg, *fwu, *fwd; } SwFwd;

#define SW_FSUB 16   /* posiciones por tanda del forward por lotes */
static void sw_fwd_range(size_t p0, size_t p1, void *c_) {
    SwFwd *c = (SwFwd *)c_;
    const TernarySwiGLU *s = c->s;
    const size_t H = s->hidden, E = s->dim;
    TernaryU8 *hq = (TernaryU8 *)(uintptr_t)&s->hq;
    float *buf = (float *)ternary_malloc64(3 * SW_FSUB * H * sizeof(float));      /* f_a, f_u, h de una tanda */
    if (!buf) abort();
    float *f_a0 = buf, *f_u0 = buf + SW_FSUB * H, *h0 = buf + 2 * SW_FSUB * H;
    uint64_t nz_tot = 0;
    for (size_t pa = p0; pa < p1; pa += SW_FSUB) {
        const size_t pb = (pa + SW_FSUB < p1) ? pa + SW_FSUB : p1, n = pb - pa;
        /* gate / up de toda la tanda (con RELU2 zg no se usa) */
        if (s->gate_mode != TERNARY_SWIGLU_GATE_RELU2)
            ternary_dense_forward_u8_batch(s->wg, c->fwg, n, c->x->q + pa * E, E, c->x->scale + pa, c->x->zp + pa, f_a0, H, s->zg + pa * H, H);
        ternary_dense_forward_u8_batch(s->wu, c->fwu, n, c->x->q + pa * E, E, c->x->scale + pa, c->x->zp + pa, f_u0, H, s->zu + pa * H, H);
        for (size_t p = pa; p < pb; p++) {
            float *f_a = f_a0 + (p - pa) * H, *f_u = f_u0 + (p - pa) * H, *h = h0 + (p - pa) * H;
            int32_t *zg = s->zg + p * H;
            if (ternary_avx512_enabled()) {
                ternary_avx512_swiglu_fwd_u8(s->gate_mode, H, f_a, f_u, zg, h);
            } else if (TN_ON(SWIGLU)) {
                ternary_neon_swiglu_fwd_u8(s->gate_mode, H, f_a, f_u, zg, h);
            } else if (s->gate_mode == TERNARY_SWIGLU_GATE_RELU2) {
                for (size_t j = 0; j < H; j++) { const float a = f_u[j]; h[j] = a > 0.0f ? a * a : 0.0f; }
            } else if (s->gate_mode == TERNARY_SWIGLU_GATE_SILU) {
                for (size_t j = 0; j < H; j++) { float sv, dsv; silu_f(f_a[j], &sv, &dsv); (void)dsv; h[j] = sv * f_u[j]; }
            } else {
                for (size_t j = 0; j < H; j++) h[j] = (float)sign_i32(zg[j]) * f_u[j];
            }
            TernaryU8 row = { hq->q + p * H, hq->scale + p, hq->zp + p, 1, H };
            ternary_u8_quantize(&row, h);
            {   /* sparsity: solo cuenta (hq == zero-point de la fila) */
                const uint8_t zb = (uint8_t)row.zp[0]; size_t nz = 0;
                for (size_t j = 0; j < H; j++) nz += (row.q[j] == zb);
                nz_tot += nz;
            }
        }
        ternary_dense_forward_u8_batch(s->wd, c->fwd, n, hq->q + pa * H, H, hq->scale + pa, hq->zp + pa, c->out + pa * E, E, s->zd + pa * E, E);
    }
    free(buf);
    __atomic_fetch_add(&c->zeros, nz_tot, __ATOMIC_RELAXED);
}

void ternary_swiglu_forward_u8(const TernarySwiGLU *s, const TernaryU8 *x, float *out, int32_t *out_raw_opt) {
    const size_t H = s->hidden, E = s->dim;
    TernaryFwdW *fwu = ternary_dense_fwdw_create(s->wu), *fwd = ternary_dense_fwdw_create(s->wd),
                *fwg = s->gate_mode != TERNARY_SWIGLU_GATE_RELU2 ? ternary_dense_fwdw_create(s->wg) : NULL;
    SwFwd c = { s, x, out, 0, fwg, fwu, fwd };
    TPROF_BEGIN("fwd_positions");
    ternary_parallel_ranges(ternary_get_compute_pool(), 0, s->rows, 0, sw_fwd_range, &c);
    TPROF_END();
    ternary_dense_fwdw_free(fwu); ternary_dense_fwdw_free(fwd); ternary_dense_fwdw_free(fwg);
    { TernarySwiGLU *sm = (TernarySwiGLU *)(uintptr_t)s; sm->zq_zeros += c.zeros; sm->zq_total += (uint64_t)s->rows * H; }
    if (out_raw_opt) memcpy(out_raw_opt, s->zd, s->rows * E * sizeof(int32_t));
}

typedef struct {
    TernarySwiGLU *s; const TernaryU8 *x; const float *grad_out; float *grad_x_out;
    size_t b0; float *xfx, *xfh; int ng, silu_mode; float sg_, su_;
    const TernaryPlanes *pwd, *pwg, *pwu;   /* planos +1/-1 de los pesos (NULL sin AVX-512) */
} SwBwdA;

#define SW_SUB 16   /* posiciones por tanda: las filas de gt/gu/ga/gx de una tanda caben en cache */
static void sw_bwd_a(size_t i0, size_t i1, void *c_) {
    SwBwdA *c = (SwBwdA *)c_;
    TernarySwiGLU *s = c->s;
    const size_t H = s->hidden, E = s->dim;
    for (size_t ia = i0; ia < i1; ia += SW_SUB) {
        const size_t ib = (ia + SW_SUB < i1) ? ia + SW_SUB : i1, n = ib - ia;
        float *gt0 = s->grad_t + ia * H, *gu0 = s->grad_u + ia * H, *ga0 = s->grad_a + ia * H, *gx0 = s->grad_xc + ia * E;
        memset(gt0, 0, n * H * sizeof(float));
        ternary_dense_u8_gradx_batch(s->wd, c->pwd, n, c->grad_out + (c->b0 + ia) * E, E, gt0, H);   /* dL/dh (estimador directo en la cuantizacion de h) */
        for (size_t pi = ia; pi < ib; pi++) {
            const size_t p = c->b0 + pi;
            float *gt = s->grad_t + pi * H, *gu = s->grad_u + pi * H, *ga = s->grad_a + pi * H;
            const float sx = c->x->scale[p];
            const int32_t *zg = s->zg + p * H, *zu = s->zu + p * H;   /* con RELU2 zg no se usa */
            if (ternary_avx512_enabled()) {
                ternary_avx512_swiglu_bwd_u8(s->gate_mode, H, zg, zu, gt, c->sg_ * sx, c->su_ * sx, gu, ga);
            } else if (TN_ON(SWIGLU)) {
                ternary_neon_swiglu_bwd_u8(s->gate_mode, H, zg, zu, gt, c->sg_ * sx, c->su_ * sx, gu, ga);
            } else
            for (size_t j = 0; j < H; j++) {
                const float a = c->sg_ * sx * (float)(c->ng ? 0 : zg[j]);
                const float u = c->su_ * sx * (float)zu[j];
                if (c->ng) {
                    gu[j] = u > 0.0f ? gt[j] * 2.0f * u : 0.0f;
                } else if (c->silu_mode) {
                    float sv, dsv; silu_f(a, &sv, &dsv);
                    gu[j] = gt[j] * sv;
                    ga[j] = gt[j] * u * dsv;
                } else {
                    gu[j] = gt[j] * (float)sign_i32(zg[j]);
                    ga[j] = gt[j] * u;                       /* sign() de la compuerta: estimador directo */
                }
            }
        }
        memset(gx0, 0, n * E * sizeof(float));
        if (!c->ng) ternary_dense_u8_gradx_batch(s->wg, c->pwg, n, ga0, H, gx0, E);
        ternary_dense_u8_gradx_batch(s->wu, c->pwu, n, gu0, H, gx0, E);
        for (size_t pi = ia; pi < ib; pi++) {
            const size_t p = c->b0 + pi;
            const float *gx = s->grad_xc + pi * E;
            for (size_t d = 0; d < E; d++) c->grad_x_out[p * E + d] += gx[d];
            ternary_dense_u8_to_f32(c->x->q + p * E, E, c->x->scale[p], c->x->zp[p], c->xfx + pi * E);
            ternary_dense_u8_to_f32(s->hq.q + p * H, H, s->hq.scale[p], s->hq.zp[p], c->xfh + pi * H);
        }
    }
}

typedef struct {
    TernarySwiGLU *s; const TernaryU8 *x; const float *grad_out;
    size_t b0, P; const float *xfx, *xfh; float *gw_g, *gw_u, *gw_d; int ng, silu_mode;
    size_t nch_d, nch_h; float *term_d, *term_g;     /* terminos de la escala: [p][o] de down y de gate */
} SwBwdB;

static void sw_bwd_b(size_t t0, size_t t1, void *c_) {
    SwBwdB *c = (SwBwdB *)c_;
    TernarySwiGLU *s = c->s;
    const size_t H = s->hidden, E = s->dim;
    for (size_t t = t0; t < t1; t++) {
        if (t < c->nch_d) {                                   /* W_down: x = h cuantizado */
            const size_t o0 = t * SWIGLU_ROW_GRAIN, o1 = (o0 + SWIGLU_ROW_GRAIN < E) ? o0 + SWIGLU_ROW_GRAIN : E;
            ternary_dense_u8_gradw(s->wd, o0, o1, c->P, c->grad_out + c->b0 * E, E, c->xfh, H,
                                   s->hq.scale + c->b0, s->zd + c->b0 * E, E, c->gw_d, c->term_d, E);
        } else if (t < c->nch_d + c->nch_h) {                 /* W_up (la escala de up no se aprende: sin parte de escala) */
            const size_t u = t - c->nch_d, o0 = u * SWIGLU_ROW_GRAIN, o1 = (o0 + SWIGLU_ROW_GRAIN < H) ? o0 + SWIGLU_ROW_GRAIN : H;
            ternary_dense_u8_gradw(s->wu, o0, o1, c->P, s->grad_u, H, c->xfx, E, c->x->scale + c->b0, s->zu + c->b0 * H, H, c->gw_u, NULL, 0);
        } else {                                              /* W_gate */
            const size_t u = t - c->nch_d - c->nch_h, o0 = u * SWIGLU_ROW_GRAIN, o1 = (o0 + SWIGLU_ROW_GRAIN < H) ? o0 + SWIGLU_ROW_GRAIN : H;
            ternary_dense_u8_gradw(s->wg, o0, o1, c->P, s->grad_a, H, c->xfx, E, c->x->scale + c->b0, s->zg + c->b0 * H, H, c->gw_g, c->silu_mode ? c->term_g : NULL, H);
        }
    }
}

void ternary_swiglu_backward_u8(TernarySwiGLU *s, const TernaryU8 *x, const float *grad_out,
                                float *const *grad_w_out, float *grad_x_out, float *grad_scale_out) {
    const size_t H = s->hidden, E = s->dim, K = s->chunk;
    const int silu_mode = (s->gate_mode == TERNARY_SWIGLU_GATE_SILU);
    const int ng = (s->gate_mode == TERNARY_SWIGLU_GATE_RELU2);   /* sin compuerta: grad_w_out[0] = W_up, [1] = W_down */
    float *const gw_u = grad_w_out[ng ? 0 : 1], *const gw_d = grad_w_out[ng ? 1 : 2];
    TPROF_BEGIN("zero_weight_grads");
    if (!ng) ternary_par_zero(grad_w_out[0], H * E * sizeof(float));
    ternary_par_zero(gw_u, H * E * sizeof(float));
    ternary_par_zero(gw_d, E * H * sizeof(float));
    TPROF_END();
    TernaryThreadPool *pool = ternary_get_compute_pool();
    const size_t nch_d = (E + SWIGLU_ROW_GRAIN - 1) / SWIGLU_ROW_GRAIN, nch_h = (H + SWIGLU_ROW_GRAIN - 1) / SWIGLU_ROW_GRAIN;
    const size_t ntask = nch_d + nch_h + (ng ? 0 : nch_h);
    float *work = (float *)ternary_malloc64(K * (E + H + E + H) * sizeof(float));
    if (!work) abort();
    float *xfx = work, *xfh = work + K * E, *term_d = work + K * (E + H), *term_g = term_d + K * E;
    float gs_g = 0.0f, gs_d = 0.0f;
    TernaryPlanes *pwd = ternary_dense_planes_create(s->wd), *pwu = ternary_dense_planes_create(s->wu), *pwg = ng ? NULL : ternary_dense_planes_create(s->wg);
    SwBwdA A = { s, x, grad_out, grad_x_out, 0, xfx, xfh, ng, silu_mode, ng ? 0.0f : s->wg->scale, s->wu->scale, pwd, pwg, pwu };
    SwBwdB B = { s, x, grad_out, 0, 0, xfx, xfh, ng ? NULL : grad_w_out[0], gw_u, gw_d, ng, silu_mode, nch_d, nch_h, term_d, term_g };
    for (size_t b0 = 0; b0 < s->rows; b0 += K) {
        const size_t P = (s->rows - b0 < K) ? (s->rows - b0) : K;
        A.b0 = B.b0 = b0; B.P = P;
        TPROF_BEGIN("bwd_positions");
        ternary_parallel_ranges(pool, 0, P, 0, sw_bwd_a, &A);
        TPROF_END();
        TPROF_BEGIN("bwd_weight_rows");
        ternary_parallel_ranges(pool, 0, ntask, 1, sw_bwd_b, &B);
        TPROF_END();
        for (size_t p = 0; p < P; p++) {                                       /* mismo orden que el calculo secuencial */
            for (size_t o = 0; o < E; o++) gs_d += term_d[p * E + o];
            if (silu_mode) for (size_t o = 0; o < H; o++) gs_g += term_g[p * H + o];
        }
    }
    free(work);
    ternary_dense_planes_free(pwd); ternary_dense_planes_free(pwu); ternary_dense_planes_free(pwg);
    if (ng) { grad_scale_out[1] += gs_d; return; }
    grad_scale_out[0] += gs_g;
    grad_scale_out[2] += gs_d;
}
