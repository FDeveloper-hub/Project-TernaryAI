#include "ternary_mem.h"
#include "ternary_avx512.h"
#include "ternary_neon_policy.h"

/* Nucleos vectoriales de la atencion flash (disposicion K/V traspuestas): AVX-512 o NEON, mismas firmas y mismos bits. */
#define TK_CALL(fn, ...) (ternary_avx512_enabled() ? ternary_avx512_##fn(__VA_ARGS__) : ternary_neon_##fn(__VA_ARGS__))
#include "ternary_prof.h"
#include "ternary_threadpool.h"
#include "ternary_gqa.h"
#include "ternary_fexp.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Posiciones por bloque de Dense (multiplo de 6: grupo de las tablas de suma de Dense). */
#ifndef GQA_BLOCK
#define GQA_BLOCK 48
#endif

/* Buffers por posicion que varios hilos escriben (una cabeza KV cada uno): alineados a linea de cache (64 B) para que dos hilos
 * no compartan linea (falso compartir). Se liberan con free(). */
static void *am64(size_t bytes) { return ternary_aligned_alloc(64, (bytes + 63) / 64 * 64); }

TernaryGQA *ternary_gqa_alloc(size_t embed_dim, size_t seq_len, size_t n_heads, size_t n_kv_heads) {
    return ternary_gqa_alloc_window(embed_dim, seq_len, n_heads, n_kv_heads, 0, 0);
}

TernaryGQA *ternary_gqa_alloc_window(size_t embed_dim, size_t seq_len, size_t n_heads, size_t n_kv_heads,
                                      size_t window, size_t sinks) {
    if (embed_dim == 0 || seq_len == 0 || n_heads == 0 || n_kv_heads == 0) return NULL;
    if (n_heads % n_kv_heads != 0) return NULL;
    if (embed_dim % n_heads != 0) return NULL;
    const size_t chunk = seq_len < GQA_BLOCK ? seq_len : GQA_BLOCK;
    if (window == 0 && sinks != 0) return NULL;
    if (sinks > chunk) return NULL;   /* los sinks viven en el primer bloque de posiciones */

    TernaryGQA *g = (TernaryGQA *)ternary_calloc64(1, sizeof(TernaryGQA));
    if (!g) return NULL;
    g->embed_dim = embed_dim; g->seq_len = seq_len;
    g->n_heads = n_heads; g->n_kv_heads = n_kv_heads;
    g->head_dim = embed_dim / n_heads;
    g->group_size = n_heads / n_kv_heads;
    g->window = window; g->sinks = sinks;

    size_t qw = n_heads * g->head_dim, kvw = n_kv_heads * g->head_dim;
    const size_t wide = qw > kvw ? (qw > embed_dim ? qw : embed_dim) : (kvw > embed_dim ? kvw : embed_dim);
    g->wq = ternary_dense_alloc(embed_dim, qw, 0);
    g->wk = ternary_dense_alloc(embed_dim, kvw, 0);
    g->wv = ternary_dense_alloc(embed_dim, kvw, 0);
    g->wo = ternary_dense_alloc(qw, embed_dim, 0);
    g->softmax = ternary_softmax_alloc((int32_t)g->head_dim, 1.0f / sqrtf((float)g->head_dim));

    g->x_pos = ternary_vec_alloc(embed_dim);
    g->qh = ternary_vec_alloc(g->head_dim);
    g->kh = ternary_vec_alloc(g->head_dim);
    g->attn_bits = ternary_vec_alloc(seq_len * qw);
    g->w_tmpf = (float *)ternary_malloc64(wide * sizeof(float));
    g->w_tmpi = (int32_t *)ternary_malloc64((qw > kvw ? qw : kvw) * sizeof(int32_t));
    g->chunk = chunk;
    g->xc = (TernaryVec **)ternary_calloc64(g->chunk, sizeof(TernaryVec *));
    g->rc = (TernaryVec **)ternary_calloc64(g->chunk, sizeof(TernaryVec *));
    g->gxc = (float *)am64(g->chunk * embed_dim * sizeof(float));
    g->views = (void **)ternary_calloc64(11 * g->chunk, sizeof(void *));
    int ok = g->wq && g->wk && g->wv && g->wo && g->softmax && g->x_pos && g->qh && g->kh && g->attn_bits &&
             g->w_tmpf && g->w_tmpi && g->xc && g->rc && g->gxc && g->views;

    if (!window) {
        g->q_z = (int32_t *)am64(seq_len * qw * sizeof(int32_t));
        g->k_z = (int32_t *)am64(seq_len * kvw * sizeof(int32_t));
        g->v_z = (int32_t *)am64(seq_len * kvw * sizeof(int32_t));
        g->q_bits = ternary_vec_alloc(seq_len * qw);
        g->k_bits = ternary_vec_alloc(seq_len * kvw);
        g->v_float = (float *)am64(seq_len * kvw * sizeof(float));
        g->scores = (int32_t *)am64(seq_len * sizeof(int32_t));
        g->probs = (float *)am64(seq_len * sizeof(float));
        g->attn_cat = (float *)am64(seq_len * qw * sizeof(float));
        g->wo_z = (int32_t *)am64(seq_len * embed_dim * sizeof(int32_t));
        g->grad_q_raw = (float *)am64(seq_len * qw * sizeof(float));
        g->grad_k_raw = (float *)am64(seq_len * kvw * sizeof(float));
        g->grad_v_raw = (float *)am64(seq_len * kvw * sizeof(float));
        g->grad_attn_cat = (float *)am64(seq_len * qw * sizeof(float));
        g->ds_scratch = (float *)am64(seq_len * sizeof(float));
        g->row_m = (int32_t *)am64(seq_len * n_heads * sizeof(int32_t));
        g->row_inv = (float *)am64(seq_len * n_heads * sizeof(float));
        g->dprobs = (float *)am64(seq_len * sizeof(float));
        ok = ok && g->q_z && g->k_z && g->v_z && g->q_bits && g->k_bits &&
             g->v_float && g->scores && g->probs && g->attn_cat && g->wo_z && g->grad_q_raw &&
             g->grad_k_raw && g->grad_v_raw && g->grad_attn_cat && g->ds_scratch && g->row_m && g->row_inv &&
             g->dprobs;
    } else {
        g->w_n = window + sinks;
        g->w_C0 = sinks > 0 ? chunk : 0;
        g->w_Lb = 2 * (window + 2 * chunk);       /* buffer deslizante: se desplaza cada ~(window + 2*chunk) posiciones */
        g->w_base = g->w_C0;
        g->w_ld = g->w_C0 + g->w_Lb + 16;
        const size_t slots = g->w_C0 + g->w_Lb + 16;
        g->w_kr = (int32_t *)ternary_calloc64(slots * kvw, sizeof(int32_t));
        g->w_qz = (int32_t *)ternary_calloc64(slots * qw, sizeof(int32_t));
        g->w_kz = (int32_t *)ternary_calloc64(slots * kvw, sizeof(int32_t));
        g->w_vz = (int32_t *)ternary_calloc64(slots * kvw, sizeof(int32_t));
        g->w_vf = (float *)ternary_calloc64(slots * kvw, sizeof(float));
        g->w_gq = (float *)ternary_calloc64(slots * qw, sizeof(float));
        g->w_gk = (float *)ternary_calloc64(slots * kvw, sizeof(float));
        g->w_gv = (float *)ternary_calloc64(slots * kvw, sizeof(float));
        g->w_qrot = (int32_t *)ternary_malloc64(chunk * qw * sizeof(int32_t));
        g->w_attn = (float *)ternary_malloc64(chunk * qw * sizeof(float));
        g->w_zo = (int32_t *)ternary_malloc64(chunk * embed_dim * sizeof(int32_t));
        g->w_o = (float *)ternary_malloc64(g->group_size * g->head_dim * sizeof(float));
        g->scores = (int32_t *)ternary_malloc64(g->w_n * sizeof(int32_t));
        g->probs = (float *)ternary_malloc64(g->w_n * sizeof(float));
        g->ds_scratch = (float *)ternary_malloc64(g->w_n * sizeof(float));
        ok = ok && g->w_kr && g->w_qz && g->w_kz && g->w_vz && g->w_vf && g->w_gq && g->w_gk && g->w_gv && g->w_qrot &&
             g->w_attn && g->w_zo && g->w_o && g->scores && g->probs && g->ds_scratch;
        if (ok && TN_VEC(ATTN)) {         /* disposicion vectorial (AVX-512 / NEON): K/V traspuestas y trabajo por grupo de cabezas */
            const size_t G = g->group_size, tp = (sinks + 15) / 16 * 16 + window + 48;
            g->tpad = tp;
            g->w_kT = (int32_t *)ternary_calloc64(kvw * g->w_ld, sizeof(int32_t));
            g->w_vT = (float *)ternary_calloc64(kvw * g->w_ld, sizeof(float));
            g->sc_blk = (int32_t *)ternary_calloc64(G * tp, sizeof(int32_t));
            g->e_blk = (float *)ternary_calloc64(G * tp, sizeof(float));
            g->p_blk = (float *)ternary_calloc64(G * tp, sizeof(float));
            g->ds_blk = (float *)ternary_calloc64(G * tp, sizeof(float));
            g->qf_blk = (float *)ternary_calloc64(G * g->head_dim, sizeof(float));
            g->qptr = (const int32_t **)ternary_calloc64(G, sizeof(int32_t *));
            g->m_blk = (int32_t *)ternary_calloc64(3 * G, sizeof(int32_t));
            g->inv_blk = (float *)ternary_calloc64(G, sizeof(float));
            ok = g->w_kT && g->w_vT && g->sc_blk && g->e_blk && g->p_blk && g->ds_blk && g->qf_blk && g->qptr && g->m_blk && g->inv_blk;
        }
    }
    if (ok) {
        for (size_t c = 0; c < g->chunk && ok; c++) {
            g->xc[c] = ternary_vec_alloc(embed_dim);
            g->rc[c] = ternary_vec_alloc(qw);
            if (!g->xc[c] || !g->rc[c]) ok = 0;
        }
    }
    if (!ok) { ternary_gqa_free(g); return NULL; }
    return g;
}

/* Estado compartible de la atencion COMPLETA: todos los buffers de tamano seq_len (Z de Q/K/V, bits, Q/K rotadas,
 * K/V traspuestas, salida de atencion, Z de Wo, gradientes, maximo/1/suma por fila). Viven en el espacio de trabajo
 * del stack y se recalculan con un forward justo antes del backward. Con base == NULL solo cuenta. */
static void bind_vec(TernaryVec *v, TernaryCarve *c, int base, int owned) {
    uint64_t *m = (uint64_t *)ternary_carve(c, v->words * sizeof(uint64_t));
    uint64_t *g = (uint64_t *)ternary_carve(c, v->words * sizeof(uint64_t));
    if (!base) return;
    if (!owned) { free(v->mask); free(v->sign); }
    v->mask = m; v->sign = g;
}
#define GQA_CARVE(ptr, type, n) do { type *q_ = (type *)ternary_carve(&c, (n) * sizeof(type)); \
    if (base) { if (!g->full_bound) free(ptr); ptr = q_; } } while (0)
static size_t gqa_full_carve(TernaryGQA *g, unsigned char *base) {
    TernaryCarve c = { base, 0 };
    const size_t T = g->seq_len, qw = g->n_heads * g->head_dim, kvw = g->n_kv_heads * g->head_dim, E = g->embed_dim;
    GQA_CARVE(g->q_z, int32_t, T * qw);  GQA_CARVE(g->k_z, int32_t, T * kvw); GQA_CARVE(g->v_z, int32_t, T * kvw);
    GQA_CARVE(g->wo_z, int32_t, T * E);
    if (g->v_float) GQA_CARVE(g->v_float, float, T * kvw);
    GQA_CARVE(g->attn_cat, float, T * qw);
    GQA_CARVE(g->grad_q_raw, float, T * qw); GQA_CARVE(g->grad_k_raw, float, T * kvw);
    GQA_CARVE(g->grad_v_raw, float, T * kvw); GQA_CARVE(g->grad_attn_cat, float, T * qw);
    GQA_CARVE(g->row_m, int32_t, T * g->n_heads); GQA_CARVE(g->row_inv, float, T * g->n_heads);
    if (g->q_rot) GQA_CARVE(g->q_rot, int32_t, T * qw);
    if (g->k_rot) GQA_CARVE(g->k_rot, int32_t, T * kvw);
    if (g->kT) { GQA_CARVE(g->kT, int32_t, kvw * g->tpad); GQA_CARVE(g->vT, float, kvw * g->tpad); }
    bind_vec(g->q_bits, &c, base != NULL, g->full_bound);
    bind_vec(g->k_bits, &c, base != NULL, g->full_bound);
    bind_vec(g->attn_bits, &c, base != NULL, g->full_bound);
    return c.off;
}

size_t ternary_gqa_state_bytes(const TernaryGQA *g) {
    if (!g->window) return gqa_full_carve((TernaryGQA *)(uintptr_t)g, NULL) + 64;
    const size_t words = (g->seq_len * g->n_heads * g->head_dim + 63) / 64;
    return 64 + 2 * words * sizeof(uint64_t);
}

void ternary_gqa_bind_state(TernaryGQA *g, void *base) {
    if (!base) return;
    if (!g->window) {
        gqa_full_carve(g, (unsigned char *)base);
        g->full_bound = 1;
        return;
    }
    TernaryVec *v = g->attn_bits;
    if (!g->attn_bound) { free(v->mask); free(v->sign); }
    const size_t words = v->words;
    v->mask = (uint64_t *)base; v->sign = v->mask + words;
    g->attn_bound = 1;
}

/* Con el estado en un espacio compartido (sucio): los bits sobrantes y el relleno de K/V traspuestas deben ser 0. */
static void gqa_full_prepare(const TernaryGQA *g) {
    if (!g->full_bound) return;
    TernaryVec *vs[3] = { g->q_bits, g->k_bits, g->attn_bits };
    for (int i = 0; i < 3; i++) {
        TernaryVec *v = vs[i];
        if (!v) continue;
        const size_t rem = v->n & 63;
        if (rem) { const uint64_t keep = (1ull << rem) - 1; v->mask[v->words - 1] &= keep; v->sign[v->words - 1] &= keep; }
    }
    if (g->kT) {
        const size_t kvw = g->n_kv_heads * g->head_dim;
        for (size_t r = 0; r < kvw; r++)
            for (size_t j = g->seq_len; j < g->tpad; j++) { g->kT[r * g->tpad + j] = 0; g->vT[r * g->tpad + j] = 0.0f; }
    }
}

/* Con el estado en el espacio compartido, los punteros no se liberan: se anulan antes del free normal. */
static void gqa_full_forget(TernaryGQA *g) {
    g->q_z = g->k_z = g->v_z = g->wo_z = g->row_m = g->q_rot = g->k_rot = NULL; g->kT = NULL;
    g->v_float = g->attn_cat = g->grad_q_raw = g->grad_k_raw = g->grad_v_raw = g->grad_attn_cat = g->row_inv = g->vT = NULL;
    TernaryVec *vs[3] = { g->q_bits, g->k_bits, g->attn_bits };
    for (int i = 0; i < 3; i++) if (vs[i]) { vs[i]->mask = NULL; vs[i]->sign = NULL; }
}

void ternary_gqa_free(TernaryGQA *g) {
    if (!g) return;
    if (g->full_bound) gqa_full_forget(g);
    ternary_dense_free(g->wq); ternary_dense_free(g->wk); ternary_dense_free(g->wv); ternary_dense_free(g->wo);
    ternary_softmax_free(g->softmax);
    free(g->rope_cos); free(g->rope_sin); free(g->q_rot); free(g->k_rot);
    ternary_vec_free(g->x_pos);
    free(g->q_z); free(g->k_z); free(g->v_z);
    if (g->q_bits) ternary_vec_free(g->q_bits);
    if (g->k_bits) ternary_vec_free(g->k_bits);
    free(g->v_float);
    if (g->qh) ternary_vec_free(g->qh);
    if (g->kh) ternary_vec_free(g->kh);
    free(g->scores); free(g->probs); free(g->attn_cat);
    if (g->attn_bits) { if (g->attn_bound) free(g->attn_bits); else ternary_vec_free(g->attn_bits); }
    free(g->wo_z);
    free(g->grad_q_raw); free(g->grad_k_raw); free(g->grad_v_raw); free(g->grad_attn_cat); free(g->ds_scratch);
    free(g->row_m); free(g->row_inv); free(g->exp_tab);
    free(g->kT); free(g->vT); free(g->sc_blk); free(g->e_blk); free(g->p_blk); free(g->ds_blk); free(g->qf_blk);
    free((void *)g->qptr); free(g->m_blk); free(g->inv_blk);
    free(g->w_kr); free(g->w_qz); free(g->w_kz); free(g->w_vz); free(g->w_vf); free(g->w_gq); free(g->w_gk); free(g->w_gv);
    free(g->w_qrot); free(g->w_attn); free(g->w_zo); free(g->w_tmpf); free(g->w_tmpi); free(g->w_o); free(g->w_kT); free(g->w_vT);
    free(g->rope_theta); free(g->rope_c); free(g->rope_s); free(g->dprobs);
    free(g->u8_tmp); free(g->smear_buf); if (g->attn_u8) ternary_u8_free(g->attn_u8);
    if (g->xc) for (size_t c = 0; c < g->chunk; c++) ternary_vec_free(g->xc[c]);
    if (g->rc) for (size_t c = 0; c < g->chunk; c++) ternary_vec_free(g->rc[c]);
    free(g->xc); free(g->rc); free(g->gxc); free(g->views);
    free(g);
}

/* Tabla de exponenciales COMPACTA para la atencion flash: exp_tab[k] = softmax->table[k] para k < exp_k0 y
 * exp_tab[exp_k0] = 0, con exp_k0 = primer indice cuya entrada es 0 (la tabla del softmax ya pone en 0 todo lo
 * menor que 1e-20): son ~260 floats a head_dim 32 en vez de 2*128*head_dim+1, asi que cabe en L1 con cualquier
 * cabeza. Mismos valores que la tabla del softmax, bit a bit. */
static int gqa_build_exp_table(TernaryGQA *g) {
    const TernarySoftmax *sm = g->softmax;
    size_t k0 = sm->table_len - 1;
    for (size_t k = 0; k < sm->table_len; k++) if (sm->table[k] == 0.0f) { k0 = k; break; }
    float *t = (float *)ternary_malloc64((k0 + 1) * sizeof(float));
    if (!t) return -1;
    for (size_t k = 0; k < k0; k++) t[k] = sm->table[k];
    t[k0] = 0.0f;
    free(g->exp_tab);
    g->exp_tab = t; g->exp_k0 = (uint32_t)k0;
    return 0;
}

int ternary_gqa_enable_rope(TernaryGQA *g, float base) {
    if (g->head_dim % 2 != 0 || base <= 1.0f) return -1;
    size_t pairs = g->head_dim / 2;
    if (g->window) {
        /* modo ventana: sin tablas de seq_len filas (serian O(T)); cos/sin de la posicion actual se calculan al vuelo con
         * la misma formula que la tabla (mismos valores). */
        double *th = (double *)ternary_malloc64(pairs * sizeof(double));
        int16_t *rc = (int16_t *)ternary_malloc64(pairs * sizeof(int16_t)), *rs = (int16_t *)ternary_malloc64(pairs * sizeof(int16_t));
        if (!th || !rc || !rs) { free(th); free(rc); free(rs); return -1; }
        for (size_t i = 0; i < pairs; i++) th[i] = pow((double)base, -2.0 * (double)i / (double)g->head_dim);
        free(g->rope_theta); free(g->rope_c); free(g->rope_s);
        g->rope_theta = th; g->rope_c = rc; g->rope_s = rs;
        free(g->rope_cos); free(g->rope_sin); g->rope_cos = g->rope_sin = NULL;
        g->rope_pairs = pairs; g->rope = 1; g->rope_base = base;
        return 0;
    }
    int16_t *cs = (int16_t *)ternary_malloc64(g->seq_len * pairs * sizeof(int16_t));
    int16_t *sn = (int16_t *)ternary_malloc64(g->seq_len * pairs * sizeof(int16_t));
    if (!cs || !sn) { free(cs); free(sn); return -1; }
    for (size_t p = 0; p < g->seq_len; p++) {
        for (size_t i = 0; i < pairs; i++) {
            double theta = pow((double)base, -2.0 * (double)i / (double)g->head_dim);
            double ang = (double)p * theta;
            cs[p * pairs + i] = (int16_t)lround(cos(ang) * TERNARY_ROPE_SCALE);
            sn[p * pairs + i] = (int16_t)lround(sin(ang) * TERNARY_ROPE_SCALE);
        }
    }
    free(g->rope_cos); free(g->rope_sin);
    g->rope_cos = cs; g->rope_sin = sn; g->rope_pairs = pairs; g->rope = 1; g->rope_base = base;
    return 0;
}

/* cos/sin (punto fijo) de la posicion pos: de la tabla, o calculados al vuelo en modo ventana. */
#define ROPE_TLS_PAIRS 1024      /* head_dim <= 2048 */
static void rope_cs(const TernaryGQA *g, size_t pos, const int16_t **c, const int16_t **s) {
    static _Thread_local int16_t tc[ROPE_TLS_PAIRS], ts[ROPE_TLS_PAIRS];   /* por hilo: varias posiciones a la vez en hilos distintos */
    const size_t P = g->rope_pairs;
    if (g->rope_cos) { *c = g->rope_cos + pos * P; *s = g->rope_sin + pos * P; return; }
    if (P > ROPE_TLS_PAIRS) abort();
    for (size_t i = 0; i < P; i++) {
        const double ang = (double)pos * g->rope_theta[i];
        tc[i] = (int16_t)lround(cos(ang) * TERNARY_ROPE_SCALE);
        ts[i] = (int16_t)lround(sin(ang) * TERNARY_ROPE_SCALE);
    }
    *c = tc; *s = ts;
}

/* Gira 'heads' cabezas de la posicion pos: enteros puros, sin desplazar
 * (solo se usara el signo, que no cambia al dividir por 256). */
static void rope_rotate_int(const TernaryGQA *g, size_t pos, const int32_t *z, size_t heads, int32_t *out) {
    size_t H = g->head_dim, P = g->rope_pairs;
    const int16_t *c, *s; rope_cs(g, pos, &c, &s);
    if (ternary_avx512_enabled()) { ternary_avx512_rope_rotate_int(z, heads, H, P, c, s, out); return; }
    if (TN_ON(ATTN)) { ternary_neon_rope_rotate_int(z, heads, H, P, c, s, out); return; }
    for (size_t h = 0; h < heads; h++)
        for (size_t i = 0; i < P; i++) {
            int32_t a = z[h * H + 2 * i], b = z[h * H + 2 * i + 1];
            out[h * H + 2 * i]     = a * c[i] - b * s[i];
            out[h * H + 2 * i + 1] = a * s[i] + b * c[i];
        }
}

/* Giro inverso (traspuesta de la rotacion) del gradiente, en float, in situ. */
static void rope_unrotate_grad(const TernaryGQA *g, size_t pos, float *grad, size_t heads) {
    size_t H = g->head_dim, P = g->rope_pairs;
    const int16_t *c, *s; rope_cs(g, pos, &c, &s);
    const float inv = 1.0f / (float)TERNARY_ROPE_SCALE;
    if (ternary_avx512_enabled()) { ternary_avx512_rope_unrotate_grad(grad, heads, H, P, c, s, inv); return; }
    if (TN_ON(ATTN)) { ternary_neon_rope_unrotate_grad(grad, heads, H, P, c, s, inv); return; }
    for (size_t h = 0; h < heads; h++)
        for (size_t i = 0; i < P; i++) {
            float ga = grad[h * H + 2 * i], gb = grad[h * H + 2 * i + 1];
            float cf = c[i] * inv, sf = s[i] * inv;
            grad[h * H + 2 * i]     =  ga * cf + gb * sf;
            grad[h * H + 2 * i + 1] = -ga * sf + gb * cf;
        }
}

static void extract_head(const TernaryVec *src, size_t offset, size_t n, TernaryVec *dst);

#ifndef TERNARY_GQA_QK_LIM_MULT
#define TERNARY_GQA_QK_LIM_MULT 128  /* recorte del puntaje entero = +-MULT*head_dim (tambien define el rango de la tabla del softmax).
                                      * Con 16 los puntajes crecen hasta el recorte, todos quedan iguales y la atencion se aplana
                                      * (plateau ~61%, sin gradiente); con >=128 (4096 a head_dim 32) no. */
#endif

int ternary_gqa_enable_rope_int(TernaryGQA *g, float base, int score_shift) {
    if (score_shift < 0 || score_shift > 16) return -1;
    if (ternary_gqa_enable_rope(g, base) != 0) return -1;
    size_t qw = g->n_heads * g->head_dim, kvw = g->n_kv_heads * g->head_dim;
    const size_t nrot = g->window ? 0 : g->seq_len;   /* modo ventana: Q/K rotadas viven en los anillos, no aqui */
    int32_t *qr = (int32_t *)ternary_malloc64((nrot * qw + 1) * sizeof(int32_t));
    int32_t *kr = (int32_t *)ternary_malloc64((nrot * kvw + 1) * sizeof(int32_t));
    TernarySoftmax *sm = ternary_softmax_alloc((int32_t)(TERNARY_GQA_QK_LIM_MULT * g->head_dim), 1.0f / sqrtf((float)g->head_dim));
    if (!qr || !kr || !sm) { free(qr); free(kr); ternary_softmax_free(sm); return -1; }
    free(g->q_rot); free(g->k_rot); ternary_softmax_free(g->softmax);
    g->q_rot = qr; g->k_rot = kr; g->softmax = sm;
    g->qk_int = 1; g->qk_shift = score_shift; g->qk_lim = (int32_t)(TERNARY_GQA_QK_LIM_MULT * g->head_dim);
    if (gqa_build_exp_table(g) != 0) return -1;
    if (TN_VEC(ATTN) && !g->kT && !g->window) {
        const size_t tp = (g->seq_len + 15) / 16 * 16, G = g->group_size, kvw2 = g->n_kv_heads * g->head_dim;
        g->tpad = tp;
        g->kT = (int32_t *)ternary_calloc64(kvw2 * tp, sizeof(int32_t));
        g->vT = (float *)ternary_calloc64(kvw2 * tp, sizeof(float));
        g->sc_blk = (int32_t *)ternary_calloc64(G * tp, sizeof(int32_t));
        g->e_blk = (float *)ternary_calloc64(G * tp, sizeof(float));
        g->p_blk = (float *)ternary_calloc64(G * tp, sizeof(float));
        g->ds_blk = (float *)ternary_calloc64(G * tp, sizeof(float));
        g->qf_blk = (float *)ternary_calloc64(G * g->head_dim, sizeof(float));
        g->qptr = (const int32_t **)ternary_calloc64(G, sizeof(int32_t *));
        g->m_blk = (int32_t *)ternary_calloc64(G, sizeof(int32_t));
        g->inv_blk = (float *)ternary_calloc64(G, sizeof(float));
        if (!g->kT || !g->vT || !g->sc_blk || !g->e_blk || !g->p_blk || !g->ds_blk || !g->qf_blk || !g->qptr || !g->m_blk || !g->inv_blk) return -1;
    }
    return 0;
}

int ternary_gqa_set_score_clip(TernaryGQA *g, int32_t lim) {
    if (!g->qk_int || lim < 1) return -1;
    TernarySoftmax *sm = ternary_softmax_alloc(lim, 1.0f / sqrtf((float)g->head_dim));
    if (!sm) return -1;
    ternary_softmax_free(g->softmax); g->softmax = sm; g->qk_lim = lim;
    if (gqa_build_exp_table(g) != 0) return -1;
    return 0;
}

/* ---- Smeared keys ---------------------------------------------------------------------------------------------
 * K'_j = (K_j + K_{j-1}) >> 1 sobre el Z entero de K (tras el desplazamiento de la ruta uint8, antes de RoPE). Las referencias
 * ESCALARES son las de abajo; el camino AVX-512 (ternary_avx512_smear_k / _smear_grad) da exactamente los mismos bits: la parte
 * entera es aritmetica entera y la de floats son productos por 1 o 0.5 (exactos) seguidos de UNA suma, sin FMA. */
int ternary_gqa_set_smear(TernaryGQA *g, int on) {
    if (!g) return -1;
    if (!on) { g->smear = 0; return 0; }
    if (!g->qk_int || g->window != 0) return -1;      /* puntaje entero (RoPE modo 1) y atencion completa */
    if (!g->smear_buf) {
        g->smear_buf = (int32_t *)ternary_calloc64(2 * g->n_kv_heads * g->head_dim, sizeof(int32_t));
        if (!g->smear_buf) return -1;
    }
    g->smear = 1;
    return 0;
}

/* Mezcla la clave de la posicion p ('in', Z entero sin rotar) con la anterior y la deja en 'out' (puede ser 'in'); recuerda 'in'
 * como "anterior" para p + 1. Las posiciones se procesan en orden 0, 1, 2, ... dentro de cada pasada. */
static void gqa_smear_k(const TernaryGQA *gc, size_t p, const int32_t *in, int32_t *out) {
    TernaryGQA *g = (TernaryGQA *)(uintptr_t)gc;      /* el estado "clave anterior" es scratch, como u8_tmp */
    const size_t kvw = g->n_kv_heads * g->head_dim;
    int32_t *prev = g->smear_buf;
    if (ternary_avx512_enabled()) { ternary_avx512_smear_k(in, out, prev, kvw, p == 0); return; }
    if (TN_ON(ELEM)) { ternary_neon_smear_k(in, out, prev, kvw, p == 0); return; }
    for (size_t d = 0; d < kvw; d++) {
        const int32_t a = in[d];
        out[d] = p == 0 ? a : (a + prev[d]) >> 1;
        prev[d] = a;
    }
}

/* Backward: gradiente respecto a K' -> respecto a K, en sitio (ver el comentario de ternary_gqa_set_smear).
 * Repartido en trozos de posiciones: la ultima posicion de cada trozo necesita la fila siguiente SIN reescribir, que es la primera del
 * trozo vecino; por eso esas filas se copian antes de repartir. Cada elemento sigue siendo own + nxt exactamente como antes. */
typedef struct { float *gk; size_t T, w, cs; const float *saved; } SmearG;
static void gqa_smear_grad_chunk(size_t c, void *ctx) {
    SmearG *S = (SmearG *)ctx;
    const size_t p0 = c * S->cs, p1 = (p0 + S->cs < S->T) ? p0 + S->cs : S->T;
    const float *nl = p1 < S->T ? S->saved + c * S->w : NULL;
    if (ternary_avx512_enabled()) { ternary_avx512_smear_grad_rows(S->gk, p0, p1, S->T, S->w, nl); return; }
    if (TN_ON(ELEM)) { ternary_neon_smear_grad_rows(S->gk, p0, p1, S->T, S->w, nl); return; }
    for (size_t p = p0; p < p1; p++) {
        float *row = S->gk + p * S->w;
        const float *nrow = (p + 1 == p1 && nl) ? nl : row + S->w;
        for (size_t d = 0; d < S->w; d++) {
            const float own = p == 0 ? row[d] : 0.5f * row[d];
            const float nxt = p + 1 < S->T ? 0.5f * nrow[d] : 0.0f;
            row[d] = own + nxt;
        }
    }
}
static void gqa_smear_grad(const TernaryGQA *g, float *gk) {
    const size_t T = g->seq_len, w = g->n_kv_heads * g->head_dim;
    TernaryThreadPool *pool = ternary_get_compute_pool();
    size_t nch = 1;
    if (pool && T * w * sizeof(float) >= TERNARY_PAR_MIN_BYTES) { nch = (size_t)ternary_threadpool_num_threads(pool) * 4; if (nch > T) nch = T; }
    if (nch <= 1) {
        SmearG S = { gk, T, w, T, NULL };
        gqa_smear_grad_chunk(0, &S);
        return;
    }
    const size_t cs = (T + nch - 1) / nch;
    nch = (T + cs - 1) / cs;
    float *saved = (float *)ternary_malloc64(nch * w * sizeof(float));
    if (!saved) abort();
    for (size_t c = 0; c + 1 < nch; c++) memcpy(saved + c * w, gk + (c + 1) * cs * w, w * sizeof(float));
    SmearG S = { gk, T, w, cs, saved };
    ternary_parallel_for(pool, 0, nch, gqa_smear_grad_chunk, &S);
    free(saved);
}

/* Llena scores[0..i] (puntaje de la consulta i, cabeza h, contra las claves j<=i). */
static void compute_scores_s(const TernaryGQA *g, size_t i, size_t h, size_t kv_h, int32_t *scores, TernaryVec *qh, TernaryVec *kh_) {
    size_t H = g->head_dim, qw = g->n_heads * H, kvw = g->n_kv_heads * H;
    if (g->qk_int) {
        const int32_t *a = g->q_rot + i * qw + h * H;
        const int32_t lim = g->qk_lim;
        if (ternary_avx512_enabled()) {
            ternary_avx512_scores_i32(a, g->k_rot + kv_h * H, kvw, H, i + 1, 16 + g->qk_shift, lim, scores);
            return;
        }
        if (TN_ON(ATTN)) {
            ternary_neon_scores_i32(a, g->k_rot + kv_h * H, kvw, H, i + 1, 16 + g->qk_shift, lim, scores);
            return;
        }
        for (size_t j = 0; j <= i; j++) {
            const int32_t *b = g->k_rot + j * kvw + kv_h * H;
            int64_t dot = 0;
            for (size_t d = 0; d < H; d++) dot += (int64_t)a[d] * (int64_t)b[d];
            dot >>= (16 + g->qk_shift);   /* TERNARY_ROPE_SCALE^2 = 2^16 */
            if (dot > lim) dot = lim; else if (dot < -lim) dot = -lim;
            scores[j] = (int32_t)dot;
        }
        return;
    }
    extract_head(g->q_bits, i * qw + h * H, H, qh);
    for (size_t j = 0; j <= i; j++) {
        extract_head(g->k_bits, j * kvw + kv_h * H, H, kh_);
        scores[j] = ternary_dot(qh, kh_);
    }
}

static void compute_scores(const TernaryGQA *g, size_t i, size_t h, size_t kv_h) {
    compute_scores_s(g, i, h, kv_h, g->scores, g->qh, g->kh);
}

/* Copia 'n' trits que empiezan en el bit bit0 de src a un TernaryVec propio, por palabras de 64 bits. */
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

/* Igual que ternary_vec_set pero en linea y sin ramas: dst[i] = sign(v). */
static inline void set_sign_i32(TernaryVec *dst, size_t i, int32_t v) {
    const uint64_t b = (uint64_t)1 << (i & 63);
    const uint64_t m = (uint64_t)(v != 0) ? b : 0, sg = (uint64_t)(v < 0) ? b : 0;
    dst->mask[i >> 6] = (dst->mask[i >> 6] & ~b) | m;
    dst->sign[i >> 6] = (dst->sign[i >> 6] & ~b) | sg;
}
static inline void set_sign_f(TernaryVec *dst, size_t i, float v) {
    const uint64_t b = (uint64_t)1 << (i & 63);
    const uint64_t m = (v != 0.0f) ? b : 0, sg = (v < 0.0f) ? b : 0;
    dst->mask[i >> 6] = (dst->mask[i >> 6] & ~b) | m;
    dst->sign[i >> 6] = (dst->sign[i >> 6] & ~b) | sg;
}


/* Copia 'width' valores de un TernaryVec grande (offset*width en adelante)
 * a un TernaryVec chico -- extrae la cabeza h de la posicion 'pos'. */
static void extract_head(const TernaryVec *big, size_t base_idx, size_t width, TernaryVec *dst) {
    for (size_t i = 0; i < width; i++) ternary_vec_set(dst, i, ternary_vec_get(big, base_idx + i));
}

/* ======================================================================================================
 * Atencion "flash" ternaria (puntaje entero). Que se tomo de FlashAttention-2 / FlashAttention-4 y como se
 * adapto al caso ternario:
 *  - FA2, normalizacion diferida: el forward acumula sum_j e_j*v_j SIN normalizar y divide una sola vez al final
 *    (H multiplicaciones por fila en vez de una por clave). Como Wo solo ve el SIGNO de la atencion y 1/suma > 0,
 *    el signo sale del acumulador sin normalizar.
 *  - FA2, estadisticas por fila: se guarda el maximo (aqui ENTERO, exacto) y 1/suma; el backward reconstruye
 *    p_j = e_j*inv sin repetir el pase del softmax. Con el truco D = dO.O (FA2) el backward tampoco necesita el
 *    vector completo de dprobs antes de empezar: ds_j = t*p_j*(dprob_j - D) sale clave por clave.
 *  - FA2, nada de la matriz N x N: por fila solo existe un vector de T enteros.
 *  - FA4, exponencial sin la unidad de exponenciales: aqui el argumento es un ENTERO k = m - s, asi que e = exp(-k*t)
 *    sale de una tabla COMPACTA (~1 KB, solo hasta donde la entrada deja de ser > 1e-20; ver gqa_build_exp_table):
 *    en escalar es una lectura, en AVX-512 un gather de L1. Los mismos valores en ambos caminos (sin polinomio:
 *    el polinomio de FA4 cuesta mas que una lectura de tabla cuando el argumento es entero).
 *  - NO se toma el reescalado en linea ni el reescalado condicional de FA4: el maximo es un entero que se obtiene
 *    exacto en un primer pase sobre la fila de puntajes (T enteros, caben en L1), asi que no hay acumulador que
 *    corregir. La suma usa 16 parciales por carril y un arbol fijo (ternary_sum16): el mismo orden que el nucleo
 *    AVX-512, asi escalar y vectorial dan los mismos bits.
 * Referencia ESCALAR: el nucleo AVX-512 es identico bit a bit a estas dos funciones.
 * ====================================================================================================== */
/* Camino AVX-512 de la atencion flash: el grupo GQA completo (G cabezas que comparten K/V) por cada bloque de claves. */
int ternary_gqa_attention_probs(const TernaryGQA *g, size_t head, float *out) {
    if (!g || !out || g->window || !g->qk_int || !g->q_rot || !g->k_rot || head >= g->n_heads) return -1;
    const size_t T = g->seq_len, kv_h = head / g->group_size;
    const float *tab = g->exp_tab; const uint32_t K0 = g->exp_k0;
    for (size_t i = 0; i < T; i++) {
        compute_scores(g, i, head, kv_h);
        int32_t m = g->scores[0];
        for (size_t j = 1; j <= i; j++) if (g->scores[j] > m) m = g->scores[j];
        float part[16] = {0};
        for (size_t j = 0; j <= i; j++) {
            const uint32_t k = (uint32_t)(m - g->scores[j]);
            const float e = tab[k < K0 ? k : K0];
            out[i * T + j] = e; part[j & 15] += e;
        }
        const float inv = 1.0f / ternary_sum16(part);
        for (size_t j = 0; j <= i; j++) out[i * T + j] *= inv;
        for (size_t j = i + 1; j < T; j++) out[i * T + j] = 0.0f;
    }
    return 0;
}

/* ---- Atencion flash repartida en hilos ----
 * Forward: cada fila i (todas las cabezas) es independiente: se reparten filas. Backward: gk/gv de una cabeza KV se acumulan sobre i
 * ascendente y sobre las cabezas de su grupo en orden, asi que se reparte POR CABEZA KV (cada tarea escribe solo su porcion de
 * grad_q/k/v y recorre i y las cabezas en el mismo orden de siempre): bit a bit igual con cualquier numero de hilos.
 * Cada tarea trae su propio scratch (nada compartido en g). */
typedef struct {
    int32_t *scores; float *probs, *dpr; TernaryVec *qh, *kh;
    int32_t *sc_blk, *m_blk; float *e_blk, *p_blk, *ds_blk, *qf_blk, *inv_blk; const int32_t **qptr;
    float *dmy;            /* descarte para la parte del backward que no se pide: G*H (gq) + 2*H (gk, gv) */
    int32_t *bsc, *bm; float *bp, *bds, *bqf, *binv; /* backward por bloques de filas: hasta 4 filas */
} AttnScr;

static void scr_free(AttnScr *s) {
    free(s->scores); free(s->probs); free(s->dpr);
    if (s->qh) ternary_vec_free(s->qh);
    if (s->kh) ternary_vec_free(s->kh);
    free(s->sc_blk); free(s->m_blk); free(s->e_blk); free(s->p_blk); free(s->ds_blk); free(s->qf_blk); free(s->inv_blk); free((void *)s->qptr); free(s->dmy);
    free(s->bsc); free(s->bm); free(s->bp); free(s->bds); free(s->bqf); free(s->binv);
}
static void scr_init(AttnScr *s, const TernaryGQA *g) {
    memset(s, 0, sizeof *s);
    const size_t T = g->seq_len, G = g->group_size, tp = g->tpad;
    s->scores = (int32_t *)ternary_malloc64(T * sizeof(int32_t));
    s->probs = (float *)ternary_malloc64(T * sizeof(float));
    s->dpr = (float *)ternary_malloc64(T * sizeof(float));
    int ok = s->scores && s->probs && s->dpr;
    if (g->kT) {
        s->sc_blk = (int32_t *)ternary_calloc64(G * tp, sizeof(int32_t)); s->e_blk = (float *)ternary_calloc64(G * tp, sizeof(float));
        s->p_blk = (float *)ternary_calloc64(G * tp, sizeof(float)); s->ds_blk = (float *)ternary_calloc64(G * tp, sizeof(float));
        s->qf_blk = (float *)ternary_calloc64(G * g->head_dim, sizeof(float)); s->qptr = (const int32_t **)ternary_calloc64(G, sizeof(int32_t *));
        s->m_blk = (int32_t *)ternary_calloc64(3 * G, sizeof(int32_t)); s->inv_blk = (float *)ternary_calloc64(G, sizeof(float));
        s->dmy = (float *)ternary_calloc64(G * g->head_dim + 2 * g->head_dim, sizeof(float));
        ok = ok && s->sc_blk && s->e_blk && s->p_blk && s->ds_blk && s->qf_blk && s->qptr && s->m_blk && s->inv_blk && s->dmy;
        if (ternary_avx512_enabled()) {
            s->bsc = (int32_t *)ternary_calloc64(4 * G * tp, sizeof(int32_t)); s->bp = (float *)ternary_calloc64(4 * G * tp, sizeof(float));
            s->bds = (float *)ternary_calloc64(4 * G * tp, sizeof(float)); s->bqf = (float *)ternary_calloc64(4 * G * g->head_dim, sizeof(float));
            s->bm = (int32_t *)ternary_calloc64(4 * G, sizeof(int32_t)); s->binv = (float *)ternary_calloc64(4 * G, sizeof(float));
            ok = ok && s->bsc && s->bp && s->bds && s->bqf && s->bm && s->binv;
        }
    } else if (!g->qk_int) {
        s->qh = ternary_vec_alloc(g->head_dim); s->kh = ternary_vec_alloc(g->head_dim);
        ok = ok && s->qh && s->kh;
    }
    if (!ok) abort();
}

static void fwd_rows_avx(const TernaryGQA *g, AttnScr *S, size_t i0, size_t i1) {
    const size_t H = g->head_dim, NQ = g->n_heads, NKV = g->n_kv_heads, G = g->group_size, qw = NQ * H, kvw = NKV * H, T = g->seq_len, tp = g->tpad;
    const float *vf = g->v_float;
    for (size_t i = i0; i < i1; i++) {
        const size_t n = i + 1;
        for (size_t kh = 0; kh < NKV; kh++) {
            for (size_t b = 0; b < G; b++) S->qptr[b] = g->q_rot + i * qw + (kh * G + b) * H;
            TK_CALL(attn_scores, S->qptr, G, g->kT + kh * H * tp, tp, H, n, 16 + g->qk_shift, g->qk_lim, S->sc_blk, tp, S->m_blk);
            TK_CALL(attn_pv, G, S->sc_blk, tp, S->m_blk, g->exp_tab, g->exp_k0, n, vf + kh * H, kvw, H, S->e_blk, tp,
                                   g->attn_cat + i * qw + kh * G * H, S->inv_blk);
            for (size_t b = 0; b < G; b++) { g->row_m[i * NQ + kh * G + b] = S->m_blk[b]; g->row_inv[i * NQ + kh * G + b] = S->inv_blk[b]; }
            if (i == T - 1 && kh == NKV - 1)
                for (size_t j = 0; j < n; j++) g->probs[j] = S->e_blk[(G - 1) * tp + j] * S->inv_blk[G - 1];  /* compat: probs de la ultima fila */
        }
    }
}

static void fwd_rows_scalar(const TernaryGQA *g, AttnScr *S, size_t i0, size_t i1) {
    const size_t H = g->head_dim, NQ = g->n_heads, qw = NQ * H, kvw = g->n_kv_heads * H, T = g->seq_len;
    const float *restrict tab = g->exp_tab; const uint32_t K0 = g->exp_k0;
    const float *vf = g->v_float;
    for (size_t i = i0; i < i1; i++) {
        const size_t n = i + 1;
        for (size_t h = 0; h < NQ; h++) {
            const size_t kv_h = h / g->group_size;
            compute_scores_s(g, i, h, kv_h, S->scores, S->qh, S->kh);
            int32_t m = S->scores[0];
            for (size_t j = 1; j < n; j++) if (S->scores[j] > m) m = S->scores[j];
            float part[16] = {0};
            for (size_t j = 0; j < n; j++) {
                const uint32_t k = (uint32_t)(m - S->scores[j]);
                const float e = tab[k < K0 ? k : K0];
                S->probs[j] = e;
                part[j & 15] += e;
            }
            const float sum = ternary_sum16(part);
            float *restrict acc = g->attn_cat + i * qw + h * H;
            memset(acc, 0, H * sizeof(float));
            for (size_t j = 0; j < n; j++) {
                const float e = S->probs[j];
                const float *restrict vj = vf + j * kvw + kv_h * H;
                for (size_t d = 0; d < H; d++) acc[d] += e * vj[d];
            }
            const float inv = 1.0f / sum;
            for (size_t d = 0; d < H; d++) acc[d] *= inv;
            g->row_m[i * NQ + h] = m; g->row_inv[i * NQ + h] = inv;
            if (i == T - 1 && h == NQ - 1) for (size_t j = 0; j < n; j++) g->probs[j] = S->probs[j] * inv;  /* compat: probs de la ultima fila */
        }
    }
}

static void attn_fwd_range(size_t i0, size_t i1, void *c_) {
    const TernaryGQA *g = (const TernaryGQA *)c_;
    AttnScr S; scr_init(&S, g);
    if (g->kT) fwd_rows_avx(g, &S, i0, i1); else fwd_rows_scalar(g, &S, i0, i1);
    scr_free(&S);
}

static void attn_forward_flash(const TernaryGQA *g) {
    ternary_parallel_ranges(ternary_get_compute_pool(), 0, g->seq_len, 4, attn_fwd_range, (void *)(uintptr_t)g);
}

/* Gradientes de la atencion flash: acumula en grad_q_raw / grad_k_raw / grad_v_raw.
 * attn_bwd_part trabaja sobre la cabeza KV kh, las filas i en [i0, i1) y SOLO las claves j en [j0, j1) (j <= i), y puede hacer solo la
 * parte de K/V (doA) o solo la de Q (doB). Cada elemento de salida se suma exactamente en el mismo orden que en la version completa
 * (grad_k/grad_v de la clave j: filas i ascendentes y cabezas ascendentes; grad_q de la fila i: claves ascendentes), asi que partir el
 * trabajo no cambia ningun bit. La parte que no se pide corre contra un buffer de descarte (el nucleo AVX-512 es el mismo de siempre).
 * j0 debe ser multiplo de 16 (las lecturas de K/V traspuestas llegan al relleno de tpad). */
static void attn_bwd_part(TernaryGQA *g, AttnScr *S, size_t kh, size_t j0, size_t j1, size_t i0, size_t i1, int doA, int doB) {
    const size_t H = g->head_dim, NQ = g->n_heads, NKV = g->n_kv_heads, G = g->group_size, qw = NQ * H, kvw = NKV * H, tp = g->tpad;
    const float it = g->softmax->inv_temp;
    const float gam = ldexpf(1.0f, -g->qk_shift) / (float)TERNARY_ROPE_SCALE;
    const float *restrict tab = g->exp_tab; const uint32_t K0 = g->exp_k0;
    const float *vf = g->v_float;
    const size_t h0 = kh * G;
    size_t istart = i0;
    const size_t RB = (g->kT && S->bsc) ? ternary_avx512_attn_bwd_rows_cap(G, H) : 0;
    if (RB >= 2) {                      /* bloques de RB filas: cada gk/gv se lee y escribe una vez por bloque (mismo orden de sumas) */
        size_t i = (i0 > j0) ? i0 : j0;
        for (; i + RB <= i1; i += RB) {
            size_t nr[4];
            for (size_t r = 0; r < RB; r++) {
                const size_t ir = i + r, jh = (ir + 1 < j1) ? ir + 1 : j1; nr[r] = jh - j0;
                int32_t *bm = S->bm + r * G; float *binv = S->binv + r * G;
                for (size_t b = 0; b < G; b++) S->qptr[b] = g->q_rot + ir * qw + (h0 + b) * H;
                int32_t *bsc = S->bsc + r * G * tp;
                ternary_avx512_attn_scores(S->qptr, G, g->kT + kh * H * tp + j0, tp, H, nr[r], 16 + g->qk_shift, g->qk_lim, bsc, tp, bm);
                for (size_t b = 0; b < G; b++) { bm[b] = g->row_m[ir * NQ + h0 + b]; binv[b] = g->row_inv[ir * NQ + h0 + b]; }
                ternary_avx512_attn_bwd_pds(G, nr[r], H, bsc, tp, bm, binv, tab, K0, g->grad_attn_cat + ir * qw + h0 * H,
                                            g->attn_cat + ir * qw + h0 * H, g->vT + kh * H * tp + j0, tp, it, gam,
                                            g->q_rot + ir * qw + h0 * H, S->bp + r * G * tp, S->bds + r * G * tp, tp, S->bqf + r * G * H);
            }
            ternary_avx512_attn_bwd_rows(RB, G, H, nr, g->grad_attn_cat + i * qw + h0 * H, qw, S->bqf, S->bp, S->bds, tp, G * tp,
                                         g->k_rot + j0 * kvw + kh * H, kvw,
                                         doB ? g->grad_q_raw + i * qw + h0 * H : S->dmy, doB ? qw : 0,
                                         g->grad_k_raw + j0 * kvw + kh * H, g->grad_v_raw + j0 * kvw + kh * H, kvw, doA);
        }
        istart = i;
        if (istart < i0) istart = i0;
    }
    for (size_t i = istart; i < i1; i++) {
        if (i < j0) continue;
        const size_t jhi = (i + 1 < j1) ? i + 1 : j1, n = jhi - j0;
        if (g->kT) {
            for (size_t b = 0; b < G; b++) S->qptr[b] = g->q_rot + i * qw + (h0 + b) * H;
            TK_CALL(attn_scores, S->qptr, G, g->kT + kh * H * tp + j0, tp, H, n, 16 + g->qk_shift, g->qk_lim, S->sc_blk, tp, S->m_blk);
            for (size_t b = 0; b < G; b++) { S->m_blk[b] = g->row_m[i * NQ + h0 + b]; S->inv_blk[b] = g->row_inv[i * NQ + h0 + b]; }
            float *gq_p = doB ? g->grad_q_raw + i * qw + h0 * H : S->dmy;
            float *gk_p = doA ? g->grad_k_raw + j0 * kvw + kh * H : S->dmy + G * H;
            float *gv_p = doA ? g->grad_v_raw + j0 * kvw + kh * H : S->dmy + G * H + H;
            const size_t gs = doA ? kvw : 0;
            TK_CALL(attn_bwd, G, n, H, S->sc_blk, tp, S->m_blk, S->inv_blk, g->exp_tab, g->exp_k0,
                                    g->grad_attn_cat + i * qw + h0 * H, g->attn_cat + i * qw + h0 * H, g->vT + kh * H * tp + j0, tp, it, gam,
                                    g->q_rot + i * qw + h0 * H, g->k_rot + j0 * kvw + kh * H, kvw,
                                    gq_p, gk_p, gv_p, gs,
                                    S->p_blk, S->ds_blk, tp, S->qf_blk);
            continue;
        }
        for (size_t hb = 0; hb < G; hb++) {
            const size_t h = h0 + hb, kv_h = kh;
            const float *ga_h = g->grad_attn_cat + i * qw + h * H;
            const float *o_h = g->attn_cat + i * qw + h * H;
            const int32_t m = g->row_m[i * NQ + h];
            const float inv = g->row_inv[i * NQ + h];
            compute_scores_s(g, i, h, kv_h, S->scores, S->qh, S->kh);
            float D = 0.0f;
            for (size_t d = 0; d < H; d++) D += ga_h[d] * o_h[d];
            float *restrict gq = g->grad_q_raw + i * qw + h * H;
            const int32_t *restrict qr = g->q_rot + i * qw + h * H;
            /* dprob_j = sum_d ga[d]*v[j][d] (d ascendente): 4 claves a la vez, cada una con su propia cadena de sumas. */
            float *restrict dpr = S->dpr;
            size_t j4 = j0;
            for (; j4 + 4 <= jhi; j4 += 4) {
                const float *v0 = vf + j4 * kvw + kv_h * H, *v1 = v0 + kvw, *v2 = v1 + kvw, *v3 = v2 + kvw;
                float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                for (size_t d = 0; d < H; d++) {
                    const float gd = ga_h[d];
                    d0 += gd * v0[d]; d1 += gd * v1[d]; d2 += gd * v2[d]; d3 += gd * v3[d];
                }
                dpr[j4] = d0; dpr[j4 + 1] = d1; dpr[j4 + 2] = d2; dpr[j4 + 3] = d3;
            }
            for (; j4 < jhi; j4++) {
                const float *restrict vj = vf + j4 * kvw + kv_h * H;
                float dot = 0.0f;
                for (size_t d = 0; d < H; d++) dot += ga_h[d] * vj[d];
                dpr[j4] = dot;
            }
            for (size_t j = j0; j < jhi; j++) {
                const uint32_t k = (uint32_t)(m - S->scores[j]);
                const float p = tab[k < K0 ? k : K0] * inv;
                const float ds = ((it * p) * (dpr[j] - D)) * gam;
                float *restrict gv_j = g->grad_v_raw + j * kvw + kv_h * H;
                float *restrict gk = g->grad_k_raw + j * kvw + kv_h * H;
                const int32_t *restrict kr = g->k_rot + j * kvw + kv_h * H;
                if (doA) {
                    for (size_t d = 0; d < H; d++) gv_j[d] += p * ga_h[d];
                    for (size_t d = 0; d < H; d++) gk[d] += ds * (float)qr[d];
                }
                if (doB) for (size_t d = 0; d < H; d++) gq[d] += ds * (float)kr[d];
            }
        }
    }
}

static void attn_bwd_range(size_t k0, size_t k1, void *c_) {
    TernaryGQA *g = (TernaryGQA *)c_;
    AttnScr S; scr_init(&S, g);
    for (size_t kh = k0; kh < k1; kh++) attn_bwd_part(g, &S, kh, 0, g->seq_len, 0, g->seq_len, 1, 1);
    scr_free(&S);
}

/* Con mas hilos que el doble de cabezas KV: tareas de claves (grad_k/grad_v, por bloque de claves) y de posiciones (grad_q, por bloque
 * de filas). Escriben salidas distintas, asi que corren todas a la vez; el resultado es bit a bit el de la version completa. */
#define ATTN_KB 32     /* claves por tarea (multiplo de 16) */
#define ATTN_IB 8      /* filas por tarea */
static void attn_bwd_split_range(size_t t0, size_t t1, void *c_) {
    TernaryGQA *g = (TernaryGQA *)c_;
    const size_t T = g->seq_len, NKV = g->n_kv_heads, nkb = (T + ATTN_KB - 1) / ATTN_KB, nib = (T + ATTN_IB - 1) / ATTN_IB;
    AttnScr S; scr_init(&S, g);
    for (size_t t = t0; t < t1; t++) {
        if (t < NKV * nkb) {
            const size_t kh = t / nkb, j0 = (t % nkb) * ATTN_KB, j1 = (j0 + ATTN_KB < T) ? j0 + ATTN_KB : T;
            attn_bwd_part(g, &S, kh, j0, j1, j0, T, 1, 0);
        } else {
            const size_t u = t - NKV * nkb, kh = u / nib, i0 = (u % nib) * ATTN_IB, i1 = (i0 + ATTN_IB < T) ? i0 + ATTN_IB : T;
            attn_bwd_part(g, &S, kh, 0, T, i0, i1, 0, 1);
        }
    }
    scr_free(&S);
}

static void attn_backward_flash(TernaryGQA *g) {
    TernaryThreadPool *pool = ternary_get_compute_pool();
    const size_t NKV = g->n_kv_heads;
    if (pool && (size_t)ternary_threadpool_num_threads(pool) > 2 * NKV) {
        const size_t T = g->seq_len, nkb = (T + ATTN_KB - 1) / ATTN_KB, nib = (T + ATTN_IB - 1) / ATTN_IB;
        ternary_parallel_ranges(pool, 0, NKV * (nkb + nib), 1, attn_bwd_split_range, g);
        return;
    }
    ternary_parallel_ranges(pool, 0, NKV, 1, attn_bwd_range, g);
}

/* ======================================================================================================
 * Atencion con VENTANA deslizante + sinks, en modo "streaming" (g->window > 0).
 * La fila i atiende a j en [j0, i] (j0 = max(0, i+1-W)) y, si hay sinks, a las posiciones [0, s), s = min(sinks, j0).
 * Memoria: un anillo de R = W + 2*chunk ranuras (mas el primer bloque aparte si hay sinks) con Z, K rotada, V y
 * gradientes por posicion, y los bits de la salida de atencion (2 bits por elemento). Lo demas se RECALCULA en el
 * backward (como FlashAttention recalcula S y P): proyecciones de Q/K/V (Dense forward, enteros exactos), el maximo
 * y la suma de cada fila, la salida O (para D = dO.O) y el Z de Wo. Con W >= T y sinks = 0 los resultados son
 * identicos bit a bit a la atencion flash completa: mismo orden de claves y de sumas.
 * ====================================================================================================== */
/* Buffer deslizante: las posiciones p >= C0 viven en la ranura C0 + (p - w_base) (contiguas, de modo que las claves de
 * una fila forman tramos fisicamente contiguos); las posiciones p < C0 (primer bloque, solo si hay sinks) viven fijas en
 * las ranuras [0, C0). w_base solo avanza (en win_ensure), nunca pasa de lo que aun se necesita. */

/* Z entero -> rango del puntaje (ruta uint8): desplazamiento aritmetico con redondeo (determinista). */
static inline int32_t u8_shift(int32_t z) { return (z + (1 << (TERNARY_GQA_U8_SHIFT - 1))) >> TERNARY_GQA_U8_SHIFT; }
static void gq_shift(const int32_t *z, size_t n, int32_t *out) {
    if (ternary_avx512_enabled()) ternary_avx512_shift_round_i32(z, n, TERNARY_GQA_U8_SHIFT, out);
    else if (TN_ON(ELEM)) ternary_neon_shift_round_i32(z, n, TERNARY_GQA_U8_SHIFT, out);
    else for (size_t i = 0; i < n; i++) out[i] = u8_shift(z[i]);
}
/* V (no ternario) en la ruta uint8: int8 simetrico con escala por posicion (como las activaciones; STE: el backward no cambia). */
static void gq_vq_row(float *v, size_t n) {
    float mx = 0.0f; for (size_t i = 0; i < n; i++) { const float a = fabsf(v[i]); if (a > mx) mx = a; }
    if (mx <= 0.0f) return;
    const float s = mx / 127.0f;
    for (size_t i = 0; i < n; i++) v[i] = roundf(v[i] / s) * s;
}
/* out[i*stride] = k * (float)z[i] */
static void gq_scale(const int32_t *z, size_t n, float k, float *out, size_t stride) {
    if (ternary_avx512_enabled()) ternary_avx512_scale_i32_f32(z, n, k, out, stride);
    else if (TN_ON(ELEM)) ternary_neon_scale_i32_f32(z, n, k, out, stride);
    else for (size_t i = 0; i < n; i++) out[i * stride] = k * (float)z[i];
}

static inline size_t wslot(const TernaryGQA *g, size_t p) { return p < g->w_C0 ? p : g->w_C0 + (p - g->w_base); }

/* Antes de escribir las posiciones hasta hi (las anteriores a 'next' ya estan escritas): si no caben, se desplaza el
 * buffer para que w_base pase a ser keep_lo (la posicion mas antigua que aun se necesita). */
static void win_ensure(const TernaryGQA *gc, size_t hi, size_t keep_lo, size_t next) {
    TernaryGQA *g = (TernaryGQA *)gc;
    if (hi < g->w_C0 || hi + 1 - g->w_base <= g->w_Lb) return;
    size_t nb = keep_lo > g->w_base ? keep_lo : g->w_base;
    if (nb > next) nb = next;
    const size_t delta = nb - g->w_base, live = next > nb ? next - nb : 0;
    const size_t qw = g->n_heads * g->head_dim, kvw = g->n_kv_heads * g->head_dim, C0 = g->w_C0;
    if (delta > 0 && live > 0) {
        memmove(g->w_kr + C0 * kvw, g->w_kr + (C0 + delta) * kvw, live * kvw * sizeof(int32_t));
        memmove(g->w_vf + C0 * kvw, g->w_vf + (C0 + delta) * kvw, live * kvw * sizeof(float));
        memmove(g->w_qz + C0 * qw, g->w_qz + (C0 + delta) * qw, live * qw * sizeof(int32_t));
        memmove(g->w_kz + C0 * kvw, g->w_kz + (C0 + delta) * kvw, live * kvw * sizeof(int32_t));
        memmove(g->w_vz + C0 * kvw, g->w_vz + (C0 + delta) * kvw, live * kvw * sizeof(int32_t));
        memmove(g->w_gq + C0 * qw, g->w_gq + (C0 + delta) * qw, live * qw * sizeof(float));
        memmove(g->w_gk + C0 * kvw, g->w_gk + (C0 + delta) * kvw, live * kvw * sizeof(float));
        memmove(g->w_gv + C0 * kvw, g->w_gv + (C0 + delta) * kvw, live * kvw * sizeof(float));
        if (g->w_kT)
            for (size_t d = 0; d < kvw; d++) {
                memmove(g->w_kT + d * g->w_ld + C0, g->w_kT + d * g->w_ld + C0 + delta, live * sizeof(int32_t));
                memmove(g->w_vT + d * g->w_ld + C0, g->w_vT + d * g->w_ld + C0 + delta, live * sizeof(float));
            }
    }
    g->w_base = nb;
}

/* Claves de la fila i: tramo A = sinks [0, s) (ranuras 0..s-1), tramo B = ventana [j0, i] (nB claves desde la ranura slB). */
typedef struct { size_t s, nB, slB, n; } WinKeys;
static WinKeys win_keys(const TernaryGQA *g, size_t i) {
    const size_t j0 = (i + 1 > g->window) ? i + 1 - g->window : 0;
    WinKeys k;
    k.s = (j0 > 0) ? (g->sinks < j0 ? g->sinks : j0) : 0;
    k.nB = i - j0 + 1; k.slB = wslot(g, j0); k.n = k.s + k.nB;
    return k;
}
#define WKSLOT(k, t) ((t) < (k).s ? (t) : (k).slB + ((t) - (k).s))

#define GQA_BLK 32          /* posiciones por bloque del backward */
#define GQA_ROW_GRAIN 16    /* filas de pesos por tarea */

/* Una matriz de pesos del backward: filas [0, out) repartidas en tareas de GQA_ROW_GRAIN filas. */
typedef struct { const TernaryDense *w; const float *gout; const int32_t *z; float *gw; float *terms; size_t out, nch; } GqJob;
typedef struct { const GqJob *jobs; int nj; size_t P; const float *xf; size_t xf_stride; const float *sx; } GqJobs;

static void gq_jobs_range(size_t t0, size_t t1, void *c_) {
    const GqJobs *c = (const GqJobs *)c_;
    for (size_t t = t0; t < t1; t++) {
        size_t tt = t; int j = 0;
        while (j < c->nj - 1 && tt >= c->jobs[j].nch) { tt -= c->jobs[j].nch; j++; }
        const GqJob *b = &c->jobs[j];
        const size_t o0 = tt * GQA_ROW_GRAIN, o1 = (o0 + GQA_ROW_GRAIN < b->out) ? o0 + GQA_ROW_GRAIN : b->out;
        ternary_dense_u8_gradw(b->w, o0, o1, c->P, b->gout, b->out, c->xf, c->xf_stride, c->sx, b->z, b->out, b->gw, b->terms, b->out);
    }
}

static void gq_run_jobs(TernaryThreadPool *pool, GqJob *jobs, int nj, size_t P, const float *xf, size_t xf_stride, const float *sx, float *const *gs) {
    size_t ntask = 0;
    for (int j = 0; j < nj; j++) { jobs[j].nch = (jobs[j].out + GQA_ROW_GRAIN - 1) / GQA_ROW_GRAIN; ntask += jobs[j].nch; }
    GqJobs c = { jobs, nj, P, xf, xf_stride, sx };
    ternary_parallel_ranges(pool, 0, ntask, 1, gq_jobs_range, &c);
    for (int j = 0; j < nj; j++) {                       /* mismo orden que el calculo secuencial: p ascendente, o ascendente */
        float acc = *gs[j];
        for (size_t p = 0; p < P; p++) for (size_t o = 0; o < jobs[j].out; o++) acc += jobs[j].terms[p * jobs[j].out + o];
        *gs[j] = acc;
    }
}

/* Scratch propio de cada tarea de la ventana (nada compartido en g mientras hay varios hilos). */
typedef struct {
    int32_t *scores; float *probs, *ds, *w_o;
    int32_t *sc_blk, *m_blk; float *e_blk, *p_blk, *ds_blk, *qf_blk, *inv_blk; const int32_t **qptr;
    float *dmy;            /* descarte de la parte del backward que no se pide: G*H (gq) + 4*H (gk, gv de cada tramo) */
} WinScr;
static void wscr_free(WinScr *S) {
    free(S->scores); free(S->probs); free(S->ds); free(S->w_o);
    free(S->sc_blk); free(S->m_blk); free(S->e_blk); free(S->p_blk); free(S->ds_blk); free(S->qf_blk); free(S->inv_blk); free((void *)S->qptr); free(S->dmy);
}
static void wscr_init(WinScr *S, const TernaryGQA *g) {
    memset(S, 0, sizeof *S);
    const size_t G = g->group_size, tp = g->tpad;
    S->scores = (int32_t *)ternary_malloc64(g->w_n * sizeof(int32_t)); S->probs = (float *)ternary_malloc64(g->w_n * sizeof(float));
    S->ds = (float *)ternary_malloc64(g->w_n * sizeof(float)); S->w_o = (float *)ternary_malloc64(G * g->head_dim * sizeof(float));
    int ok = S->scores && S->probs && S->ds && S->w_o;
    if (g->w_kT) {
        S->sc_blk = (int32_t *)ternary_calloc64(G * tp, sizeof(int32_t)); S->e_blk = (float *)ternary_calloc64(G * tp, sizeof(float));
        S->p_blk = (float *)ternary_calloc64(G * tp, sizeof(float)); S->ds_blk = (float *)ternary_calloc64(G * tp, sizeof(float));
        S->qf_blk = (float *)ternary_calloc64(G * g->head_dim, sizeof(float)); S->qptr = (const int32_t **)ternary_calloc64(G, sizeof(int32_t *));
        S->m_blk = (int32_t *)ternary_calloc64(3 * G, sizeof(int32_t)); S->inv_blk = (float *)ternary_calloc64(G, sizeof(float));
        S->dmy = (float *)ternary_calloc64(G * g->head_dim + 4 * g->head_dim, sizeof(float));
        ok = ok && S->sc_blk && S->e_blk && S->p_blk && S->ds_blk && S->qf_blk && S->qptr && S->m_blk && S->inv_blk && S->dmy;
    }
    if (!ok) abort();
}

/* Puntajes, maximo, exponenciales (en probs) y 1/suma de una cabeza. La suma usa 16 parciales por carril y el arbol fijo
 * de ternary_sum16; cada tramo empieza en el carril 0 (carril = (t < s ? t : t - s) & 15), igual que el nucleo AVX-512. */
static int32_t win_row_stats(const TernaryGQA *g, WinScr *S, const WinKeys *wk, const int32_t *qh, size_t kv_h, float *inv_out) {
    const size_t H = g->head_dim, kvw = g->n_kv_heads * H;
    const int32_t lim = g->qk_lim; const int sh = 16 + g->qk_shift;
    const float *restrict tab = g->exp_tab; const uint32_t K0 = g->exp_k0;
    int32_t m = 0;
    for (size_t t = 0; t < wk->n; t++) {
        const int32_t *b = g->w_kr + WKSLOT(*wk, t) * kvw + kv_h * H;
        int64_t dot = 0;
        for (size_t d = 0; d < H; d++) dot += (int64_t)qh[d] * (int64_t)b[d];
        dot >>= sh;
        if (dot > lim) dot = lim; else if (dot < -lim) dot = -lim;
        S->scores[t] = (int32_t)dot;
        if (t == 0 || (int32_t)dot > m) m = (int32_t)dot;
    }
    float part[16] = {0};
    for (size_t t = 0; t < wk->n; t++) {
        const uint32_t k = (uint32_t)(m - S->scores[t]);
        const float e = tab[k < K0 ? k : K0];
        S->probs[t] = e; part[(t < wk->s ? t : t - wk->s) & 15] += e;
    }
    *inv_out = 1.0f / ternary_sum16(part);
    return m;
}

/* acc = (sum_t e_t * v_t) * inv, t ascendente (mismo orden que la atencion flash completa). */
static void win_row_pv(const TernaryGQA *g, WinScr *S, const WinKeys *wk, size_t kv_h, float inv, float *restrict acc) {
    const size_t H = g->head_dim, kvw = g->n_kv_heads * H;
    memset(acc, 0, H * sizeof(float));
    for (size_t t = 0; t < wk->n; t++) {
        const float e = S->probs[t];
        const float *restrict vj = g->w_vf + WKSLOT(*wk, t) * kvw + kv_h * H;
        for (size_t d = 0; d < H; d++) acc[d] += e * vj[d];
    }
    for (size_t d = 0; d < H; d++) acc[d] *= inv;
}

/* Camino AVX-512 de una fila: el grupo GQA completo por tramo de claves. */
static void win_avx_scores(const TernaryGQA *g, WinScr *S, const WinKeys *wk, size_t kh, const int32_t *qrow) {
    const size_t H = g->head_dim, G = g->group_size, tp = g->tpad, ld = g->w_ld, spA = (wk->s + 15) / 16 * 16;
    const int32_t *kTb = g->w_kT + kh * H * ld;
    int32_t *mA = S->m_blk + G, *mB = S->m_blk + 2 * G;
    for (size_t b = 0; b < G; b++) S->qptr[b] = qrow + (kh * G + b) * H;
    TK_CALL(attn_scores, S->qptr, G, kTb, ld, H, wk->s, 16 + g->qk_shift, g->qk_lim, S->sc_blk, tp, mA);
    TK_CALL(attn_scores, S->qptr, G, kTb + wk->slB, ld, H, wk->nB, 16 + g->qk_shift, g->qk_lim, S->sc_blk + spA, tp, mB);
    for (size_t b = 0; b < G; b++) S->m_blk[b] = mA[b] > mB[b] ? mA[b] : mB[b];
}

static void win_row_forward(const TernaryGQA *g, WinScr *S, size_t i, const int32_t *qrow, float *out) {
    const WinKeys wk = win_keys(g, i); const size_t H = g->head_dim;
    if (g->w_kT) {
        const size_t G = g->group_size, kvw = g->n_kv_heads * H, tp = g->tpad, spA = (wk.s + 15) / 16 * 16;
        for (size_t kh = 0; kh < g->n_kv_heads; kh++) {
            win_avx_scores(g, S, &wk, kh, qrow);
            TK_CALL(attn_pv2, G, S->sc_blk, tp, S->m_blk, g->exp_tab, g->exp_k0, wk.s, wk.nB, spA,
                                    g->w_vf + kh * H, g->w_vf + wk.slB * kvw + kh * H, kvw, H, S->e_blk, tp,
                                    out + kh * G * H, S->inv_blk);
        }
        return;
    }
    for (size_t h = 0; h < g->n_heads; h++) {
        const size_t kv_h = h / g->group_size; float inv;
        win_row_stats(g, S, &wk, qrow + h * H, kv_h, &inv);
        win_row_pv(g, S, &wk, kv_h, inv, out + h * H);
    }
}

static void win_row_backward_kv(TernaryGQA *g, WinScr *S, size_t kvh, size_t i, const int32_t *qrow, const float *ga_row, float *gq_row) {
    const WinKeys wk = win_keys(g, i);
    const size_t H = g->head_dim, kvw = g->n_kv_heads * H;
    const float it = g->softmax->inv_temp;
    const float gam = ldexpf(1.0f, -g->qk_shift) / (float)TERNARY_ROPE_SCALE;
    if (g->w_kT) {
        const size_t G = g->group_size, tp = g->tpad, ld = g->w_ld, spA = (wk.s + 15) / 16 * 16;
        {
            const size_t kh = kvh;
            win_avx_scores(g, S, &wk, kh, qrow);
            TK_CALL(attn_pv2, G, S->sc_blk, tp, S->m_blk, g->exp_tab, g->exp_k0, wk.s, wk.nB, spA,
                                    g->w_vf + kh * H, g->w_vf + wk.slB * kvw + kh * H, kvw, H, S->e_blk, tp,
                                    S->w_o, S->inv_blk);                       /* O recalculada */
            const float *vTA = g->w_vT + kh * H * ld;
            TK_CALL(attn_bwd2, G, wk.s, wk.nB, spA, H, S->sc_blk, tp, S->m_blk, S->inv_blk, g->exp_tab, g->exp_k0,
                                     ga_row + kh * G * H, S->w_o, vTA, vTA + wk.slB, ld, it, gam,
                                     qrow + kh * G * H, g->w_kr + kh * H, g->w_kr + wk.slB * kvw + kh * H, kvw,
                                     gq_row + kh * G * H, g->w_gk + kh * H, g->w_gk + wk.slB * kvw + kh * H,
                                     g->w_gv + kh * H, g->w_gv + wk.slB * kvw + kh * H, kvw,
                                     S->p_blk, S->ds_blk, tp, S->qf_blk);
        }
        return;
    }
    float *restrict dpr = S->ds;
    const size_t n = wk.n;
    for (size_t h = kvh * g->group_size; h < (kvh + 1) * g->group_size; h++) {
        const size_t kv_h = kvh;
        const float *ga_h = ga_row + h * H;
        const int32_t *restrict qr = qrow + h * H;
        float *restrict gq = gq_row + h * H;
        float inv;
        win_row_stats(g, S, &wk, qr, kv_h, &inv);
        win_row_pv(g, S, &wk, kv_h, inv, S->w_o);          /* O recalculada */
        float D = 0.0f;
        for (size_t d = 0; d < H; d++) D += ga_h[d] * S->w_o[d];
        size_t t4 = 0;
        for (; t4 + 4 <= n; t4 += 4) {
            const float *v0 = g->w_vf + WKSLOT(wk, t4) * kvw + kv_h * H, *v1 = g->w_vf + WKSLOT(wk, t4 + 1) * kvw + kv_h * H;
            const float *v2 = g->w_vf + WKSLOT(wk, t4 + 2) * kvw + kv_h * H, *v3 = g->w_vf + WKSLOT(wk, t4 + 3) * kvw + kv_h * H;
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
            for (size_t d = 0; d < H; d++) {
                const float gd = ga_h[d];
                d0 += gd * v0[d]; d1 += gd * v1[d]; d2 += gd * v2[d]; d3 += gd * v3[d];
            }
            dpr[t4] = d0; dpr[t4 + 1] = d1; dpr[t4 + 2] = d2; dpr[t4 + 3] = d3;
        }
        for (; t4 < n; t4++) {
            const float *restrict vj = g->w_vf + WKSLOT(wk, t4) * kvw + kv_h * H;
            float dot = 0.0f;
            for (size_t d = 0; d < H; d++) dot += ga_h[d] * vj[d];
            dpr[t4] = dot;
        }
        for (size_t t = 0; t < n; t++) {
            const float p = S->probs[t] * inv;
            const float ds = ((it * p) * (dpr[t] - D)) * gam;
            const size_t sl = WKSLOT(wk, t);
            float *restrict gv_j = g->w_gv + sl * kvw + kv_h * H;
            float *restrict gk = g->w_gk + sl * kvw + kv_h * H;
            const int32_t *restrict kr = g->w_kr + sl * kvw + kv_h * H;
            for (size_t d = 0; d < H; d++) gv_j[d] += p * ga_h[d];
            for (size_t d = 0; d < H; d++) {
                gq[d] += ds * (float)kr[d];
                gk[d] += ds * (float)qr[d];
            }
        }
    }
}

/* Proyecta Q/K/V de la posicion p (Dense forward entero) hacia el anillo; la Q rotada va a qrot_out. Si keep_z, deja
 * los Z crudos en la ranura (el backward los necesita para Wq/Wk/Wv hacia atras). */
static void win_project(const TernaryGQA *g, const TernaryVec *x, const TernaryU8 *xu, size_t p, int32_t *qrot_out, int keep_z,
                        const TernaryCapabilities *caps, float *tmpf, int32_t *tmpi, int32_t *u8t,
                        const int32_t *zq, const int32_t *zk, const int32_t *zv) {   /* zq/zk/zv != NULL: Z ya calculados por lotes (solo ruta uint8) */
    const size_t E = g->embed_dim, qw = g->n_heads * g->head_dim, kvw = g->n_kv_heads * g->head_dim, sl = wslot(g, p);
    float sx = 1.0f;
    if (xu) sx = xu->scale[p]; else extract_words(x, p * E, E, g->x_pos);
    const uint8_t *xq = xu ? xu->q + p * E : NULL; const int32_t xzp = xu ? xu->zp[p] : 0;
    int32_t *qz = keep_z ? g->w_qz + sl * qw : tmpi;
    if (xu) {
        if (zq) qz = (int32_t *)(uintptr_t)zq; else ternary_dense_forward_u8_row(g->wq, xq, sx, xzp, tmpf, qz);
        gq_shift(qz, qw, u8t);
        rope_rotate_int(g, p, u8t, g->n_heads, qrot_out);
    } else {
        ternary_dense_forward_auto(g->wq, g->x_pos, tmpf, qz, caps);
        rope_rotate_int(g, p, qz, g->n_heads, qrot_out);
    }
    int32_t *kz = g->w_kz + sl * kvw, *vz = g->w_vz + sl * kvw;
    if (!keep_z) { kz = tmpi; }
    if (xu) {
        if (zk) kz = (int32_t *)(uintptr_t)zk; else ternary_dense_forward_u8_row(g->wk, xq, sx, xzp, tmpf, kz);
        gq_shift(kz, kvw, u8t);
        rope_rotate_int(g, p, u8t, g->n_kv_heads, g->w_kr + sl * kvw);
    } else {
        ternary_dense_forward_auto(g->wk, g->x_pos, tmpf, kz, caps);
        rope_rotate_int(g, p, kz, g->n_kv_heads, g->w_kr + sl * kvw);
    }
    if (!keep_z) vz = tmpi;
    if (xu) { if (zv) vz = (int32_t *)(uintptr_t)zv; else ternary_dense_forward_u8_row(g->wv, xq, sx, xzp, tmpf, vz); }
    else ternary_dense_forward_auto(g->wv, g->x_pos, tmpf, vz, caps);
    float *vf = g->w_vf + sl * kvw;
    gq_scale(vz, kvw, g->wv->scale * (xu ? sx : 1.0f), vf, 1);
    gq_vq_row(vf, kvw);
    if (g->w_kT) {
        const int32_t *kr = g->w_kr + sl * kvw;
        for (size_t i = 0; i < kvw; i++) { g->w_kT[i * g->w_ld + sl] = kr[i]; g->w_vT[i * g->w_ld + sl] = vf[i]; }
    }
}

static void win_caps(TernaryCapabilities *caps) {
    memset(caps, 0, sizeof(*caps)); caps->num_cores = 1; caps->dot_impl_name = "scalar";
    TernaryThreadPool *pl = ternary_get_compute_pool();
    if (pl && ternary_threadpool_num_threads(pl) > 1) caps->cpu_pool = pl;
}

/* ---- ventana repartida en hilos ----
 * Por bloque de posiciones: proyecciones por posicion (ruta uint8), atencion por posicion (forward) o por cabeza KV (backward: gk/gv se acumulan
 * en orden de fila), Wo y Wq/Wk/Wv como en la atencion completa (fase por posicion + fase por filas de pesos, terminos de la escala sumados en
 * el orden de siempre). Mismo resultado con cualquier numero de hilos. */
typedef struct {
    TernaryGQA *g; const TernaryVec *x; const TernaryU8 *xu; size_t p0, S; float *out; int32_t *raw; int keep_z;
    const float *grad_out; float *xf; float *grad_x_out;
    const TernaryPlanes *pwo, *pwq, *pwk, *pwv;   /* planos +1/-1 de los pesos para el backward (NULL sin AVX-512) */
    const TernaryFwdW *pfq, *pfk, *pfv, *pfo;      /* pesos int8 para el forward entero por lotes (NULL sin VNNI) */
} WinCtx;
#define WIN_SUB 16   /* posiciones por tanda del grad_x por lotes */

static void win_proj_range(size_t c0, size_t c1, void *c_) {
    WinCtx *c = (WinCtx *)c_;
    TernaryGQA *g = c->g;
    const size_t E = g->embed_dim, qw = g->n_heads * g->head_dim, kvw = g->n_kv_heads * g->head_dim;
    size_t wide = qw > kvw ? qw : kvw; if (E > wide) wide = E;
    float *tmpf = (float *)ternary_malloc64(wide * sizeof(float));
    int32_t *tmpi = (int32_t *)ternary_malloc64(wide * sizeof(int32_t)), *u8t = (int32_t *)ternary_malloc64(wide * sizeof(int32_t));
    int32_t *zs = c->keep_z ? NULL : (int32_t *)ternary_malloc64(WIN_SUB * (qw + 2 * kvw) * sizeof(int32_t));   /* Z de la tanda si no se guardan */
    if (!tmpf || !tmpi || !u8t || (!c->keep_z && !zs)) abort();
    for (size_t ka = c0; ka < c1; ka += WIN_SUB) {
        const size_t kb = (ka + WIN_SUB < c1) ? ka + WIN_SUB : c1, n = kb - ka;
        /* Z de Q/K/V de toda la tanda por lotes (las casillas de la tanda deben ser consecutivas: no lo son al cruzar el borde de los sinks) */
        const size_t sl0 = wslot(g, c->p0 + ka);
        int batch = c->pfq && (wslot(g, c->p0 + kb - 1) - sl0 == n - 1);
        int32_t *zq = NULL, *zk = NULL, *zv = NULL;
        if (batch) {
            zq = c->keep_z ? g->w_qz + sl0 * qw : zs;
            zk = c->keep_z ? g->w_kz + sl0 * kvw : zs + WIN_SUB * qw;
            zv = c->keep_z ? g->w_vz + sl0 * kvw : zs + WIN_SUB * (qw + kvw);
            const size_t p0 = c->p0 + ka;
            ternary_dense_forward_u8_batch(g->wq, c->pfq, n, c->xu->q + p0 * E, E, c->xu->scale + p0, c->xu->zp + p0, NULL, 0, zq, qw);
            ternary_dense_forward_u8_batch(g->wk, c->pfk, n, c->xu->q + p0 * E, E, c->xu->scale + p0, c->xu->zp + p0, NULL, 0, zk, kvw);
            ternary_dense_forward_u8_batch(g->wv, c->pfv, n, c->xu->q + p0 * E, E, c->xu->scale + p0, c->xu->zp + p0, NULL, 0, zv, kvw);
        }
        for (size_t k = ka; k < kb; k++) {
            const size_t p = c->p0 + k, sl = wslot(g, p), i = k - ka;
            win_project(g, c->x, c->xu, p, g->w_qrot + k * qw, c->keep_z, NULL, tmpf, tmpi, u8t,
                        batch ? zq + i * qw : NULL, batch ? zk + i * kvw : NULL, batch ? zv + i * kvw : NULL);
            if (c->keep_z) {
                memset(g->w_gq + sl * qw, 0, qw * sizeof(float));
                memset(g->w_gk + sl * kvw, 0, kvw * sizeof(float));
                memset(g->w_gv + sl * kvw, 0, kvw * sizeof(float));
            }
        }
    }
    free(tmpf); free(tmpi); free(u8t); free(zs);
}

static void win_rows_fwd_range(size_t c0, size_t c1, void *c_) {
    WinCtx *c = (WinCtx *)c_;
    TernaryGQA *g = c->g;
    const size_t qw = g->n_heads * g->head_dim;
    WinScr S; wscr_init(&S, g);
    for (size_t k = c0; k < c1; k++) win_row_forward(g, &S, c->p0 + k, g->w_qrot + k * qw, g->w_attn + k * qw);
    wscr_free(&S);
}

static void win_out_range(size_t c0, size_t c1, void *c_) {
    WinCtx *c = (WinCtx *)c_;
    TernaryGQA *g = c->g;
    const size_t E = g->embed_dim, qw = g->n_heads * g->head_dim;
    for (size_t ka = c0; ka < c1; ka += WIN_SUB) {
        const size_t kb = (ka + WIN_SUB < c1) ? ka + WIN_SUB : c1, i0 = c->p0 + ka;
        for (size_t k = ka; k < kb; k++) {
            const size_t i = c->p0 + k;
            TernaryU8 row = { g->attn_u8->q + i * qw, g->attn_u8->scale + i, g->attn_u8->zp + i, 1, qw };
            ternary_u8_quantize(&row, g->w_attn + k * qw);
        }
        ternary_dense_forward_u8_batch(g->wo, c->pfo, kb - ka, g->attn_u8->q + i0 * qw, qw, g->attn_u8->scale + i0, g->attn_u8->zp + i0,
                                       c->out + i0 * E, E, c->raw ? c->raw + i0 * E : g->w_zo + ka * E, E);
    }
}

static void win_rows_bwd_range(size_t k0, size_t k1, void *c_) {
    WinCtx *c = (WinCtx *)c_;
    TernaryGQA *g = c->g;
    const size_t qw = g->n_heads * g->head_dim;
    WinScr S; wscr_init(&S, g);
    for (size_t kh = k0; kh < k1; kh++)
        for (size_t k = 0; k < c->S; k++)
            win_row_backward_kv(g, &S, kh, c->p0 + k, g->w_qrot + k * qw, g->w_attn + k * qw, g->w_gq + wslot(g, c->p0 + k) * qw);
    wscr_free(&S);
}

/* ---- atencion con ventana hacia atras, partida para mas hilos que el doble de cabezas KV ----
 * Igual que en la atencion completa: gk/gv de cada clave se suman en el mismo orden de siempre (filas ascendentes, cabezas ascendentes) y
 * gq de cada fila en el de siempre (tramo de sinks y luego ventana, claves ascendentes); por eso se parte en tareas de CLAVES (por bloque de
 * posiciones, solo gk/gv) y tareas de FILAS (solo gq), que escriben salidas distintas y corren a la vez. Lo que la fila necesita de TODAS sus
 * claves (maximo, 1/suma y la salida O para D = dO.O) se calcula antes, una vez por fila y cabeza KV, con las mismas funciones del camino completo.
 * La parte que una tarea no pide corre contra un buffer de descarte (mismo nucleo AVX-512). */
#define WIN_KB 32      /* posiciones por tarea de claves (multiplo de 16) */
#define WIN_RB 8       /* filas por tarea de filas */
typedef struct {
    WinCtx *W; float *ochunk; int32_t *mrow; float *invrow;      /* por fila del bloque: O (qw), maximo y 1/suma por cabeza (NQ) */
    size_t nAblk, wb_first, wb_cnt, nkb, nrb;
} WinSplit;

static void win_prestats_range(size_t t0, size_t t1, void *c_) {
    WinSplit *c = (WinSplit *)c_; WinCtx *W = c->W; TernaryGQA *g = W->g;
    const size_t H = g->head_dim, G = g->group_size, NQ = g->n_heads, qw = NQ * H, kvw = g->n_kv_heads * H;
    WinScr S; wscr_init(&S, g);
    for (size_t t = t0; t < t1; t++) {
        const size_t kh = t / W->S, k = t % W->S, i = W->p0 + k;
        const WinKeys wk = win_keys(g, i);
        const int32_t *qrow = g->w_qrot + k * qw;
        float *o = c->ochunk + k * qw + kh * G * H;
        if (g->w_kT) {
            const size_t tp = g->tpad, spA = (wk.s + 15) / 16 * 16;
            win_avx_scores(g, &S, &wk, kh, qrow);
            TK_CALL(attn_pv2, G, S.sc_blk, tp, S.m_blk, g->exp_tab, g->exp_k0, wk.s, wk.nB, spA,
                                    g->w_vf + kh * H, g->w_vf + wk.slB * kvw + kh * H, kvw, H, S.e_blk, tp, o, S.inv_blk);
            for (size_t b = 0; b < G; b++) { c->mrow[k * NQ + kh * G + b] = S.m_blk[b]; c->invrow[k * NQ + kh * G + b] = S.inv_blk[b]; }
        } else {
            for (size_t h = kh * G; h < (kh + 1) * G; h++) {
                float inv;
                const int32_t m = win_row_stats(g, &S, &wk, qrow + h * H, kh, &inv);
                win_row_pv(g, &S, &wk, kh, inv, o + (h - kh * G) * H);
                c->mrow[k * NQ + h] = m; c->invrow[k * NQ + h] = inv;
            }
        }
    }
    wscr_free(&S);
}

/* Fila k del bloque, cabeza KV kh, claves en las posiciones [a, b): doA = gk/gv de esas claves, doB = gq de la fila (entonces [a, b) cubre todo). */
static void win_bwd_part(WinSplit *c, WinScr *S, size_t kh, size_t k, size_t a, size_t b, int doA, int doB) {
    WinCtx *W = c->W; TernaryGQA *g = W->g;
    const size_t i = W->p0 + k;
    const WinKeys wk = win_keys(g, i);
    const size_t H = g->head_dim, G = g->group_size, NQ = g->n_heads, qw = NQ * H, kvw = g->n_kv_heads * H;
    const float it = g->softmax->inv_temp;
    const float gam = ldexpf(1.0f, -g->qk_shift) / (float)TERNARY_ROPE_SCALE;
    const size_t jw = i + 1 - wk.nB;                                  /* primera posicion de la ventana */
    const size_t a0 = a, a1 = (b < wk.s) ? b : wk.s, nA = a1 > a0 ? a1 - a0 : 0;               /* tramo de sinks dentro de [a, b) */
    const size_t b0 = a > jw ? a : jw, b1 = (b < i + 1) ? b : i + 1, nBp = b1 > b0 ? b1 - b0 : 0;   /* tramo de ventana dentro de [a, b) */
    if (nA + nBp == 0) return;
    const size_t slb = nBp ? wslot(g, b0) : 0;
    const int32_t *qrow = g->w_qrot + k * qw;
    const float *ga_row = g->w_attn + k * qw, *ochunk = c->ochunk + k * qw;
    float *gq_row = g->w_gq + wslot(g, i) * qw;
    if (g->w_kT) {
        const size_t tp = g->tpad, ld = g->w_ld, spA = (nA + 15) / 16 * 16;
        const int32_t *kTb = g->w_kT + kh * H * ld;
        int32_t *mA = S->m_blk + G, *mB = S->m_blk + 2 * G;
        for (size_t bb = 0; bb < G; bb++) S->qptr[bb] = qrow + (kh * G + bb) * H;
        TK_CALL(attn_scores, S->qptr, G, kTb + a0, ld, H, nA, 16 + g->qk_shift, g->qk_lim, S->sc_blk, tp, mA);
        TK_CALL(attn_scores, S->qptr, G, kTb + slb, ld, H, nBp, 16 + g->qk_shift, g->qk_lim, S->sc_blk + spA, tp, mB);
        for (size_t bb = 0; bb < G; bb++) { S->m_blk[bb] = c->mrow[k * NQ + kh * G + bb]; S->inv_blk[bb] = c->invrow[k * NQ + kh * G + bb]; }
        float *dq = S->dmy, *dk = S->dmy + G * H;
        float *gq_p = doB ? gq_row + kh * G * H : dq;
        float *gkA = doA ? g->w_gk + a0 * kvw + kh * H : dk, *gkB = doA ? g->w_gk + slb * kvw + kh * H : dk + H;
        float *gvA = doA ? g->w_gv + a0 * kvw + kh * H : dk + 2 * H, *gvB = doA ? g->w_gv + slb * kvw + kh * H : dk + 3 * H;
        const float *vT = g->w_vT + kh * H * ld;
        TK_CALL(attn_bwd2, G, nA, nBp, spA, H, S->sc_blk, tp, S->m_blk, S->inv_blk, g->exp_tab, g->exp_k0,
                                 ga_row + kh * G * H, ochunk + kh * G * H, vT + a0, vT + slb, ld, it, gam,
                                 qrow + kh * G * H, g->w_kr + a0 * kvw + kh * H, g->w_kr + slb * kvw + kh * H, kvw,
                                 gq_p, gkA, gkB, gvA, gvB, doA ? kvw : 0,
                                 S->p_blk, S->ds_blk, tp, S->qf_blk);
        return;
    }
    const int sh = 16 + g->qk_shift; const int32_t lim = g->qk_lim;
    const float *restrict tab = g->exp_tab; const uint32_t K0 = g->exp_k0;
    for (size_t h = kh * G; h < (kh + 1) * G; h++) {
        const float *ga_h = ga_row + h * H;
        const int32_t *restrict qr = qrow + h * H;
        float *restrict gq = gq_row + h * H;
        const int32_t m = c->mrow[k * NQ + h]; const float inv = c->invrow[k * NQ + h];
        float D = 0.0f;
        for (size_t d = 0; d < H; d++) D += ga_h[d] * ochunk[h * H + d];
        for (size_t u = 0; u < nA + nBp; u++) {
            const size_t sl = u < nA ? a0 + u : wslot(g, b0 + (u - nA));
            const int32_t *restrict kr = g->w_kr + sl * kvw + kh * H;
            int64_t dot = 0;
            for (size_t d = 0; d < H; d++) dot += (int64_t)qr[d] * (int64_t)kr[d];
            dot >>= sh;
            if (dot > lim) dot = lim; else if (dot < -lim) dot = -lim;
            const uint32_t kk = (uint32_t)(m - (int32_t)dot);
            const float p = tab[kk < K0 ? kk : K0] * inv;
            const float *restrict vj = g->w_vf + sl * kvw + kh * H;
            float dpr = 0.0f;
            for (size_t d = 0; d < H; d++) dpr += ga_h[d] * vj[d];
            const float ds = ((it * p) * (dpr - D)) * gam;
            if (doA) {
                float *restrict gv_j = g->w_gv + sl * kvw + kh * H, *restrict gk = g->w_gk + sl * kvw + kh * H;
                for (size_t d = 0; d < H; d++) gv_j[d] += p * ga_h[d];
                for (size_t d = 0; d < H; d++) gk[d] += ds * (float)qr[d];
            }
            if (doB) for (size_t d = 0; d < H; d++) gq[d] += ds * (float)kr[d];
        }
    }
}

static void win_bwd_split_range(size_t t0, size_t t1, void *c_) {
    WinSplit *c = (WinSplit *)c_; WinCtx *W = c->W; TernaryGQA *g = W->g;
    const size_t NKV = g->n_kv_heads, S_ = W->S, ntA = NKV * c->nkb;
    WinScr S; wscr_init(&S, g);
    for (size_t t = t0; t < t1; t++) {
        if (t < ntA) {                                                 /* claves: bloque de posiciones, filas ascendentes */
            const size_t kh = t / c->nkb, u = t % c->nkb;
            const size_t blk = u < c->nAblk ? u : c->wb_first + (u - c->nAblk);
            const size_t a = blk * WIN_KB, b = a + WIN_KB;
            for (size_t k = 0; k < S_; k++) if (W->p0 + k >= a) win_bwd_part(c, &S, kh, k, a, b, 1, 0);
        } else {                                                       /* filas: gq con todas las claves */
            const size_t v = t - ntA, kh = v / c->nrb, r0 = (v % c->nrb) * WIN_RB, r1 = (r0 + WIN_RB < S_) ? r0 + WIN_RB : S_;
            for (size_t k = r0; k < r1; k++) win_bwd_part(c, &S, kh, k, 0, W->p0 + S_, 0, 1);
        }
    }
    wscr_free(&S);
}

/* Un bloque de filas de la ventana hacia atras con el reparto partido (mismos bits que win_rows_bwd_range). */
static void win_bwd_rows_split(TernaryThreadPool *pool, WinCtx *W) {
    TernaryGQA *g = W->g;
    const size_t NKV = g->n_kv_heads, NQ = g->n_heads, qw = NQ * g->head_dim, S = W->S;
    WinSplit c;
    c.W = W;
    c.ochunk = (float *)ternary_malloc64(S * qw * sizeof(float));
    c.mrow = (int32_t *)ternary_malloc64(S * NQ * sizeof(int32_t));
    c.invrow = (float *)ternary_malloc64(S * NQ * sizeof(float));
    if (!c.ochunk || !c.mrow || !c.invrow) abort();
    ternary_parallel_ranges(pool, 0, NKV * S, 1, win_prestats_range, &c);
    const size_t end = W->p0 + S;
    const size_t lo_win = (W->p0 + 1 > g->window) ? W->p0 + 1 - g->window : 0;
    c.nAblk = (g->sinks + WIN_KB - 1) / WIN_KB;
    const size_t wb0 = lo_win / WIN_KB, wb1 = (end + WIN_KB - 1) / WIN_KB;
    c.wb_first = wb0 < c.nAblk ? c.nAblk : wb0;
    c.wb_cnt = wb1 > c.wb_first ? wb1 - c.wb_first : 0;
    c.nkb = c.nAblk + c.wb_cnt;
    c.nrb = (S + WIN_RB - 1) / WIN_RB;
    ternary_parallel_ranges(pool, 0, NKV * (c.nkb + c.nrb), 1, win_bwd_split_range, &c);
    free(c.ochunk); free(c.mrow); free(c.invrow);
}

static void win_forward(const TernaryGQA *g, const TernaryVec *x, const TernaryU8 *xu, float *out, int32_t *out_raw_opt) {
    const size_t E = g->embed_dim, qw = g->n_heads * g->head_dim, C = g->chunk;
    TernaryCapabilities caps; win_caps(&caps);
    TernaryThreadPool *pool = ternary_get_compute_pool();
    ((TernaryGQA *)g)->w_base = g->w_C0;
    if (!xu) { memset(g->attn_bits->mask, 0, g->attn_bits->words * sizeof(uint64_t)); memset(g->attn_bits->sign, 0, g->attn_bits->words * sizeof(uint64_t)); }
    TernaryFwdW *pfq = xu ? ternary_dense_fwdw_create(g->wq) : NULL, *pfk = xu ? ternary_dense_fwdw_create(g->wk) : NULL,
                *pfv = xu ? ternary_dense_fwdw_create(g->wv) : NULL, *pfo = xu ? ternary_dense_fwdw_create(g->wo) : NULL;
    WinCtx W = { (TernaryGQA *)g, x, xu, 0, 0, out, out_raw_opt, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, pfq, pfk, pfv, pfo };
    for (size_t p0 = 0; p0 < g->seq_len; p0 += C) {
        const size_t S = (g->seq_len - p0 < C) ? (g->seq_len - p0) : C;
        W.p0 = p0; W.S = S;
        { const size_t j0 = (p0 + 1 > g->window) ? p0 + 1 - g->window : 0;
          win_ensure(g, p0 + S - 1, j0 > g->w_C0 ? j0 : g->w_C0, p0); }
        TPROF_BEGIN("w_fwd_proj");
        if (xu) ternary_parallel_ranges(pool, 0, S, 1, win_proj_range, &W);
        else for (size_t c = 0; c < S; c++) win_project(g, x, xu, p0 + c, g->w_qrot + c * qw, 0, &caps, g->w_tmpf, g->w_tmpi, g->u8_tmp, NULL, NULL, NULL);
        TPROF_END();
        TPROF_BEGIN("w_fwd_rows");
        ternary_parallel_ranges(pool, 0, S, 1, win_rows_fwd_range, &W);
        TPROF_END();
        if (xu) { TPROF_BEGIN("w_fwd_out"); ternary_parallel_ranges(pool, 0, S, 1, win_out_range, &W); TPROF_END(); continue; }
        for (size_t c = 0; c < S; c++) {
            const size_t i = p0 + c;
            for (size_t k = 0; k < qw; k++) set_sign_f(g->attn_bits, i * qw + k, g->w_attn[c * qw + k]);
        }
        for (size_t c = 0; c < S; c++) {
            const size_t i = p0 + c;
            extract_words(g->attn_bits, i * qw, qw, g->rc[0]);
            ternary_dense_forward_auto(g->wo, g->rc[0], out + i * E, out_raw_opt ? out_raw_opt + i * E : g->w_zo, &caps);
        }
    }
    ternary_dense_fwdw_free(pfq); ternary_dense_fwdw_free(pfk); ternary_dense_fwdw_free(pfv); ternary_dense_fwdw_free(pfo);
}

/* ---- finalizacion de un bloque (Wq/Wk/Wv hacia atras) ---- */
static void win_fin_pos(size_t k0, size_t k1, void *c_) {
    WinCtx *c = (WinCtx *)c_;
    TernaryGQA *g = c->g;
    const size_t E = g->embed_dim, H = g->head_dim, qw = g->n_heads * H, kvw = g->n_kv_heads * H;
    for (size_t ka = k0; ka < k1; ka += WIN_SUB) {
        const size_t kb = (ka + WIN_SUB < k1) ? ka + WIN_SUB : k1, n = kb - ka;
        for (size_t k = ka; k < kb; k++) {
            const size_t p = c->p0 + k, sl = wslot(g, p);
            float *gq = g->w_gq + sl * qw, *gk = g->w_gk + sl * kvw;
            rope_unrotate_grad(g, p, gq, g->n_heads);
            rope_unrotate_grad(g, p, gk, g->n_kv_heads);
            const float sx = c->xu->scale[p];
            {   /* mismo arreglo de escala que en ternary_gqa_backward_u8: Q/K salen de z>>8, no de scale*sx*z */
                const float fq = 1.0f / ((float)(1 << TERNARY_GQA_U8_SHIFT) * g->wq->scale * sx), fk = 1.0f / ((float)(1 << TERNARY_GQA_U8_SHIFT) * g->wk->scale * sx);
                for (size_t i = 0; i < qw; i++) gq[i] *= fq;
                for (size_t i = 0; i < kvw; i++) gk[i] *= fk;
            }
        }
        memset(g->gxc + ka * E, 0, n * E * sizeof(float));
        /* las posiciones de la tanda ocupan casillas consecutivas salvo al cruzar el borde de los sinks: ahi, de una en una */
        const size_t sa = wslot(g, c->p0 + ka);
        const size_t step = (wslot(g, c->p0 + kb - 1) - sa == n - 1) ? n : 1;
        for (size_t k = ka; k < kb; k += step) {
            const size_t m = (kb - k < step) ? kb - k : step, sl = wslot(g, c->p0 + k);
            float *gx = g->gxc + k * E;
            ternary_dense_u8_gradx_batch(g->wq, c->pwq, m, g->w_gq + sl * qw, qw, gx, E);
            ternary_dense_u8_gradx_batch(g->wk, c->pwk, m, g->w_gk + sl * kvw, kvw, gx, E);
            ternary_dense_u8_gradx_batch(g->wv, c->pwv, m, g->w_gv + sl * kvw, kvw, gx, E);
        }
        for (size_t k = ka; k < kb; k++) {
            const size_t p = c->p0 + k;
            ternary_dense_u8_to_f32(c->xu->q + p * E, E, c->xu->scale[p], c->xu->zp[p], c->xf + k * E);
            float *dst = c->grad_x_out + p * E;
            const float *gx = g->gxc + k * E;
            for (size_t d = 0; d < E; d++) dst[d] += gx[d];
        }
    }
}

static void win_finalize(TernaryGQA *g, const TernaryVec *x, const TernaryU8 *xu, size_t q0, float *const *grad_w_out, float *grad_x_out,
                         float *gs_wq, float *gs_wk, float *gs_wv, float *xf, float *terms,
                         const TernaryPlanes *pwq, const TernaryPlanes *pwk, const TernaryPlanes *pwv) {
    const size_t E = g->embed_dim, H = g->head_dim, qw = g->n_heads * H, kvw = g->n_kv_heads * H, C = g->chunk;
    const size_t S = (g->seq_len - q0 < C) ? (g->seq_len - q0) : C;
    if (xu) {
        TernaryThreadPool *pool = ternary_get_compute_pool();
        const size_t sl = wslot(g, q0);
        WinCtx W = { g, NULL, xu, q0, S, NULL, NULL, 0, NULL, xf, grad_x_out, NULL, pwq, pwk, pwv, NULL, NULL, NULL, NULL };
        ternary_parallel_ranges(pool, 0, S, 1, win_fin_pos, &W);
        GqJob jobs[3] = {
            { g->wq, g->w_gq + sl * qw, g->w_qz + sl * qw, grad_w_out[0], terms, qw, 0 },
            { g->wk, g->w_gk + sl * kvw, g->w_kz + sl * kvw, grad_w_out[1], terms + C * qw, kvw, 0 },
            { g->wv, g->w_gv + sl * kvw, g->w_vz + sl * kvw, grad_w_out[2], terms + C * (qw + kvw), kvw, 0 } };
        float *gsp[3] = { gs_wq, gs_wk, gs_wv };
        gq_run_jobs(pool, jobs, 3, S, xf, E, xu->scale + q0, gsp);
        return;
    }
    const TernaryVec **xcv = (const TernaryVec **)g->views;
    const float **gqv = (const float **)(g->views + 3 * C), **gkv = gqv + C, **gvv = gkv + C;
    const int32_t **zqv = (const int32_t **)(g->views + 7 * C), **zkv = zqv + C, **zvv = zkv + C;
    float **gxv = (float **)(g->views + 10 * C);
    for (size_t c = 0; c < S; c++) {
        const size_t p = q0 + c, sl = wslot(g, p);
        rope_unrotate_grad(g, p, g->w_gq + sl * qw, g->n_heads);
        rope_unrotate_grad(g, p, g->w_gk + sl * kvw, g->n_kv_heads);
        extract_words(x, p * E, E, g->xc[c]);
        xcv[c] = g->xc[c];
        gqv[c] = g->w_gq + sl * qw;  zqv[c] = g->w_qz + sl * qw;
        gkv[c] = g->w_gk + sl * kvw; zkv[c] = g->w_kz + sl * kvw;
        gvv[c] = g->w_gv + sl * kvw; zvv[c] = g->w_vz + sl * kvw;
        gxv[c] = g->gxc + c * E;
    }
    memset(g->gxc, 0, S * E * sizeof(float));
    ternary_dense_backward_block(g->wq, S, xcv, gqv, zqv, grad_w_out[0], gxv, gs_wq, NULL);
    ternary_dense_backward_block(g->wk, S, xcv, gkv, zkv, grad_w_out[1], gxv, gs_wk, NULL);
    ternary_dense_backward_block(g->wv, S, xcv, gvv, zvv, grad_w_out[2], gxv, gs_wv, NULL);
    for (size_t c = 0; c < S; c++)
        for (size_t d = 0; d < E; d++) grad_x_out[(q0 + c) * E + d] += g->gxc[c * E + d];
}

static void win_bwd_o_range(size_t k0, size_t k1, void *c_) {
    WinCtx *c = (WinCtx *)c_;
    TernaryGQA *g = c->g;
    const size_t E = g->embed_dim, qw = g->n_heads * g->head_dim;
    for (size_t ka = k0; ka < k1; ka += WIN_SUB) {
        const size_t kb = (ka + WIN_SUB < k1) ? ka + WIN_SUB : k1, i0 = c->p0 + ka;
        /* Z de Wo recalculado (solo interesa el entero crudo) */
        ternary_dense_forward_u8_batch(g->wo, c->pfo, kb - ka, g->attn_u8->q + i0 * qw, qw, g->attn_u8->scale + i0, g->attn_u8->zp + i0,
                                       NULL, 0, g->w_zo + ka * E, E);
        ternary_dense_u8_gradx_batch(g->wo, c->pwo, kb - ka, c->grad_out + i0 * E, E, g->w_attn + ka * qw, qw);
        for (size_t k = ka; k < kb; k++) {
            const size_t i = c->p0 + k;
            ternary_dense_u8_to_f32(g->attn_u8->q + i * qw, qw, g->attn_u8->scale[i], g->attn_u8->zp[i], c->xf + k * qw);
        }
    }
}

static void win_backward(TernaryGQA *g, const TernaryVec *x, const TernaryU8 *xu, const float *grad_out, float *const *grad_w_out,
                         float *grad_x_out, float *grad_scale_out) {
    const size_t E = g->embed_dim, qw = g->n_heads * g->head_dim, kvw = g->n_kv_heads * g->head_dim;
    const size_t C = g->chunk, T = g->seq_len;
    TernaryCapabilities caps; win_caps(&caps);
    TernaryThreadPool *pool = ternary_get_compute_pool();
    float gs_wo = 0.0f, gs_wq = 0.0f, gs_wk = 0.0f, gs_wv = 0.0f;
    ternary_par_zero(grad_w_out[0], qw * E * sizeof(float));
    ternary_par_zero(grad_w_out[1], kvw * E * sizeof(float));
    ternary_par_zero(grad_w_out[2], kvw * E * sizeof(float));
    ternary_par_zero(grad_w_out[3], E * qw * sizeof(float));
    const TernaryVec **rcv = (const TernaryVec **)(g->views + C);
    const float **gov = (const float **)(g->views + 2 * C);
    const int32_t **zov = (const int32_t **)(g->views + 6 * C);
    float **gxv = (float **)(g->views + 10 * C);
    const size_t XW = qw > E ? qw : E;
    float *work = xu ? (float *)ternary_malloc64(C * (XW + E + qw + 2 * kvw) * sizeof(float)) : NULL;
    if (xu && !work) abort();
    float *xf = work, *terms = work ? work + C * XW : NULL;
    TernaryPlanes *pwo = xu ? ternary_dense_planes_create(g->wo) : NULL, *pwq = xu ? ternary_dense_planes_create(g->wq) : NULL,
                  *pwk = xu ? ternary_dense_planes_create(g->wk) : NULL, *pwv = xu ? ternary_dense_planes_create(g->wv) : NULL;
    TernaryFwdW *pfq = xu ? ternary_dense_fwdw_create(g->wq) : NULL, *pfk = xu ? ternary_dense_fwdw_create(g->wk) : NULL,
                *pfv = xu ? ternary_dense_fwdw_create(g->wv) : NULL, *pfo = xu ? ternary_dense_fwdw_create(g->wo) : NULL;
    WinCtx W = { g, x, xu, 0, 0, NULL, NULL, 1, grad_out, xf, grad_x_out, pwo, pwq, pwk, pwv, pfq, pfk, pfv, pfo };
    g->w_base = g->w_C0;
    size_t nf = g->w_C0 ? C : 0;      /* primer bloque aun sin finalizar (el primero se deja para el final si hay sinks) */
    for (size_t p0 = 0; p0 < T; p0 += C) {
        const size_t S = (T - p0 < C) ? (T - p0) : C;
        W.p0 = p0; W.S = S;
        /* A1: Wo hacia atras (Z de Wo recalculado) -> gradiente de la salida de atencion del bloque */
        memset(g->w_attn, 0, S * qw * sizeof(float));
        TPROF_BEGIN("w_bwd_o");
        if (xu) {
            ternary_parallel_ranges(pool, 0, S, 0, win_bwd_o_range, &W);
            GqJob job = { g->wo, grad_out + p0 * E, g->w_zo, grad_w_out[3], terms, E, 0 };
            float *gsp = &gs_wo;
            gq_run_jobs(pool, &job, 1, S, xf, qw, g->attn_u8->scale + p0, &gsp);
        } else {
        for (size_t c = 0; c < S; c++) {
            const size_t i = p0 + c;
            extract_words(g->attn_bits, i * qw, qw, g->rc[c]);
            ternary_dense_forward_auto(g->wo, g->rc[c], g->w_tmpf, g->w_zo + c * E, &caps);
            rcv[c] = g->rc[c]; gov[c] = grad_out + i * E; zov[c] = g->w_zo + c * E; gxv[c] = g->w_attn + c * qw;
        }
        ternary_dense_backward_block(g->wo, S, rcv, gov, zov, grad_w_out[3], gxv, &gs_wo, NULL);
        }
        TPROF_END();
        /* proyecciones del bloque hacia el buffer (recalculadas), con sus gradientes en cero */
        { const size_t j0 = (p0 + 1 > g->window) ? p0 + 1 - g->window : 0;
          size_t lo = nf < j0 ? nf : j0; if (lo < g->w_C0) lo = g->w_C0;
          win_ensure(g, p0 + S - 1, lo, p0); }
        TPROF_BEGIN("w_bwd_proj");
        if (xu) ternary_parallel_ranges(pool, 0, S, 1, win_proj_range, &W);
        else
        for (size_t c = 0; c < S; c++) {
            const size_t p = p0 + c, sl = wslot(g, p);
            win_project(g, x, xu, p, g->w_qrot + c * qw, 1, &caps, g->w_tmpf, g->w_tmpi, g->u8_tmp, NULL, NULL, NULL);
            memset(g->w_gq + sl * qw, 0, qw * sizeof(float));
            memset(g->w_gk + sl * kvw, 0, kvw * sizeof(float));
            memset(g->w_gv + sl * kvw, 0, kvw * sizeof(float));
        }
        TPROF_END();
        TPROF_BEGIN("w_bwd_rows");
        if (pool && (size_t)ternary_threadpool_num_threads(pool) > 2 * g->n_kv_heads) win_bwd_rows_split(pool, &W);
        else ternary_parallel_ranges(pool, 0, g->n_kv_heads, 1, win_rows_bwd_range, &W);
        TPROF_END();
        TPROF_BEGIN("w_bwd_fin");
        /* bloques cuyas claves ya no reciben gradiente de filas futuras */
        const size_t last = p0 + S - 1;
        while (nf < T) {
            const size_t Sf = (T - nf < C) ? (T - nf) : C;
            const size_t bend = nf + Sf - 1;
            if (!(bend + g->window - 1 <= last)) break;
            win_finalize(g, x, xu, nf, grad_w_out, grad_x_out, &gs_wq, &gs_wk, &gs_wv, xf, terms, pwq, pwk, pwv);
            nf += C;
        }
        TPROF_END();
    }
    TPROF_BEGIN("w_bwd_fin");
    while (nf < T) { win_finalize(g, x, xu, nf, grad_w_out, grad_x_out, &gs_wq, &gs_wk, &gs_wv, xf, terms, pwq, pwk, pwv); nf += C; }
    if (g->w_C0) win_finalize(g, x, xu, 0, grad_w_out, grad_x_out, &gs_wq, &gs_wk, &gs_wv, xf, terms, pwq, pwk, pwv);
    TPROF_END();
    free(work);
    ternary_dense_planes_free(pwo); ternary_dense_planes_free(pwq); ternary_dense_planes_free(pwk); ternary_dense_planes_free(pwv);
    ternary_dense_fwdw_free(pfq); ternary_dense_fwdw_free(pfk); ternary_dense_fwdw_free(pfv); ternary_dense_fwdw_free(pfo);
    grad_scale_out[0] += gs_wq; grad_scale_out[1] += gs_wk; grad_scale_out[2] += gs_wv; grad_scale_out[3] += gs_wo;
}

void ternary_gqa_forward(const TernaryGQA *g, const TernaryVec *x, float *out, int32_t *out_raw_opt) {
    if (g->window) { win_forward(g, x, NULL, out, out_raw_opt); return; }
    gqa_full_prepare(g);
    size_t E = g->embed_dim, H = g->head_dim, NQ = g->n_heads, NKV = g->n_kv_heads;
    size_t qw = NQ * H, kvw = NKV * H;
    TernaryCapabilities caps; memset(&caps, 0, sizeof(caps)); caps.num_cores = 1; caps.dot_impl_name = "scalar";
    { TernaryThreadPool *pl = ternary_get_compute_pool();
      if (pl && ternary_threadpool_num_threads(pl) > 1) caps.cpu_pool = pl; }  /* enteros: mismos bits con o sin pool */

    /* --- Paso 1: proyectar Q, K, V en cada posicion y cuantizar --- */
    for (size_t p = 0; p < g->seq_len; p++) {
        extract_words(x, p * E, E, g->x_pos);
        float *const tmp_out = g->w_tmpf;
        int32_t *qz = g->q_z + p * qw, *kz = g->k_z + p * kvw, *vz = g->v_z + p * kvw;

        int32_t *const rot = g->w_tmpi;
        ternary_dense_forward_auto(g->wq, g->x_pos, tmp_out, qz, &caps);
        const int32_t *qs = qz;
        if (g->rope) { rope_rotate_int(g, p, qz, NQ, rot); qs = rot; }
        if (ternary_avx512_enabled()) ternary_avx512_pack_sign_i32(qs, qw, g->q_bits->mask, g->q_bits->sign, p * qw);
        else for (size_t i = 0; i < qw; i++) set_sign_i32(g->q_bits, p * qw + i, qs[i]);
        if (g->qk_int) memcpy(g->q_rot + p * qw, qs, qw * sizeof(int32_t));

        ternary_dense_forward_auto(g->wk, g->x_pos, tmp_out, kz, &caps);
        const int32_t *ks = kz;
        const int32_t *kmix = kz;
        if (g->smear) { gqa_smear_k(g, p, kz, g->smear_buf + kvw); kmix = g->smear_buf + kvw; }   /* k_z queda SIN mezclar: es el Z del STE de Wk */
        if (g->rope) { rope_rotate_int(g, p, kmix, NKV, rot); ks = rot; } else ks = kmix;
        if (ternary_avx512_enabled()) ternary_avx512_pack_sign_i32(ks, kvw, g->k_bits->mask, g->k_bits->sign, p * kvw);
        else for (size_t i = 0; i < kvw; i++) set_sign_i32(g->k_bits, p * kvw + i, ks[i]);
        if (g->qk_int) memcpy(g->k_rot + p * kvw, ks, kvw * sizeof(int32_t));
        if (g->kT) { if (ternary_avx512_enabled()) ternary_avx512_scatter_i32(ks, kvw, g->kT + p, g->tpad); else if (TN_ON(ELEM)) ternary_neon_scatter_i32(ks, kvw, g->kT + p, g->tpad); else for (size_t i = 0; i < kvw; i++) g->kT[i * g->tpad + p] = ks[i]; }

        ternary_dense_forward_auto(g->wv, g->x_pos, tmp_out, vz, &caps);
        gq_scale(vz, kvw, g->wv->scale, g->v_float + p * kvw, 1);
        gq_vq_row(g->v_float + p * kvw, kvw);
        if (g->vT) for (size_t i = 0; i < kvw; i++) g->vT[i * g->tpad + p] = g->v_float[p * kvw + i];
    }

    /* --- Paso 2: atencion causal por cabeza --- */
    const float *vf = g->v_float;   /* V como floats (en la ruta uint8 ya cuantizada a 8 bits) */
    if (g->qk_int) {
        attn_forward_flash(g);
        for (size_t i = 0; i < g->seq_len; i++)
            for (size_t i2 = 0; i2 < qw; i2++) set_sign_f(g->attn_bits, i * qw + i2, g->attn_cat[i * qw + i2]);
    } else
    for (size_t i = 0; i < g->seq_len; i++) {
        for (size_t h = 0; h < NQ; h++) {
            size_t kv_h = h / g->group_size;
            compute_scores(g, i, h, kv_h);
            ternary_softmax_forward(g->softmax, g->scores, g->seq_len, i + 1, g->probs);

            float *restrict acc = g->attn_cat + i * qw + h * H;
            memset(acc, 0, H * sizeof(float));
            for (size_t j = 0; j <= i; j++) {
                const float pr = g->probs[j];
                const float *restrict vj = vf + j * kvw + kv_h * H;
                for (size_t d = 0; d < H; d++) acc[d] += pr * vj[d];
            }
        }
        for (size_t i2 = 0; i2 < qw; i2++) set_sign_f(g->attn_bits, i * qw + i2, g->attn_cat[i * qw + i2]);
    }

    /* --- Paso 3: proyeccion de salida Wo, posicion por posicion --- */
    for (size_t p = 0; p < g->seq_len; p++) {
        extract_words(g->attn_bits, p * qw, qw, g->rc[0]);
        ternary_dense_forward_auto(g->wo, g->rc[0], out + p * E, g->wo_z + p * E, &caps);
    }
    if (out_raw_opt) memcpy(out_raw_opt, g->wo_z, g->seq_len * E * sizeof(int32_t));
}

static void gq_bwd_rope(size_t p0, size_t p1, void *c_);
void ternary_gqa_backward(TernaryGQA *g, const TernaryVec *x, const float *grad_out,
                           float *const *grad_w_out, float *grad_x_out, float *grad_scale_out,
                           const int32_t *out_raw_opt) {
    size_t E = g->embed_dim, H = g->head_dim, NQ = g->n_heads;
    size_t qw = NQ * H, kvw = g->n_kv_heads * H;
    const size_t C = g->chunk;
    (void)out_raw_opt;  /* ya lo tenemos en g->wo_z, calculado en el forward inmediatamente anterior */
    if (g->window) { win_backward(g, x, NULL, grad_out, grad_w_out, grad_x_out, grad_scale_out); return; }

    memset(g->grad_q_raw, 0, g->seq_len * qw * sizeof(float));
    memset(g->grad_k_raw, 0, g->seq_len * kvw * sizeof(float));
    memset(g->grad_v_raw, 0, g->seq_len * kvw * sizeof(float));
    memset(g->grad_attn_cat, 0, g->seq_len * qw * sizeof(float));
    float gs_wo = 0.0f, gs_wq = 0.0f, gs_wk = 0.0f, gs_wv = 0.0f;
    /* grad_w_out[m] es el acumulador (contrato: se SOBREESCRIBE con el total de la muestra): en cero al empezar,
     * y cada bloque de posiciones suma directo ahi (ternary_dense_backward_block). Sin buffers propios. */
    memset(grad_w_out[0], 0, qw * E * sizeof(float));
    memset(grad_w_out[1], 0, kvw * E * sizeof(float));
    memset(grad_w_out[2], 0, kvw * E * sizeof(float));
    memset(grad_w_out[3], 0, E * qw * sizeof(float));

    /* Vistas por muestra del bloque actual (todas apuntan a memoria de la capa o del llamador). */
    const TernaryVec **xcv = (const TernaryVec **)g->views, **rcv = xcv + C;
    const float **gov = (const float **)(g->views + 2 * C), **gqv = gov + C, **gkv = gqv + C, **gvv = gkv + C;
    const int32_t **zov = (const int32_t **)(g->views + 6 * C), **zqv = zov + C, **zkv = zqv + C, **zvv = zkv + C;
    float **gxv = (float **)(g->views + 10 * C);

    /* --- Pase A1: Wo hacia atras para TODAS las posiciones (por bloques): da grad_attn_cat completo. Los
     * acumuladores de cada elemento siguen en orden ascendente de posicion, como antes. --- */
    for (size_t p0 = 0; p0 < g->seq_len; p0 += C) {
        const size_t S = (g->seq_len - p0 < C) ? (g->seq_len - p0) : C;
        for (size_t c = 0; c < S; c++) {
            const size_t i = p0 + c;
            extract_words(g->attn_bits, i * qw, qw, g->rc[c]);
            rcv[c] = g->rc[c];
            gov[c] = grad_out + i * E;
            zov[c] = g->wo_z + i * E;
            gxv[c] = g->grad_attn_cat + i * qw;
        }
        ternary_dense_backward_block(g->wo, S, rcv, gov, zov, grad_w_out[3], gxv, &gs_wo, NULL);
    }

    /* --- Pase A2: gradiente de Q (completo por posicion) y acumulacion de los de K/V (completos recien al
     * terminar todo el pase). --- */
    const float *vf = g->v_float;
    if (g->qk_int) attn_backward_flash(g);
    else
    for (size_t i = 0; i < g->seq_len; i++) {
        float *grad_attn_i = g->grad_attn_cat + i * qw;

        for (size_t h = 0; h < NQ; h++) {
            size_t kv_h = h / g->group_size;
            const float *ga_h = grad_attn_i + h * H;

            compute_scores(g, i, h, kv_h);
            ternary_softmax_forward(g->softmax, g->scores, i + 1, i + 1, g->probs);

            float *const dprobs = g->dprobs;
            /* dprobs[j] = sum_d ga[d]*v[j][d], sumando d en orden como siempre (no se puede reasociar sin
             * cambiar bits); para no depender de la latencia de una sola cadena de sumas se calculan 4 filas j a
             * la vez, cada una con su propia cadena (mismo orden dentro de cada una). */
            size_t j = 0;
            for (; j + 4 <= i + 1; j += 4) {
                const float *v0 = vf + j * kvw + kv_h * H, *v1 = v0 + kvw, *v2 = v1 + kvw, *v3 = v2 + kvw;
                float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                for (size_t d = 0; d < H; d++) {
                    const float gd = ga_h[d];
                    d0 += gd * v0[d]; d1 += gd * v1[d]; d2 += gd * v2[d]; d3 += gd * v3[d];
                }
                dprobs[j] = d0; dprobs[j + 1] = d1; dprobs[j + 2] = d2; dprobs[j + 3] = d3;
            }
            for (; j <= i; j++) {
                const float *restrict vj = vf + j * kvw + kv_h * H;
                float dot = 0.0f;
                for (size_t d = 0; d < H; d++) dot += ga_h[d] * vj[d];
                dprobs[j] = dot;
            }
            if (ternary_avx512_enabled()) ternary_avx512_outer_add(g->grad_v_raw + kv_h * H, kvw, H, i + 1, g->probs, ga_h);
            else if (TN_ON(ATTN)) ternary_neon_outer_add(g->grad_v_raw + kv_h * H, kvw, H, i + 1, g->probs, ga_h);
            else
            for (size_t jj = 0; jj <= i; jj++) {
                float *restrict gv_j = g->grad_v_raw + jj * kvw + kv_h * H;
                const float pj = g->probs[jj];
                for (size_t d = 0; d < H; d++) gv_j[d] += pj * ga_h[d];
            }
            ternary_softmax_backward(g->softmax, g->probs, dprobs, i + 1, g->ds_scratch);

            if (g->qk_int) {
                /* score = q_real.k_real / 2^shift, con q_real = q_rot / 256 */
                const float gam = ldexpf(1.0f, -g->qk_shift) / (float)TERNARY_ROPE_SCALE;
                float *restrict gq = g->grad_q_raw + i * qw + h * H;
                const int32_t *restrict qr = g->q_rot + i * qw + h * H;
                if (TN_VEC(ATTN)) {
                    for (size_t j = 0; j <= i; j++) g->ds_scratch[j] *= gam;
                    TK_CALL(qk_grad, gq, qr, g->k_rot + kv_h * H, g->grad_k_raw + kv_h * H, kvw, H, i + 1, g->ds_scratch);
                } else
                for (size_t j = 0; j <= i; j++) {
                    const float ds = g->ds_scratch[j] * gam;
                    const int32_t *restrict kr = g->k_rot + j * kvw + kv_h * H;
                    float *restrict gk = g->grad_k_raw + j * kvw + kv_h * H;
                    for (size_t d = 0; d < H; d++) {
                        gq[d] += ds * (float)kr[d];
                        gk[d] += ds * (float)qr[d];
                    }
                }
            } else
            for (size_t j = 0; j <= i; j++) {
                float ds = g->ds_scratch[j];
                for (size_t d = 0; d < H; d++) {
                    int qd = ternary_vec_get(g->q_bits, i * qw + h * H + d);
                    int kd = ternary_vec_get(g->k_bits, j * kvw + kv_h * H + d);
                    g->grad_q_raw[i * qw + h * H + d] += ds * (float)kd;
                    g->grad_k_raw[j * kvw + kv_h * H + d] += ds * (float)qd;
                }
            }
        }
    }

    /* RoPE: el gradiente de Q/K esta respecto a los bits ya rotados; se
     * gira de vuelta (R^T) al espacio de q_z/k_z antes de pasar por Wq/Wk. */
    if (g->rope) ternary_parallel_ranges(ternary_get_compute_pool(), 0, g->seq_len, 0, gq_bwd_rope, g);
    if (g->smear) gqa_smear_grad(g, g->grad_k_raw);

    /* --- Pase B: ahora que grad_k_raw/grad_v_raw estan completos (recibieron
     * el aporte de TODAS las consultas i >= j), Wq/Wk/Wv hacia atras por bloques de posiciones. Los tres
     * suman en el mismo grad_x temporal de la muestra, en el mismo orden de siempre (Wq, Wk, Wv). --- */
    for (size_t p0 = 0; p0 < g->seq_len; p0 += C) {
        const size_t S = (g->seq_len - p0 < C) ? (g->seq_len - p0) : C;
        for (size_t c = 0; c < S; c++) {
            const size_t p = p0 + c;
            extract_words(x, p * E, E, g->xc[c]);
            xcv[c] = g->xc[c];
            gqv[c] = g->grad_q_raw + p * qw;  zqv[c] = g->q_z + p * qw;
            gkv[c] = g->grad_k_raw + p * kvw; zkv[c] = g->k_z + p * kvw;
            gvv[c] = g->grad_v_raw + p * kvw; zvv[c] = g->v_z + p * kvw;
            gxv[c] = g->gxc + c * E;
        }
        memset(g->gxc, 0, S * E * sizeof(float));
        ternary_dense_backward_block(g->wq, S, xcv, gqv, zqv, grad_w_out[0], gxv, &gs_wq, NULL);
        ternary_dense_backward_block(g->wk, S, xcv, gkv, zkv, grad_w_out[1], gxv, &gs_wk, NULL);
        ternary_dense_backward_block(g->wv, S, xcv, gvv, zvv, grad_w_out[2], gxv, &gs_wv, NULL);
        for (size_t c = 0; c < S; c++)
            for (size_t d = 0; d < E; d++) grad_x_out[(p0 + c) * E + d] += g->gxc[c * E + d];
    }

    grad_scale_out[0] += gs_wq; grad_scale_out[1] += gs_wk; grad_scale_out[2] += gs_wv; grad_scale_out[3] += gs_wo;
}


/* ======================= ruta uint8 (atencion completa) ======================= */
static int u8_ready(const TernaryGQA *gc, const char *who) {
    TernaryGQA *g = (TernaryGQA *)(uintptr_t)gc;
    if (!g->qk_int) {
        ternary_diag("%s: la ruta uint8 de GQA requiere puntaje entero (enable_rope_int).\n", who);
        return 0;
    }
    const size_t qw = g->n_heads * g->head_dim, kvw = g->n_kv_heads * g->head_dim;
    if (!g->u8_tmp) g->u8_tmp = (int32_t *)ternary_malloc64((qw > kvw ? qw : kvw) * sizeof(int32_t));
    if (!g->attn_u8) g->attn_u8 = ternary_u8_alloc(g->seq_len, qw);
    if (!g->u8_tmp || !g->attn_u8) { ternary_diag("%s: sin memoria\n", who); return 0; }
    return 1;
}


/* ---- ruta uint8 de atencion completa, repartida en hilos ----
 * Cada posicion es independiente salvo la mezcla de claves (smear), que necesita el Z de K de la posicion anterior: por eso K se
 * hace en dos fases (proyecciones | mezcla + RoPE + almacenamiento). El backward va por bloques de GQA_BLK posiciones: fase por
 * posicion (grad_x, escala del gradiente) y fase por FILAS de cada matriz de pesos (gw, con p ascendente); los terminos de la escala
 * se suman en serie en el orden de siempre. Resultado bit a bit igual con cualquier numero de hilos (y igual al codigo serial). */
typedef struct { const TernaryGQA *g; const TernaryU8 *x; float *out; const TernaryFwdW *fwq, *fwk, *fwv, *fwo; } GqFwd;
#define GQ_FSUB 16   /* posiciones por tanda del forward por lotes */

static void gq_fwd_proj(size_t p0, size_t p1, void *c_) {
    GqFwd *c = (GqFwd *)c_;
    const TernaryGQA *g = c->g;
    const size_t E = g->embed_dim, H = g->head_dim, NQ = g->n_heads, NKV = g->n_kv_heads, qw = NQ * H, kvw = NKV * H;
    const size_t wide = qw > kvw ? qw : kvw;
    int32_t *sh = (int32_t *)ternary_malloc64(wide * sizeof(int32_t)), *rot = (int32_t *)ternary_malloc64(wide * sizeof(int32_t));
    if (!sh || !rot) abort();
    for (size_t pa = p0; pa < p1; pa += GQ_FSUB) {
        const size_t pb = (pa + GQ_FSUB < p1) ? pa + GQ_FSUB : p1, n = pb - pa;
        const uint8_t *xq = c->x->q + pa * E; const float *xs = c->x->scale + pa; const int32_t *xz = c->x->zp + pa;
        ternary_dense_forward_u8_batch(g->wq, c->fwq, n, xq, E, xs, xz, NULL, 0, g->q_z + pa * qw, qw);
        ternary_dense_forward_u8_batch(g->wk, c->fwk, n, xq, E, xs, xz, NULL, 0, g->k_z + pa * kvw, kvw);
        ternary_dense_forward_u8_batch(g->wv, c->fwv, n, xq, E, xs, xz, NULL, 0, g->v_z + pa * kvw, kvw);
        for (size_t p = pa; p < pb; p++) {
            const float sx = c->x->scale[p];
            int32_t *qz = g->q_z + p * qw, *vz = g->v_z + p * kvw;
            gq_shift(qz, qw, sh);
            const int32_t *qs = sh;
            if (g->rope) { rope_rotate_int(g, p, sh, NQ, rot); qs = rot; }
            memcpy(g->q_rot + p * qw, qs, qw * sizeof(int32_t));
            gq_scale(vz, kvw, g->wv->scale * sx, g->v_float + p * kvw, 1);
            gq_vq_row(g->v_float + p * kvw, kvw);
            if (g->vT) for (size_t i = 0; i < kvw; i++) g->vT[i * g->tpad + p] = g->v_float[p * kvw + i];
        }
    }
    free(sh); free(rot);
}

static void gq_fwd_keys(size_t p0, size_t p1, void *c_) {
    GqFwd *c = (GqFwd *)c_;
    const TernaryGQA *g = c->g;
    const size_t NKV = g->n_kv_heads, kvw = NKV * g->head_dim;
    int32_t *sh = (int32_t *)ternary_malloc64(kvw * sizeof(int32_t)), *rot = (int32_t *)ternary_malloc64(kvw * sizeof(int32_t)), *prev = (int32_t *)ternary_calloc64(kvw, sizeof(int32_t));
    if (!sh || !rot || !prev) abort();
    for (size_t p = p0; p < p1; p++) {
        gq_shift(g->k_z + p * kvw, kvw, sh);
        if (g->smear) {
            if (p > 0) gq_shift(g->k_z + (p - 1) * kvw, kvw, prev);
            if (ternary_avx512_enabled()) ternary_avx512_smear_k(sh, sh, prev, kvw, p == 0);
            else if (TN_ON(ELEM)) ternary_neon_smear_k(sh, sh, prev, kvw, p == 0);
            else if (p > 0) for (size_t d = 0; d < kvw; d++) sh[d] = (sh[d] + prev[d]) >> 1;
        }
        const int32_t *ks = sh;
        if (g->rope) { rope_rotate_int(g, p, sh, NKV, rot); ks = rot; }
        memcpy(g->k_rot + p * kvw, ks, kvw * sizeof(int32_t));
        if (g->kT) { if (ternary_avx512_enabled()) ternary_avx512_scatter_i32(ks, kvw, g->kT + p, g->tpad); else if (TN_ON(ELEM)) ternary_neon_scatter_i32(ks, kvw, g->kT + p, g->tpad); else for (size_t i = 0; i < kvw; i++) g->kT[i * g->tpad + p] = ks[i]; }
    }
    free(sh); free(rot); free(prev);
}

static void gq_fwd_out(size_t p0, size_t p1, void *c_) {
    GqFwd *c = (GqFwd *)c_;
    const TernaryGQA *g = c->g;
    const size_t E = g->embed_dim, qw = g->n_heads * g->head_dim;
    TernaryU8 *au = g->attn_u8;
    for (size_t pa = p0; pa < p1; pa += GQ_FSUB) {
        const size_t pb = (pa + GQ_FSUB < p1) ? pa + GQ_FSUB : p1;
        for (size_t p = pa; p < pb; p++) {
            TernaryU8 row = { au->q + p * qw, au->scale + p, au->zp + p, 1, qw };
            ternary_u8_quantize(&row, g->attn_cat + p * qw);
        }
        ternary_dense_forward_u8_batch(g->wo, c->fwo, pb - pa, au->q + pa * qw, qw, au->scale + pa, au->zp + pa, c->out + pa * E, E, g->wo_z + pa * E, E);
    }
}

void ternary_gqa_forward_u8(const TernaryGQA *g, const TernaryU8 *x, float *out, int32_t *out_raw_opt) {
    if (!u8_ready(g, "ternary_gqa_forward_u8")) { memset(out, 0, g->seq_len * g->embed_dim * sizeof(float)); return; }
    if (g->window) { win_forward(g, NULL, x, out, out_raw_opt); return; }
    gqa_full_prepare(g);
    const size_t E = g->embed_dim;
    TernaryThreadPool *pool = ternary_get_compute_pool();
    TernaryFwdW *fwq = ternary_dense_fwdw_create(g->wq), *fwk = ternary_dense_fwdw_create(g->wk),
                *fwv = ternary_dense_fwdw_create(g->wv), *fwo = ternary_dense_fwdw_create(g->wo);
    GqFwd c = { g, x, out, fwq, fwk, fwv, fwo };
    TPROF_BEGIN("fwd_proj_qkv");
    ternary_parallel_ranges(pool, 0, g->seq_len, 0, gq_fwd_proj, &c);
    TPROF_END();
    TPROF_BEGIN("fwd_keys");
    ternary_parallel_ranges(pool, 0, g->seq_len, 0, gq_fwd_keys, &c);
    TPROF_END();
    TPROF_BEGIN("attention");
    attn_forward_flash(g);
    TPROF_END();
    TPROF_BEGIN("quantize_proj_o");
    ternary_parallel_ranges(pool, 0, g->seq_len, 0, gq_fwd_out, &c);
    TPROF_END();
    ternary_dense_fwdw_free(fwq); ternary_dense_fwdw_free(fwk); ternary_dense_fwdw_free(fwv); ternary_dense_fwdw_free(fwo);
    if (out_raw_opt) memcpy(out_raw_opt, g->wo_z, g->seq_len * E * sizeof(int32_t));
}

typedef struct { TernaryGQA *g; const TernaryU8 *x; const float *grad_out; float *grad_x_out; size_t b0; float *xf;
                 const TernaryPlanes *pwo, *pwq, *pwk, *pwv; } GqBwd;   /* planos +1/-1 de los pesos (NULL sin AVX-512) */
#define GQ_SUB 16   /* posiciones por tanda del grad_x por lotes */

static void gq_bwd_o(size_t i0, size_t i1, void *c_) {
    GqBwd *c = (GqBwd *)c_;
    TernaryGQA *g = c->g;
    const size_t E = g->embed_dim, qw = g->n_heads * g->head_dim;
    for (size_t ia = i0; ia < i1; ia += GQ_SUB) {
        const size_t ib = (ia + GQ_SUB < i1) ? ia + GQ_SUB : i1;
        ternary_dense_u8_gradx_batch(g->wo, c->pwo, ib - ia, c->grad_out + (c->b0 + ia) * E, E, g->grad_attn_cat + (c->b0 + ia) * qw, qw);
        for (size_t pi = ia; pi < ib; pi++) {
            const size_t p = c->b0 + pi;
            ternary_dense_u8_to_f32(g->attn_u8->q + p * qw, qw, g->attn_u8->scale[p], g->attn_u8->zp[p], c->xf + pi * qw);
        }
    }
}

static void gq_bwd_rope(size_t p0, size_t p1, void *c_) {
    TernaryGQA *g = (TernaryGQA *)c_;
    const size_t qw = g->n_heads * g->head_dim, kvw = g->n_kv_heads * g->head_dim;
    for (size_t p = p0; p < p1; p++) {
        rope_unrotate_grad(g, p, g->grad_q_raw + p * qw, g->n_heads);
        rope_unrotate_grad(g, p, g->grad_k_raw + p * kvw, g->n_kv_heads);
    }
}

static void gq_bwd_x(size_t i0, size_t i1, void *c_) {
    GqBwd *c = (GqBwd *)c_;
    TernaryGQA *g = c->g;
    const size_t E = g->embed_dim, qw = g->n_heads * g->head_dim, kvw = g->n_kv_heads * g->head_dim;
    for (size_t ia = i0; ia < i1; ia += GQ_SUB) {
        const size_t ib = (ia + GQ_SUB < i1) ? ia + GQ_SUB : i1, n = ib - ia, pa = c->b0 + ia;
        for (size_t pi = ia; pi < ib; pi++) {
            const size_t p = c->b0 + pi;
            const float sx = c->x->scale[p];
            /* Q/K salen de z>>8 (no de scale*sx*z): dL/dz = g/256, y Dense espera dL/dz_float = dL/dz/(scale*sx). */
            const float fq = 1.0f / ((float)(1 << TERNARY_GQA_U8_SHIFT) * g->wq->scale * sx), fk = 1.0f / ((float)(1 << TERNARY_GQA_U8_SHIFT) * g->wk->scale * sx);
            float *gqp = g->grad_q_raw + p * qw, *gkp = g->grad_k_raw + p * kvw;
            for (size_t i = 0; i < qw; i++) gqp[i] *= fq;
            for (size_t i = 0; i < kvw; i++) gkp[i] *= fk;
        }
        float *gx = c->grad_x_out + pa * E;   /* los tres suman en grad_x de cada posicion, en este orden */
        ternary_dense_u8_gradx_batch(g->wq, c->pwq, n, g->grad_q_raw + pa * qw, qw, gx, E);
        ternary_dense_u8_gradx_batch(g->wk, c->pwk, n, g->grad_k_raw + pa * kvw, kvw, gx, E);
        ternary_dense_u8_gradx_batch(g->wv, c->pwv, n, g->grad_v_raw + pa * kvw, kvw, gx, E);
        for (size_t pi = ia; pi < ib; pi++) {
            const size_t p = c->b0 + pi;
            ternary_dense_u8_to_f32(c->x->q + p * E, E, c->x->scale[p], c->x->zp[p], c->xf + pi * E);
        }
    }
}

void ternary_gqa_backward_u8(TernaryGQA *g, const TernaryU8 *x, const float *grad_out,
                             float *const *grad_w_out, float *grad_x_out, float *grad_scale_out) {
    if (!u8_ready(g, "ternary_gqa_backward_u8")) return;
    if (g->window) { win_backward(g, NULL, x, grad_out, grad_w_out, grad_x_out, grad_scale_out); return; }
    const size_t E = g->embed_dim, H = g->head_dim, NQ = g->n_heads;
    const size_t qw = NQ * H, kvw = g->n_kv_heads * H, T = g->seq_len;
    ternary_par_zero(g->grad_q_raw, T * qw * sizeof(float));
    ternary_par_zero(g->grad_k_raw, T * kvw * sizeof(float));
    ternary_par_zero(g->grad_v_raw, T * kvw * sizeof(float));
    ternary_par_zero(g->grad_attn_cat, T * qw * sizeof(float));
    float gs_wo = 0.0f, gs_wq = 0.0f, gs_wk = 0.0f, gs_wv = 0.0f;
    TPROF_BEGIN("zero_weight_grads");
    ternary_par_zero(grad_w_out[0], qw * E * sizeof(float));
    ternary_par_zero(grad_w_out[1], kvw * E * sizeof(float));
    ternary_par_zero(grad_w_out[2], kvw * E * sizeof(float));
    ternary_par_zero(grad_w_out[3], E * qw * sizeof(float));
    TPROF_END();
    TernaryThreadPool *pool = ternary_get_compute_pool();
    const size_t K = GQA_BLK, XW = qw > E ? qw : E;
    const size_t tsz = E + qw + 2 * kvw;                        /* terminos de la escala: wo | wq | wk | wv */
    float *work = (float *)ternary_malloc64(K * (XW + tsz) * sizeof(float));
    if (!work) abort();
    float *xf = work, *terms = work + K * XW;
    TernaryPlanes *pwo = ternary_dense_planes_create(g->wo), *pwq = ternary_dense_planes_create(g->wq),
                  *pwk = ternary_dense_planes_create(g->wk), *pwv = ternary_dense_planes_create(g->wv);
    GqBwd B = { g, x, grad_out, grad_x_out, 0, xf, pwo, pwq, pwk, pwv };

    /* A1: Wo hacia atras -> grad_attn_cat */
    TPROF_BEGIN("bwd_proj_o");
    for (size_t b0 = 0; b0 < T; b0 += K) {
        const size_t P = (T - b0 < K) ? (T - b0) : K;
        B.b0 = b0;
        ternary_parallel_ranges(pool, 0, P, 0, gq_bwd_o, &B);
        GqJob job = { g->wo, grad_out + b0 * E, g->wo_z + b0 * E, grad_w_out[3], terms, E, 0 };
        float *gsp = &gs_wo;
        gq_run_jobs(pool, &job, 1, P, xf, qw, g->attn_u8->scale + b0, &gsp);
    }
    TPROF_END();
    /* A2: atencion (flash) */
    TPROF_BEGIN("bwd_attention");
    attn_backward_flash(g);
    TPROF_END();
    if (g->rope) {
        TPROF_BEGIN("bwd_rope");
        ternary_parallel_ranges(pool, 0, T, 0, gq_bwd_rope, g);
        TPROF_END();
    }
    if (g->smear) { TPROF_BEGIN("bwd_smear"); gqa_smear_grad(g, g->grad_k_raw); TPROF_END(); }
    /* B: Wq, Wk, Wv hacia atras; los tres suman en grad_x de la posicion (en ese orden) */
    TPROF_BEGIN("bwd_proj_qkv");
    for (size_t b0 = 0; b0 < T; b0 += K) {
        const size_t P = (T - b0 < K) ? (T - b0) : K;
        B.b0 = b0;
        ternary_parallel_ranges(pool, 0, P, 0, gq_bwd_x, &B);
        GqJob jobs[3] = {
            { g->wq, g->grad_q_raw + b0 * qw, g->q_z + b0 * qw, grad_w_out[0], terms, qw, 0 },
            { g->wk, g->grad_k_raw + b0 * kvw, g->k_z + b0 * kvw, grad_w_out[1], terms + K * qw, kvw, 0 },
            { g->wv, g->grad_v_raw + b0 * kvw, g->v_z + b0 * kvw, grad_w_out[2], terms + K * (qw + kvw), kvw, 0 } };
        float *gsp[3] = { &gs_wq, &gs_wk, &gs_wv };
        gq_run_jobs(pool, jobs, 3, P, xf, E, x->scale + b0, gsp);
    }
    TPROF_END();
    free(work);
    ternary_dense_planes_free(pwo); ternary_dense_planes_free(pwq); ternary_dense_planes_free(pwk); ternary_dense_planes_free(pwv);
    grad_scale_out[0] += gs_wq; grad_scale_out[1] += gs_wk; grad_scale_out[2] += gs_wv; grad_scale_out[3] += gs_wo;
}
