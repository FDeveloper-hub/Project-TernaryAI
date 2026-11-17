/* Sesiones y ejecucion de un token con cache KV. */
#define _POSIX_C_SOURCE 200809L
#include "ternary_mem.h"
#include "ternary_infer_internal.h"
#include "ternary_threadpool.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int P_EMBED, P_NORM, P_QUANT, P_PROJQ, P_PROJK, P_PROJV, P_ATTN, P_PROJO, P_FFN_UP, P_FFN_ACT, P_FFN_DOWN, P_HEAD, P_ROPE, P_A_SC, P_A_SM, P_A_PV;
long long ternaryai_infer_n_fast = 0, ternaryai_infer_n_slow = 0;
static int prof_ready = 0;
static void prof_init(void) {
    if (prof_ready) return;
    prof_ready = 1;
    P_EMBED = ternaryai_infer_prof_id("embed"); P_NORM = ternaryai_infer_prof_id("rmsnorm"); P_QUANT = ternaryai_infer_prof_id("quantize");
    P_PROJQ = ternaryai_infer_prof_id("proj_q+rope"); P_PROJK = ternaryai_infer_prof_id("proj_k+rope+cache"); P_PROJV = ternaryai_infer_prof_id("proj_v+cache");
    P_ATTN = ternaryai_infer_prof_id("attention"); P_PROJO = ternaryai_infer_prof_id("proj_o"); P_FFN_UP = ternaryai_infer_prof_id("ffn_up");
    P_FFN_ACT = ternaryai_infer_prof_id("ffn_act+quant"); P_FFN_DOWN = ternaryai_infer_prof_id("ffn_down"); P_HEAD = ternaryai_infer_prof_id("head"); P_ROPE = ternaryai_infer_prof_id("rope");
    P_A_SC = ternaryai_infer_prof_id("  att:puntajes"); P_A_SM = ternaryai_infer_prof_id("  att:softmax"); P_A_PV = ternaryai_infer_prof_id("  att:pv");
}

/* ---------------------------------------------------------------- exponencial compartida (igual que ternary_fexp.h) */
#define FEXP_CUTOFF (-66.0f)
static inline float fexp2_core(float y) {
    const float fl = rintf(y);
    const float f = y - fl;
    float p = 1.535336188319500e-4f;
    p = p * f; p = p + 1.339887440266574e-3f;
    p = p * f; p = p + 9.618437357674640e-3f;
    p = p * f; p = p + 5.550332471162809e-2f;
    p = p * f; p = p + 2.402264791363012e-1f;
    p = p * f; p = p + 6.931472028550421e-1f;
    float r = f * p; r = r + 1.0f;
    const int32_t e = ((int32_t)fl + 127) << 23;
    float two; memcpy(&two, &e, sizeof(two));
    return r * two;
}
static inline void silu_f(float a, float *sv) {
    const float y = -fabsf(a) * 1.44269504088896341f;
    const float e = (y < FEXP_CUTOFF) ? 0.0f : fexp2_core(y);
    const float sg = (a >= 0.0f) ? 1.0f / (1.0f + e) : e / (1.0f + e);
    *sv = a * sg;
}
static inline float sum16(const float *part) {
    float t8[8], t4[4];
    for (int l = 0; l < 8; l++) t8[l] = part[l] + part[l + 8];
    for (int l = 0; l < 4; l++) t4[l] = t8[l] + t8[l + 4];
    const float a = t4[0] + t4[2], b = t4[1] + t4[3];
    return a + b;
}

/* ---------------------------------------------------------------- opciones y vida de la sesion */
void ternaryai_infer_session_opts_default(const TernaryInferModel *m, TernaryInferSessionOpts *o) {
    memset(o, 0, sizeof(*o));
    o->max_context = m->seq_len;
    for (int i = 0; i < m->n_top; i++) {
        if (m->top[i].kind != TERNARYAI_INFER_TOP_RESIDUAL) continue;
        for (int j = 0; j < m->top[i].n_sub; j++) if (m->top[i].sub[j].kind == TERNARYAI_INFER_SUB_GQA) {
            o->attn_window = m->top[i].sub[j].gqa.window; o->sinks = m->top[i].sub[j].gqa.sinks; i = m->n_top; break;
        }
    }
    o->ctx_policy = TERNARYAI_INFER_CTX_ERROR;
}

static void *xalloc(size_t n) { void *p = ternary_calloc64(1, n ? n : 1); return p; }

static void session_free_bufs(TernaryInferSession *s) {
    if (s->lc) for (int i = 0; i < s->m->n_attn; i++) { free(s->lc[i].kT); free(s->lc[i].v8); free(s->lc[i].vs); free(s->lc[i].prev); free(s->lc[i].k16); }
    free(s->lc); free(s->hist); free(s->z); free(s->x); free(s->y); free(s->f); free(s->tmp); free(s->q8); free(s->q8b);
    free(s->zi); free(s->zi2); free(s->qr); free(s->kr); free(s->sh); free(s->attn); free(s->vrow); free(s->sc); free(s->ee); free(s->hbuf);
    for (int i = 0; i < s->xs_n; i++) { TernaryInferScratch *w = &s->xs[i]; free(w->x); free(w->y); free(w->f); free(w->q8); free(w->q8b); free(w->zi); free(w->zi2); free(w->qr); free(w->kr); free(w->sh); free(w->attn); free(w->vrow); free(w->hbuf); }
    free(s->xs);
    free(s->lg); free(s->probs_rec); free(s->probs_n); free(s->scr);
}

void ternaryai_infer_session_free(TernaryInferSession *s) {
    if (!s) return;
    session_free_bufs(s);
    free(s);
}

static int session_alloc(TernaryInferSession *s) {
    const TernaryInferModel *m = s->m;
    const int C = s->o.max_context;
    const int E = m->dim;
    const int wide = (m->max_qw > m->max_hidden ? m->max_qw : m->max_hidden);
    const int w8 = (wide > E ? wide : E) + 128;
    const int rows16 = ((m->max_rows_body + 15) / 16) * 16 + 16;
    int maxG = 1;
    for (int i = 0; i < m->n_top; i++) if (m->top[i].kind == TERNARYAI_INFER_TOP_RESIDUAL)
        for (int j = 0; j < m->top[i].n_sub; j++) if (m->top[i].sub[j].kind == TERNARYAI_INFER_SUB_GQA && m->top[i].sub[j].gqa.group > maxG) maxG = m->top[i].sub[j].gqa.group;
    s->lc = (TernaryInferLayerCache *)xalloc((size_t)(m->n_attn ? m->n_attn : 1) * sizeof(TernaryInferLayerCache));
    s->z = (int32_t *)xalloc(((size_t)((m->vocab + 15) / 16) * 16 + 16) * 4);
    s->x = (float *)xalloc((size_t)(E + 16) * 4); s->y = (float *)xalloc((size_t)(E + 16) * 4);
    s->f = (float *)xalloc((size_t)(wide > E ? wide : E) * 4 + 64); s->tmp = (float *)xalloc((size_t)(E + 16) * 4);
    s->q8 = (uint8_t *)xalloc((size_t)w8); s->q8b = (uint8_t *)xalloc((size_t)w8);
    s->zi = (int32_t *)xalloc((size_t)rows16 * 4); s->zi2 = (int32_t *)xalloc((size_t)rows16 * 4);
    s->qr = (int32_t *)xalloc((size_t)(m->max_qw + 16) * 4); s->kr = (int32_t *)xalloc((size_t)(m->max_kvw + 16) * 4);
    s->sh = (int32_t *)xalloc((size_t)((m->max_qw > m->max_kvw ? m->max_qw : m->max_kvw) + 16) * 4);
    s->attn = (float *)xalloc((size_t)(m->max_qw + 16) * 4); s->vrow = (float *)xalloc((size_t)(m->max_kvw + 16) * 4);
    s->sc = (int32_t *)xalloc((size_t)(m->max_qw / (m->max_hd ? m->max_hd : 1) + 1) * (size_t)(C + 32) * 4); (void)maxG; s->ee = (float *)xalloc((size_t)(C + 32) * (size_t)(m->max_qw / (m->max_hd ? m->max_hd : 1) + 1) * 4);
    s->scr = (int64_t *)xalloc((size_t)(m->max_qw / (m->max_hd ? m->max_hd : 1) + 1) * (size_t)(C + 32) * 8);
    s->hbuf = (float *)xalloc((size_t)(m->max_hidden + 16) * 4);
    s->lg = (float *)xalloc((size_t)(m->vocab + 16) * 4);
    s->hist_cap = 256; s->hist = (int32_t *)xalloc((size_t)s->hist_cap * 4);
    if (s->o.record_attn) {
        s->probs_rec = (float *)xalloc((size_t)m->n_attn * (size_t)(m->max_qw / (m->max_hd ? m->max_hd : 1) + 1) * (size_t)(C + 16) * 4);
        s->probs_n = (int *)xalloc((size_t)(m->n_attn ? m->n_attn : 1) * sizeof(int));
    }
    if (!s->lc || !s->z || !s->x || !s->y || !s->f || !s->tmp || !s->q8 || !s->q8b || !s->zi || !s->zi2 || !s->qr || !s->kr || !s->sh ||
        !s->attn || !s->vrow || !s->sc || !s->ee || !s->scr || !s->hbuf || !s->lg || !s->hist) return 0;
    if (s->o.record_attn && (!s->probs_rec || !s->probs_n)) return 0;
    for (int i = 0; i < m->n_top; i++) {
        if (m->top[i].kind != TERNARYAI_INFER_TOP_RESIDUAL) continue;
        for (int j = 0; j < m->top[i].n_sub; j++) {
            const TernaryInferSub *sb = &m->top[i].sub[j];
            if (sb->kind != TERNARYAI_INFER_SUB_GQA) continue;
            TernaryInferLayerCache *c = &s->lc[sb->gqa_idx];
            c->tp = ((C + 15) / 16) * 16 + 16;
            c->v8 = (int8_t *)xalloc((size_t)sb->gqa.n_kv * (size_t)C * (size_t)sb->gqa.hd);
            c->vs = (float *)xalloc((size_t)C * 4);
            c->prev = (int32_t *)xalloc((size_t)sb->gqa.kvw * 4);
            /* TERNARYAI_INFER_NO_INT16=1 fuerza el camino int64 (pruebas) */
            c->k16ok = (sb->gqa.hd % 2 == 0 && sb->gqa.kvw % 2 == 0 && sb->gqa.hd <= 256 && sb->gqa.group <= 8 && !getenv("TERNARYAI_INFER_NO_INT16"));
            c->k16kind = ternaryai_infer_kern->scores16 ? 1 : 2;
            if (c->k16ok) c->k16 = (int16_t *)xalloc((size_t)(sb->gqa.kvw / 2) * (size_t)c->tp * 4);
            if (!c->k16ok) c->kT = (int32_t *)xalloc((size_t)sb->gqa.kvw * (size_t)c->tp * 4);     /* sin camino int16: K solo en int32 */
            if (!c->v8 || !c->vs || !c->prev || (c->k16ok && !c->k16) || (!c->k16ok && !c->kT)) return 0;
        }
    }
    return 1;
}

/* Crea la cache int32 de claves (solo si alguna clave no cabe en int16 o el camino int16 no sirve para este paso); se llena desde k16. */
int ternaryai_infer_kt_promote(TernaryInferLayerCache *c, int kvw) {
    if (c->kT) return 1;
    c->kT = (int32_t *)xalloc((size_t)kvw * (size_t)c->tp * 4);
    if (!c->kT) return 0;
    if (c->k16 && c->k16ok)
        for (int r = 0; r < kvw; r++)
            for (int p = 0; p < c->tp; p++)
                c->kT[(size_t)r * c->tp + p] = (c->k16kind == 1) ? c->k16[((size_t)(r >> 1) * c->tp + (size_t)p) * 2 + (size_t)(r & 1)] : c->k16[(size_t)r * c->tp + (size_t)p];
    return 1;
}

TernaryInferSession *ternaryai_infer_session_new(const TernaryInferModel *m, const TernaryInferSessionOpts *opts) {
    if (!m) return NULL;
    prof_init();
    TernaryInferSession *s = (TernaryInferSession *)ternary_calloc64(1, sizeof(TernaryInferSession));
    if (!s) return NULL;
    s->m = m;
    if (opts) s->o = *opts; else ternaryai_infer_session_opts_default(m, &s->o);
    if (s->o.max_context < 1) s->o.max_context = m->seq_len;
    if (s->o.ctx_policy != TERNARYAI_INFER_CTX_SLIDE && s->o.max_context > m->seq_len) s->o.max_context = m->seq_len;
    if (s->o.sinks < 0 || s->o.sinks >= s->o.max_context) s->o.sinks = 0;
    if (s->o.attn_window < 0) s->o.attn_window = 0;
    if (s->o.keep_tokens <= 0 || s->o.keep_tokens >= s->o.max_context) s->o.keep_tokens = s->o.max_context / 2 > 0 ? s->o.max_context / 2 : 1;
    s->rng = 0x9E3779B97F4A7C15ULL;
    if (!session_alloc(s)) { ternaryai_infer_session_free(s); return NULL; }
    return s;
}

/* Cota de las claves y validez del camino int16 al vaciar la cache. */
static void kcache_reset(TernaryInferSession *s) {
    for (int i = 0; i < s->m->n_attn; i++) { s->lc[i].kmax = 0; s->lc[i].k16ok = (s->lc[i].k16 != NULL); }
}

void ternaryai_infer_session_reset(TernaryInferSession *s) {
    if (!s) return;
    s->n = 0; s->next_pos = 0; s->hist_n = 0; s->have_head = 0;
    kcache_reset(s);
}

int ternaryai_infer_session_len(const TernaryInferSession *s) { return s->n; }
int ternaryai_infer_session_next_position(const TernaryInferSession *s) { return s->next_pos; }
int ternaryai_infer_session_history(const TernaryInferSession *s, const int32_t **ids) { if (ids) *ids = s->hist; return s->hist_n; }
const TernaryInferModel *ternaryai_infer_session_model(const TernaryInferSession *s) { return s->m; }
void ternaryai_infer_session_seed(TernaryInferSession *s, uint64_t seed) { s->rng = seed * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL; if (!s->rng) s->rng = 1; }

TernaryInferSession *ternaryai_infer_session_fork(const TernaryInferSession *src) {
    if (!src) return NULL;
    TernaryInferSession *s = ternaryai_infer_session_new(src->m, &src->o);
    if (!s) return NULL;
    const TernaryInferModel *m = src->m;
    s->n = src->n; s->next_pos = src->next_pos; s->rng = src->rng; s->have_head = src->have_head; s->head_k = src->head_k;
    for (int i = 0; i < m->n_top; i++) {
        if (m->top[i].kind != TERNARYAI_INFER_TOP_RESIDUAL) continue;
        for (int j = 0; j < m->top[i].n_sub; j++) {
            const TernaryInferSub *sb = &m->top[i].sub[j];
            if (sb->kind != TERNARYAI_INFER_SUB_GQA) continue;
            const TernaryInferLayerCache *a = &src->lc[sb->gqa_idx]; TernaryInferLayerCache *b = &s->lc[sb->gqa_idx];
            if (a->kT) {
                if (!b->kT) { b->k16ok = 0; if (!ternaryai_infer_kt_promote(b, sb->gqa.kvw)) { ternaryai_infer_session_free(s); return NULL; } }
                memcpy(b->kT, a->kT, (size_t)sb->gqa.kvw * (size_t)a->tp * 4);
            }
            memcpy(b->v8, a->v8, (size_t)sb->gqa.n_kv * (size_t)src->o.max_context * (size_t)sb->gqa.hd);
            memcpy(b->vs, a->vs, (size_t)src->o.max_context * 4);
            memcpy(b->prev, a->prev, (size_t)sb->gqa.kvw * 4);
            if (a->k16 && b->k16) memcpy(b->k16, a->k16, (size_t)(sb->gqa.kvw / 2) * (size_t)a->tp * 4);
            b->kmax = a->kmax; b->k16ok = a->k16ok; b->k16kind = a->k16kind;
        }
    }
    memcpy(s->z, src->z, ((size_t)((m->vocab + 15) / 16) * 16) * 4);
    if (src->hist_n > s->hist_cap) {
        int32_t *h = (int32_t *)realloc(s->hist, (size_t)src->hist_n * 4);
        if (!h) { ternaryai_infer_session_free(s); return NULL; }
        s->hist = h; s->hist_cap = src->hist_n;
    }
    memcpy(s->hist, src->hist, (size_t)src->hist_n * 4); s->hist_n = src->hist_n;
    return s;
}

/* ---------------------------------------------------------------- historia */
static void hist_push(TernaryInferSession *s, int32_t id) {
    if (s->hist_n == s->hist_cap) {
        if (s->o.max_history > 0 && s->hist_n >= s->o.max_history) {
            const int drop = s->hist_n - s->o.max_history + 1;
            memmove(s->hist, s->hist + drop, (size_t)(s->hist_n - drop) * 4); s->hist_n -= drop;
        } else {
            int nc = s->hist_cap * 2;
            int32_t *h = (int32_t *)realloc(s->hist, (size_t)nc * 4);
            if (!h) return;
            s->hist = h; s->hist_cap = nc;
        }
    }
    if (s->o.max_history > 0 && s->hist_n >= s->o.max_history) {
        memmove(s->hist, s->hist + 1, (size_t)(s->hist_n - 1) * 4); s->hist_n--;
    }
    s->hist[s->hist_n++] = id;
}

/* ---------------------------------------------------------------- una capa de atencion, una posicion */
static void rope_rot(const TernaryInferModel *m, int pos, const int32_t *z, int heads, int hd, int32_t *out) {
    const int P = hd / 2;
    const int16_t *c = m->rope_c + (size_t)pos * P, *sn = m->rope_s + (size_t)pos * P;
    for (int h = 0; h < heads; h++) {
        const int32_t *zh = z + h * hd; int32_t *oh = out + h * hd;
        for (int i = 0; i < P; i++) {
            const int32_t a = zh[2 * i], b = zh[2 * i + 1];
            oh[2 * i] = a * c[i] - b * sn[i];
            oh[2 * i + 1] = a * sn[i] + b * c[i];
        }
    }
}

/* Un carril = un token en vuelo: su sesion, su posicion y su memoria de trabajo. */
typedef struct {
    TernaryInferSession *s;
    int32_t id;
    int pos, n;                 /* posicion absoluta y ranura de la cache */
    int want_head;
    TernaryInferScratch w;
    const float *src;           /* entrada de la subcapa actual */
    float sx; int32_t zp;       /* cuantizacion de src */
    float sa; int32_t za;       /* cuantizacion de la salida de atencion / del hidden */
} Lane;

static void lane_scratch(TernaryInferSession *s, int k, TernaryInferScratch *w) {
    if (k == 0) {
        w->x = s->x; w->y = s->y; w->f = s->f; w->q8 = s->q8; w->q8b = s->q8b; w->zi = s->zi; w->zi2 = s->zi2; w->qr = s->qr; w->kr = s->kr;
        w->sh = s->sh; w->attn = s->attn; w->vrow = s->vrow; w->hbuf = s->hbuf;
    } else *w = s->xs[k - 1];
}

/* Matriz sobre todos los carriles: q/zp/z los da el llamador por carril. */
#define LANE_MATMUL(mat, QF, ZPF, ZF)                                                                                 \
    do {                                                                                                              \
        const uint8_t *qp_[TERNARYAI_INFER_MAX_LANES]; int32_t zp_[TERNARYAI_INFER_MAX_LANES]; int32_t *zz_[TERNARYAI_INFER_MAX_LANES]; \
        for (int b_ = 0; b_ < B; b_++) { Lane *L_ = &Ln[b_]; qp_[b_] = (QF); zp_[b_] = (ZPF); zz_[b_] = (ZF); }          \
        ternaryai_infer_matmul(&(mat), qp_, zp_, zz_, B);                                                             \
    } while (0)

/* Puntajes con claves int16 planas [fila][pos] (portable; el compilador usa multiplicaciones de 16 bits). Mismos valores que el
 * camino int64 cuando hd*max|q|*max|k| < 2^31 (lo comprueba el llamador). */
static void scores16_planar(const int16_t *k16, int tp, int hd, int start, int cnt, const int32_t *const *q, int G,
                            int shift, int32_t lim, int32_t **sc) {
    const int sh = 16 + shift > 31 ? 31 : 16 + shift;
    for (int b = 0; b < G; b++) {
        int32_t *acc = sc[b];
        for (int j = 0; j < cnt; j++) acc[j] = 0;
        for (int d = 0; d < hd; d++) {
            const int32_t qd = (int16_t)q[b][d]; const int16_t *row = k16 + (size_t)d * tp + start;
            for (int j = 0; j < cnt; j++) acc[j] += qd * (int32_t)row[j];
        }
        for (int j = 0; j < cnt; j++) { int32_t v = acc[j] >> sh; if (v > lim) v = lim; else if (v < -lim) v = -lim; acc[j] = v; }
    }
}

typedef struct {
    const TernaryInferGqa *g; TernaryInferSession *s; TernaryInferLayerCache *c; Lane *L;
    int C, nk, W, S, windowed, cnt, startB, stride, use16; float *invs;
} AttnCtx;

/* Una cabeza KV (sus G cabezas de consulta): puntajes, softmax y suma ponderada de V. Cada cabeza KV usa su propio tramo del scratch de la sesion
 * y escribe solo sus cabezas: el reparto en hilos no cambia ningun bit. */
static void attn_kv_task(size_t k0, size_t k1, void *ctx_) {
    AttnCtx *x = (AttnCtx *)ctx_;
    const TernaryInferGqa *g = x->g; TernaryInferSession *s = x->s; TernaryInferLayerCache *c = x->c; Lane *L = x->L;
    const int C = x->C, nk = x->nk, W = x->W, S = x->S, windowed = x->windowed, cnt = x->cnt, startB = x->startB, stride = x->stride, use16 = x->use16;
    float *invs = x->invs;
    const int hd = g->hd, G = g->group;
    const uint32_t K0 = g->exp_k0; const float *tab = g->exp_tab;
    for (size_t kh_ = k0; kh_ < k1; kh_++) {
        const int kh = (int)kh_;
        int32_t *sc[8]; const int32_t *qp[8];
        int32_t *scb = s->sc;
        int64_t *scr = s->scr + (size_t)kh * (size_t)stride;

            for (int b = 0; b < G && b < 8; b++) { sc[b] = scb + (size_t)(kh * G + b) * stride; qp[b] = L->w.qr + (size_t)(kh * G + b) * hd; }
            const int32_t *kT = c->kT ? c->kT + (size_t)kh * hd * c->tp : NULL;
            TERNARYAI_INFER_PROF_BEGIN(P_A_SC);          /* sub-ambitos solo fiables con 1 hilo (el perfilador no es multihilo) */
            if (use16 && c->k16kind == 2) {
                const int16_t *k16 = c->k16 + (size_t)kh * hd * c->tp;
                void (*const s16p)(const int16_t *, int, int, int, int, const int32_t *const *, int, int, int32_t, int32_t **) =
                    ternaryai_infer_kern->scores16p ? ternaryai_infer_kern->scores16p : scores16_planar;
                if (!windowed) s16p(k16, c->tp, hd, 0, nk, qp, G, g->shift, g->qk_lim, sc);
                else {
                    int32_t *scB[8];
                    if (S > 0) s16p(k16, c->tp, hd, 0, S, qp, G, g->shift, g->qk_lim, sc);
                    for (int b = 0; b < G; b++) scB[b] = sc[b] + S;
                    s16p(k16, c->tp, hd, startB, W, qp, G, g->shift, g->qk_lim, scB);
                }
            } else if (use16) {
                const int16_t *k16 = c->k16 + (size_t)kh * (hd / 2) * c->tp * 2;
                if (!windowed) ternaryai_infer_kern->scores16(k16, c->tp, hd, 0, nk, qp, G, g->shift, g->qk_lim, sc);
                else {
                    int32_t *scB[8];
                    if (S > 0) ternaryai_infer_kern->scores16(k16, c->tp, hd, 0, S, qp, G, g->shift, g->qk_lim, sc);
                    for (int b = 0; b < G; b++) scB[b] = sc[b] + S;
                    ternaryai_infer_kern->scores16(k16, c->tp, hd, startB, W, qp, G, g->shift, g->qk_lim, scB);
                }
            } else if (!windowed) {
                ternaryai_infer_kern->scores(kT, c->tp, hd, 0, nk, qp, G, g->shift, g->qk_lim, sc, scr);
            } else {
                int32_t *scB[8];
                if (S > 0) ternaryai_infer_kern->scores(kT, c->tp, hd, 0, S, qp, G, g->shift, g->qk_lim, sc, scr);
                for (int b = 0; b < G; b++) scB[b] = sc[b] + S;
                ternaryai_infer_kern->scores(kT, c->tp, hd, startB, W, qp, G, g->shift, g->qk_lim, scB, scr);
            }
            TERNARYAI_INFER_PROF_END(P_A_SC);
            TERNARYAI_INFER_PROF_BEGIN(P_A_SM);
            for (int b = 0; b < G; b++) {                  /* softmax de cada cabeza: ee por cabeza, 1/suma aparte */
                const int h = kh * G + b;
                const int32_t *sb = sc[b];
                float *ee = s->ee + (size_t)h * stride;
                float tot;
                if (ternaryai_infer_kern->softmax) tot = ternaryai_infer_kern->softmax(sb, cnt, tab, K0, ee);
                else {
                    int32_t mmax = sb[0];
                    for (int j = 1; j < cnt; j++) if (sb[j] > mmax) mmax = sb[j];
                    float part[16] = {0};
                    for (int j = 0; j < cnt; j++) {
                        const uint32_t k = (uint32_t)(mmax - sb[j]);
                        const float e = tab[k < K0 ? k : K0];
                        ee[j] = e; part[j & 15] += e;
                    }
                    tot = sum16(part);
                }
                invs[h] = 1.0f / tot;
            }
            TERNARYAI_INFER_PROF_END(P_A_SM);
            TERNARYAI_INFER_PROF_BEGIN(P_A_PV);
                /* suma ponderada de V de todas las cabezas a la vez (cadenas independientes: se solapa la latencia) */
        {
            const int h0 = kh * G, nc = G;
            const float *eep[64], *eep2[64]; const int8_t *vp[64], *vp2[64]; float *ap[64];
            for (int i = 0; i < nc; i++) {
                const int h = h0 + i;
                eep[i] = s->ee + (size_t)h * stride; vp[i] = c->v8 + (size_t)(h / G) * C * hd; ap[i] = L->w.attn + (size_t)h * hd;
                eep2[i] = eep[i] + S; vp2[i] = vp[i] + (size_t)startB * hd;
                for (int d = 0; d < hd; d++) ap[i][d] = 0.0f;
            }
            if (ternaryai_infer_kern->pv_multi) {
                if (!windowed) ternaryai_infer_kern->pv_multi(nc, eep, cnt, vp, c->vs, hd, ap);
                else { if (S > 0) ternaryai_infer_kern->pv_multi(nc, eep, S, vp, c->vs, hd, ap); ternaryai_infer_kern->pv_multi(nc, eep2, W, vp2, c->vs + startB, hd, ap); }
            } else {
                for (int i = 0; i < nc; i++) {
                    float *acc = ap[i];
                    if (!windowed) { for (int j = 0; j < cnt; j++) { const float e = eep[i][j], sj = c->vs[j]; const int8_t *vj = vp[i] + (size_t)j * hd; for (int d = 0; d < hd; d++) { const float vv = (float)vj[d] * sj; acc[d] += e * vv; } } }
                    else {
                        for (int j = 0; j < S; j++) { const float e = eep[i][j], sj = c->vs[j]; const int8_t *vj = vp[i] + (size_t)j * hd; for (int d = 0; d < hd; d++) { const float vv = (float)vj[d] * sj; acc[d] += e * vv; } }
                        for (int j = 0; j < W; j++) { const float e = eep2[i][j], sj = c->vs[startB + j]; const int8_t *vj = vp2[i] + (size_t)j * hd; for (int d = 0; d < hd; d++) { const float vv = (float)vj[d] * sj; acc[d] += e * vv; } }
                    }
                }
            }
        }
        TERNARYAI_INFER_PROF_END(P_A_PV);
    }
}

/* Atencion de una capa para B carriles (un token por carril; los carriles de una misma sesion son posiciones consecutivas). */
static void gqa_lanes(Lane *Ln, int B, const TernaryInferGqa *g, int li) {
    const int hd = g->hd, qw = g->qw, kvw = g->kvw;
    const TernaryInferModel *m = Ln[0].s->m;

    TERNARYAI_INFER_PROF_BEGIN(P_PROJQ);
    LANE_MATMUL(g->wq, L_->w.q8, L_->zp, L_->w.zi);
    for (int b = 0; b < B; b++) {
        Lane *L = &Ln[b]; int32_t *zi = L->w.zi, *sh = L->w.sh;
        for (int i = 0; i < qw; i++) sh[i] = (zi[i] + 128) >> 8;
        rope_rot(m, L->pos, sh, g->n_heads, hd, L->w.qr);
    }
    TERNARYAI_INFER_PROF_END(P_PROJQ);

    TERNARYAI_INFER_PROF_BEGIN(P_PROJK);
    LANE_MATMUL(g->wk, L_->w.q8, L_->zp, L_->w.zi);
    for (int b = 0; b < B; b++) {                       /* en orden: el smear de un carril usa la clave del anterior */
        Lane *L = &Ln[b]; TernaryInferLayerCache *c = &L->s->lc[li]; int32_t *zi = L->w.zi, *sh = L->w.sh;
        for (int i = 0; i < kvw; i++) sh[i] = (zi[i] + 128) >> 8;
        if (g->smear) {
            for (int i = 0; i < kvw; i++) { const int32_t a = sh[i]; sh[i] = (L->n == 0) ? a : (a + c->prev[i]) >> 1; c->prev[i] = a; }
        }
        rope_rot(m, L->pos, sh, g->n_kv, hd, L->w.kr);
        int fits = 1; int32_t mx = c->kmax;
        for (int i = 0; i < kvw; i++) { const int32_t v = L->w.kr[i], a = v < 0 ? -v : v; if (a > mx) mx = a; if (a > TERNARYAI_INFER_K16_MAX) fits = 0; }
        if (!fits && !c->kT && !ternaryai_infer_kt_promote(c, kvw)) { L->s->oom = 1; continue; }
        if (c->kT) for (int i = 0; i < kvw; i++) c->kT[(size_t)i * c->tp + L->n] = L->w.kr[i];
        if (c->k16 && c->k16ok) {
            if (fits) {
                for (int i = 0; i < kvw; i++) {
                    const int32_t v = L->w.kr[i];
                    if (c->k16kind == 1) c->k16[((size_t)(i >> 1) * c->tp + (size_t)L->n) * 2 + (size_t)(i & 1)] = (int16_t)v;
                    else c->k16[(size_t)i * c->tp + (size_t)L->n] = (int16_t)v;
                }
                c->kmax = mx;
            } else c->k16ok = 0;                    /* kT ya existe y tiene todas las claves */
        }
    }
    TERNARYAI_INFER_PROF_END(P_PROJK);

    TERNARYAI_INFER_PROF_BEGIN(P_PROJV);
    LANE_MATMUL(g->wv, L_->w.q8, L_->zp, L_->w.zi);
    for (int b = 0; b < B; b++) {
        Lane *L = &Ln[b]; TernaryInferLayerCache *c = &L->s->lc[li]; const int C = L->s->o.max_context; int32_t *zi = L->w.zi;
        const float k = g->wv.scale * L->sx;
        float *v = L->w.vrow;
        float mx = 0.0f; int8_t vq[4096];       /* kvw <= 4096 (comprobado al cargar) */
        for (int i = 0; i < kvw; i++) { v[i] = k * (float)zi[i]; const float a = fabsf(v[i]); if (a > mx) mx = a; }
        float sc = 0.0f;
        for (int i = 0; i < kvw; i++) vq[i] = 0;
        if (mx > 0.0f) { sc = mx / 127.0f; for (int i = 0; i < kvw; i++) vq[i] = (int8_t)roundf(v[i] / sc); }
        c->vs[L->n] = sc;
        for (int kh = 0; kh < g->n_kv; kh++) memcpy(c->v8 + ((size_t)kh * C + L->n) * hd, vq + kh * hd, (size_t)hd);
    }
    TERNARYAI_INFER_PROF_END(P_PROJV);

    TERNARYAI_INFER_PROF_BEGIN(P_ATTN);
    for (int bl = 0; bl < B; bl++) {
        Lane *L = &Ln[bl]; TernaryInferSession *s = L->s; TernaryInferLayerCache *c = &s->lc[li];
        const int C = s->o.max_context, nk = L->n + 1;
        const int W = s->o.attn_window, S = s->o.sinks;
        const int windowed = (W > 0 && nk > W + S);
        const int cnt = windowed ? S + W : nk;
        const int startB = windowed ? nk - W : 0;
        const int stride = C + 32;
        const int nh = g->n_heads;
        int use16 = 0;
        if (c->k16ok && (c->k16kind == 2 || ternaryai_infer_kern->scores16)) {          /* acumulador int32 exacto si hd*max|q|*max|k| < 2^31 */
            int32_t qm = 0;
            for (int i = 0; i < qw; i++) { const int32_t v = L->w.qr[i], a = v < 0 ? -v : v; if (a > qm) qm = a; }
            use16 = (qm <= 32767) && ((int64_t)hd * qm * c->kmax < 2147483647LL);
        }
        if (use16) ternaryai_infer_n_fast++; else { ternaryai_infer_n_slow++; if (!c->kT && !ternaryai_infer_kt_promote(c, kvw)) { s->oom = 1; continue; } }
        float invs[256];                     /* 1/suma por cabeza (n_heads <= 256) */
        {
            AttnCtx ax = { g, s, c, L, C, nk, W, S, windowed, cnt, startB, stride, use16, invs };
            const size_t work = (size_t)cnt * (size_t)hd * (size_t)g->n_heads;
            TernaryThreadPool *pool = ternary_get_compute_pool();
            if (pool && work >= 262144 && g->n_kv > 1) ternary_parallel_ranges(pool, 0, (size_t)g->n_kv, 1, attn_kv_task, &ax);
            else attn_kv_task(0, (size_t)g->n_kv, &ax);
        }
        for (int h = 0; h < nh; h++) {
            float *o = L->w.attn + (size_t)h * hd; const float inv = invs[h];
            for (int d = 0; d < hd; d++) o[d] = o[d] * inv;
            if (s->probs_rec) {
                const float *ee = s->ee + (size_t)h * stride;
                float *pr = s->probs_rec + ((size_t)li * g->n_heads + (size_t)h) * (size_t)(C + 16);
                for (int j = 0; j < cnt; j++) pr[j] = ee[j] * inv;
                s->probs_n[li] = cnt;
            }
        }
    }
    TERNARYAI_INFER_PROF_END(P_ATTN);

    TERNARYAI_INFER_PROF_BEGIN(P_PROJO);
    for (int b = 0; b < B; b++) ternaryai_infer_quantize_row(Ln[b].w.attn, qw, Ln[b].w.q8b, &Ln[b].sa, &Ln[b].za);
    LANE_MATMUL(g->wo, L_->w.q8b, L_->za, L_->w.zi);
    for (int b = 0; b < B; b++) {
        Lane *L = &Ln[b]; const float k = g->wo.scale * L->sa;
        for (int o = 0; o < m->dim; o++) L->w.f[o] = k * (float)L->w.zi[o];
    }
    TERNARYAI_INFER_PROF_END(P_PROJO);
}

/* ---------------------------------------------------------------- feed-forward (ReLU^2 / silu / sign) */
static void ffn_lanes(Lane *Ln, int B, const TernaryInferFfn *f) {
    const int H = f->hidden;
    TERNARYAI_INFER_PROF_BEGIN(P_FFN_UP);
    LANE_MATMUL(f->up, L_->w.q8, L_->zp, L_->w.zi);
    if (f->gate_mode != 2) LANE_MATMUL(f->gate, L_->w.q8, L_->zp, L_->w.zi2);
    TERNARYAI_INFER_PROF_END(P_FFN_UP);
    TERNARYAI_INFER_PROF_BEGIN(P_FFN_ACT);
    for (int b = 0; b < B; b++) {
        Lane *L = &Ln[b]; float *h = L->w.hbuf; const int32_t *zi = L->w.zi, *zi2 = L->w.zi2;
        const float ku = f->up.scale * L->sx;
        if (f->gate_mode == 2) {
            if (ternaryai_infer_kern->relu2) ternaryai_infer_kern->relu2(zi, H, ku, h);
            else for (int j = 0; j < H; j++) { const float a = ku * (float)zi[j]; const float r = a > 0.0f ? a : 0.0f; h[j] = r * r; }   /* sin salto: vectoriza */
        } else if (f->gate_mode == 0) {
            const float kg = f->gate.scale * L->sx;
            for (int j = 0; j < H; j++) { float sv; silu_f(kg * (float)zi2[j], &sv); h[j] = sv * (ku * (float)zi[j]); }
        } else {
            for (int j = 0; j < H; j++) { const int32_t zg = zi2[j]; h[j] = (float)(zg > 0) - (float)(zg < 0); h[j] *= ku * (float)zi[j]; }
        }
        ternaryai_infer_quantize_row(h, H, L->w.q8b, &L->sa, &L->za);
    }
    TERNARYAI_INFER_PROF_END(P_FFN_ACT);
    TERNARYAI_INFER_PROF_BEGIN(P_FFN_DOWN);
    LANE_MATMUL(f->down, L_->w.q8b, L_->za, L_->w.zi);
    for (int b = 0; b < B; b++) {
        Lane *L = &Ln[b]; const float k = f->down.scale * L->sa; const int dim = L->s->m->dim;
        for (int o = 0; o < dim; o++) L->w.f[o] = k * (float)L->w.zi[o];
    }
    TERNARYAI_INFER_PROF_END(P_FFN_DOWN);
}

/* ---------------------------------------------------------------- contexto lleno */
static int step_token(TernaryInferSession *s, int32_t id, int want_head, int record_hist);

static int make_room(TernaryInferSession *s) {
    const int C = s->o.max_context;
    if (s->n < C) return TERNARYAI_INFER_OK;
    switch (s->o.ctx_policy) {
    case TERNARYAI_INFER_CTX_ERROR: return TERNARYAI_INFER_E_CTX_FULL;
    case TERNARYAI_INFER_CTX_TRUNCATE: {
        const int keep = s->o.keep_tokens;
        if (s->hist_n < keep) return TERNARYAI_INFER_E_CTX_FULL;
        int32_t *tail = (int32_t *)ternary_malloc64((size_t)keep * 4);
        if (!tail) return TERNARYAI_INFER_E_NOMEM;
        memcpy(tail, s->hist + (s->hist_n - keep), (size_t)keep * 4);
        s->n = 0; s->next_pos = 0; s->have_head = 0;
        kcache_reset(s);
        int rc = TERNARYAI_INFER_OK;
        for (int i = 0; i < keep && rc == TERNARYAI_INFER_OK; i++) rc = step_token(s, tail[i], 0, 0);
        free(tail);
        return rc;
    }
    case TERNARYAI_INFER_CTX_SLIDE: {
        const int S = s->o.sinks;
        int ev = (C - S) / 4; if (ev < 1) ev = 1;
        if (S + ev > s->n) ev = s->n - S;
        if (ev < 1) return TERNARYAI_INFER_E_CTX_FULL;
        const TernaryInferModel *m = s->m;
        for (int i = 0; i < m->n_top; i++) {
            if (m->top[i].kind != TERNARYAI_INFER_TOP_RESIDUAL) continue;
            for (int j = 0; j < m->top[i].n_sub; j++) {
                const TernaryInferSub *sb = &m->top[i].sub[j];
                if (sb->kind != TERNARYAI_INFER_SUB_GQA) continue;
                TernaryInferLayerCache *c = &s->lc[sb->gqa_idx];
                const int rest = s->n - S - ev;
                if (c->kT)
                    for (int r = 0; r < sb->gqa.kvw; r++) {
                        int32_t *row = c->kT + (size_t)r * c->tp;
                        memmove(row + S, row + S + ev, (size_t)rest * 4);
                    }
                if (c->k16 && c->k16kind == 1)
                    for (int r = 0; r < sb->gqa.kvw / 2; r++) {
                        int16_t *row = c->k16 + (size_t)r * c->tp * 2;
                        memmove(row + 2 * (size_t)S, row + 2 * (size_t)(S + ev), (size_t)rest * 4);
                    }
                else if (c->k16)
                    for (int r = 0; r < sb->gqa.kvw; r++) {
                        int16_t *row = c->k16 + (size_t)r * c->tp;
                        memmove(row + S, row + S + ev, (size_t)rest * 2);
                    }
                for (int kh = 0; kh < sb->gqa.n_kv; kh++) {
                    int8_t *vb = c->v8 + (size_t)kh * C * sb->gqa.hd;
                    memmove(vb + (size_t)S * sb->gqa.hd, vb + (size_t)(S + ev) * sb->gqa.hd, (size_t)rest * sb->gqa.hd);
                }
                memmove(c->vs + S, c->vs + S + ev, (size_t)rest * 4);
            }
        }
        s->n -= ev;
        return TERNARYAI_INFER_OK;
    }
    }
    return TERNARYAI_INFER_E_ARG;
}

/* ---------------------------------------------------------------- tokens en vuelo */
/* Carriles extra de una sesion (prefill por bloques). */
static int ensure_extra_lanes(TernaryInferSession *s, int need) {
    if (need <= s->xs_n) return 1;
    const TernaryInferModel *m = s->m;
    const int E = m->dim;
    const int wide = (m->max_qw > m->max_hidden ? m->max_qw : m->max_hidden);
    const int w8 = (wide > E ? wide : E) + 128;
    const int rows16 = ((m->max_rows_body + 15) / 16) * 16 + 16;
    TernaryInferScratch *xs = (TernaryInferScratch *)realloc(s->xs, (size_t)need * sizeof(*xs));
    if (!xs) return 0;
    s->xs = xs;
    for (int i = s->xs_n; i < need; i++) {
        TernaryInferScratch *w = &xs[i];
        w->x = (float *)xalloc((size_t)(E + 16) * 4); w->y = (float *)xalloc((size_t)(E + 16) * 4);
        w->f = (float *)xalloc((size_t)(wide > E ? wide : E) * 4 + 64);
        w->q8 = (uint8_t *)xalloc((size_t)w8); w->q8b = (uint8_t *)xalloc((size_t)w8);
        w->zi = (int32_t *)xalloc((size_t)rows16 * 4); w->zi2 = (int32_t *)xalloc((size_t)rows16 * 4);
        w->qr = (int32_t *)xalloc((size_t)(m->max_qw + 16) * 4); w->kr = (int32_t *)xalloc((size_t)(m->max_kvw + 16) * 4);
        w->sh = (int32_t *)xalloc((size_t)((m->max_qw > m->max_kvw ? m->max_qw : m->max_kvw) + 16) * 4);
        w->attn = (float *)xalloc((size_t)(m->max_qw + 16) * 4); w->vrow = (float *)xalloc((size_t)(m->max_kvw + 16) * 4);
        w->hbuf = (float *)xalloc((size_t)(m->max_hidden + 16) * 4);
        s->xs_n = i + 1;
        if (!w->x || !w->y || !w->f || !w->q8 || !w->q8b || !w->zi || !w->zi2 || !w->qr || !w->kr || !w->sh || !w->attn || !w->vrow || !w->hbuf) return 0;
    }
    return 1;
}

/* Pasa B carriles por todo el modelo. Precondiciones: ids validos, hay sitio en la cache para cada carril, RoPE cubre las posiciones. */
static void run_lanes(Lane *Ln, int B) {
    const TernaryInferModel *m = Ln[0].s->m;
    const int E = m->dim;
    ternaryai_infer_model_sync((TernaryInferModel *)m);      /* pesos en el orden del nucleo activo (casi siempre ya lo estan) */
    TERNARYAI_INFER_PROF_BEGIN(P_EMBED);
    for (int b = 0; b < B; b++) {
        /* fila del embedding a 2 bits, desempaquetada al vuelo (4 pesos por byte) */
        const uint8_t *e = m->emb + (size_t)Ln[b].id * (size_t)m->emb_rb; float *x = Ln[b].w.x;
        const float es = m->emb_scale;
        int d = 0;
        for (; d + 4 <= E; d += 4) {
            const unsigned by = e[d >> 2];
            x[d] = es * (float)((int)(by & 3) - 1); x[d + 1] = es * (float)((int)((by >> 2) & 3) - 1);
            x[d + 2] = es * (float)((int)((by >> 4) & 3) - 1); x[d + 3] = es * (float)((int)(by >> 6) - 1);
        }
        for (; d < E; d++) x[d] = es * (float)((int)((e[d >> 2] >> (2 * (d & 3))) & 3) - 1);
    }
    TERNARYAI_INFER_PROF_END(P_EMBED);
    for (int t = 0; t < m->n_top; t++) {
        const TernaryInferTop *tp = &m->top[t];
        if (tp->kind == TERNARYAI_INFER_TOP_NORM) {
            TERNARYAI_INFER_PROF_BEGIN(P_NORM);
            for (int b = 0; b < B; b++) { ternaryai_infer_rmsnorm_row(&tp->norm, Ln[b].w.x, Ln[b].w.y); memcpy(Ln[b].w.x, Ln[b].w.y, (size_t)E * 4); }
            TERNARYAI_INFER_PROF_END(P_NORM);
            continue;
        }
        for (int b = 0; b < B; b++) Ln[b].src = Ln[b].w.x;
        for (int j = 0; j < tp->n_sub; j++) {
            const TernaryInferSub *sb = &tp->sub[j];
            if (sb->kind == TERNARYAI_INFER_SUB_NORM) {
                TERNARYAI_INFER_PROF_BEGIN(P_NORM);
                for (int b = 0; b < B; b++) { ternaryai_infer_rmsnorm_row(&sb->norm, Ln[b].src, Ln[b].w.y); Ln[b].src = Ln[b].w.y; }
                TERNARYAI_INFER_PROF_END(P_NORM);
                continue;
            }
            TERNARYAI_INFER_PROF_BEGIN(P_QUANT);
            for (int b = 0; b < B; b++) ternaryai_infer_quantize_row(Ln[b].src, E, Ln[b].w.q8, &Ln[b].sx, &Ln[b].zp);
            TERNARYAI_INFER_PROF_END(P_QUANT);
            if (sb->kind == TERNARYAI_INFER_SUB_GQA) gqa_lanes(Ln, B, &sb->gqa, sb->gqa_idx);
            else ffn_lanes(Ln, B, &sb->ffn);
            for (int b = 0; b < B; b++) Ln[b].src = Ln[b].w.f;
        }
        for (int b = 0; b < B; b++) { float *x = Ln[b].w.x; const float *r = Ln[b].src; for (int d = 0; d < E; d++) x[d] = x[d] + r[d]; }
    }
    /* cabeza de salida de los carriles que la piden (cada uno con los z de su sesion) */
    int hb = 0; Lane *hl[TERNARYAI_INFER_MAX_LANES];
    for (int b = 0; b < B; b++) if (Ln[b].want_head) hl[hb++] = &Ln[b];
    if (hb) {
        TERNARYAI_INFER_PROF_BEGIN(P_HEAD);
        const uint8_t *qp[TERNARYAI_INFER_MAX_LANES]; int32_t zp[TERNARYAI_INFER_MAX_LANES]; int32_t *zz[TERNARYAI_INFER_MAX_LANES]; float sxs[TERNARYAI_INFER_MAX_LANES];
        for (int b = 0; b < hb; b++) { ternaryai_infer_quantize_row(hl[b]->w.x, E, hl[b]->w.q8, &sxs[b], &zp[b]); qp[b] = hl[b]->w.q8; zz[b] = hl[b]->s->z; }
        ternaryai_infer_matmul(&m->head, qp, zp, zz, hb);
        for (int b = 0; b < hb; b++) { hl[b]->s->head_k = m->head.scale * sxs[b]; hl[b]->s->have_head = 1; }
        TERNARYAI_INFER_PROF_END(P_HEAD);
    }
}

/* Reparto GRUESO del lote entre hilos: cada hilo corre el modelo entero para su subgrupo de carriles (sesiones distintas, buffers propios),
 * sin sincronizar por operacion. Los carriles son independientes, asi que los bits no cambian con el reparto. Dentro de una tarea del pool
 * las operaciones repartidas (matmul, atencion) corren en serie. */
typedef struct { Lane *ln; int B, parts; } LaneSplit;
static void lane_split_task(size_t t0, size_t t1, void *c_) {
    LaneSplit *c = (LaneSplit *)c_;
    for (size_t t = t0; t < t1; t++) {
        const int lo = (int)((size_t)c->B * t / (size_t)c->parts), hi = (int)((size_t)c->B * (t + 1) / (size_t)c->parts);
        for (int a = lo; a < hi; a += TERNARYAI_INFER_MAX_LANES) run_lanes(c->ln + a, (hi - a < TERNARYAI_INFER_MAX_LANES) ? hi - a : TERNARYAI_INFER_MAX_LANES);
    }
}
static void run_lanes_split(Lane *Ln, int B) {
    TernaryThreadPool *pool = ternary_get_compute_pool();
    const int nt = pool ? ternary_threadpool_num_threads(pool) : 1;
    if (nt < 2 || B < 2 || ternaryai_infer_prof_on) { for (int a = 0; a < B; a += TERNARYAI_INFER_MAX_LANES) run_lanes(Ln + a, (B - a < TERNARYAI_INFER_MAX_LANES) ? B - a : TERNARYAI_INFER_MAX_LANES); return; }
    ternaryai_infer_model_sync((TernaryInferModel *)Ln[0].s->m);   /* el cambio de formato de pesos, antes de repartir */
    LaneSplit c = { Ln, B, nt < B ? nt : B };
    ternary_parallel_static(pool, (size_t)c.parts, lane_split_task, &c);
}

/* Un token de una sesion (con la politica de contexto llena). */
static int step_token(TernaryInferSession *s, int32_t id, int want_head, int record_hist) {
    const TernaryInferModel *m = s->m;
    if (id < 0 || id >= m->vocab) return TERNARYAI_INFER_E_RANGE;
    int rc = make_room(s);
    if (rc != TERNARYAI_INFER_OK) return rc;
    const int pos = s->next_pos;
    ternaryai_infer_rope_ensure((TernaryInferModel *)(uintptr_t)m, pos);
    if (m->rope_len <= pos && m->rope_hd) return TERNARYAI_INFER_E_NOMEM;
    Lane ln; memset(&ln, 0, sizeof(ln));
    ln.s = s; ln.id = id; ln.pos = pos; ln.n = s->n; ln.want_head = want_head; lane_scratch(s, 0, &ln.w);
    run_lanes(&ln, 1);
    if (s->oom) return TERNARYAI_INFER_E_NOMEM;
    s->n++; s->next_pos++;
    if (record_hist) hist_push(s, id);
    return TERNARYAI_INFER_OK;
}

int ternaryai_infer_feed1(TernaryInferSession *s, int32_t id) { return s ? step_token(s, id, 1, 1) : TERNARYAI_INFER_E_ARG; }

/* Prefill por bloques de carriles: las proyecciones de un bloque comparten la lectura de los pesos. Mismos bits que token a token. */
int ternaryai_infer_feed(TernaryInferSession *s, const int32_t *ids, int n) {
    if (!s || (n > 0 && !ids)) return TERNARYAI_INFER_E_ARG;
    const TernaryInferModel *m = s->m;
    for (int i = 0; i < n; i++) if (ids[i] < 0 || ids[i] >= m->vocab) return TERNARYAI_INFER_E_RANGE;
    int i = 0;
    while (i < n) {
        int B = n - i; if (B > TERNARYAI_INFER_MAX_LANES) B = TERNARYAI_INFER_MAX_LANES;
        if (s->n + B > s->o.max_context) B = s->o.max_context - s->n;     /* el bloque no cruza el limite del contexto */
        if (B < 2 || !ensure_extra_lanes(s, B - 1)) {
            int rc = step_token(s, ids[i], i == n - 1, 1);
            if (rc != TERNARYAI_INFER_OK) return rc;
            i++; continue;
        }
        ternaryai_infer_rope_ensure((TernaryInferModel *)(uintptr_t)m, s->next_pos + B - 1);
        if (m->rope_len <= s->next_pos + B - 1 && m->rope_hd) return TERNARYAI_INFER_E_NOMEM;
        Lane ln[TERNARYAI_INFER_MAX_LANES]; memset(ln, 0, sizeof(ln));
        for (int b = 0; b < B; b++) {
            ln[b].s = s; ln[b].id = ids[i + b]; ln[b].pos = s->next_pos + b; ln[b].n = s->n + b;
            ln[b].want_head = (i + b == n - 1); lane_scratch(s, b, &ln[b].w);
        }
        run_lanes(ln, B);
        if (s->oom) return TERNARYAI_INFER_E_NOMEM;
        s->n += B; s->next_pos += B;
        for (int b = 0; b < B; b++) hist_push(s, ids[i + b]);
        i += B;
    }
    return TERNARYAI_INFER_OK;
}

/* Un token en cada una de varias sesiones DISTINTAS, compartiendo la lectura de los pesos. rcs[i] recibe el resultado de la sesion i. */
#define STEP_PAR_MAX 64      /* con varios hilos y hasta tantos carriles, se preparan todos y se reparten por grupos completos */
void ternaryai_infer_step_many(TernaryInferSession **ss, const int32_t *ids, int n, int *rcs) {
    const int G = (n > STEP_PAR_MAX) ? TERNARYAI_INFER_MAX_LANES : STEP_PAR_MAX;     /* carriles preparados antes de correr */
    for (int g0 = 0; g0 < n; g0 += G) {
        const int cnt = (n - g0 < G) ? n - g0 : G;
        Lane ln[STEP_PAR_MAX]; int idx[STEP_PAR_MAX]; int B = 0;
        for (int k = 0; k < cnt; k++) {
            TernaryInferSession *s = ss[g0 + k]; const TernaryInferModel *m = s->m;
            int dup = 0;
            for (int q = 0; q < B; q++) if (ln[q].s == s) dup = 1;
            if (dup || ids[g0 + k] < 0 || ids[g0 + k] >= m->vocab) { rcs[g0 + k] = dup ? step_token(s, ids[g0 + k], 1, 1) : TERNARYAI_INFER_E_RANGE; continue; }
            int rc = make_room(s);
            if (rc == TERNARYAI_INFER_OK) {
                ternaryai_infer_rope_ensure((TernaryInferModel *)(uintptr_t)m, s->next_pos);
                if (m->rope_len <= s->next_pos && m->rope_hd) rc = TERNARYAI_INFER_E_NOMEM;
            }
            if (rc != TERNARYAI_INFER_OK) { rcs[g0 + k] = rc; continue; }
            memset(&ln[B], 0, sizeof(Lane));
            ln[B].s = s; ln[B].id = ids[g0 + k]; ln[B].pos = s->next_pos; ln[B].n = s->n; ln[B].want_head = 1; lane_scratch(s, 0, &ln[B].w);
            idx[B++] = g0 + k;
        }
        if (B) run_lanes_split(ln, B);
        for (int b = 0; b < B; b++) {
            TernaryInferSession *s = ln[b].s;
            if (s->oom) { rcs[idx[b]] = TERNARYAI_INFER_E_NOMEM; continue; }
            s->n++; s->next_pos++; hist_push(s, ln[b].id); rcs[idx[b]] = TERNARYAI_INFER_OK;
        }
    }
}

/* Para ternaryai_infer_score: procesa un token y deja los logits (uso interno de ternary_infer_gen.c). */
int ternaryai_infer_step_internal(TernaryInferSession *s, int32_t id, int want_head) { return step_token(s, id, want_head, 1); }

/* ---------------------------------------------------------------- salida del ultimo token */
int ternaryai_infer_argmax(const TernaryInferSession *s, int32_t *id_out) {
    if (!s || !id_out) return TERNARYAI_INFER_E_ARG;
    if (!s->have_head) return TERNARYAI_INFER_E_STATE;
    *id_out = ternaryai_infer_kern->argmax(s->z, s->m->vocab);
    return TERNARYAI_INFER_OK;
}

int ternaryai_infer_logits(const TernaryInferSession *s, float *out) {
    if (!s || !out) return TERNARYAI_INFER_E_ARG;
    if (!s->have_head) return TERNARYAI_INFER_E_STATE;
    for (int v = 0; v < s->m->vocab; v++) out[v] = s->head_k * (float)s->z[v];
    return TERNARYAI_INFER_OK;
}

int ternaryai_infer_attn_probs(const TernaryInferSession *s, int attn_layer, int head, float *out, int cap, int *n_out) {
    if (!s || !s->probs_rec || attn_layer < 0 || attn_layer >= s->m->n_attn || !out) return TERNARYAI_INFER_E_ARG;
    int nh = 0;
    for (int i = 0; i < s->m->n_top && !nh; i++) if (s->m->top[i].kind == TERNARYAI_INFER_TOP_RESIDUAL)
        for (int j = 0; j < s->m->top[i].n_sub; j++) if (s->m->top[i].sub[j].kind == TERNARYAI_INFER_SUB_GQA && s->m->top[i].sub[j].gqa_idx == attn_layer) { nh = s->m->top[i].sub[j].gqa.n_heads; break; }
    if (head < 0 || head >= nh) return TERNARYAI_INFER_E_ARG;
    const int cnt = s->probs_n[attn_layer];
    if (cnt == 0) return TERNARYAI_INFER_E_STATE;
    const float *pr = s->probs_rec + ((size_t)attn_layer * nh + (size_t)head) * (size_t)(s->o.max_context + 16);
    const int k = cnt < cap ? cnt : cap;
    memcpy(out, pr, (size_t)k * 4);
    if (n_out) *n_out = cnt;
    return TERNARYAI_INFER_OK;
}
