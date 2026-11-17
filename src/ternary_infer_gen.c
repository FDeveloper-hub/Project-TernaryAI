/* Muestreo, generacion, evaluacion y texto. */
#define _POSIX_C_SOURCE 200809L
#include "ternary_mem.h"
#include "ternary_threadpool.h"
#include "ternary_infer_internal.h"
#include <math.h>
#include <stdio.h>

int ternaryai_infer_step_internal(TernaryInferSession *s, int32_t id, int want_head);

/* ---------------------------------------------------------------- generador pseudoaleatorio (xorshift64*) */
static uint64_t rng_next(uint64_t *st) {
    uint64_t x = *st;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *st = x;
    return x * 0x2545F4914F6CDD1DULL;
}
static double rng_unit(uint64_t *st) { return (double)(rng_next(st) >> 11) * (1.0 / 9007199254740992.0); }   /* [0, 1) */

void ternaryai_infer_sampler_default(TernaryInferSampler *p) {
    memset(p, 0, sizeof(*p));
    p->temperature = 0.0f; p->top_p = 1.0f; p->repeat_penalty = 1.0f;
}
void ternaryai_infer_gen_params_default(TernaryInferGenParams *p) {
    memset(p, 0, sizeof(*p));
    p->max_new_tokens = 64;
    p->think_open_id = p->think_close_id = -1;
    ternaryai_infer_sampler_default(&p->sampler);
}

/* ---------------------------------------------------------------- logits y probabilidades */
typedef struct { float v; int i; } Cand;
static int cand_cmp(const void *a, const void *b) {
    const Cand *x = (const Cand *)a, *y = (const Cand *)b;
    if (x->v > y->v) return -1;
    if (x->v < y->v) return 1;
    return x->i - y->i;
}

int ternaryai_infer_top_probs(const TernaryInferSession *s, int n, int32_t *ids_out, float *probs_out) {
    if (!s || n <= 0 || !ids_out || !probs_out) return TERNARYAI_INFER_E_ARG;
    if (!s->have_head) return TERNARYAI_INFER_E_STATE;
    const int V = s->m->vocab;
    if (n > V) n = V;
    float mx = -INFINITY;
    for (int v = 0; v < V; v++) { const float l = s->head_k * (float)s->z[v]; if (l > mx) mx = l; }
    double Z = 0.0;
    for (int v = 0; v < V; v++) Z += exp((double)(s->head_k * (float)s->z[v] - mx));
    Cand *c = (Cand *)ternary_malloc64((size_t)V * sizeof(Cand));
    if (!c) return TERNARYAI_INFER_E_NOMEM;
    for (int v = 0; v < V; v++) { c[v].v = (float)s->z[v]; c[v].i = v; }
    qsort(c, (size_t)V, sizeof(Cand), cand_cmp);     /* head_k > 0: el orden de z es el de los logits */
    for (int k = 0; k < n; k++) {
        ids_out[k] = c[k].i;
        probs_out[k] = (float)(exp((double)(s->head_k * c[k].v - mx)) / Z);
    }
    free(c);
    return n;
}

/* ---------------------------------------------------------------- muestreo */
static int needs_logit_work(const TernaryInferSampler *p) {
    return p->logit_bias || p->allowed || (p->banned && p->n_banned > 0) || p->repeat_penalty != 1.0f ||
           p->freq_penalty != 0.0f || p->presence_penalty != 0.0f;
}

int ternaryai_infer_sample(TernaryInferSession *s, const TernaryInferSampler *p, int32_t *id_out) {
    if (!s || !p || !id_out) return TERNARYAI_INFER_E_ARG;
    if (!s->have_head) return TERNARYAI_INFER_E_STATE;
    const int V = s->m->vocab;
    if (p->temperature <= 0.0f && !needs_logit_work(p)) return ternaryai_infer_argmax(s, id_out);

    float *lg = s->lg;
    for (int v = 0; v < V; v++) lg[v] = s->head_k * (float)s->z[v];
    if (p->logit_bias) for (int v = 0; v < V; v++) lg[v] += p->logit_bias[v];
    if (p->repeat_penalty != 1.0f || p->freq_penalty != 0.0f || p->presence_penalty != 0.0f) {
        int from = 0;
        if (p->penalty_last_n > 0 && s->hist_n > p->penalty_last_n) from = s->hist_n - p->penalty_last_n;
        int *cnt = (int *)ternary_calloc64((size_t)V, sizeof(int));
        if (!cnt) return TERNARYAI_INFER_E_NOMEM;
        for (int i = from; i < s->hist_n; i++) if (s->hist[i] >= 0 && s->hist[i] < V) cnt[s->hist[i]]++;
        for (int v = 0; v < V; v++) {
            if (!cnt[v]) continue;
            if (p->repeat_penalty != 1.0f) lg[v] = lg[v] > 0.0f ? lg[v] / p->repeat_penalty : lg[v] * p->repeat_penalty;
            lg[v] -= p->freq_penalty * (float)cnt[v] + p->presence_penalty;
        }
        free(cnt);
    }
    if (p->allowed) for (int v = 0; v < V; v++) if (!p->allowed[v]) lg[v] = -INFINITY;
    if (p->banned) for (int k = 0; k < p->n_banned; k++) if (p->banned[k] >= 0 && p->banned[k] < V) lg[p->banned[k]] = -INFINITY;

    if (p->temperature <= 0.0f) {     /* voraz con modificadores */
        int best = -1; float bv = -INFINITY;
        for (int v = 0; v < V; v++) if (lg[v] > bv) { bv = lg[v]; best = v; }
        if (best < 0) return TERNARYAI_INFER_E_STATE;
        *id_out = best;
        return TERNARYAI_INFER_OK;
    }

    Cand *c = (Cand *)ternary_malloc64((size_t)V * sizeof(Cand));
    double *pr = (double *)ternary_malloc64((size_t)V * sizeof(double));
    if (!c || !pr) { free(c); free(pr); return TERNARYAI_INFER_E_NOMEM; }
    int nc = 0;
    for (int v = 0; v < V; v++) if (lg[v] > -INFINITY) { c[nc].v = lg[v] / p->temperature; c[nc].i = v; nc++; }
    if (nc == 0) { free(c); free(pr); return TERNARYAI_INFER_E_STATE; }
    const int sorted_needed = (p->top_k > 0 && p->top_k < nc) || (p->top_p > 0.0f && p->top_p < 1.0f) || p->min_p > 0.0f;
    if (sorted_needed) qsort(c, (size_t)nc, sizeof(Cand), cand_cmp);
    if (p->top_k > 0 && p->top_k < nc) nc = p->top_k;
    float mx = c[0].v;
    if (!sorted_needed) for (int k = 1; k < nc; k++) if (c[k].v > mx) mx = c[k].v;
    double Z = 0.0;
    for (int k = 0; k < nc; k++) { pr[k] = exp((double)(c[k].v - mx)); Z += pr[k]; }
    for (int k = 0; k < nc; k++) pr[k] /= Z;
    if (p->min_p > 0.0f) {          /* c esta ordenado: pr[0] es el maximo */
        const double thr = (double)p->min_p * pr[0];
        int keep = nc;
        for (int k = 0; k < nc; k++) if (pr[k] < thr) { keep = k; break; }
        nc = keep > 0 ? keep : 1;
    }
    if (p->top_p > 0.0f && p->top_p < 1.0f) {
        double cum = 0.0; int keep = nc;
        for (int k = 0; k < nc; k++) { cum += pr[k]; if (cum >= (double)p->top_p) { keep = k + 1; break; } }
        nc = keep;
    }
    double tot = 0.0;
    for (int k = 0; k < nc; k++) tot += pr[k];
    double r = rng_unit(&s->rng) * tot, acc = 0.0;
    int pick = nc - 1;
    for (int k = 0; k < nc; k++) { acc += pr[k]; if (r < acc) { pick = k; break; } }
    *id_out = c[pick].i;
    free(c); free(pr);
    return TERNARYAI_INFER_OK;
}

/* ---------------------------------------------------------------- generacion */
typedef struct {
    TernaryInferSession *s; int32_t *out; int cap, count; TernaryInferStopReason why; int active; TernaryInferSampler samp; uint8_t *mask;
    int in_think, think_n, think_init;      /* presupuesto de razonamiento */
} GenState;

static inline int think_on(const TernaryInferGenParams *p) { return p->think_budget > 0 && p->think_open_id >= 0 && p->think_close_id >= 0; }

static int is_stop_id(const TernaryInferGenParams *p, int32_t id) {
    for (int k = 0; k < p->n_stop_ids; k++) if (p->stop_ids[k] == id) return 1;
    return 0;
}

/* Primera mitad de un paso: elige y emite el token. Devuelve 1 si hay que procesarlo (*id_out), 0 si no hay nada que procesar.
 * *finish_after != 0: tras procesarlo (si hay algo) la generacion de esta sesion termino (g->why ya esta puesto). */
static int gen_pick(GenState *g, const TernaryInferGenParams *p, int32_t *id_out, int *finish_after) {
    TernaryInferSession *s = g->s;
    *finish_after = 0;
    if (g->count >= p->max_new_tokens) { g->why = TERNARYAI_INFER_STOP_MAX_TOKENS; *finish_after = 1; return 0; }
    if (!s->have_head) { g->why = TERNARYAI_INFER_STOP_ERROR; *finish_after = 1; return 0; }
    g->samp = p->sampler;
    if (p->mask_fn) {
        memset(g->mask, 1, (size_t)s->m->vocab);
        if (p->mask_fn(p->user, s, g->mask)) g->samp.allowed = g->mask;
    }
    int32_t id;
    const int thk = think_on(p);
    if (thk && !g->think_init) { g->think_init = 1; g->in_think = p->think_inside ? 1 : 0; g->think_n = 0; }
    if (thk && g->in_think && g->think_n >= p->think_budget) id = p->think_close_id;          /* presupuesto agotado: se cierra el razonamiento */
    else if (ternaryai_infer_sample(s, &g->samp, &id) != TERNARYAI_INFER_OK) { g->why = TERNARYAI_INFER_STOP_ERROR; *finish_after = 1; return 0; }
    if (thk) {
        if (id == p->think_open_id) { g->in_think = 1; g->think_n = 0; }
        else if (id == p->think_close_id) g->in_think = 0;
        else if (g->in_think) g->think_n++;
    }
    *id_out = id;
    const int stop = is_stop_id(p, id);
    if (stop && !p->include_stop) { g->why = TERNARYAI_INFER_STOP_TOKEN; *finish_after = 1; return 1; }   /* el token de parada entra al contexto */
    if (g->count >= g->cap) { g->why = TERNARYAI_INFER_STOP_MAX_TOKENS; *finish_after = 1; return 0; }
    g->out[g->count++] = id;
    if (p->on_token && p->on_token(id, p->user)) { g->why = TERNARYAI_INFER_STOP_CALLBACK; *finish_after = 1; return 1; }
    int seq_hit = 0;
    for (int k = 0, off = 0; k < p->n_stop_seq && !seq_hit; k++) {
        const int L = p->stop_seq_len[k];
        if (L > 0 && g->count >= L && !memcmp(g->out + g->count - L, p->stop_seq + off, (size_t)L * sizeof(int32_t))) {
            seq_hit = 1;
            if (!p->include_stop) g->count -= L;
        }
        off += L;
    }
    if (stop) { g->why = TERNARYAI_INFER_STOP_TOKEN; *finish_after = 1; }
    else if (seq_hit) { g->why = TERNARYAI_INFER_STOP_SEQUENCE; *finish_after = 1; }
    return 1;
}

/* Un paso de una sola sesion. Devuelve 1 si la generacion termino. */
static int gen_step(GenState *g, const TernaryInferGenParams *p) {
    int32_t id; int fin;
    if (gen_pick(g, p, &id, &fin)) {
        int rc = ternaryai_infer_step_internal(g->s, id, 1);
        if (rc != TERNARYAI_INFER_OK) { g->why = TERNARYAI_INFER_STOP_ERROR; return 1; }
    }
    return fin;
}

int ternaryai_infer_generate(TernaryInferSession *s, const TernaryInferGenParams *p, int32_t *out, int cap, TernaryInferStopReason *why) {
    if (!s || !p || (cap > 0 && !out)) return TERNARYAI_INFER_E_ARG;
    GenState g; memset(&g, 0, sizeof(g));
    g.s = s; g.out = out; g.cap = cap; g.active = 1;
    if (p->mask_fn) { g.mask = (uint8_t *)ternary_malloc64((size_t)s->m->vocab); if (!g.mask) return TERNARYAI_INFER_E_NOMEM; }
    while (!gen_step(&g, p)) {}
    free(g.mask);
    if (why) *why = g.why;
    if (g.why == TERNARYAI_INFER_STOP_ERROR && g.count == 0 && !s->have_head) return TERNARYAI_INFER_E_STATE;
    return g.count;
}

/* Un paso completo de las sesiones [lo, hi): elegir y emitir el token de cada una, procesarlas juntas (comparten la lectura de los pesos)
 * y anotar el resultado. Solo toca los GenState y las sesiones de ese tramo. */
static void gen_step_range(GenState *g, TernaryInferSession **ss, const TernaryInferGenParams *p, int lo, int hi) {
    for (int c0 = lo; c0 < hi; c0 += TERNARYAI_INFER_MAX_LANES) {
        const int c1 = (c0 + TERNARYAI_INFER_MAX_LANES < hi) ? c0 + TERNARYAI_INFER_MAX_LANES : hi;
        TernaryInferSession *bs[TERNARYAI_INFER_MAX_LANES]; int32_t bid[TERNARYAI_INFER_MAX_LANES];
        int bi[TERNARYAI_INFER_MAX_LANES], bfin[TERNARYAI_INFER_MAX_LANES], brc[TERNARYAI_INFER_MAX_LANES];
        int nb = 0;
        for (int i = c0; i < c1; i++) {
            if (!g[i].active) continue;
            int32_t id; int fin;
            if (gen_pick(&g[i], p, &id, &fin)) { bs[nb] = ss[i]; bid[nb] = id; bi[nb] = i; bfin[nb] = fin; nb++; }
            else if (fin) g[i].active = 0;
        }
        if (nb) ternaryai_infer_step_many(bs, bid, nb, brc);
        for (int k = 0; k < nb; k++) {
            const int i = bi[k];
            if (brc[k] != TERNARYAI_INFER_OK) { g[i].why = TERNARYAI_INFER_STOP_ERROR; g[i].active = 0; }
            else if (bfin[k]) g[i].active = 0;
        }
    }
}

typedef struct { GenState *g; TernaryInferSession **ss; const TernaryInferGenParams *p; int n, parts; } StepPar;
static void step_part(size_t b, size_t e, void *c_) {
    StepPar *c = (StepPar *)c_;
    for (size_t t = b; t < e; t++) gen_step_range(c->g, c->ss, c->p, (int)((size_t)c->n * t / (size_t)c->parts), (int)((size_t)c->n * (t + 1) / (size_t)c->parts));
}

int ternaryai_infer_generate_batch(TernaryInferSession **ss, int n, const TernaryInferGenParams *p, int32_t **outs, const int *caps, int *counts, TernaryInferStopReason *whys) {
    if (!ss || n <= 0 || !p || !outs || !caps) return TERNARYAI_INFER_E_ARG;
    GenState *g = (GenState *)ternary_calloc64((size_t)n, sizeof(GenState));
    if (!g) return TERNARYAI_INFER_E_NOMEM;
    int left = n;
    for (int i = 0; i < n; i++) {
        g[i].s = ss[i]; g[i].out = outs[i]; g[i].cap = caps[i]; g[i].active = 1;
        if (p->mask_fn) { g[i].mask = (uint8_t *)ternary_malloc64((size_t)ss[i]->m->vocab); if (!g[i].mask) { left = -1; } }
    }
    if (left < 0) { for (int i = 0; i < n; i++) free(g[i].mask); free(g); return TERNARYAI_INFER_E_NOMEM; }
    TernaryThreadPool *gpool = ternary_get_compute_pool();
    const int nthr = gpool ? ternary_threadpool_num_threads(gpool) : 1;
    if (nthr > 1 && n >= 2 && nthr <= 30 && !p->on_token && !p->mask_fn) {
        /* Sin callbacks de usuario, cada hilo avanza SUS sesiones un paso completo en una sola region de partes fijas: las mismas sesiones
         * siempre en el mismo hilo (su cache queda en su nucleo) y una sola sincronizacion por paso. Cada sesion es independiente, asi que
         * los resultados no dependen del reparto. */
        StepPar sp = { g, ss, p, n, nthr < n ? nthr : n };
        while (left > 0) {
            ternary_parallel_static(gpool, (size_t)sp.parts, step_part, &sp);
            left = 0;
            for (int i = 0; i < n; i++) left += g[i].active;
        }
    } else {
        while (left > 0) {
            gen_step_range(g, ss, p, 0, n);
            left = 0;
            for (int i = 0; i < n; i++) left += g[i].active;
        }
    }
    for (int i = 0; i < n; i++) {
        if (counts) counts[i] = g[i].count;
        if (whys) whys[i] = g[i].why;
        free(g[i].mask);
    }
    free(g);
    return TERNARYAI_INFER_OK;
}

int ternaryai_infer_generate_n(const TernaryInferSession *s, int n, const TernaryInferGenParams *p, int32_t **outs, const int *caps, int *counts, TernaryInferStopReason *whys) {
    if (!s || n <= 0) return TERNARYAI_INFER_E_ARG;
    TernaryInferSession **ss = (TernaryInferSession **)ternary_calloc64((size_t)n, sizeof(TernaryInferSession *));
    if (!ss) return TERNARYAI_INFER_E_NOMEM;
    int rc = TERNARYAI_INFER_OK;
    for (int i = 0; i < n && rc == TERNARYAI_INFER_OK; i++) {
        ss[i] = ternaryai_infer_session_fork(s);
        if (!ss[i]) { rc = TERNARYAI_INFER_E_NOMEM; break; }
        uint64_t st = s->rng + (uint64_t)(i + 1) * 0xA24BAED4963EE407ULL;
        ternaryai_infer_session_seed(ss[i], rng_next(&st));
    }
    if (rc == TERNARYAI_INFER_OK) rc = ternaryai_infer_generate_batch(ss, n, p, outs, caps, counts, whys);
    for (int i = 0; i < n; i++) ternaryai_infer_session_free(ss[i]);
    free(ss);
    return rc;
}

/* ---------------------------------------------------------------- evaluacion */
static double logprob_of(const TernaryInferSession *s, int32_t id) {
    const int V = s->m->vocab;
    float mx = -INFINITY;
    for (int v = 0; v < V; v++) { const float l = s->head_k * (float)s->z[v]; if (l > mx) mx = l; }
    double Z = 0.0;
    for (int v = 0; v < V; v++) Z += exp((double)(s->head_k * (float)s->z[v] - mx));
    return (double)(s->head_k * (float)s->z[id] - mx) - log(Z);
}

int ternaryai_infer_score(TernaryInferSession *s, const int32_t *ids, int n, float *logprob_out, double *ppl_out) {
    if (!s || n < 0 || (n > 0 && (!ids || !logprob_out))) return TERNARYAI_INFER_E_ARG;
    double sum = 0.0; int cnt = 0;
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= s->m->vocab) return TERNARYAI_INFER_E_RANGE;
        if (s->have_head) { const double lp = logprob_of(s, ids[i]); logprob_out[i] = (float)lp; sum += lp; cnt++; }
        else logprob_out[i] = NAN;
        int rc = ternaryai_infer_step_internal(s, ids[i], 1);
        if (rc != TERNARYAI_INFER_OK) return rc;
    }
    if (ppl_out) *ppl_out = cnt ? exp(-sum / (double)cnt) : NAN;
    return cnt;
}

/* ---------------------------------------------------------------- texto */
int ternaryai_infer_encode(const TernaryInferModel *m, const char *text, int parse_special, int32_t *ids, int cap, int *n_out) {
    if (!m || !text) return TERNARYAI_INFER_E_ARG;
    if (!m->tok) return TERNARYAI_INFER_E_NOTOK;
    size_t len = 0; char err[160];
    if (!ternary_tokenizer_encode_ex(m->tok, (const uint8_t *)text, strlen(text), parse_special, ids, (size_t)(cap > 0 ? cap : 0), &len, err, sizeof err)) return TERNARYAI_INFER_E_ARG;
    if (n_out) *n_out = (int)len;
    return (int)len > cap ? TERNARYAI_INFER_E_NOMEM : TERNARYAI_INFER_OK;
}

int ternaryai_infer_decode(const TernaryInferModel *m, const int32_t *ids, int n, int skip_special, char *out, int cap, int *len_out) {
    if (!m || (n > 0 && !ids)) return TERNARYAI_INFER_E_ARG;
    if (!m->tok) return TERNARYAI_INFER_E_NOTOK;
    size_t len = 0; char err[160];
    if (!ternary_tokenizer_decode_ex(m->tok, ids, (size_t)n, skip_special, (uint8_t *)out, (size_t)(cap > 0 ? cap - 1 : 0), &len, err, sizeof err)) return TERNARYAI_INFER_E_ARG;
    if (len_out) *len_out = (int)len;
    if (cap > 0) out[len < (size_t)cap - 1 ? len : (size_t)cap - 1] = 0;
    return (int)len >= cap ? TERNARYAI_INFER_E_NOMEM : TERNARYAI_INFER_OK;
}

int ternaryai_infer_special_id(const TernaryInferModel *m, const char *text) {
    if (!m || !m->tok || !text) return -1;
    return (int)ternary_tokenizer_special_id(m->tok, (const uint8_t *)text, strlen(text));
}

int ternaryai_infer_chat_encode(const TernaryInferModel *m, const char *const *roles, const char *const *texts, int n_msgs,
                   int add_assistant_prompt, int32_t *ids, int cap, int *n_out) {
    if (!m || !roles || !texts || n_msgs < 0) return TERNARYAI_INFER_E_ARG;
    if (!m->tok) return TERNARYAI_INFER_E_NOTOK;
    int n = 0;
    for (int i = 0; i < n_msgs; i++) {
        int rid;
        if (!strcmp(roles[i], "user")) rid = m->chat_user_id;
        else if (!strcmp(roles[i], "assistant")) rid = m->chat_asst_id;
        else { char tag[64]; snprintf(tag, sizeof tag, "<|%s|>", roles[i]); rid = ternaryai_infer_special_id(m, tag); }
        if (rid < 0) return TERNARYAI_INFER_E_UNSUPPORTED;
        if (n < cap) ids[n] = rid;
        n++;
        size_t len = 0; char err[160];
        if (!ternary_tokenizer_encode_ex(m->tok, (const uint8_t *)texts[i], strlen(texts[i]), 0, ids ? ids + (n < cap ? n : cap) : NULL,
                                         (size_t)(cap > n ? cap - n : 0), &len, err, sizeof err)) return TERNARYAI_INFER_E_ARG;
        n += (int)len;
    }
    if (add_assistant_prompt) { if (m->chat_asst_id < 0) return TERNARYAI_INFER_E_UNSUPPORTED; if (n < cap) ids[n] = m->chat_asst_id; n++; }
    if (n_out) *n_out = n;
    return n > cap ? TERNARYAI_INFER_E_NOMEM : TERNARYAI_INFER_OK;
}

int ternaryai_infer_chat_reply(TernaryInferSession *s, const char *const *roles, const char *const *texts, int n_msgs,
                  const TernaryInferGenParams *p, char *out, int cap, TernaryInferStopReason *why) {
    if (!s || !p || !out || cap < 1) return TERNARYAI_INFER_E_ARG;
    const TernaryInferModel *m = s->m;
    int need = 0;
    int rc = ternaryai_infer_chat_encode(m, roles, texts, n_msgs, 1, NULL, 0, &need);
    if (rc != TERNARYAI_INFER_OK && rc != TERNARYAI_INFER_E_NOMEM) return rc;
    int32_t *ids = (int32_t *)ternary_malloc64((size_t)(need + 1) * 4);
    if (!ids) return TERNARYAI_INFER_E_NOMEM;
    rc = ternaryai_infer_chat_encode(m, roles, texts, n_msgs, 1, ids, need, &need);
    if (rc != TERNARYAI_INFER_OK) { free(ids); return rc; }
    ternaryai_infer_session_reset(s);
    rc = ternaryai_infer_feed(s, ids, need);
    free(ids);
    if (rc != TERNARYAI_INFER_OK) return rc;
    TernaryInferGenParams q = *p;
    int32_t stops[64]; int ns = 0;
    for (int k = 0; k < p->n_stop_ids && ns < 63; k++) stops[ns++] = p->stop_ids[k];
    if (m->chat_end_id >= 0) stops[ns++] = m->chat_end_id;
    q.stop_ids = stops; q.n_stop_ids = ns;
    int budget = s->o.max_context - s->n; if (budget < 1) budget = 1;
    if (q.max_new_tokens > budget) q.max_new_tokens = budget;
    int32_t *gen = (int32_t *)ternary_malloc64((size_t)(q.max_new_tokens + 1) * 4);
    if (!gen) return TERNARYAI_INFER_E_NOMEM;
    int ng = ternaryai_infer_generate(s, &q, gen, q.max_new_tokens, why);
    if (ng < 0) { free(gen); return ng; }
    int len = 0;
    rc = ternaryai_infer_decode(m, gen, ng, 1, out, cap, &len);
    free(gen);
    if (rc != TERNARYAI_INFER_OK && rc != TERNARYAI_INFER_E_NOMEM) return rc;
    return len < cap ? len : cap - 1;
}

/* ---------------------------------------------------------------- conversacion configurable */
void ternaryai_infer_chat_opts_default(TernaryInferChatOpts *o) {
    if (!o) return;
    memset(o, 0, sizeof(*o));
    o->after_role = "\n"; o->after_msg = "\n"; o->stop_on_user = 1;
}

static int emit_text(const TernaryInferModel *m, const char *txt, int32_t *ids, int cap, int *n) {
    size_t len = 0; char err[160];
    if (!txt || !*txt) return TERNARYAI_INFER_OK;
    if (!ternary_tokenizer_encode_ex(m->tok, (const uint8_t *)txt, strlen(txt), 0, ids ? ids + (*n < cap ? *n : cap) : NULL,
                                     (size_t)(cap > *n ? cap - *n : 0), &len, err, sizeof err)) return TERNARYAI_INFER_E_ARG;
    *n += (int)len;
    return TERNARYAI_INFER_OK;
}

static int role_id(const TernaryInferModel *m, const char *role) {
    char tag[80];
    snprintf(tag, sizeof tag, "<|%s|>", role);
    return ternaryai_infer_special_id(m, tag);
}

/* Cada mensaje se codifica como UN solo tramo "<after_role>texto<after_msg>" entre marcadores especiales, igual que hace el cargador de
 * datasets al codificar la conversacion (BPE no es independiente por trozos: "\nHola\n" no da los mismos ids que "\n", "Hola", "\n").
 * El turno del asistente a generar es solo el marcador: el separador tras el rol forma parte de lo que el modelo aprendio a generar
 * (el cargador entrena el asistente desde el token que sigue al marcador). Con force_think se agrega tras el marcador el tramo after_role
 * y <|think|>, que es como el dataset abre el razonamiento. */
int ternaryai_infer_chat_encode_fmt(const TernaryInferModel *m, const TernaryInferChatOpts *o, const char *const *roles, const char *const *texts,
                   int n_msgs, int add_assistant_prompt, int32_t *ids, int cap, int *n_out) {
    if (!m || n_msgs < 0 || (n_msgs > 0 && (!roles || !texts)) || cap < 0 || (cap > 0 && !ids)) return TERNARYAI_INFER_E_ARG;
    if (!m->tok) return TERNARYAI_INFER_E_NOTOK;
    TernaryInferChatOpts d; ternaryai_infer_chat_opts_default(&d);
    const char *ar = (o && o->after_role) ? o->after_role : d.after_role;
    const char *am = (o && o->after_msg) ? o->after_msg : d.after_msg;
    const size_t lar = strlen(ar), lam = strlen(am);
    int n = 0, rc;
    for (int i = 0; i < n_msgs; i++) {
        const int rid = role_id(m, roles[i]);
        if (rid < 0) return TERNARYAI_INFER_E_UNSUPPORTED;
        if (n < cap) ids[n] = rid;
        n++;
        const size_t lt = strlen(texts[i]);
        char *seg = (char *)malloc(lar + lt + lam + 1);
        if (!seg) return TERNARYAI_INFER_E_NOMEM;
        memcpy(seg, ar, lar); memcpy(seg + lar, texts[i], lt); memcpy(seg + lar + lt, am, lam); seg[lar + lt + lam] = 0;
        rc = emit_text(m, seg, ids, cap, &n);
        free(seg);
        if (rc != TERNARYAI_INFER_OK) return rc;
    }
    if (add_assistant_prompt) {
        if (m->chat_asst_id < 0) return TERNARYAI_INFER_E_UNSUPPORTED;
        if (n < cap) ids[n] = m->chat_asst_id;
        n++;
        if (o && o->force_think) {
            const int op = ternaryai_infer_special_id(m, "<|think|>");
            if (op < 0) return TERNARYAI_INFER_E_UNSUPPORTED;
            if ((rc = emit_text(m, ar, ids, cap, &n)) != TERNARYAI_INFER_OK) return rc;
            if (n < cap) ids[n] = op;
            n++;
        }
    }
    if (n_out) *n_out = n;
    return n > cap ? TERNARYAI_INFER_E_NOMEM : TERNARYAI_INFER_OK;
}

int ternaryai_infer_think_split(const TernaryInferModel *m, const int32_t *ids, int n, int think_inside,
                   int32_t *answer_out, int *n_answer, int32_t *think_out, int *n_think) {
    if (!m || n < 0 || (n > 0 && !ids)) return TERNARYAI_INFER_E_ARG;
    const int op = ternaryai_infer_special_id(m, "<|think|>"), cl = ternaryai_infer_special_id(m, "<|/think|>");
    if (op < 0 || cl < 0) return TERNARYAI_INFER_E_UNSUPPORTED;
    int na = 0, nt = 0, in = think_inside ? 1 : 0;
    for (int i = 0; i < n; i++) {
        if (ids[i] == op) { in = 1; continue; }
        if (ids[i] == cl) { in = 0; continue; }
        if (in) { if (think_out) think_out[nt] = ids[i]; nt++; }
        else { if (answer_out) answer_out[na] = ids[i]; na++; }
    }
    if (n_answer) *n_answer = na;
    if (n_think) *n_think = nt;
    return TERNARYAI_INFER_OK;
}

int ternaryai_infer_chat_reply_ex(TernaryInferSession *s, const TernaryInferChatOpts *o, const char *const *roles, const char *const *texts, int n_msgs,
                  const TernaryInferGenParams *p, char *out, int cap, char *think_out, int think_cap, TernaryInferStopReason *why) {
    if (!s || !p || !out || cap < 1) return TERNARYAI_INFER_E_ARG;
    const TernaryInferModel *m = s->m;
    TernaryInferChatOpts d; ternaryai_infer_chat_opts_default(&d);
    if (!o) o = &d;
    int open_id = -1, close_id = -1;
    if (o->force_think || o->think_budget > 0 || !o->show_think || think_out) {
        open_id = ternaryai_infer_special_id(m, "<|think|>"); close_id = ternaryai_infer_special_id(m, "<|/think|>");
        if ((o->force_think || o->think_budget > 0) && (open_id < 0 || close_id < 0)) return TERNARYAI_INFER_E_UNSUPPORTED;
    }
    int need = 0;
    int rc = ternaryai_infer_chat_encode_fmt(m, o, roles, texts, n_msgs, 1, NULL, 0, &need);
    if (rc != TERNARYAI_INFER_OK && rc != TERNARYAI_INFER_E_NOMEM) return rc;
    int32_t *ids = (int32_t *)ternary_malloc64((size_t)(need + 2) * 4);
    if (!ids) return TERNARYAI_INFER_E_NOMEM;
    rc = ternaryai_infer_chat_encode_fmt(m, o, roles, texts, n_msgs, 1, ids, need, &need);
    if (rc != TERNARYAI_INFER_OK) { free(ids); return rc; }
    ternaryai_infer_session_reset(s);
    rc = ternaryai_infer_feed(s, ids, need);
    free(ids);
    if (rc != TERNARYAI_INFER_OK) return rc;
    TernaryInferGenParams q = *p;
    int32_t stops[66]; int ns = 0;
    for (int k = 0; k < p->n_stop_ids && ns < 63; k++) stops[ns++] = p->stop_ids[k];
    if (m->chat_end_id >= 0) stops[ns++] = m->chat_end_id;
    if (o->stop_on_user) { const int u = m->chat_user_id; if (u >= 0) stops[ns++] = u; }
    q.stop_ids = stops; q.n_stop_ids = ns;
    if (o->think_budget > 0) { q.think_budget = o->think_budget; q.think_open_id = open_id; q.think_close_id = close_id; q.think_inside = o->force_think; }
    int budget = s->o.max_context - s->n; if (budget < 1) budget = 1;
    if (q.max_new_tokens > budget) q.max_new_tokens = budget;
    int32_t *gen = (int32_t *)ternary_malloc64((size_t)(q.max_new_tokens + 1) * 4);
    if (!gen) return TERNARYAI_INFER_E_NOMEM;
    int ng = ternaryai_infer_generate(s, &q, gen, q.max_new_tokens, why);
    if (ng < 0) { free(gen); return ng; }
    int len = 0;
    if (open_id >= 0 && close_id >= 0 && (think_out || !o->show_think)) {
        int32_t *ans = (int32_t *)ternary_malloc64((size_t)(ng + 1) * 4), *thk = (int32_t *)ternary_malloc64((size_t)(ng + 1) * 4);
        if (!ans || !thk) { free(ans); free(thk); free(gen); return TERNARYAI_INFER_E_NOMEM; }
        int na = 0, nt = 0;
        ternaryai_infer_think_split(m, gen, ng, o->force_think, ans, &na, thk, &nt);
        if (think_out && think_cap > 0) {
            int tl = 0;
            const int rt = ternaryai_infer_decode(m, thk, nt, 1, think_out, think_cap, &tl);
            if (rt != TERNARYAI_INFER_OK && rt != TERNARYAI_INFER_E_NOMEM) { free(ans); free(thk); free(gen); return rt; }
        }
        if (o->show_think) rc = ternaryai_infer_decode(m, gen, ng, 0, out, cap, &len);
        else rc = ternaryai_infer_decode(m, ans, na, 1, out, cap, &len);
        free(ans); free(thk);
    } else {
        rc = ternaryai_infer_decode(m, gen, ng, o->show_think ? 0 : 1, out, cap, &len);
        if (think_out && think_cap > 0) think_out[0] = 0;
    }
    free(gen);
    if (rc != TERNARYAI_INFER_OK && rc != TERNARYAI_INFER_E_NOMEM) return rc;
    if (len >= cap) len = cap - 1;
    {   /* el modelo genera los separadores del turno; no son parte de la respuesta */
        const char *ar = o->after_role ? o->after_role : d.after_role, *am = o->after_msg ? o->after_msg : d.after_msg;
        const int lar = (int)strlen(ar), lam = (int)strlen(am);
        if (lam > 0 && len >= lam && !memcmp(out + len - lam, am, (size_t)lam)) len -= lam;
        if (lar > 0 && len >= lar && !memcmp(out, ar, (size_t)lar)) { memmove(out, out + lar, (size_t)(len - lar)); len -= lar; }
        out[len] = 0;
    }
    return len;
}
