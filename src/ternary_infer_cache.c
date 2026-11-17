/* Guardado / carga de la cache de una sesion (p. ej. un prompt de sistema ya procesado). */
#include "ternary_mem.h"
#include "ternary_infer_internal.h"
#include <stdio.h>

#define MAGIC "TICACHE2"

static int wr(FILE *f, const void *p, size_t n) { return n == 0 || fwrite(p, 1, n, f) == n; }
static int rd(FILE *f, void *p, size_t n) { return n == 0 || fread(p, 1, n, f) == n; }

int ternaryai_infer_session_save(const TernaryInferSession *s, const char *path) {
    if (!s || !path) return TERNARYAI_INFER_E_ARG;
    FILE *f = fopen(path, "wb");
    if (!f) return TERNARYAI_INFER_E_IO;
    const TernaryInferModel *m = s->m;
    int32_t hdr[6] = { s->o.max_context, s->n, s->next_pos, s->have_head, s->hist_n, m->vocab };
    int ok = wr(f, MAGIC, 8) && wr(f, &m->fingerprint, 8) && wr(f, hdr, sizeof hdr) && wr(f, &s->head_k, 4) && wr(f, &s->rng, 8)
          && wr(f, s->hist, (size_t)s->hist_n * 4) && wr(f, s->z, (size_t)m->vocab * 4);
    for (int i = 0; i < m->n_top && ok; i++) {
        if (m->top[i].kind != TERNARYAI_INFER_TOP_RESIDUAL) continue;
        for (int j = 0; j < m->top[i].n_sub && ok; j++) {
            const TernaryInferSub *sb = &m->top[i].sub[j];
            if (sb->kind != TERNARYAI_INFER_SUB_GQA) continue;
            const TernaryInferLayerCache *c = &s->lc[sb->gqa_idx];
            for (int r = 0; r < sb->gqa.kvw && ok; r++) {          /* K como int32 (viene de kT, o de k16 si kT no existe) */
                if (c->kT) ok = wr(f, c->kT + (size_t)r * c->tp, (size_t)s->n * 4);
                else for (int p = 0; p < s->n && ok; p++) {
                    const int32_t v = (c->k16kind == 1) ? c->k16[((size_t)(r >> 1) * c->tp + (size_t)p) * 2 + (size_t)(r & 1)] : c->k16[(size_t)r * c->tp + (size_t)p];
                    ok = wr(f, &v, 4);
                }
            }
            for (int h = 0; h < sb->gqa.n_kv && ok; h++)
                ok = wr(f, c->v8 + (size_t)h * s->o.max_context * sb->gqa.hd, (size_t)s->n * sb->gqa.hd);
            ok = ok && wr(f, c->vs, (size_t)s->n * 4);
            ok = ok && wr(f, c->prev, (size_t)sb->gqa.kvw * 4);
        }
    }
    if (fclose(f) != 0) ok = 0;
    return ok ? TERNARYAI_INFER_OK : TERNARYAI_INFER_E_IO;
}

static void seterr(char *e, size_t c, const char *msg) { if (e && c) snprintf(e, c, "%s", msg); }

TernaryInferSession *ternaryai_infer_session_load(const TernaryInferModel *m, const char *path, const TernaryInferSessionOpts *opts, char *err, size_t errcap) {
    if (!m || !path) { seterr(err, errcap, "argumento invalido"); return NULL; }
    FILE *f = fopen(path, "rb");
    if (!f) { seterr(err, errcap, "no se pudo abrir el archivo"); return NULL; }
    char mg[8]; uint64_t fp; int32_t hdr[6]; float hk; uint64_t rng;
    if (!rd(f, mg, 8) || memcmp(mg, MAGIC, 8) || !rd(f, &fp, 8) || !rd(f, hdr, sizeof hdr) || !rd(f, &hk, 4) || !rd(f, &rng, 8)) {
        fclose(f); seterr(err, errcap, "archivo de cache invalido"); return NULL;
    }
    if (fp != m->fingerprint || hdr[5] != m->vocab) { fclose(f); seterr(err, errcap, "la cache pertenece a otro modelo"); return NULL; }
    TernaryInferSessionOpts o;
    if (opts) o = *opts; else { ternaryai_infer_session_opts_default(m, &o); o.max_context = hdr[0]; }
    const int n = hdr[1], oldC = hdr[0];
    if (n > o.max_context) { fclose(f); seterr(err, errcap, "la cache no cabe en max_context"); return NULL; }
    TernaryInferSession *s = ternaryai_infer_session_new(m, &o);
    if (!s) { fclose(f); seterr(err, errcap, "memoria insuficiente"); return NULL; }
    if (n > s->o.max_context) { ternaryai_infer_session_free(s); fclose(f); seterr(err, errcap, "la cache no cabe en max_context"); return NULL; }
    s->n = n; s->next_pos = hdr[2]; s->have_head = hdr[3]; s->head_k = hk; s->rng = rng;
    int ok = 1;
    if (hdr[4] > s->hist_cap) {
        int32_t *h = (int32_t *)realloc(s->hist, (size_t)hdr[4] * 4);
        if (!h) ok = 0; else { s->hist = h; s->hist_cap = hdr[4]; }
    }
    ok = ok && rd(f, s->hist, (size_t)hdr[4] * 4);
    s->hist_n = hdr[4];
    ok = ok && rd(f, s->z, (size_t)m->vocab * 4);
    for (int i = 0; i < m->n_top && ok; i++) {
        if (m->top[i].kind != TERNARYAI_INFER_TOP_RESIDUAL) continue;
        for (int j = 0; j < m->top[i].n_sub && ok; j++) {
            const TernaryInferSub *sb = &m->top[i].sub[j];
            if (sb->kind != TERNARYAI_INFER_SUB_GQA) continue;
            TernaryInferLayerCache *c = &s->lc[sb->gqa_idx];
            int32_t *tmp = (int32_t *)ternary_malloc64((size_t)(sb->gqa.kvw * (n ? n : 1)) * 4);
            if (!tmp) { ok = 0; break; }
            ok = rd(f, tmp, (size_t)sb->gqa.kvw * (size_t)n * 4);
            for (int h = 0; h < sb->gqa.n_kv && ok; h++)
                ok = rd(f, c->v8 + (size_t)h * s->o.max_context * sb->gqa.hd, (size_t)n * sb->gqa.hd);
            ok = ok && rd(f, c->vs, (size_t)n * 4);
            ok = ok && rd(f, c->prev, (size_t)sb->gqa.kvw * 4);
            if (ok) {                                           /* K: int16 si cabe (lo normal); si no, tambien int32 */
                int32_t mx = 0;
                for (size_t k = 0; k < (size_t)sb->gqa.kvw * (size_t)n; k++) { const int32_t a = tmp[k] < 0 ? -tmp[k] : tmp[k]; if (a > mx) mx = a; }
                const int fits = (mx <= TERNARYAI_INFER_K16_MAX);
                if ((!fits || !c->k16) && !ternaryai_infer_kt_promote(c, sb->gqa.kvw)) ok = 0;
                for (int r = 0; r < sb->gqa.kvw && ok; r++)
                    for (int p = 0; p < n; p++) {
                        const int32_t v = tmp[(size_t)r * (size_t)n + (size_t)p];
                        if (c->kT) c->kT[(size_t)r * c->tp + (size_t)p] = v;
                        if (c->k16 && fits) {
                            if (c->k16kind == 1) c->k16[((size_t)(r >> 1) * c->tp + (size_t)p) * 2 + (size_t)(r & 1)] = (int16_t)v;
                            else c->k16[(size_t)r * c->tp + (size_t)p] = (int16_t)v;
                        }
                    }
                c->kmax = mx; c->k16ok = (c->k16 != NULL && fits);
            }
            free(tmp);
        }
    }
    (void)oldC;
    fclose(f);
    if (!ok) { ternaryai_infer_session_free(s); seterr(err, errcap, "archivo de cache truncado"); return NULL; }
    return s;
}
