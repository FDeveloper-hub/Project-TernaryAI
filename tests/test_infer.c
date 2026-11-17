/* Pruebas funcionales del motor de inferencia: test_infer carpeta_modelo (sin argumento: usa $TERNARYAI_TEST_MODEL; si no hay modelo, se salta). */
#include "../include/ternary_infer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tmp_path.h"   /* hay plataformas sin /tmp: usa $TMPDIR */
static int fails = 0, total = 0;
#define CHECK(c, msg) do { total++; if (!(c)) { fails++; printf("FALLA: %s (linea %d)\n", msg, __LINE__); } } while (0)
static TernaryInferModel *M; static TernaryInferModelInfo MI;
static void ids_fill(int32_t *a, int n, uint32_t seed) { for (int i = 0; i < n; i++) { seed = seed * 1664525u + 1013904223u; a[i] = (int)((seed >> 8) % (uint32_t)MI.vocab); } }
static int same_logits(TernaryInferSession *a, TernaryInferSession *b) {
    float *x = malloc(MI.vocab * 4), *y = malloc(MI.vocab * 4); ternaryai_infer_logits(a, x); ternaryai_infer_logits(b, y);
    int r = !memcmp(x, y, MI.vocab * 4); free(x); free(y); return r;
}
static int same_logits_m(TernaryInferSession *a, TernaryInferSession *b) { return same_logits(a, b); }
static int cb_count; static int cb(int32_t id, void *u) { (void)id; (void)u; return ++cb_count >= 5; }
static int mask_even(void *u, const TernaryInferSession *s, uint8_t *al) { (void)u; (void)s; for (int v = 0; v < MI.vocab; v++) al[v] = (v % 2 == 0); return 1; }

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : getenv("TERNARYAI_TEST_MODEL");
    if (!dir) { printf("test_infer: sin modelo (pasar la carpeta o TERNARYAI_TEST_MODEL); se salta\n"); return 0; }
    char err[256]; M = ternaryai_infer_model_load(dir, err, sizeof err);
    if (!M) { printf("no carga: %s\n", err); return 1; }
    ternaryai_infer_model_info(M, &MI);
    int32_t ids[512]; ids_fill(ids, 512, 7);
    const int C = MI.seq_len;

    /* feed por bloques == token a token; fork; reset */
    TernaryInferSession *a = ternaryai_infer_session_new(M, NULL), *b = ternaryai_infer_session_new(M, NULL);
    CHECK(ternaryai_infer_feed(a, ids, 40) == 0, "feed");
    for (int i = 0; i < 40; i++) ternaryai_infer_feed1(b, ids[i]);
    CHECK(same_logits(a, b), "feed vs feed1");
    CHECK(ternaryai_infer_session_len(a) == 40 && ternaryai_infer_session_next_position(a) == 40, "len");
    TernaryInferSession *f = ternaryai_infer_session_fork(a);
    CHECK(same_logits(a, f), "fork igual");
    ternaryai_infer_feed1(a, ids[50]); ternaryai_infer_feed1(f, ids[50]);
    CHECK(same_logits(a, f), "fork avanza igual");
    ternaryai_infer_feed1(f, ids[51]);
    CHECK(!same_logits(a, f), "fork independiente");
    ternaryai_infer_session_reset(b); ternaryai_infer_feed(b, ids, 40); ternaryai_infer_feed1(b, ids[50]);
    CHECK(same_logits(a, b), "reset reproduce");

    /* sin logits */
    TernaryInferSession *e = ternaryai_infer_session_new(M, NULL); int32_t t;
    CHECK(ternaryai_infer_argmax(e, &t) == TERNARYAI_INFER_E_STATE, "argmax sin estado");

    /* determinismo y reproducibilidad del muestreo */
    TernaryInferGenParams gp; ternaryai_infer_gen_params_default(&gp); gp.max_new_tokens = 20;
    gp.sampler.temperature = 0.9f; gp.sampler.top_k = 40; gp.sampler.top_p = 0.95f; gp.sampler.min_p = 0.01f; gp.sampler.repeat_penalty = 1.1f;
    int32_t o1[20], o2[20], o3[20]; TernaryInferStopReason w;
    TernaryInferSession *s1 = ternaryai_infer_session_new(M, NULL), *s2 = ternaryai_infer_session_new(M, NULL), *s3 = ternaryai_infer_session_new(M, NULL);
    ternaryai_infer_feed(s1, ids, 10); ternaryai_infer_feed(s2, ids, 10); ternaryai_infer_feed(s3, ids, 10);
    ternaryai_infer_session_seed(s1, 42); ternaryai_infer_session_seed(s2, 42); ternaryai_infer_session_seed(s3, 43);
    int n1 = ternaryai_infer_generate(s1, &gp, o1, 20, &w), n2 = ternaryai_infer_generate(s2, &gp, o2, 20, &w), n3 = ternaryai_infer_generate(s3, &gp, o3, 20, &w);
    CHECK(n1 == 20 && n2 == 20 && !memcmp(o1, o2, 80), "misma semilla = mismos tokens");
    CHECK(n3 == 20 && memcmp(o1, o3, 80), "otra semilla = otros tokens");

    /* voraz == argmax y == generate con temperatura 0 */
    TernaryInferGenParams gg; ternaryai_infer_gen_params_default(&gg); gg.max_new_tokens = 12;
    TernaryInferSession *g1 = ternaryai_infer_session_new(M, NULL), *g2 = ternaryai_infer_session_new(M, NULL);
    ternaryai_infer_feed(g1, ids, 10); ternaryai_infer_feed(g2, ids, 10);
    int32_t og[12], oh[12];
    ternaryai_infer_generate(g1, &gg, og, 12, &w);
    for (int i = 0; i < 12; i++) { ternaryai_infer_argmax(g2, &oh[i]); ternaryai_infer_feed1(g2, oh[i]); }
    CHECK(!memcmp(og, oh, 48), "generate voraz == argmax manual");
    CHECK(w == TERNARYAI_INFER_STOP_MAX_TOKENS, "motivo max_tokens");

    /* stop id, stop seq, include_stop, callback, mask, banned, allowed */
    TernaryInferGenParams sp = gg; int32_t stop = og[4]; sp.stop_ids = &stop; sp.n_stop_ids = 1;
    TernaryInferSession *g3 = ternaryai_infer_session_new(M, NULL); ternaryai_infer_feed(g3, ids, 10);
    int32_t os[12]; int ns = ternaryai_infer_generate(g3, &sp, os, 12, &w);
    int first = 0; while (first < 12 && og[first] != stop) first++;
    CHECK(w == TERNARYAI_INFER_STOP_TOKEN && ns == first, "stop id (excluido)");
    sp.include_stop = 1; ternaryai_infer_session_reset(g3); ternaryai_infer_feed(g3, ids, 10); ns = ternaryai_infer_generate(g3, &sp, os, 12, &w);
    CHECK(ns == first + 1 && os[ns - 1] == stop, "stop id (incluido)");
    TernaryInferGenParams qp = gg; int32_t seq2[2] = { og[2], og[3] }; int sl = 2; qp.stop_seq = seq2; qp.stop_seq_len = &sl; qp.n_stop_seq = 1;
    ternaryai_infer_session_reset(g3); ternaryai_infer_feed(g3, ids, 10); ns = ternaryai_infer_generate(g3, &qp, os, 12, &w);
    CHECK(w == TERNARYAI_INFER_STOP_SEQUENCE, "stop sequence");
    TernaryInferGenParams cp = gg; cp.on_token = cb; cb_count = 0;
    ternaryai_infer_session_reset(g3); ternaryai_infer_feed(g3, ids, 10); ns = ternaryai_infer_generate(g3, &cp, os, 12, &w);
    CHECK(ns == 5 && w == TERNARYAI_INFER_STOP_CALLBACK, "streaming + parada por callback");
    TernaryInferGenParams mp = gg; mp.mask_fn = mask_even; ternaryai_infer_session_reset(g3); ternaryai_infer_feed(g3, ids, 10);
    ns = ternaryai_infer_generate(g3, &mp, os, 12, &w); int ok = ns == 12; for (int i = 0; i < ns; i++) ok &= (os[i] % 2 == 0);
    CHECK(ok, "mascara por paso");
    TernaryInferGenParams bp = gg; int32_t ban[1] = { og[0] }; bp.sampler.banned = ban; bp.sampler.n_banned = 1;
    ternaryai_infer_session_reset(g3); ternaryai_infer_feed(g3, ids, 10); ns = ternaryai_infer_generate(g3, &bp, os, 12, &w);
    CHECK(os[0] != og[0], "token prohibido");
    float *bias = calloc(MI.vocab, 4); bias[100] = 1e6f; TernaryInferGenParams ip = gg; ip.sampler.logit_bias = bias;
    ternaryai_infer_session_reset(g3); ternaryai_infer_feed(g3, ids, 10); ns = ternaryai_infer_generate(g3, &ip, os, 12, &w);
    CHECK(os[0] == 100 && os[5] == 100, "logit_bias");

    /* batch == individual; generate_n */
    TernaryInferSession *bs[3]; int32_t *bo[3]; int caps[3] = { 12, 12, 12 }, cn[3]; TernaryInferStopReason bw[3]; int32_t buf[3][12];
    for (int i = 0; i < 3; i++) { bs[i] = ternaryai_infer_session_new(M, NULL); ternaryai_infer_feed(bs[i], ids + 3 * i, 10); bo[i] = buf[i]; }
    ternaryai_infer_generate_batch(bs, 3, &gg, bo, caps, cn, bw);
    for (int i = 0; i < 3; i++) {
        TernaryInferSession *x = ternaryai_infer_session_new(M, NULL); int32_t r[12]; ternaryai_infer_feed(x, ids + 3 * i, 10); ternaryai_infer_generate(x, &gg, r, 12, &w);
        CHECK(cn[i] == 12 && !memcmp(r, buf[i], 48), "batch == individual"); ternaryai_infer_session_free(x); ternaryai_infer_session_free(bs[i]);
    }
    int32_t cand[3][12]; int32_t *co[3] = { cand[0], cand[1], cand[2] }; int cc[3]; TernaryInferStopReason cw[3];
    ternaryai_infer_session_seed(g1, 5); TernaryInferSession *base = ternaryai_infer_session_new(M, NULL); ternaryai_infer_feed(base, ids, 10);
    ternaryai_infer_generate_n(base, 3, &gp, co, caps, cc, cw);
    CHECK(ternaryai_infer_session_len(base) == 10, "generate_n no altera la sesion base");

    /* score / perplexity */
    TernaryInferSession *sc = ternaryai_infer_session_new(M, NULL); float lp[20]; double ppl;
    int ns2 = ternaryai_infer_score(sc, ids, 20, lp, &ppl);
    CHECK(ns2 == 19 && isnan(lp[0]) && ppl > 1.0 && lp[5] <= 0.0f, "score/perplexity");
    int32_t tid[5]; float tp[5]; CHECK(ternaryai_infer_top_probs(sc, 5, tid, tp) == 5 && tp[0] >= tp[1], "top_probs");
    int32_t am; ternaryai_infer_argmax(sc, &am); CHECK(am == tid[0], "top_probs[0] == argmax");

    /* politicas de contexto */
    TernaryInferSessionOpts o; ternaryai_infer_session_opts_default(M, &o); o.ctx_policy = TERNARYAI_INFER_CTX_ERROR;
    TernaryInferSession *pe = ternaryai_infer_session_new(M, &o);
    CHECK(ternaryai_infer_feed(pe, ids, C) == 0, "llenar contexto");
    CHECK(ternaryai_infer_feed1(pe, ids[C]) == TERNARYAI_INFER_E_CTX_FULL, "ERROR al llenarse");
    o.ctx_policy = TERNARYAI_INFER_CTX_TRUNCATE; o.keep_tokens = 32;
    TernaryInferSession *pt = ternaryai_infer_session_new(M, &o); ternaryai_infer_feed(pt, ids, C); CHECK(ternaryai_infer_feed1(pt, ids[C]) == 0, "TRUNCATE sigue");
    TernaryInferSession *ref = ternaryai_infer_session_new(M, NULL); ternaryai_infer_feed(ref, ids + C - 32, 32); ternaryai_infer_feed1(ref, ids[C]);
    CHECK(ternaryai_infer_session_len(pt) == 33 && same_logits(pt, ref), "TRUNCATE == reprocesar los ultimos keep");
    o.ctx_policy = TERNARYAI_INFER_CTX_SLIDE; o.sinks = 4; o.max_context = 64; o.attn_window = 0;
    TernaryInferSession *pl = ternaryai_infer_session_new(M, &o); int rc = ternaryai_infer_feed(pl, ids, 300);
    CHECK(rc == 0 && ternaryai_infer_session_len(pl) <= 64 && ternaryai_infer_session_next_position(pl) == 300, "SLIDE con sinks");
    float *lg = malloc(MI.vocab * 4); ternaryai_infer_logits(pl, lg); int fin = 1; for (int v = 0; v < MI.vocab; v++) fin &= isfinite(lg[v]); CHECK(fin, "SLIDE logits finitos");
    /* ventana vs contexto completo: con ventana >= largo es identica */
    TernaryInferSessionOpts ow; ternaryai_infer_session_opts_default(M, &ow); ow.attn_window = 200; ow.sinks = 2;
    TernaryInferSession *pw = ternaryai_infer_session_new(M, &ow), *pf = ternaryai_infer_session_new(M, NULL); ow.attn_window = 0; ow.sinks = 0;
    ternaryai_infer_feed(pw, ids, 60); TernaryInferSessionOpts ofull; ternaryai_infer_session_opts_default(M, &ofull); ofull.attn_window = 0; ofull.sinks = 0; TernaryInferSession *pf2 = ternaryai_infer_session_new(M, &ofull); ternaryai_infer_feed(pf2, ids, 60);
    CHECK(same_logits(pw, pf2), "ventana grande == atencion completa");
    TernaryInferSessionOpts on; ternaryai_infer_session_opts_default(M, &on); on.attn_window = 8; on.sinks = 2; TernaryInferSession *pn = ternaryai_infer_session_new(M, &on); ternaryai_infer_feed(pn, ids, 60);
    CHECK(!same_logits(pn, pf2), "ventana pequena cambia el resultado");
    (void)pf;

    /* atencion registrada */
    TernaryInferSessionOpts orc; ternaryai_infer_session_opts_default(M, &orc); orc.record_attn = 1; TernaryInferSession *pr = ternaryai_infer_session_new(M, &orc); ternaryai_infer_feed(pr, ids, 30);
    float ap[256]; int na; CHECK(ternaryai_infer_attn_probs(pr, 0, 0, ap, 256, &na) == 0 && na > 0, "attn_probs"); double ssum = 0; for (int i = 0; i < na; i++) ssum += ap[i];
    CHECK(fabs(ssum - 1.0) < 1e-3, "probabilidades de atencion suman 1");

    /* guardar / cargar cache */
    const char *path = tmp_path("ternaryai_infer_test.cache");
    CHECK(ternaryai_infer_session_save(a, path) == 0, "save");
    TernaryInferSession *ld = ternaryai_infer_session_load(M, path, NULL, err, sizeof err);
    CHECK(ld && same_logits(a, ld) && ternaryai_infer_session_len(ld) == ternaryai_infer_session_len(a), "load restaura");
    if (ld) { ternaryai_infer_feed1(ld, ids[70]); ternaryai_infer_feed1(a, ids[70]); CHECK(same_logits(a, ld), "tras cargar sigue identico"); }
    CHECK(ternaryai_infer_session_load(M, tmp_path("no_existe.cache"), NULL, err, sizeof err) == NULL, "load de archivo inexistente falla");

    /* escalar == AVX-512 */
    if (ternaryai_infer_cpu_has_avx512()) {
        TernaryInferSession *x1 = ternaryai_infer_session_new(M, NULL); ternaryai_infer_feed(x1, ids, 100); float *l1 = malloc(MI.vocab * 4); ternaryai_infer_logits(x1, l1);
        ternaryai_infer_set_backend(TERNARYAI_INFER_BACKEND_SCALAR);
        TernaryInferSession *x2 = ternaryai_infer_session_new(M, NULL); ternaryai_infer_feed(x2, ids, 100); float *l2 = malloc(MI.vocab * 4); ternaryai_infer_logits(x2, l2);
        CHECK(!memcmp(l1, l2, MI.vocab * 4), "escalar bit-identico a AVX-512");
        ternaryai_infer_set_backend(TERNARYAI_INFER_BACKEND_AVX512);
    }

    /* escalar == NEON (ARM): mismos logits y misma generacion */
    if (ternaryai_infer_cpu_has_neon()) {
        const int act1 = ternaryai_infer_get_backend();
        ternaryai_infer_set_backend(TERNARYAI_INFER_BACKEND_NEON);
        CHECK(ternaryai_infer_get_backend() == TERNARYAI_INFER_BACKEND_NEON, "backend NEON activo");
        TernaryInferSession *x1 = ternaryai_infer_session_new(M, NULL); ternaryai_infer_feed(x1, ids, 100); float *l1 = malloc(MI.vocab * 4); ternaryai_infer_logits(x1, l1);
        TernaryInferGenParams gp; ternaryai_infer_gen_params_default(&gp); gp.max_new_tokens = 24; int32_t gn[24], gs[24]; TernaryInferStopReason wn, ws;
        int nn = ternaryai_infer_generate(x1, &gp, gn, 24, &wn);
        ternaryai_infer_set_backend(TERNARYAI_INFER_BACKEND_SCALAR);
        TernaryInferSession *x2 = ternaryai_infer_session_new(M, NULL); ternaryai_infer_feed(x2, ids, 100); float *l2 = malloc(MI.vocab * 4); ternaryai_infer_logits(x2, l2);
        int ns = ternaryai_infer_generate(x2, &gp, gs, 24, &ws);
        CHECK(!memcmp(l1, l2, MI.vocab * 4), "escalar bit-identico a NEON");
        CHECK(nn == ns && nn > 0 && !memcmp(gn, gs, (size_t)nn * 4), "NEON: misma generacion que el escalar");
        free(l1); free(l2); ternaryai_infer_session_free(x1); ternaryai_infer_session_free(x2);
        ternaryai_infer_set_backend((TernaryInferBackend)act1);
    }

    /* modelo autocontenido (.ternam): mismas salidas que la carpeta, con los dos ordenes de pesos y los dos nucleos */
    {
        const char *tp[3] = { tmp_path("ternaryai_t_row.ternam"), tmp_path("ternaryai_t_avx.ternam"), tmp_path("ternaryai_t_neon.ternam") };
        const int lay[3] = { TERNARYAI_INFER_LAYOUT_ROW, TERNARYAI_INFER_LAYOUT_AVX, TERNARYAI_INFER_LAYOUT_NEON };
        const int act0 = ternaryai_infer_get_backend();
        for (int li = 0; li < 3; li++) {
            CHECK(ternaryai_infer_export_ternam(dir, tp[li], lay[li], err, sizeof err) == 0, "export .ternam");
            for (int bk = 0; bk < 3; bk++) {
                if (bk == 1 && !ternaryai_infer_cpu_has_avx512()) continue;
                if (bk == 2 && !ternaryai_infer_cpu_has_neon()) continue;
                ternaryai_infer_set_backend(bk == 1 ? TERNARYAI_INFER_BACKEND_AVX512 : bk == 2 ? TERNARYAI_INFER_BACKEND_NEON : TERNARYAI_INFER_BACKEND_SCALAR);
                TernaryInferModel *T = ternaryai_infer_model_load(tp[li], err, sizeof err);
                CHECK(T != NULL, "carga .ternam");
                if (!T) continue;
                TernaryInferModelInfo TI; ternaryai_infer_model_info(T, &TI);
                CHECK(TI.fingerprint == MI.fingerprint && TI.vocab == MI.vocab && TI.has_tokenizer == MI.has_tokenizer && TI.weight_bytes == MI.weight_bytes, ".ternam: mismos datos del modelo");
                TernaryInferSession *sa = ternaryai_infer_session_new(M, NULL), *sb = ternaryai_infer_session_new(T, NULL);
                ternaryai_infer_feed(sa, ids, 60); ternaryai_infer_feed(sb, ids, 60);
                float *la = malloc(MI.vocab * 4), *lb = malloc(MI.vocab * 4); ternaryai_infer_logits(sa, la); ternaryai_infer_logits(sb, lb);
                CHECK(!memcmp(la, lb, MI.vocab * 4), ".ternam: logits identicos a la carpeta");
                TernaryInferGenParams g2; ternaryai_infer_gen_params_default(&g2); g2.max_new_tokens = 24;
                int32_t ga[24], gb[24]; TernaryInferStopReason wa, wb;
                int na2 = ternaryai_infer_generate(sa, &g2, ga, 24, &wa), nb2 = ternaryai_infer_generate(sb, &g2, gb, 24, &wb);
                CHECK(na2 == nb2 && na2 > 0 && !memcmp(ga, gb, (size_t)na2 * 4), ".ternam: misma generacion");
                if (li == 0 && bk == 0) {                           /* una cache guardada con el modelo carpeta carga en el .ternam (misma huella) */
                    CHECK(ternaryai_infer_session_save(sa, path) == 0, "save cache");
                    TernaryInferSession *lc = ternaryai_infer_session_load(T, path, NULL, err, sizeof err);
                    CHECK(lc && same_logits_m(sa, lc), "cache de la carpeta carga en el .ternam");
                    if (lc) ternaryai_infer_session_free(lc);
                }
                free(la); free(lb); ternaryai_infer_session_free(sa); ternaryai_infer_session_free(sb); ternaryai_infer_model_free(T);
            }
        }
        /* re-exportar un .ternam a otro orden; archivos danados se rechazan con un mensaje */
        CHECK(ternaryai_infer_export_ternam(tp[0], tmp_path("ternaryai_t_re.ternam"), TERNARYAI_INFER_LAYOUT_AVX, err, sizeof err) == 0, "re-exportar .ternam");
        { TernaryInferModel *T = ternaryai_infer_model_load(tmp_path("ternaryai_t_re.ternam"), err, sizeof err); CHECK(T != NULL, "carga el re-exportado"); if (T) ternaryai_infer_model_free(T); }
        FILE *fz = fopen(tp[0], "rb"); long fl = 0; unsigned char *buf = NULL;
        if (fz) { fseek(fz, 0, SEEK_END); fl = ftell(fz); fseek(fz, 0, SEEK_SET); buf = malloc((size_t)fl); if (fread(buf, 1, (size_t)fl, fz) != (size_t)fl) fl = 0; fclose(fz); }
        CHECK(buf && fl > 200, "leer el .ternam");
        if (buf && fl > 200) {
            const char *bad = tmp_path("ternaryai_t_bad.ternam"); TernaryInferModel *T;
            FILE *fo = fopen(bad, "wb"); fwrite(buf, 1, (size_t)fl - 1, fo); fclose(fo);                 /* truncado */
            T = ternaryai_infer_model_load(bad, err, sizeof err); CHECK(T == NULL && err[0], "rechaza un .ternam truncado"); if (T) ternaryai_infer_model_free(T);
            buf[fl / 2] ^= 0x5A; fo = fopen(bad, "wb"); fwrite(buf, 1, (size_t)fl, fo); fclose(fo); buf[fl / 2] ^= 0x5A;   /* un byte cambiado */
            T = ternaryai_infer_model_load(bad, err, sizeof err); CHECK(T == NULL && err[0], "rechaza un .ternam con un byte cambiado"); if (T) ternaryai_infer_model_free(T);
            buf[0] = 'X'; fo = fopen(bad, "wb"); fwrite(buf, 1, (size_t)fl, fo); fclose(fo); buf[0] = 'T';  /* cabecera invalida */
            T = ternaryai_infer_model_load(bad, err, sizeof err); CHECK(T == NULL && err[0], "rechaza una cabecera invalida"); if (T) ternaryai_infer_model_free(T);
            fo = fopen(bad, "wb"); fclose(fo);                                                           /* vacio */
            T = ternaryai_infer_model_load(bad, err, sizeof err); CHECK(T == NULL && err[0], "rechaza un archivo vacio"); if (T) ternaryai_infer_model_free(T);
            remove(bad);
        }
        free(buf);
        remove(tp[0]); remove(tp[1]); remove(tp[2]); remove(tmp_path("ternaryai_t_re.ternam"));
        ternaryai_infer_set_backend((TernaryInferBackend)act0);
    }
    printf("%d/%d pruebas correctas\n", total - fails, total);
    return fails ? 1 : 0;
}
