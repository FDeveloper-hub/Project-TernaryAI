/* bench modelo [--noavx] [--secs S]: tres metricas (decodificacion, con prefill, lote). */
#include "../include/ternary_infer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint32_t lcg = 12345;
static int rnd(int v) { lcg = lcg * 1664525u + 1013904223u; return (int)((lcg >> 8) % (uint32_t)v); }
int main(int argc, char **argv) {
    if (argc < 2) return 2;
    double secs = 1.0; int noavx = 0, prof = 0;
    for (int i = 2; i < argc; i++) { if (!strcmp(argv[i], "--noavx")) noavx = 1; else if (!strcmp(argv[i], "--secs")) secs = atof(argv[++i]); else if (!strcmp(argv[i], "--prof")) prof = 1; }
    if (noavx) ternaryai_infer_set_backend(TERNARYAI_INFER_BACKEND_SCALAR);
    char err[256]; TernaryInferModel *m = ternaryai_infer_model_load(argv[1], err, sizeof err);
    if (!m) { fprintf(stderr, "error: %s\n", err); return 1; }
    TernaryInferModelInfo mi; ternaryai_infer_model_info(m, &mi);
    printf("backend=%s vocab=%d dim=%d seq=%d pesos=%.1f KB\n", ternaryai_infer_backend_name(), mi.vocab, mi.dim, mi.seq_len, mi.weight_bytes / 1024.0);
    int32_t *ids = (int32_t *)malloc((size_t)mi.seq_len * 4);
    for (int i = 0; i < mi.seq_len; i++) ids[i] = rnd(mi.vocab);
    const int ctxs[] = { 16, 64, mi.seq_len / 2 };
    for (int ci = 0; ci < 3; ci++) {
        const int P = ctxs[ci], N = mi.seq_len - P - 1 > 32 ? 32 : mi.seq_len - P - 1;
        if (N < 1) continue;
        TernaryInferSession *s = ternaryai_infer_session_new(m, NULL);
        /* 1. decodificacion pura: se repite prompt+N pasos; solo se mide la parte de pasos */
        double bdec = 0, ball = 0, bpre = 0;                /* mejor de 8 ventanas: el entorno es ruidoso */
        for (int w = 0; w < 8; w++) {
            uint64_t tdec = 0, tall = 0; long ntok = 0; uint64_t t0 = ternaryai_infer_now_ns();
            while (ternaryai_infer_now_ns() - t0 < (uint64_t)(secs * 1e9 / 4)) {
                ternaryai_infer_session_reset(s);
                uint64_t a = ternaryai_infer_now_ns(); ternaryai_infer_feed(s, ids, P); uint64_t b = ternaryai_infer_now_ns();
                for (int k = 0; k < N; k++) { int32_t t; ternaryai_infer_argmax(s, &t); ternaryai_infer_feed1(s, t); }
                uint64_t c = ternaryai_infer_now_ns();
                tdec += c - b; tall += c - a; ntok += N;
            }
            const double d = ntok / (tdec / 1e9), al = ntok / (tall / 1e9), pr = (double)(ntok / N) * P / ((tall - tdec) / 1e9);
            if (d > bdec) bdec = d; if (al > ball) ball = al; if (pr > bpre) bpre = pr;
        }
        printf("ctx %4d  decodificacion: %9.0f tok/s | con prefill(%d): %9.0f tok/s | prefill: %9.0f tok/s\n", P, bdec, P, ball, bpre);
        ternaryai_infer_session_free(s);
    }
    for (int B = 8; B <= 64; B *= 8) {
        const int P = 32, N = 32;
        TernaryInferSession **ss = (TernaryInferSession **)malloc((size_t)B * sizeof(TernaryInferSession *));
        int32_t **outs = (int32_t **)malloc((size_t)B * sizeof(int32_t *)); int *caps = (int *)malloc((size_t)B * 4), *cnt = (int *)malloc((size_t)B * 4);
        TernaryInferStopReason *wh = (TernaryInferStopReason *)malloc((size_t)B * sizeof(TernaryInferStopReason));
        for (int i = 0; i < B; i++) { ss[i] = ternaryai_infer_session_new(m, NULL); outs[i] = (int32_t *)malloc(N * 4); caps[i] = N; }
        TernaryInferGenParams gp; ternaryai_infer_gen_params_default(&gp); gp.max_new_tokens = N;
        double bl = 0;
        for (int w = 0; w < 8; w++) {
            long ntok = 0; uint64_t t0 = ternaryai_infer_now_ns(), tt = 0;
            while (ternaryai_infer_now_ns() - t0 < (uint64_t)(secs * 1e9 / 4)) {
                for (int i = 0; i < B; i++) { ternaryai_infer_session_reset(ss[i]); ternaryai_infer_feed(ss[i], ids + i, P); }
                uint64_t a = ternaryai_infer_now_ns(); ternaryai_infer_generate_batch(ss, B, &gp, outs, caps, cnt, wh); tt += ternaryai_infer_now_ns() - a;
                for (int i = 0; i < B; i++) ntok += cnt[i];
            }
            if (ntok / (tt / 1e9) > bl) bl = ntok / (tt / 1e9);
        }
        printf("lote %3d (prompt %d, +%d): %9.0f tok/s\n", B, P, N, bl);
        for (int i = 0; i < B; i++) { ternaryai_infer_session_free(ss[i]); free(outs[i]); }
        free(ss); free(outs); free(caps); free(cnt); free(wh);
    }
    if (prof) {
        ternaryai_infer_prof_enable(1); ternaryai_infer_prof_reset();
        TernaryInferSession *s = ternaryai_infer_session_new(m, NULL); ternaryai_infer_feed(s, ids, 64);
        for (int k = 0; k < 2000; k++) { int32_t t; ternaryai_infer_argmax(s, &t); if (ternaryai_infer_session_len(s) > mi.seq_len - 2) { ternaryai_infer_session_reset(s); ternaryai_infer_feed(s, ids, 64); } ternaryai_infer_feed1(s, t); }
        char buf[8192]; ternaryai_infer_prof_report(buf, sizeof buf); puts(buf);
    }
    return 0;
}
