/* Nucleos NEON de atencion, softmax y RoPE contra las referencias ESCALARES de la libreria (transcritas de ternary_gqa.c y
 * ternary_softmax.c): mismos bits (memcmp). Formas al azar con colas parciales. Sin NEON en la compilacion o en la CPU se omite. */
#include "ternary_neon.h"
#include "ternary_fexp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 88172645463325252ull;
static uint64_t rnd64(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static float rndf(void) { int k = (int)(rnd64() % 7); if (k == 0) return 0.0f; if (k == 1) return -0.0f; return (float)((int)(rnd64() % 2001) - 1000) * 0.0137f; }
static int32_t rndi(int32_t lim) { return (int32_t)(rnd64() % (2 * (uint64_t)lim + 1)) - lim; }
static size_t rup16(size_t n) { return (n + 15) & ~(size_t)15; }
static int fails = 0;
#define FAIL(...) do { if (fails < 12) { printf("FALLA: "); printf(__VA_ARGS__); printf("\n"); } fails++; } while (0)
static void *A(size_t n) { void *p = calloc(n ? n : 1, 1); if (!p) abort(); return p; }
static float *FA(size_t n) { return (float *)A(n * 4); }
static int32_t *IA(size_t n) { return (int32_t *)A(n * 4); }
static void fillf(float *p, size_t n) { for (size_t i = 0; i < n; i++) p[i] = rndf(); }
static void filli(int32_t *p, size_t n, int32_t lim) { for (size_t i = 0; i < n; i++) p[i] = rndi(lim); }
static void cmpf(const char *nm, const float *a, const float *b, size_t n, const char *ctx) {
    for (size_t i = 0; i < n; i++) if (memcmp(&a[i], &b[i], 4)) { FAIL("%s [%zu] %a vs %a (%s)", nm, i, a[i], b[i], ctx); return; }
}
static void cmpi(const char *nm, const int32_t *a, const int32_t *b, size_t n, const char *ctx) {
    for (size_t i = 0; i < n; i++) if (a[i] != b[i]) { FAIL("%s [%zu] %d vs %d (%s)", nm, i, a[i], b[i], ctx); return; }
}

static const size_t NS[] = {1, 2, 3, 4, 5, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 100, 129};
static const size_t HS[] = {1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 17, 24, 31, 32, 33, 48, 64, 100};
static const size_t GS_[] = {1, 2, 3, 4, 5, 8};
#define NN (sizeof(NS) / sizeof(NS[0]))
#define NH (sizeof(HS) / sizeof(HS[0]))
#define NG (sizeof(GS_) / sizeof(GS_[0]))

static float *make_tab(uint32_t *K0) {
    const uint32_t len = 40 + (uint32_t)(rnd64() % 160);
    float *t = FA(len);
    const float c = 0.02f + 0.001f * (float)(rnd64() % 50);
    for (uint32_t k = 0; k < len; k++) t[k] = expf(-c * (float)k);
    *K0 = len - 1;
    return t;
}

/* ---------------------------------------------------------------------------------------------------------------- */
static void t_scores_i32(void) {
    for (int it = 0; it < 400; it++) {
        const size_t H = HS[rnd64() % NH], cnt = (it % 7 == 0) ? 0 : 1 + rnd64() % 70, ks = H + rnd64() % 9;
        const int shift = 16 + (int)(rnd64() % 6);
        const int32_t lim = 1 + (int32_t)(rnd64() % 3000), big = (it % 3 == 0) ? (1 << 28) : (1 << 14);
        int32_t *a = IA(H), *kb = IA(ks * (cnt + 1) + 1), *o = IA(cnt + 4), *r = IA(cnt + 4);
        filli(a, H, big); filli(kb, ks * (cnt + 1), big);
        for (size_t i = 0; i < cnt + 4; i++) o[i] = r[i] = 0x5a5a5a5a;
        ternary_neon_scores_i32(a, kb, ks, H, cnt, shift, lim, o);
        for (size_t j = 0; j < cnt; j++) {
            int64_t dot = 0;
            for (size_t d = 0; d < H; d++) dot += (int64_t)a[d] * (int64_t)kb[j * ks + d];
            dot >>= shift; if (dot > lim) dot = lim; else if (dot < -lim) dot = -lim;
            r[j] = (int32_t)dot;
        }
        char ctx[96]; snprintf(ctx, sizeof ctx, "H=%zu cnt=%zu ks=%zu", H, cnt, ks);
        cmpi("scores_i32", o, r, cnt + 4, ctx);
        free(a); free(kb); free(o); free(r);
    }
}

static void t_qk_grad_outer(void) {
    for (int it = 0; it < 400; it++) {
        const size_t H = HS[rnd64() % NH], cnt = 1 + rnd64() % 40, ks = H + rnd64() % 9;
        char ctx[96]; snprintf(ctx, sizeof ctx, "H=%zu cnt=%zu ks=%zu", H, cnt, ks);
        int32_t *qv = IA(H), *kb = IA(ks * cnt + 1);
        filli(qv, H, 1 << 18); filli(kb, ks * cnt, 1 << 18);
        float *ds = FA(cnt), *gq1 = FA(H), *gq2 = FA(H), *gk1 = FA(ks * cnt + 1), *gk2 = FA(ks * cnt + 1);
        fillf(ds, cnt); fillf(gq1, H); memcpy(gq2, gq1, H * 4); fillf(gk1, ks * cnt); memcpy(gk2, gk1, ks * cnt * 4);
        ternary_neon_qk_grad(gq1, qv, kb, gk1, ks, H, cnt, ds);
        for (size_t j = 0; j < cnt; j++)
            for (size_t d = 0; d < H; d++) { gq2[d] += ds[j] * (float)kb[j * ks + d]; gk2[j * ks + d] += ds[j] * (float)qv[d]; }
        cmpf("qk_grad gq", gq1, gq2, H, ctx); cmpf("qk_grad gk", gk1, gk2, ks * cnt + 1, ctx);
        float *p = FA(cnt), *ga = FA(H), *d1 = FA(ks * cnt + 1), *d2 = FA(ks * cnt + 1);
        fillf(p, cnt); fillf(ga, H); fillf(d1, ks * cnt); memcpy(d2, d1, ks * cnt * 4);
        ternary_neon_outer_add(d1, ks, H, cnt, p, ga);
        for (size_t j = 0; j < cnt; j++) for (size_t d = 0; d < H; d++) d2[j * ks + d] += p[j] * ga[d];
        cmpf("outer_add", d1, d2, ks * cnt + 1, ctx);
        free(qv); free(kb); free(ds); free(gq1); free(gq2); free(gk1); free(gk2); free(p); free(ga); free(d1); free(d2);
    }
}

/* ---- atencion flash ------------------------------------------------------------------------------------------- */
typedef struct {
    size_t G, H, n, ld, scs;
    int32_t *kT, **qrow, *qbuf, *m, *sc;
    int shift; int32_t lim;
} AS;

static void t_attn_scores(void) {
    for (int it = 0; it < 300; it++) {
        const size_t G = GS_[rnd64() % NG], H = HS[rnd64() % NH], n = NS[rnd64() % NN], ld = rup16(n) + 16 * (rnd64() % 2), scs = rup16(n) + 16;
        const int shift = 16 + (int)(rnd64() % 6); const int32_t lim = 1 + (int32_t)(rnd64() % 3000);
        const int32_t big = (it % 3 == 0) ? (1 << 28) : (1 << 14);
        int32_t *kT = IA(H * ld), *qb = IA(G * H), *sc = IA(G * scs), *m = IA(3 * G), *rsc = IA(G * scs), *rm = IA(G);
        const int32_t **q = (const int32_t **)A(G * sizeof(int32_t *));
        filli(kT, H * ld, big); filli(qb, G * H, big);
        for (size_t b = 0; b < G; b++) q[b] = qb + b * H;
        for (size_t i = 0; i < G * scs; i++) sc[i] = rsc[i] = 0x5a5a5a5a;
        ternary_neon_attn_scores(q, G, kT, ld, H, n, 16 + shift - 16, lim, sc, scs, m);
        for (size_t b = 0; b < G; b++) {
            int32_t mx = 0;
            for (size_t j = 0; j < n; j++) {
                int64_t dot = 0;
                for (size_t d = 0; d < H; d++) dot += (int64_t)q[b][d] * (int64_t)kT[d * ld + j];
                dot >>= shift; if (dot > lim) dot = lim; else if (dot < -lim) dot = -lim;
                rsc[b * scs + j] = (int32_t)dot; if (j == 0 || (int32_t)dot > mx) mx = (int32_t)dot;
            }
            rm[b] = mx;
        }
        char ctx[96]; snprintf(ctx, sizeof ctx, "G=%zu H=%zu n=%zu ld=%zu", G, H, n, ld);
        for (size_t b = 0; b < G; b++) cmpi("attn_scores sc", sc + b * scs, rsc + b * scs, n, ctx);
        cmpi("attn_scores m", m, rm, G, ctx);
        free(kT); free(qb); free(sc); free(m); free(rsc); free(rm); free(q);
    }
}

/* e = tab[min(m - sc, K0)] y suma por 16 carriles; part[lane(t)] */
static void ref_pv(size_t G, const int32_t *sc, size_t scs, const int32_t *m, const float *tab, uint32_t K0, size_t nA, size_t nB, size_t spA,
                   const float *vA, const float *vB, size_t vs, size_t H, float *e, size_t es, float *out, float *inv_out) {
    for (size_t b = 0; b < G; b++) {
        float part[16] = {0};
        for (size_t t = 0; t < nA + nB; t++) {
            const size_t col = t < nA ? t : spA + (t - nA);
            const uint32_t k = (uint32_t)(m[b] - sc[b * scs + col]);
            const float ev = tab[k < K0 ? k : K0];
            e[b * es + col] = ev; part[(t < nA ? t : t - nA) & 15] += ev;
        }
        const float inv = 1.0f / ternary_sum16(part);
        inv_out[b] = inv;
        float *acc = out + b * H; memset(acc, 0, H * 4);
        for (size_t t = 0; t < nA + nB; t++) {
            const size_t col = t < nA ? t : spA + (t - nA);
            const float *vj = t < nA ? vA + t * vs : vB + (t - nA) * vs;
            const float ev = e[b * es + col];
            for (size_t d = 0; d < H; d++) acc[d] += ev * vj[d];
        }
        for (size_t d = 0; d < H; d++) acc[d] *= inv;
    }
}

static void gen_sc(int32_t *sc, int32_t *m, size_t G, size_t scs, size_t nA, size_t nB, size_t spA, int32_t lim) {
    for (size_t i = 0; i < G * scs; i++) sc[i] = rndi(lim);
    for (size_t b = 0; b < G; b++) {
        int32_t mx = INT32_MIN;
        for (size_t t = 0; t < nA + nB; t++) { const size_t col = t < nA ? t : spA + (t - nA); if (sc[b * scs + col] > mx) mx = sc[b * scs + col]; }
        m[b] = mx;
        if (rnd64() % 4 == 0) for (size_t t = 0; t < nA + nB; t += 3) { const size_t col = t < nA ? t : spA + (t - nA); sc[b * scs + col] = mx; }
    }
}

static void t_attn_pv(int two) {
    for (int it = 0; it < 300; it++) {
        const size_t G = GS_[rnd64() % NG], H = HS[rnd64() % NH];
        size_t nA = 0, nB = NS[rnd64() % NN], spA = 0;
        if (two) { nA = (rnd64() % 5 == 0) ? 0 : NS[rnd64() % NN]; spA = rup16(nA); }
        else { nA = nB; nB = 0; spA = rup16(nA); }
        const size_t scs = spA + rup16(nB) + 16, es = scs, vs = H + rnd64() % 9;
        uint32_t K0; float *tab = make_tab(&K0);
        const int32_t lim = 5 + (int32_t)(rnd64() % 400);
        int32_t *sc = IA(G * scs), *m = IA(3 * G);
        gen_sc(sc, m, G, scs, nA, nB, spA, lim);
        float *vA = FA(vs * (nA + 1)), *vB = FA(vs * (nB + 1));
        fillf(vA, vs * (nA + 1)); fillf(vB, vs * (nB + 1));
        float *e1 = FA(G * es), *e2 = FA(G * es), *o1 = FA(G * H), *o2 = FA(G * H), *i1 = FA(G), *i2 = FA(G);
        char ctx[128]; snprintf(ctx, sizeof ctx, "%s G=%zu H=%zu nA=%zu nB=%zu", two ? "pv2" : "pv", G, H, nA, nB);
        if (two) ternary_neon_attn_pv2(G, sc, scs, m, tab, K0, nA, nB, spA, vA, vB, vs, H, e1, es, o1, i1);
        else ternary_neon_attn_pv(G, sc, scs, m, tab, K0, nA, vA, vs, H, e1, es, o1, i1);
        ref_pv(G, sc, scs, m, tab, K0, nA, nB, spA, vA, vB, vs, H, e2, es, o2, i2);
        for (size_t b = 0; b < G; b++) {
            cmpf("pv e", e1 + b * es, e2 + b * es, nA ? nA : 0, ctx);
            if (nB) cmpf("pv e(B)", e1 + b * es + spA, e2 + b * es + spA, nB, ctx);
        }
        cmpf("pv out", o1, o2, G * H, ctx); cmpf("pv inv", i1, i2, G, ctx);
        free(tab); free(sc); free(m); free(vA); free(vB); free(e1); free(e2); free(o1); free(o2); free(i1); free(i2);
    }
}

/* backward de la fila, referencia: scalar attn_bwd_part (cada cabeza b ascendente, claves ascendentes; A y luego B) */
static void ref_bwd(size_t G, size_t nA, size_t nB, size_t spA, size_t H, const int32_t *sc, size_t scs, const int32_t *m, const float *inv,
                    const float *tab, uint32_t K0, const float *ga, const float *o, const float *vTA, const float *vTB, size_t ld, float itf, float gam,
                    const int32_t *qr, const int32_t *krA, const int32_t *krB, size_t ks, float *gq, float *gkA, float *gkB, float *gvA, float *gvB, size_t gs) {
    for (size_t b = 0; b < G; b++) {
        const float *gab = ga + b * H, *ob = o + b * H;
        float D = 0.0f;
        for (size_t d = 0; d < H; d++) D += gab[d] * ob[d];
        for (int seg = 0; seg < 2; seg++) {
            const size_t n = seg ? nB : nA, so = seg ? spA : 0;
            const float *vT = seg ? vTB : vTA; const int32_t *kr = seg ? krB : krA; float *gk = seg ? gkB : gkA, *gv = seg ? gvB : gvA;
            for (size_t j = 0; j < n; j++) {
                float dot = 0.0f;
                for (size_t d = 0; d < H; d++) dot += gab[d] * vT[d * ld + j];
                const uint32_t k = (uint32_t)(m[b] - sc[b * scs + so + j]);
                const float p = tab[k < K0 ? k : K0] * inv[b];
                const float ds = ((itf * p) * (dot - D)) * gam;
                for (size_t d = 0; d < H; d++) gv[j * gs + d] += p * gab[d];
                for (size_t d = 0; d < H; d++) gk[j * gs + d] += ds * (float)qr[b * H + d];
                for (size_t d = 0; d < H; d++) gq[b * H + d] += ds * (float)kr[j * ks + d];
            }
        }
    }
}

static void t_attn_bwd(int two) {
    for (int it = 0; it < 250; it++) {
        const size_t G = GS_[rnd64() % NG], H = HS[rnd64() % NH];
        size_t nA, nB, spA;
        if (two) { nB = NS[rnd64() % NN]; nA = (rnd64() % 5 == 0) ? 0 : NS[rnd64() % NN]; spA = rup16(nA); }
        else { nA = NS[rnd64() % NN]; nB = 0; spA = rup16(nA); }
        const size_t scs = spA + rup16(nB) + 16, es = scs, ld = rup16(spA > nA ? nA : nA) + 16, ldB = rup16(nB) + 16, ldm = ld > ldB ? ld : ldB, ks = H + rnd64() % 9, gs = H + rnd64() % 9;
        uint32_t K0; float *tab = make_tab(&K0);
        const int32_t lim = 5 + (int32_t)(rnd64() % 400);
        int32_t *sc = IA(G * scs), *m = IA(3 * G);
        gen_sc(sc, m, G, scs, nA, nB, spA, lim);
        /* inv coherente con la suma (cualquier valor positivo sirve para la comparacion de bits) */
        float *inv = FA(G); for (size_t b = 0; b < G; b++) inv[b] = 1.0f / (1.0f + (float)(rnd64() % 40) * 0.37f);
        float *ga = FA(G * H), *o = FA(G * H), *vTA = FA(H * ldm), *vTB = FA(H * ldm);
        fillf(ga, G * H); fillf(o, G * H); fillf(vTA, H * ldm); fillf(vTB, H * ldm);
        int32_t *qr = IA(G * H), *krA = IA(ks * (nA + 1)), *krB = IA(ks * (nB + 1));
        filli(qr, G * H, 1 << 18); filli(krA, ks * (nA + 1), 1 << 18); filli(krB, ks * (nB + 1), 1 << 18);
        const float itf = 0.25f + 0.5f * (float)(rnd64() % 5), gam = ldexpf(1.0f, -(int)(rnd64() % 4)) / 256.0f;
        const size_t kA = gs * (nA + 1), kB = gs * (nB + 1);
        float *gq1 = FA(G * H), *gq2 = FA(G * H), *gkA1 = FA(kA), *gkA2 = FA(kA), *gvA1 = FA(kA), *gvA2 = FA(kA);
        float *gkB1 = FA(kB), *gkB2 = FA(kB), *gvB1 = FA(kB), *gvB2 = FA(kB);
        fillf(gq1, G * H); memcpy(gq2, gq1, G * H * 4);
        fillf(gkA1, kA); memcpy(gkA2, gkA1, kA * 4); fillf(gvA1, kA); memcpy(gvA2, gvA1, kA * 4);
        fillf(gkB1, kB); memcpy(gkB2, gkB1, kB * 4); fillf(gvB1, kB); memcpy(gvB2, gvB1, kB * 4);
        float *p = FA(G * es + 32), *ds = FA(G * es + 32), *qf = FA(G * H + 16);
        char ctx[128]; snprintf(ctx, sizeof ctx, "%s G=%zu H=%zu nA=%zu nB=%zu ks=%zu gs=%zu", two ? "bwd2" : "bwd", G, H, nA, nB, ks, gs);
        if (two) ternary_neon_attn_bwd2(G, nA, nB, spA, H, sc, scs, m, inv, tab, K0, ga, o, vTA, vTB, ld > ldB ? ld : ldB, itf, gam, qr, krA, krB, ks, gq1, gkA1, gkB1, gvA1, gvB1, gs, p, ds, es, qf);
        else ternary_neon_attn_bwd(G, nA, H, sc, scs, m, inv, tab, K0, ga, o, vTA, ldm, itf, gam, qr, krA, ks, gq1, gkA1, gvA1, gs, p, ds, es, qf);
        ref_bwd(G, nA, nB, spA, H, sc, scs, m, inv, tab, K0, ga, o, vTA, vTB, ldm, itf, gam, qr, krA, krB, ks, gq2, gkA2, gkB2, gvA2, gvB2, gs);
        cmpf("bwd gq", gq1, gq2, G * H, ctx); cmpf("bwd gkA", gkA1, gkA2, kA, ctx); cmpf("bwd gvA", gvA1, gvA2, kA, ctx);
        if (two) { cmpf("bwd gkB", gkB1, gkB2, kB, ctx); cmpf("bwd gvB", gvB1, gvB2, kB, ctx); }
        free(tab); free(sc); free(m); free(inv); free(ga); free(o); free(vTA); free(vTB); free(qr); free(krA); free(krB);
        free(gq1); free(gq2); free(gkA1); free(gkA2); free(gvA1); free(gvA2); free(gkB1); free(gkB2); free(gvB1); free(gvB2); free(p); free(ds); free(qf);
    }
}

/* ---- softmax y RoPE ------------------------------------------------------------------------------------------- */
static void t_softmax(void) {
    for (int it = 0; it < 400; it++) {
        const size_t n = 1 + rnd64() % 150; const size_t len = 30 + rnd64() % 200;
        float *tab = FA(len); const float c = 0.02f + 0.001f * (float)(rnd64() % 50);
        for (size_t k = 0; k < len; k++) tab[k] = expf(-c * (float)k);
        int32_t *sc = IA(n); filli(sc, n, 20 + (int32_t)(rnd64() % 400));
        float *o1 = FA(n + 4), *o2 = FA(n + 4);
        for (size_t i = 0; i < n + 4; i++) o1[i] = o2[i] = 7.0f;
        ternary_neon_softmax_fwd(sc, n, tab, len, o1);
        int32_t m = sc[0]; for (size_t i = 1; i < n; i++) if (sc[i] > m) m = sc[i];
        float part[16] = {0};
        for (size_t i = 0; i < n; i++) { int64_t k = (int64_t)m - (int64_t)sc[i]; if (k >= (int64_t)len) k = (int64_t)len - 1; const float e = tab[k]; o2[i] = e; part[i & 15] += e; }
        const float inv = 1.0f / ternary_sum16(part);
        for (size_t i = 0; i < n; i++) o2[i] *= inv;
        char ctx[64]; snprintf(ctx, sizeof ctx, "n=%zu len=%zu", n, len);
        cmpf("softmax_fwd", o1, o2, n + 4, ctx);
        float *y = FA(n), *g = FA(n), *d1 = FA(n + 4), *d2 = FA(n + 4);
        fillf(y, n); fillf(g, n);
        for (size_t i = 0; i < n; i++) y[i] = fabsf(y[i]) * 0.1f;
        const float itemp = 0.25f * (float)(1 + rnd64() % 8);
        for (size_t i = 0; i < n + 4; i++) d1[i] = d2[i] = 7.0f;
        ternary_neon_softmax_bwd(y, g, n, itemp, d1);
        float pp[16] = {0}; for (size_t j = 0; j < n; j++) pp[j & 15] += g[j] * y[j];
        const float dot = ternary_sum16(pp);
        for (size_t i = 0; i < n; i++) d2[i] = itemp * y[i] * (g[i] - dot);
        cmpf("softmax_bwd", d1, d2, n + 4, ctx);
        free(tab); free(sc); free(o1); free(o2); free(y); free(g); free(d1); free(d2);
    }
}

static void t_rope(void) {
    for (int it = 0; it < 400; it++) {
        const size_t heads = 1 + rnd64() % 6, H = 2 * (1 + rnd64() % 40), P = 1 + rnd64() % (H / 2);
        int16_t *c = (int16_t *)A(P * 2), *s = (int16_t *)A(P * 2);
        for (size_t i = 0; i < P; i++) { c[i] = (int16_t)rndi(256); s[i] = (int16_t)rndi(256); }
        int32_t *z = IA(heads * H), *o1 = IA(heads * H + 4), *o2 = IA(heads * H + 4);
        filli(z, heads * H, 1 << 20);
        for (size_t i = 0; i < heads * H + 4; i++) o1[i] = o2[i] = 0x1234567;
        ternary_neon_rope_rotate_int(z, heads, H, P, c, s, o1);
        for (size_t h = 0; h < heads; h++) for (size_t i = 0; i < P; i++) {
            const int32_t a = z[h * H + 2 * i], b = z[h * H + 2 * i + 1];
            o2[h * H + 2 * i] = a * c[i] - b * s[i]; o2[h * H + 2 * i + 1] = a * s[i] + b * c[i];
        }
        char ctx[64]; snprintf(ctx, sizeof ctx, "heads=%zu H=%zu P=%zu", heads, H, P);
        cmpi("rope_rotate_int", o1, o2, heads * H + 4, ctx);
        float *g1 = FA(heads * H + 4), *g2 = FA(heads * H + 4);
        fillf(g1, heads * H); for (size_t i = heads * H; i < heads * H + 4; i++) g1[i] = 3.0f;
        memcpy(g2, g1, (heads * H + 4) * 4);
        const float inv = 1.0f / 256.0f;
        ternary_neon_rope_unrotate_grad(g1, heads, H, P, c, s, inv);
        for (size_t h = 0; h < heads; h++) for (size_t i = 0; i < P; i++) {
            const float ga = g2[h * H + 2 * i], gb = g2[h * H + 2 * i + 1], cf = c[i] * inv, sf = s[i] * inv;
            g2[h * H + 2 * i] = ga * cf + gb * sf; g2[h * H + 2 * i + 1] = -ga * sf + gb * cf;
        }
        cmpf("rope_unrotate_grad", g1, g2, heads * H + 4, ctx);
        free(c); free(s); free(z); free(o1); free(o2); free(g1); free(g2);
    }
}

int main(int argc, char **argv) {
    if (!ternary_neon_enabled()) { printf("NEON no disponible: test omitido\n"); return 0; }
    const char *only = argc > 1 ? argv[1] : "";
#define RUN(name, call) do { if (!*only || !strcmp(only, name)) { const int f0 = fails; call; printf("  %-14s %s\n", name, fails == f0 ? "ok" : "FALLA"); } } while (0)
    RUN("scores_i32", t_scores_i32());
    RUN("qk_grad", t_qk_grad_outer());
    RUN("attn_scores", t_attn_scores());
    RUN("attn_pv", t_attn_pv(0));
    RUN("attn_pv2", t_attn_pv(1));
    RUN("attn_bwd", t_attn_bwd(0));
    RUN("attn_bwd2", t_attn_bwd(1));
    RUN("softmax", t_softmax());
    RUN("rope", t_rope());
    if (fails) { printf("test_neon_attn: %d fallos\n", fails); return 1; }
    printf("test_neon_attn: OK\n");
    return 0;
}
