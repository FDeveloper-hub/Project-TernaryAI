/* Nucleos AVX-512 contra el bucle escalar de referencia (a += g * x con x en {-1,0,+1}, mismo orden de suma):
 * deben dar el MISMO valor float (== ; solo puede diferir el signo de un cero). Formas al azar con colas de tile
 * parciales. Si la CPU no tiene AVX-512 el test se salta. */
#include <math.h>
#include "ternary_avx512.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint64_t rs = 88172645463325252ull;
static uint64_t rnd64(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static float rndf(void) { int k = (int)(rnd64() % 7); if (k == 0) return 0.0f; if (k == 1) return -0.0f; return (float)((int)(rnd64() % 2001) - 1000) * 0.0137f; }
static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { if (fails < 10) printf("FALLA: %s (S=%zu n=%zu outd=%zu)\n", msg, S, n, outd); fails++; } } while (0)
/* pack_thr y rmsnorm contra el escalar. */
static int test_elementwise(void) {
    int bad = 0;
    for (int it = 0; it < 300; it++) {
        size_t n = 1 + rnd64() % 300;
        float *x = malloc(n * 4), *go = malloc(n * 4);
        for (size_t i = 0; i < n; i++) { x[i] = rndf(); go[i] = rndf(); }
        float thr = (it & 1) ? 0.0f : 0.37f;
        size_t w = (n + 63) / 64;
        uint64_t *am = calloc(w, 8), *as = calloc(w, 8), *bm = calloc(w, 8), *bs = calloc(w, 8);
        for (size_t i = 0; i < n; i++) { int v = (fabsf(x[i]) > thr) ? (x[i] > 0 ? 1 : -1) : 0; if (v) { am[i >> 6] |= 1ull << (i & 63); if (v < 0) as[i >> 6] |= 1ull << (i & 63); } }
        ternary_avx512_pack_thr(x, n, thr, bm, bs);
        for (size_t k = 0; k < w; k++) if (am[k] != bm[k] || as[k] != bs[k]) bad++;
        /* rmsnorm: ganancia = los planos a */
        size_t nt = (n + 15) / 16, rows = 1 + rnd64() % 3;
        uint16_t *pm = malloc(2 * nt * 2);
        ternary_avx512_split_planes(pm, pm + nt, n, am, as);
        float *xr = malloc(rows * n * 4), *gr = malloc(rows * n * 4);
        for (size_t i = 0; i < rows * n; i++) { xr[i] = rndf(); gr[i] = rndf(); }
        float *o1 = malloc(rows * n * 4), *o2 = malloc(rows * n * 4);
        const float eps = 1e-5f, sc = 0.7f;
        for (size_t r = 0; r < rows; r++) {
            double ss = 0; for (size_t j = 0; j < n; j++) ss += (double)xr[r*n+j] * xr[r*n+j];
            float f = sc * (float)(1.0 / sqrt(ss / n + eps));
            for (size_t j = 0; j < n; j++) { int g = (am[j>>6] >> (j&63) & 1) ? ((as[j>>6] >> (j&63) & 1) ? -1 : 1) : 0; o1[r*n+j] = g == 0 ? 0.0f : (g > 0 ? f * xr[r*n+j] : -f * xr[r*n+j]); }
        }
        ternary_avx512_rmsnorm_forward(rows, n, eps, sc, pm, pm + nt, xr, o2);
        for (size_t i = 0; i < rows * n; i++) if (o1[i] != o2[i]) bad++;
        /* backward */
        float *gg1 = calloc(n, 4), *gg2 = calloc(n, 4), *gx1 = calloc(rows * n, 4), *gx2 = calloc(rows * n, 4);
        double gs1 = 0, gs2 = 0;
        for (size_t r = 0; r < rows; r++) {
            const float *xi = xr + r*n, *g = gr + r*n;
            double ss = 0; for (size_t j = 0; j < n; j++) ss += (double)xi[j] * xi[j];
            float rr = (float)(1.0 / sqrt(ss / n + eps)); double dot = 0;
            for (size_t j = 0; j < n; j++) { int gv = (am[j>>6] >> (j&63) & 1) ? ((as[j>>6] >> (j&63) & 1) ? -1 : 1) : 0; dot += (double)gv * (double)g[j] * (double)xi[j]; gg1[j] += g[j] * sc * xi[j] * rr; }
            gs1 += dot * (double)rr;
            float c = (float)(dot * (double)rr * (double)rr / (double)n);
            for (size_t k = 0; k < n; k++) { int gv = (am[k>>6] >> (k&63) & 1) ? ((as[k>>6] >> (k&63) & 1) ? -1 : 1) : 0; gx1[r*n+k] += sc * rr * ((float)gv * g[k] - xi[k] * c); }
        }
        ternary_avx512_rmsnorm_backward(rows, n, eps, sc, pm, pm + nt, xr, gr, gg2, gx2, &gs2);
        for (size_t j = 0; j < n; j++) if (gg1[j] != gg2[j]) bad++;
        for (size_t i = 0; i < rows * n; i++) if (gx1[i] != gx2[i]) bad++;
        if (fabs(gs1 - gs2) > 1e-9 * (1 + fabs(gs1))) bad++;
        free(x); free(go); free(am); free(as); free(bm); free(bs); free(pm); free(xr); free(gr); free(o1); free(o2); free(gg1); free(gg2); free(gx1); free(gx2);
    }
    return bad;
}

static int test_counter_bwd1(void) {
    int bad = 0;
    for (int it = 0; it < 400; it++) {
        size_t n = 1 + rnd64() % 200;
        /* bwd1_row */
        size_t nt = (n + 15) / 16, w = (n + 63) / 64;
        uint64_t *xm = calloc(w, 8), *xs = calloc(w, 8), *wm = calloc(w, 8), *ws = calloc(w, 8);
        for (size_t i = 0; i < n; i++) { if (rnd64() & 1) { xm[i>>6] |= 1ull << (i&63); if (rnd64() & 1) xs[i>>6] |= 1ull << (i&63); } if (rnd64() & 1) { wm[i>>6] |= 1ull << (i&63); if (rnd64() & 1) ws[i>>6] |= 1ull << (i&63); } }
        uint16_t *pl = malloc(4 * nt * 2);
        ternary_avx512_split_planes(pl, pl + nt, n, xm, xs); ternary_avx512_split_planes(pl + 2 * nt, pl + 3 * nt, n, wm, ws);
        float gv = rndf(), gsv = rndf();
        for (int acc = 0; acc < 2; acc++) {
            float *r1 = malloc(n * 4), *r2 = malloc(n * 4), *x1 = malloc(n * 4), *x2 = malloc(n * 4);
            for (size_t i = 0; i < n; i++) { r1[i] = r2[i] = rndf(); x1[i] = x2[i] = rndf(); }
            for (size_t i = 0; i < n; i++) {
                float xf = (xm[i>>6] >> (i&63) & 1) ? ((xs[i>>6] >> (i&63) & 1) ? -1.0f : 1.0f) : 0.0f, wf = (wm[i>>6] >> (i&63) & 1) ? ((ws[i>>6] >> (i&63) & 1) ? -1.0f : 1.0f) : 0.0f;
                if (acc) r1[i] += gv * xf; else r1[i] = gv * xf;
                x1[i] += gsv * wf;
            }
            ternary_avx512_bwd1_row(n, pl, pl + nt, pl + 2 * nt, pl + 3 * nt, gv, gsv, r2, x2, acc);
            for (size_t i = 0; i < n; i++) if (memcmp(&r1[i], &r2[i], 4) || memcmp(&x1[i], &x2[i], 4)) bad++;
            free(r1); free(r2); free(x1); free(x2);
        }
        free(xm); free(xs); free(wm); free(ws); free(pl);
    }
    return bad;
}

static int test_attn_embed(void) {
    int bad = 0;
    for (int it = 0; it < 300; it++) {
        size_t H = 1 + rnd64() % 70, cnt = 1 + rnd64() % 9, ks = H + rnd64() % 5;
        int32_t *a = malloc(H * 4), *kb = malloc(cnt * ks * 4), *o1 = malloc(cnt * 4), *o2 = malloc(cnt * 4);
        for (size_t i = 0; i < H; i++) a[i] = (int32_t)(rnd64() % 200001) - 100000;
        for (size_t i = 0; i < cnt * ks; i++) kb[i] = (int32_t)(rnd64() % 200001) - 100000;
        int sh = 16 + (int)(rnd64() % 3); int32_t lim = 1 + (int32_t)(rnd64() % 5000);
        for (size_t j = 0; j < cnt; j++) { int64_t d = 0; for (size_t i = 0; i < H; i++) d += (int64_t)a[i] * kb[j * ks + i]; d >>= sh; if (d > lim) d = lim; else if (d < -lim) d = -lim; o1[j] = (int32_t)d; }
        ternary_avx512_scores_i32(a, kb, ks, H, cnt, sh, lim, o2);
        for (size_t j = 0; j < cnt; j++) if (o1[j] != o2[j]) bad++;
        float *ds = malloc(cnt * 4), *ga = malloc(H * 4);
        for (size_t j = 0; j < cnt; j++) ds[j] = rndf();
        for (size_t i = 0; i < H; i++) ga[i] = rndf();
        float *gq1 = calloc(H, 4), *gq2 = calloc(H, 4), *gk1 = calloc(cnt * ks, 4), *gk2 = calloc(cnt * ks, 4), *v1 = calloc(cnt * ks, 4), *v2 = calloc(cnt * ks, 4);
        for (size_t j = 0; j < cnt; j++) for (size_t d = 0; d < H; d++) { gq1[d] += ds[j] * (float)kb[j*ks+d]; gk1[j*ks+d] += ds[j] * (float)a[d]; v1[j*ks+d] += ds[j] * ga[d]; }
        ternary_avx512_qk_grad(gq2, a, kb, gk2, ks, H, cnt, ds);
        ternary_avx512_outer_add(v2, ks, H, cnt, ds, ga);
        for (size_t d = 0; d < H; d++) if (memcmp(&gq1[d], &gq2[d], 4)) bad++;
        for (size_t i = 0; i < cnt * ks; i++) if (memcmp(&gk1[i], &gk2[i], 4) || memcmp(&v1[i], &v2[i], 4)) bad++;
        /* unpack */
        size_t nb = 64 + rnd64() % 400, off = rnd64() % 40, n = 1 + rnd64() % (nb - off), w = (nb + 63) / 64;
        uint64_t *m = calloc(w, 8), *sg = calloc(w, 8);
        for (size_t i = 0; i < nb; i++) if (rnd64() & 1) { m[i>>6] |= 1ull << (i&63); if (rnd64() & 1) sg[i>>6] |= 1ull << (i&63); }
        float *u1 = malloc(n * 4), *u2 = malloc(n * 4); int32_t *r2 = malloc(n * 4); float sc = 0.37f;
        ternary_avx512_unpack_row(m, sg, nb, off, n, sc, u2, r2);
        for (size_t i = 0; i < n; i++) { size_t b = off + i; int wv = (m[b>>6] >> (b&63) & 1) ? ((sg[b>>6] >> (b&63) & 1) ? -1 : 1) : 0; u1[i] = sc * (float)wv; if (memcmp(&u1[i], &u2[i], 4) || r2[i] != wv) bad++; }
        free(a); free(kb); free(o1); free(o2); free(ds); free(ga); free(gq1); free(gq2); free(gk1); free(gk2); free(v1); free(v2); free(m); free(sg); free(u1); free(u2); free(r2);
    }
    return bad;
}

int main(void) {
    if (!ternary_avx512_enabled()) { printf("test_avx512: CPU sin AVX-512 (o TERNARYAI_NO_AVX512): salteado\n"); return 0; }
    int cases = 0;
    for (int it = 0; it < 600; it++) {
        size_t S = 1 + rnd64() % 37, n = 1 + rnd64() % 300, outd = 1 + rnd64() % 41;
        size_t nt16 = (n + 15) / 16, words = (nt16 * 16 + 63) / 64 + 1;
        uint64_t *xm = malloc(S * words * 8), *xs = malloc(S * words * 8), *wm = malloc(outd * words * 8), *ws = malloc(outd * words * 8);
        for (size_t i = 0; i < S * words; i++) { xm[i] = rnd64(); xs[i] = rnd64(); }
        for (size_t i = 0; i < outd * words; i++) { wm[i] = rnd64(); ws[i] = rnd64(); }
        uint16_t *xp = malloc(S * nt16 * 2), *xn = malloc(S * nt16 * 2), *wp = malloc(outd * nt16 * 2), *wn = malloc(outd * nt16 * 2);
        for (size_t s = 0; s < S; s++) ternary_avx512_split_planes(xp + s * nt16, xn + s * nt16, n, xm + s * words, xs + s * words);
        for (size_t o = 0; o < outd; o++) ternary_avx512_split_planes(wp + o * nt16, wn + o * nt16, n, wm + o * words, ws + o * words);
        /* referencia de x[s][i], w[o][i] */
        #define XV(s, i) ((int)((xm[(s) * words + ((i) >> 6)] >> ((i) & 63)) & 1u) * (1 - 2 * (int)((xs[(s) * words + ((i) >> 6)] >> ((i) & 63)) & 1u)))
        #define WV(o, i) ((int)((wm[(o) * words + ((i) >> 6)] >> ((i) & 63)) & 1u) * (1 - 2 * (int)((ws[(o) * words + ((i) >> 6)] >> ((i) & 63)) & 1u)))
        /* grad_w con ld > n */
        size_t ld = n + (rnd64() % 20);
        float *gT = malloc(outd * S * 4), *A = malloc(outd * ld * 4), *B = malloc(outd * ld * 4);
        for (size_t i = 0; i < outd * S; i++) gT[i] = rndf();
        for (size_t i = 0; i < outd * ld; i++) A[i] = B[i] = rndf();
        for (size_t o = 0; o < outd; o++) for (size_t i = 0; i < n; i++) { float a = A[o * ld + i]; for (size_t s = 0; s < S; s++) a += gT[o * S + s] * (float)XV(s, i); A[o * ld + i] = a; }
        ternary_avx512_gradw(B, ld, outd, n, S, gT, xp, xn);
        for (size_t o = 0; o < outd; o++) for (size_t i = 0; i < ld; i++) CHECK(A[o * ld + i] == B[o * ld + i], "gradw distinto");
        /* grad_x */
        float *gs = malloc(S * outd * 4); float **gxa = malloc(S * sizeof(float *)), **gxb = malloc(S * sizeof(float *));
        for (size_t i = 0; i < S * outd; i++) gs[i] = rndf();
        for (size_t s = 0; s < S; s++) { gxa[s] = malloc((n + 20) * 4); gxb[s] = malloc((n + 20) * 4); for (size_t i = 0; i < n + 20; i++) gxa[s][i] = gxb[s][i] = rndf(); }
        for (size_t s = 0; s < S; s++) for (size_t i = 0; i < n; i++) { float a = gxa[s][i]; for (size_t o = 0; o < outd; o++) a += gs[s * outd + o] * (float)WV(o, i); gxa[s][i] = a; }
        ternary_avx512_gradx(gxb, S, outd, n, gs, wp, wn);
        for (size_t s = 0; s < S; s++) for (size_t i = 0; i < n + 20; i++) CHECK(gxa[s][i] == gxb[s][i], "gradx distinto");
        cases++;
        for (size_t s = 0; s < S; s++) { free(gxa[s]); free(gxb[s]); }
        free(gxa); free(gxb); free(gs); free(gT); free(A); free(B); free(xm); free(xs); free(wm); free(ws); free(xp); free(xn); free(wp); free(wn);
    }
    { int b = test_attn_embed(); if (b) { printf("FALLA attn/embed: %d\n", b); fails += b; } }
    { int b = test_counter_bwd1(); if (b) { printf("FALLA votes/bwd1: %d\n", b); fails += b; } }
    { int b = test_elementwise(); if (b) { printf("FALLA elementwise: %d\n", b); fails += b; } }
    printf("test_avx512: %d formas al azar, grad_w y grad_x: %s\n", cases, fails ? "FALLAS" : "igual al escalar (valor float ==)");
    return fails ? 1 : 0;
}
