/* Nucleos NEON elementales, smear, cuantizacion uint8, R2 y normalizacion por elemento contra las referencias ESCALARES de la libreria
 * (transcritas de ternary_dense.c, ternary_u8.c y ternary_gqa.c): mismos bits (memcmp). Sin NEON se omite. */
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
static int fails = 0;
#define FAIL(...) do { if (fails < 12) { printf("FALLA: "); printf(__VA_ARGS__); printf("\n"); } fails++; } while (0)
static void *A(size_t n) { if (n > ((size_t)1 << 28)) abort(); void *p = calloc(n ? n : 1, 1); if (!p) abort(); return p; }
static float *FA(size_t n) { return (float *)A(n * 4); }
static int32_t *IA(size_t n) { return (int32_t *)A(n * 4); }
static void fillf(float *p, size_t n) { for (size_t i = 0; i < n; i++) p[i] = rndf(); }
static void cmpf(const char *nm, const float *a, const float *b, size_t n, const char *ctx) {
    for (size_t i = 0; i < n; i++) if (memcmp(&a[i], &b[i], 4)) { FAIL("%s [%zu] %a vs %a (%s)", nm, i, a[i], b[i], ctx); return; }
}
static void cmpm(const char *nm, const void *a, const void *b, size_t bytes, const char *ctx) {
    if (memcmp(a, b, bytes)) { const unsigned char *x = a, *y = b; size_t i = 0; while (x[i] == y[i]) i++; FAIL("%s byte %zu: %02x vs %02x (%s)", nm, i, x[i], y[i], ctx); }
}
static const size_t NS[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 129, 257};
#define NN (sizeof(NS) / sizeof(NS[0]))

static void t_simple(void) {
    for (int it = 0; it < 500; it++) {
        const size_t n = NS[rnd64() % NN]; char ctx[64]; snprintf(ctx, sizeof ctx, "n=%zu", n);
        float *a = FA(n + 4), *b = FA(n + 4), *o1 = FA(n + 4), *o2 = FA(n + 4);
        fillf(a, n); fillf(b, n);
        for (size_t i = 0; i < n + 4; i++) o1[i] = o2[i] = 9.0f;
        ternary_neon_add_f32(o1, a, b, n); for (size_t i = 0; i < n; i++) o2[i] = a[i] + b[i];
        cmpf("add_f32", o1, o2, n + 4, ctx);
        fillf(o1, n); memcpy(o2, o1, n * 4);
        ternary_neon_acc_add2(o1, a, b, n); for (size_t i = 0; i < n; i++) o2[i] += (a[i] + b[i]);
        cmpf("acc_add2", o1, o2, n, ctx);
        const float s = rndf(); fillf(o1, n); memcpy(o2, o1, n * 4);
        ternary_neon_scale_f32(o1, n, s); for (size_t i = 0; i < n; i++) o2[i] *= s;
        cmpf("scale_f32", o1, o2, n, ctx);
        int32_t *z = IA(n + 4), *i1 = IA(n + 4), *i2 = IA(n + 4);
        for (size_t i = 0; i < n; i++) z[i] = (it % 3 == 0) ? (int32_t)rnd64() : rndi(1 << 20);
        const int sh = 1 + (int)(rnd64() % 20);
        ternary_neon_shift_round_i32(z, n, sh, i1);
        for (size_t i = 0; i < n; i++) i2[i] = (int32_t)((uint32_t)z[i] + (1u << (sh - 1))) >> sh;
        if (memcmp(i1, i2, n * 4)) FAIL("shift_round_i32 (%s)", ctx);
        const size_t stride = 1 + rnd64() % 4;
        float *s1 = FA(n * stride + 4), *s2 = FA(n * stride + 4);
        for (size_t i = 0; i < n * stride + 4; i++) s1[i] = s2[i] = 7.0f;
        ternary_neon_sign_f32(z, n, s1, stride); for (size_t i = 0; i < n; i++) s2[i * stride] = (float)((z[i] > 0) - (z[i] < 0));
        cmpf("sign_f32", s1, s2, n * stride + 4, ctx);
        const float k = rndf();
        for (size_t i = 0; i < n * stride + 4; i++) s1[i] = s2[i] = 7.0f;
        ternary_neon_scale_i32_f32(z, n, k, s1, stride); for (size_t i = 0; i < n; i++) s2[i * stride] = k * (float)z[i];
        cmpf("scale_i32_f32", s1, s2, n * stride + 4, ctx);
        int32_t *d1 = IA(n * stride + 4), *d2 = IA(n * stride + 4);
        for (size_t i = 0; i < n * stride + 4; i++) d1[i] = d2[i] = 77;
        ternary_neon_scatter_i32(z, n, d1, stride); for (size_t i = 0; i < n; i++) d2[i * stride] = z[i];
        if (memcmp(d1, d2, (n * stride + 4) * 4)) FAIL("scatter_i32 (%s)", ctx);
        for (int first = 0; first < 2; first++) for (int alias = 0; alias < 2; alias++) {
            int32_t *in1 = IA(n + 4), *in2 = IA(n + 4), *p1 = IA(n + 4), *p2 = IA(n + 4), *oo1 = IA(n + 4), *oo2 = IA(n + 4);
            for (size_t i = 0; i < n; i++) { in1[i] = in2[i] = rndi(1 << 22); p1[i] = p2[i] = rndi(1 << 22); }
            for (size_t i = 0; i < n + 4; i++) oo1[i] = oo2[i] = 5;
            ternary_neon_smear_k(in1, alias ? in1 : oo1, p1, n, first);
            for (size_t i = 0; i < n; i++) { const int32_t aa = in2[i]; (alias ? in2 : oo2)[i] = first ? aa : (int32_t)((uint32_t)aa + (uint32_t)p2[i]) >> 1; p2[i] = aa; }
            if (memcmp(p1, p2, n * 4) || memcmp(alias ? in1 : oo1, alias ? in2 : oo2, (n + 4) * 4)) FAIL("smear_k first=%d alias=%d (%s)", first, alias, ctx);
            free(in1); free(in2); free(p1); free(p2); free(oo1); free(oo2);
        }
        free(a); free(b); free(o1); free(o2); free(z); free(i1); free(i2); free(s1); free(s2); free(d1); free(d2);
    }
    for (int it = 0; it < 300; it++) {
        const size_t T = 1 + rnd64() % 20, w = 1 + rnd64() % 40, p0 = rnd64() % T, p1 = p0 + 1 + rnd64() % (T - p0);
        float *g1 = FA(T * w + 4), *g2 = FA(T * w + 4); fillf(g1, T * w); memcpy(g2, g1, (T * w + 4) * 4);
        float *nl = NULL; if (p1 < T && (it & 1)) { nl = FA(w); memcpy(nl, g1 + p1 * w, w * 4); }
        ternary_neon_smear_grad_rows(g1, p0, p1, T, w, nl);
        for (size_t p = p0; p < p1; p++) { float *row = g2 + p * w; const float *nrow = (p + 1 == p1 && nl) ? nl : row + w;
            for (size_t d = 0; d < w; d++) { const float own = p == 0 ? row[d] : 0.5f * row[d]; const float nxt = p + 1 < T ? 0.5f * nrow[d] : 0.0f; row[d] = own + nxt; } }
        char ctx[64]; snprintf(ctx, sizeof ctx, "T=%zu w=%zu p0=%zu p1=%zu", T, w, p0, p1);
        cmpf("smear_grad_rows", g1, g2, T * w + 4, ctx);
        free(g1); free(g2); free(nl);
    }
}

static void ref_quant(const float *xr, size_t n, uint8_t *qr, float *sc_out, int32_t *zp_out) {
    float lo = 0.0f, hi = 0.0f;
    for (size_t i = 0; i < n; i++) { const float v = xr[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    float sc = (hi - lo) / 255.0f;
    if (!(sc > 0.0f) || !isfinite(sc)) { *sc_out = 1.0f; *zp_out = 0; memset(qr, 0, n); return; }
    int zp = (int)rintf(-lo / sc); if (zp < 0) zp = 0; if (zp > 255) zp = 255;
    *sc_out = sc; *zp_out = zp;
    const float inv = 1.0f / sc;
    for (size_t i = 0; i < n; i++) { float qf = rintf(xr[i] * inv) + (float)zp; if (qf < 0.0f) qf = 0.0f; if (qf > 255.0f) qf = 255.0f; qr[i] = (uint8_t)qf; }
}
static void t_quant(void) {
    for (int it = 0; it < 600; it++) {
        const size_t n = NS[1 + rnd64() % (NN - 1)]; char ctx[64]; snprintf(ctx, sizeof ctx, "n=%zu it=%d", n, it);
        float *x = FA(n); fillf(x, n);
        const int kind = it % 8;
        for (size_t i = 0; i < n; i++) {
            if (kind == 1) x[i] = fabsf(x[i]) + 0.5f;            /* todo positivo */
            else if (kind == 2) x[i] = -fabsf(x[i]) - 0.5f;      /* todo negativo */
            else if (kind == 3) x[i] = 0.0f;
            else if (kind == 4) x[i] *= 1e-6f;
            else if (kind == 5) x[i] *= 1e6f;
            else if (kind == 6 && i == n / 2) x[i] = NAN;
            else if (kind == 7 && i == 0) x[i] = INFINITY;
        }
        uint8_t *q1 = (uint8_t *)A(n + 8), *q2 = (uint8_t *)A(n + 8); memset(q1, 0xAB, n + 8); memset(q2, 0xAB, n + 8);
        float s1 = 0, s2 = 0; int32_t z1 = -1, z2 = -2;
        ternary_neon_u8_quantize_row(x, n, q1, &s1, &z1); ref_quant(x, n, q2, &s2, &z2);
        if (memcmp(&s1, &s2, 4) || z1 != z2) FAIL("quantize scale/zp %a/%d vs %a/%d (%s)", s1, z1, s2, z2, ctx);
        cmpm("quantize q", q1, q2, n + 8, ctx);
        free(x); free(q1); free(q2);
    }
}

static void ref_r2(uint64_t *mask, uint64_t *sign, int8_t *cnt, size_t base, const float *g, size_t n, float beta, float inc, int ti, int tin, int tout, int ri, int pre) {
    float inv = 1.0f;
    if (!pre) { const double ms = ternary_canon_sumsq(g, n) / (double)n; inv = 1.0f / ((float)sqrt(ms) + 1e-12f); }
    const float cb = beta, cg = inc * 127.0f;
    for (size_t i = 0; i < n; i++) {
        float gh = g[i] * inv;
        gh = gh > 3.0f ? 3.0f : (gh < -3.0f ? -3.0f : gh);
        float x = cb * (float)cnt[i] - cg * gh;
        if (!(x == x)) x = 0.0f;
        int v = (int)rintf(x); v = v > 127 ? 127 : (v < -127 ? -127 : v);
        const size_t idx = base + i, word = idx >> 6; const uint64_t m = (uint64_t)1 << (idx & 63);
        const int cur = (mask[word] & m) ? ((sign[word] & m) ? -1 : 1) : 0, dir = v > 0 ? 1 : -1;
        const int th = cur == 0 ? tin : (dir == -cur ? tout : ti);
        if (v >= th || v <= -th) {
            int nuevo = cur + dir; if (nuevo > 1) nuevo = 1; else if (nuevo < -1) nuevo = -1;
            if (nuevo == 0) { mask[word] &= ~m; sign[word] &= ~m; }
            else { mask[word] |= m; if (nuevo < 0) sign[word] |= m; else sign[word] &= ~m; }
            v = -dir * ri;
        }
        cnt[i] = (int8_t)v;
    }
}
static void t_r2(void) {
    for (int it = 0; it < 1500; it++) {
        const size_t n = (it % 4 == 0) ? 1 + rnd64() % 300 : NS[1 + rnd64() % (NN - 1)], base = (it % 5 == 0) ? 0 : rnd64() % 200;
        const size_t words = (base + n + 63) / 64 + 2;
        uint64_t *m1 = (uint64_t *)A(words * 8), *s1 = (uint64_t *)A(words * 8);
        const int dense = it % 3;
        for (size_t w = 0; w < words; w++) { m1[w] = rnd64(); s1[w] = rnd64() & m1[w]; if (dense == 1) m1[w] = ~0ull, s1[w] = rnd64(); if (dense == 2) m1[w] = s1[w] = 0; }
        uint64_t *m2 = (uint64_t *)A(words * 8), *s2 = (uint64_t *)A(words * 8); memcpy(m2, m1, words * 8); memcpy(s2, s1, words * 8);
        int8_t *c1 = (int8_t *)A(n + 16), *c2 = (int8_t *)A(n + 16);
        for (size_t i = 0; i < n + 16; i++) c1[i] = c2[i] = (int8_t)rndi(127);
        float *g = FA(n); fillf(g, n);
        const float gsc = (it % 7 == 0) ? 1e-4f : ((it % 7 == 1) ? 100.0f : 1.0f);
        for (size_t i = 0; i < n; i++) g[i] *= gsc;
        if (it % 11 == 0 && n > 2) g[n / 2] = NAN;
        const int ti = 1 + (int)(rnd64() % 127), tin = 1 + (int)(rnd64() % 127), tout = 1 + (int)(rnd64() % 127), ri = (int)(rnd64() % 80);
        const float beta = 0.9f + 0.001f * (float)(rnd64() % 100), inc = 0.005f * (float)(1 + rnd64() % 20);
        const int pre = (it % 3) == 0;
        char ctx[96]; snprintf(ctx, sizeof ctx, "n=%zu base=%zu pre=%d it=%d", n, base, pre, it);
        ternary_neon_r2_row(m1, s1, c1, base, g, n, beta, inc, ti, tin, tout, ri, pre);
        ref_r2(m2, s2, c2, base, g, n, beta, inc, ti, tin, tout, ri, pre);
        cmpm("r2 mask", m1, m2, words * 8, ctx); cmpm("r2 sign", s1, s2, words * 8, ctx); cmpm("r2 cnt", c1, c2, n + 16, ctx);
        free(m1); free(s1); free(m2); free(s2); free(c1); free(c2); free(g);
    }
}

static inline int en_code(float x) {
    if (!(x >= 0x1p-16f)) return 15;
    int e; const float m = frexpf(x, &e);
    int r = -e + (m < 0.70710678f ? 1 : 0);
    return r < 0 ? 0 : (r > 15 ? 15 : r);
}
static void ref_elemnorm(uint8_t *cr, float *rmax, float *gr, float *v, size_t W, float beta2, float omb, float bc, float eps) {
    const float rm = *rmax; float mx = 0.0f;
    for (size_t k = 0; k < W; k++) {
        const int c = (k & 1) ? (cr[k >> 1] >> 4) : (cr[k >> 1] & 15);
        const float vh = ldexpf(rm, -c);
        const float x = gr[k] * gr[k];
        const float vn = beta2 * vh + omb * x;
        v[k] = vn; if (vn > mx) mx = vn;
    }
    *rmax = mx;
    for (size_t k = 0; k < W; k++) {
        const int c = mx > 0.0f ? en_code(v[k] / mx) : 15;
        if (k & 1) cr[k >> 1] = (uint8_t)((cr[k >> 1] & 0x0F) | (c << 4)); else cr[k >> 1] = (uint8_t)((cr[k >> 1] & 0xF0) | c);
        const float vq = ldexpf(mx, -c);
        gr[k] = gr[k] / (sqrtf(vq / bc) + eps);
    }
}
static void t_elemnorm(void) {
    for (int it = 0; it < 1500; it++) {
        const size_t W = (it % 3 == 0) ? 1 + rnd64() % 300 : NS[1 + rnd64() % (NN - 1)], st = (W + 1) / 2;
        const float beta2 = (it % 2) ? 0.99f : 0.999f, omb = 1.0f - beta2;
        const float bc = (float)(1.0 - pow((double)beta2, (double)(1 + rnd64() % 50)));
        float *g1 = FA(W + 4), *g2 = FA(W + 4), *v1 = FA(W + 4), *v2 = FA(W + 4);
        const int kind = it % 6;
        if (kind == 0 || kind == 4) {                 /* sin historia, g^2 con razones justo en las fronteras 2^-k * 0.7071 (casos dudosos) */
            const float gmax = 0.01f * (float)(1 + rnd64() % 100);
            for (size_t k = 0; k < W; k++) {
                const int lev = (int)(rnd64() % 18);
                float gg = gmax * sqrtf(ldexpf(0.70710678f, -lev));
                for (int u = (int)(rnd64() % 9); u > 0; u--) gg = nextafterf(gg, (rnd64() & 1) ? 1e9f : -1e9f);
                g1[k] = (rnd64() & 1) ? gg : -gg;
            }
            g1[rnd64() % W] = gmax;                       /* el maximo: las demas razones v/mx quedan justo en las fronteras */
        } else {
            for (size_t k = 0; k < W; k++) g1[k] = rndf() * ((kind == 2) ? 1e-12f : ((kind == 3) ? 1e-17f : 1.0f));
            if (kind == 5 && W > 3) g1[W / 2] = NAN;
        }
        memcpy(g2, g1, (W + 4) * 4);
        uint8_t *c1 = (uint8_t *)A(st + 8), *c2 = (uint8_t *)A(st + 8);
        float rm1 = 0.0f;
        if (kind != 0) { for (size_t i = 0; i < st; i++) c1[i] = (uint8_t)rnd64(); rm1 = (it % 4 == 0) ? 0.0f : (float)(1 + rnd64() % 1000) * 1e-3f * ((kind == 2) ? 1e-12f : ((kind == 3) ? 1e-17f : 1.0f)); }
        memcpy(c2, c1, st + 8); float rm2 = rm1;
        double gm = 0; for (size_t k = 0; k < W; k++) gm += (double)g1[k] * (double)g1[k];
        const float eps = (float)(1e-3 * sqrt(gm / (double)W) + 1e-20);
        char ctx[96]; snprintf(ctx, sizeof ctx, "W=%zu kind=%d it=%d", W, kind, it);
        ternary_neon_elemnorm_row(c1, &rm1, g1, v1, W, beta2, omb, bc, eps);
        ref_elemnorm(c2, &rm2, g2, v2, W, beta2, omb, bc, eps);
        if (memcmp(&rm1, &rm2, 4)) FAIL("elemnorm rmax %a vs %a (%s)", rm1, rm2, ctx);
        cmpm("elemnorm cr", c1, c2, st + 8, ctx); cmpf("elemnorm g", g1, g2, W + 4, ctx);
        for (size_t k = 0; k < W; k++) if (v1[k] == v1[k] && memcmp(&v1[k], &v2[k], 4)) { FAIL("elemnorm v[%zu] (%s)", k, ctx); break; }
        free(g1); free(g2); free(v1); free(v2); free(c1); free(c2);
    }
}

int main(int argc, char **argv) {
    if (!ternary_neon_enabled()) { printf("NEON no disponible: test omitido\n"); return 0; }
    const char *only = argc > 1 ? argv[1] : "";
#define RUN(name, call) do { if (!*only || !strcmp(only, name)) { const int f0 = fails; call; printf("  %-12s %s\n", name, fails == f0 ? "ok" : "FALLA"); } } while (0)
    RUN("simple", t_simple());
    RUN("quantize", t_quant());
    RUN("r2", t_r2());
    RUN("elemnorm", t_elemnorm());
    if (fails) { printf("test_neon_elem: %d fallos\n", fails); return 1; }
    printf("test_neon_elem: OK\n");
    return 0;
}
