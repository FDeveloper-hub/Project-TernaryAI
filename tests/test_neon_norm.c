/* Nucleos NEON de RMSNorm, SwiGLU, entropia cruzada y dot_gz contra las referencias ESCALARES de la libreria (transcritas de
 * ternary_rmsnorm.c, ternary_swiglu.c, ternary_loss.c y ternary_fexp.h): mismos bits (memcmp). Sin NEON se omite. */
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
static void *A(size_t n) { void *p = calloc(n ? n : 1, 1); if (!p) abort(); return p; }
static float *FA(size_t n) { return (float *)A(n * 4); }
static int32_t *IA(size_t n) { return (int32_t *)A(n * 4); }
static void fillf(float *p, size_t n) { for (size_t i = 0; i < n; i++) p[i] = rndf(); }
static void filli(int32_t *p, size_t n, int32_t lim) { for (size_t i = 0; i < n; i++) p[i] = rndi(lim); }
static void cmpf(const char *nm, const float *a, const float *b, size_t n, const char *ctx) {
    for (size_t i = 0; i < n; i++) if (memcmp(&a[i], &b[i], 4)) { FAIL("%s [%zu] %a vs %a (%s)", nm, i, a[i], b[i], ctx); return; }
}
static int sign_i32(int32_t v) { return v > 0 ? 1 : (v < 0 ? -1 : 0); }

static const size_t DS[] = {1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 17, 20, 31, 32, 33, 48, 63, 64, 65, 100, 127, 130};
#define ND (sizeof(DS) / sizeof(DS[0]))

/* ---- RMSNorm ---- */
static void t_rms(void) {
    for (int it = 0; it < 300; it++) {
        const size_t D = DS[rnd64() % ND], rows = 1 + rnd64() % 4, nt = (D + 15) / 16;
        const float eps = 1e-5f * (float)(1 + rnd64() % 9), scale = 0.01f * (float)(1 + rnd64() % 100);
        uint16_t *gp = (uint16_t *)A(nt * 2 * 2), *gm = gp + nt; int *g = (int *)A(D * sizeof(int));
        for (size_t j = 0; j < D; j++) { g[j] = (int)(rnd64() % 3) - 1; if (g[j] > 0) gp[j >> 4] |= (uint16_t)(1u << (j & 15)); if (g[j] < 0) gm[j >> 4] |= (uint16_t)(1u << (j & 15)); }
        float *x = FA(rows * D), *go = FA(rows * D);
        fillf(x, rows * D); fillf(go, rows * D);
        char ctx[96]; snprintf(ctx, sizeof ctx, "D=%zu rows=%zu", D, rows);
        float *o1 = FA(rows * D + 4), *o2 = FA(rows * D + 4);
        for (size_t i = 0; i < rows * D + 4; i++) o1[i] = o2[i] = 9.0f;
        ternary_neon_rmsnorm_forward(rows, D, eps, scale, gp, gm, x, o1);
        for (size_t i = 0; i < rows; i++) {
            const double ss = ternary_canon_sumsq(x + i * D, D);
            const float r = (float)(1.0 / sqrt(ss / (double)D + (double)eps)), f = scale * r;
            for (size_t j = 0; j < D; j++) o2[i * D + j] = (g[j] == 0) ? 0.0f : (g[j] > 0 ? f * x[i * D + j] : -f * x[i * D + j]);
        }
        cmpf("rms_fwd", o1, o2, rows * D + 4, ctx);
        /* backward (una o varias filas; grad_gain parte de cero, gx acumula) */
        float *gg1 = FA(D), *gg2 = FA(D), *gx1 = FA(rows * D), *gx2 = FA(rows * D);
        fillf(gx1, rows * D); memcpy(gx2, gx1, rows * D * 4);
        double gs1 = 0.5, gs2 = 0.5;
        const int with_gx = (it % 4) != 0;
        ternary_neon_rmsnorm_backward(rows, D, eps, scale, gp, gm, x, go, gg1, with_gx ? gx1 : NULL, &gs1);
        for (size_t i = 0; i < rows; i++) {
            const float *xi = x + i * D, *goi = go + i * D;
            const double ss = ternary_canon_sumsq(xi, D);
            const float r = (float)(1.0 / sqrt(ss / (double)D + (double)eps));
            double dl[16] = {0};
            for (size_t j = 0; j < D; j++) { dl[j & 15] += (double)g[j] * (double)goi[j] * (double)xi[j]; gg2[j] += goi[j] * scale * xi[j] * r; }
            const double dot = ternary_canon_reduce_d16(dl);
            gs2 += dot * (double)r;
            if (with_gx) {
                const float cc = (float)(dot * (double)r * (double)r / (double)D);
                for (size_t k = 0; k < D; k++) gx2[i * D + k] += scale * r * ((float)g[k] * goi[k] - xi[k] * cc);
            }
        }
        cmpf("rms_bwd gain", gg1, gg2, D, ctx); cmpf("rms_bwd gx", gx1, gx2, rows * D, ctx);
        if (memcmp(&gs1, &gs2, 8)) FAIL("rms_bwd gscale %a vs %a (%s)", gs1, gs2, ctx);
        free(gp); free(g); free(x); free(go); free(o1); free(o2); free(gg1); free(gg2); free(gx1); free(gx2);
    }
}

/* ---- SwiGLU ---- */
static inline void silu_f(float a, float *sv, float *dsv) {
    const float y = -fabsf(a) * 1.44269504088896341f;
    const float e = (y < TERNARY_FEXP_CUTOFF) ? 0.0f : ternary_fexp2_core(y);
    const float sg = (a >= 0.0f) ? 1.0f / (1.0f + e) : e / (1.0f + e);
    *sv = a * sg; *dsv = sg * (1.0f + a * (1.0f - sg));
}
static const size_t HS[] = {1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 64, 100, 129};
#define NHS (sizeof(HS) / sizeof(HS[0]))

static void t_swiglu(void) {
    for (int it = 0; it < 450; it++) {
        const int mode = it % 3;
        const size_t H = HS[rnd64() % NHS];
        const int32_t D = 1 + (int32_t)(rnd64() % 300);
        const int32_t zr = (it % 5 == 0) ? 3 * D : D;        /* a veces fuera de [-D, D] para ejercitar el recorte de la tabla */
        float *silu = FA(2 * D + 1), *dsilu = FA(2 * D + 1);
        for (int32_t z = -D; z <= D; z++) {
            const double a = 0.01 * (double)z, sig = 1.0 / (1.0 + exp(-a));
            silu[z + D] = (float)(a * sig); dsilu[z + D] = (float)(sig * (1.0 + a * (1.0 - sig)));
        }
        int32_t *zg = IA(H), *zu = IA(H); filli(zg, H, zr); filli(zu, H, D);
        if (it % 7 == 0) for (size_t j = 0; j < H; j += 2) zg[j] = 0;
        if (it % 11 == 0) { for (size_t j = 0; j < H; j += 3) zg[j] = INT32_MAX; for (size_t j = 1; j < H; j += 3) zg[j] = INT32_MIN + 5; }
        const float su = 0.003f * (float)(1 + rnd64() % 50);
        char ctx[96]; snprintf(ctx, sizeof ctx, "mode=%d H=%zu D=%d", mode, H, D);
        float *h1 = FA(H + 4), *h2 = FA(H + 4);
        for (size_t i = 0; i < H + 4; i++) h1[i] = h2[i] = 9.0f;
        ternary_neon_swiglu_prod_z(mode, H, zg, zu, su, silu, D, h1);
        for (size_t j = 0; j < H; j++) {
            if (mode == 0) { int64_t k = (int64_t)zg[j] + D; if (k < 0) k = 0; else if (k > 2 * D) k = 2 * D; h2[j] = silu[k] * (su * (float)zu[j]); }
            else if (mode == 2) { const float a = su * (float)zu[j]; h2[j] = a > 0.0f ? a * a : 0.0f; }
            else h2[j] = (float)(sign_i32(zg[j]) * zu[j]);
        }
        cmpf("swiglu_prod_z", h1, h2, H + 4, ctx);
        /* backward z */
        float *gt = FA(H); fillf(gt, H);
        float *gu1 = FA(H + 4), *ga1 = FA(H + 4), *gu2 = FA(H + 4), *ga2 = FA(H + 4);
        for (size_t i = 0; i < H + 4; i++) gu1[i] = gu2[i] = ga1[i] = ga2[i] = 9.0f;
        ternary_neon_swiglu_bwd_z(mode, H, zg, zu, gt, su, silu, dsilu, D, gu1, ga1);
        for (size_t j = 0; j < H; j++) {
            const float u = su * (float)zu[j];
            if (mode == 2) gu2[j] = u > 0.0f ? gt[j] * 2.0f * u : 0.0f;
            else if (mode == 0) { int64_t k = (int64_t)zg[j] + D; if (k < 0) k = 0; else if (k > 2 * D) k = 2 * D; gu2[j] = gt[j] * silu[k]; ga2[j] = gt[j] * u * dsilu[k]; }
            else { gu2[j] = gt[j] * (float)sign_i32(zg[j]); ga2[j] = gt[j] * u; }
        }
        cmpf("swiglu_bwd_z gu", gu1, gu2, H + 4, ctx);
        if (mode != 2) cmpf("swiglu_bwd_z ga", ga1, ga2, H + 4, ctx);
        /* ruta uint8 */
        float *fa = FA(H), *fu = FA(H); fillf(fa, H); fillf(fu, H);
        if (it % 3 == 0) for (size_t j = 0; j < H; j++) { fa[j] *= 40.0f; }       /* |a| grande: exp con y < CUTOFF */
        for (size_t i = 0; i < H + 4; i++) h1[i] = h2[i] = 9.0f;
        ternary_neon_swiglu_fwd_u8(mode, H, fa, fu, zg, h1);
        for (size_t j = 0; j < H; j++) {
            if (mode == 2) { const float a = fu[j]; h2[j] = a > 0.0f ? a * a : 0.0f; }
            else if (mode == 0) { float sv, dsv; silu_f(fa[j], &sv, &dsv); (void)dsv; h2[j] = sv * fu[j]; }
            else h2[j] = (float)sign_i32(zg[j]) * fu[j];
        }
        cmpf("swiglu_fwd_u8", h1, h2, H + 4, ctx);
        const float kg = 0.002f * (float)(1 + rnd64() % 40), ku = 0.002f * (float)(1 + rnd64() % 40);
        for (size_t i = 0; i < H + 4; i++) gu1[i] = gu2[i] = ga1[i] = ga2[i] = 9.0f;
        if (it % 3 == 0) for (size_t j = 0; j < H; j++) zg[j] = rndi(20000);
        ternary_neon_swiglu_bwd_u8(mode, H, zg, zu, gt, kg, ku, gu1, ga1);
        for (size_t j = 0; j < H; j++) {
            const float a = kg * (float)(mode == 2 ? 0 : zg[j]);
            const float u = ku * (float)zu[j];
            if (mode == 2) gu2[j] = u > 0.0f ? gt[j] * 2.0f * u : 0.0f;
            else if (mode == 0) { float sv, dsv; silu_f(a, &sv, &dsv); gu2[j] = gt[j] * sv; ga2[j] = gt[j] * u * dsv; }
            else { gu2[j] = gt[j] * (float)sign_i32(zg[j]); ga2[j] = gt[j] * u; }
        }
        cmpf("swiglu_bwd_u8 gu", gu1, gu2, H + 4, ctx);
        if (mode != 2) cmpf("swiglu_bwd_u8 ga", ga1, ga2, H + 4, ctx);
        free(silu); free(dsilu); free(zg); free(zu); free(h1); free(h2); free(gt); free(gu1); free(gu2); free(ga1); free(ga2); free(fa); free(fu);
    }
}

/* ---- entropia cruzada y dot_gz ---- */
static void t_xent(void) {
    static const size_t VS[] = {1, 2, 3, 4, 5, 15, 16, 17, 31, 33, 64, 100, 257, 512, 1000};
    for (int it = 0; it < 300; it++) {
        const size_t V = VS[rnd64() % (sizeof(VS) / sizeof(VS[0]))], t = rnd64() % V, n = 1 + rnd64() % 50;
        float *l = FA(V); for (size_t v = 0; v < V; v++) l[v] = (float)rndi(4000) * 0.01f * ((it % 4 == 0) ? 5.0f : 1.0f);
        if (it % 5 == 0 && V > 2) l[V - 1] = l[0];        /* empates: el argmax es el primer maximo */
        float *g1 = FA(V + 4), *g2 = FA(V + 4);
        for (size_t i = 0; i < V + 4; i++) g1[i] = g2[i] = 9.0f;
        int ok1 = -1, ok2 = -1; double lo1 = 0;
        char ctx[96]; snprintf(ctx, sizeof ctx, "V=%zu t=%zu n=%zu", V, t, n);
        const int with_g = (it % 3) != 0;
        ternary_neon_xent_row(l, V, t, n, with_g ? g1 : NULL, &ok1, &lo1);
        size_t arg = 0; float m = l[0];
        for (size_t v = 1; v < V; v++) if (l[v] > m) { m = l[v]; arg = v; }
        ok2 = (arg == t);
        double dl[16] = {0};
        for (size_t v = 0; v < V; v++) dl[v & 15] += (double)ternary_expf_nonpos(l[v] - m);
        const double sum = ternary_canon_reduce_d16(dl), lo2 = log(sum) - (double)(l[t] - m);
        if (with_g) {
            const double inv = 1.0 / (sum * (double)n);
            for (size_t v = 0; v < V; v++) g2[v] = (float)((double)ternary_expf_nonpos(l[v] - m) * inv);
            g2[t] -= (float)(1.0 / (double)n);
        }
        if (ok1 != ok2) FAIL("xent ok %d vs %d (%s)", ok1, ok2, ctx);
        if (memcmp(&lo1, &lo2, 8)) FAIL("xent loss %a vs %a (%s)", lo1, lo2, ctx);
        cmpf("xent g", g1, g2, V + 4, ctx);
        free(l); free(g1); free(g2);
    }
    for (int it = 0; it < 300; it++) {
        const size_t n = 1 + rnd64() % 300; float *g = FA(n); int32_t *z = IA(n); fillf(g, n); filli(z, n, 1 << 20);
        const double sx = 0.001 * (double)(1 + rnd64() % 50);
        double d[16] = {0}, e[16] = {0};
        for (size_t j = 0; j < n; j++) { d[j & 15] += (double)g[j] * (double)z[j]; e[j & 15] += (double)g[j] * sx * (double)z[j]; }
        const double r0 = ternary_canon_reduce_d16(d), r1 = ternary_canon_reduce_d16(e);
        const double a0 = ternary_neon_dot_gz(g, z, n, 0, 0.0), a1 = ternary_neon_dot_gz(g, z, n, 1, sx);
        if (memcmp(&a0, &r0, 8)) FAIL("dot_gz %a vs %a n=%zu", a0, r0, n);
        if (memcmp(&a1, &r1, 8)) FAIL("dot_gz sx %a vs %a n=%zu", a1, r1, n);
        free(g); free(z);
    }
}

int main(int argc, char **argv) {
    if (!ternary_neon_enabled()) { printf("NEON no disponible: test omitido\n"); return 0; }
    const char *only = argc > 1 ? argv[1] : "";
#define RUN(name, call) do { if (!*only || !strcmp(only, name)) { const int f0 = fails; call; printf("  %-12s %s\n", name, fails == f0 ? "ok" : "FALLA"); } } while (0)
    RUN("rmsnorm", t_rms());
    RUN("swiglu", t_swiglu());
    RUN("xent_dot", t_xent());
    if (fails) { printf("test_neon_norm: %d fallos\n", fails); return 1; }
    printf("test_neon_norm: OK\n");
    return 0;
}
