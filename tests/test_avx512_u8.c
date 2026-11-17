/* Nucleos AVX-512 de la ruta uint8 contra el escalar: deben dar los MISMOS bits (excepto el signo de un cero).
 * Formas al azar con colas parciales. Sin AVX-512 en la CPU el test se salta. */
#include "ternary_avx512.h"
#include "ternary_u8.h"
#include "ternary_fexp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 88172645463325252ull;
static uint64_t rnd64(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static float rndf(void) { int k = (int)(rnd64() % 7); if (k == 0) return 0.0f; if (k == 1) return -0.0f; return (float)((int)(rnd64() % 2001) - 1000) * 0.0137f; }
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { if (fails < 10) { printf("FALLA (linea %d): ", __LINE__); printf(__VA_ARGS__); printf("\n"); } fails++; } } while (0)
static int same(float a, float b) { return a == b || (a != a && b != b); }   /* -0 == +0 cuenta como igual */

static void test_dense_kernels(void) {
    for (int it = 0; it < 400; it++) {
        const size_t n = 1 + rnd64() % 400, w = (n + 63) / 64;
        uint64_t *wm = calloc(w, 8), *ws = calloc(w, 8);
        int8_t *wv = malloc(n); uint8_t *q = malloc(n); float *xf = malloc(n * 4), *xr = malloc(n * 4);
        for (size_t i = 0; i < n; i++) {
            int v = (int)(rnd64() % 3) - 1; wv[i] = (int8_t)v;
            if (v) { wm[i >> 6] |= 1ull << (i & 63); if (v < 0) ws[i >> 6] |= 1ull << (i & 63); }
            q[i] = (uint8_t)(rnd64() % 256);
        }
        const int32_t zp = (int32_t)(rnd64() % 256);
        CHECK(ternary_avx512_dot_u8(wm, ws, n, q, zp) == ternary_dot_u8_scalar(wm, ws, n, q, zp), "dot_u8 n=%zu", n);
        CHECK(ternary_avx512_dot_u8(wm, ws, n, q, zp) == ternary_dot_u8_reference(wv, n, q, zp), "dot_u8 vs referencia n=%zu", n);
        const float sx = 0.003f * (float)(1 + rnd64() % 50);
        ternary_avx512_u8_to_f32(q, n, sx, zp, xf);
        for (size_t i = 0; i < n; i++) xr[i] = sx * (float)((int32_t)q[i] - zp);
        CHECK(!memcmp(xf, xr, n * 4), "u8_to_f32 n=%zu", n);
        float *a = malloc(n * 4), *b = malloc(n * 4);
        for (size_t i = 0; i < n; i++) a[i] = b[i] = rndf();
        const float g = rndf();
        ternary_avx512_outer_acc(a, n, g, xf);
        for (size_t i = 0; i < n; i++) b[i] += g * xf[i];
        for (size_t i = 0; i < n; i++) CHECK(same(a[i], b[i]), "outer_acc n=%zu i=%zu", n, i);
        for (size_t i = 0; i < n; i++) a[i] = b[i] = rndf();
        const float gs = rndf();
        ternary_avx512_gradx_row(a, n, wm, ws, gs);
        for (size_t i = 0; i < n; i++) if (wv[i]) b[i] += wv[i] < 0 ? -gs : gs;
        for (size_t i = 0; i < n; i++) CHECK(same(a[i], b[i]), "gradx_row n=%zu i=%zu", n, i);
        free(wm); free(ws); free(wv); free(q); free(xf); free(xr); free(a); free(b);
    }
}


static int sgn(int32_t v) { return v > 0 ? 1 : (v < 0 ? -1 : 0); }
static void silu_ref(float a, float *sv, float *dsv) {
    const float y = -fabsf(a) * 1.44269504088896341f;
    const float e = (y < TERNARY_FEXP_CUTOFF) ? 0.0f : ternary_fexp2_core(y);
    const float sg = (a >= 0.0f) ? 1.0f / (1.0f + e) : e / (1.0f + e);
    *sv = a * sg; *dsv = sg * (1.0f + a * (1.0f - sg));
}
static void test_swiglu(void) {
    for (int it = 0; it < 300; it++) {
        const size_t H = 1 + rnd64() % 200; const int32_t D = 1 + (int32_t)(rnd64() % 200);
        int32_t *zg = malloc(H * 4), *zu = malloc(H * 4);
        float *tab = malloc((2 * D + 1) * 4), *dtab = malloc((2 * D + 1) * 4), *fa = malloc(H * 4), *fu = malloc(H * 4), *gt = malloc(H * 4);
        for (int32_t k = 0; k <= 2 * D; k++) { tab[k] = rndf(); dtab[k] = rndf(); }
        for (size_t j = 0; j < H; j++) {
            zg[j] = (int32_t)(rnd64() % (2 * D + 21)) - D - 10; zu[j] = (int32_t)(rnd64() % (2 * D + 1)) - D;
            if (it % 5 == 0 && j % 7 == 0) zg[j] = 0;
            fa[j] = (it % 4 == 0 ? 40.0f : 3.0f) * rndf(); fu[j] = rndf(); gt[j] = rndf();
        }
        const float su = 0.01f * (float)(1 + rnd64() % 30), sx = 0.02f * (float)(1 + rnd64() % 9), sg_ = 0.013f * (float)(1 + rnd64() % 30);
        for (int mode = 0; mode < 3; mode++) {
            float *a = malloc(H * 4), *b = malloc(H * 4), *ga = malloc(H * 4), *gb = malloc(H * 4), *ua = malloc(H * 4), *ub = malloc(H * 4);
            /* forward con tablas */
            ternary_avx512_swiglu_prod_z(mode, H, zg, zu, su, tab, D, a);
            for (size_t j = 0; j < H; j++) {
                if (mode == 0) { int64_t k = (int64_t)zg[j] + D; if (k < 0) k = 0; else if (k > 2 * D) k = 2 * D; b[j] = tab[k] * (su * (float)zu[j]); }
                else if (mode == 2) { const float x = su * (float)zu[j]; b[j] = x > 0.0f ? x * x : 0.0f; }
                else b[j] = (float)(sgn(zg[j]) * zu[j]);
            }
            for (size_t j = 0; j < H; j++) CHECK(same(a[j], b[j]), "prod_z mode=%d H=%zu j=%zu", mode, H, j);
            /* backward con tablas */
            memset(ga, 0, H * 4); memset(gb, 0, H * 4); memset(ua, 0, H * 4); memset(ub, 0, H * 4);
            ternary_avx512_swiglu_bwd_z(mode, H, zg, zu, gt, su, tab, dtab, D, ua, ga);
            for (size_t j = 0; j < H; j++) {
                const float u = su * (float)zu[j];
                if (mode == 2) ub[j] = u > 0.0f ? gt[j] * 2.0f * u : 0.0f;
                else if (mode == 0) { int64_t k = (int64_t)zg[j] + D; if (k < 0) k = 0; else if (k > 2 * D) k = 2 * D; ub[j] = gt[j] * tab[k]; gb[j] = gt[j] * u * dtab[k]; }
                else { ub[j] = gt[j] * (float)sgn(zg[j]); gb[j] = gt[j] * u; }
            }
            for (size_t j = 0; j < H; j++) { CHECK(same(ua[j], ub[j]), "bwd_z gu mode=%d j=%zu", mode, j); if (mode != 2) CHECK(same(ga[j], gb[j]), "bwd_z ga mode=%d j=%zu", mode, j); }
            /* forward uint8 */
            ternary_avx512_swiglu_fwd_u8(mode, H, fa, fu, zg, a);
            for (size_t j = 0; j < H; j++) {
                if (mode == 2) b[j] = fu[j] > 0.0f ? fu[j] * fu[j] : 0.0f;
                else if (mode == 0) { float sv, dsv; silu_ref(fa[j], &sv, &dsv); b[j] = sv * fu[j]; }
                else b[j] = (float)sgn(zg[j]) * fu[j];
            }
            for (size_t j = 0; j < H; j++) CHECK(same(a[j], b[j]), "fwd_u8 mode=%d H=%zu j=%zu a=%g fa=%g", mode, H, j, a[j], fa[j]);
            /* backward uint8 */
            memset(ga, 0, H * 4); memset(gb, 0, H * 4); memset(ua, 0, H * 4); memset(ub, 0, H * 4);
            ternary_avx512_swiglu_bwd_u8(mode, H, zg, zu, gt, sg_ * sx, su * sx, ua, ga);
            for (size_t j = 0; j < H; j++) {
                const float aa = sg_ * sx * (float)zg[j], u = su * sx * (float)zu[j];
                if (mode == 2) ub[j] = u > 0.0f ? gt[j] * 2.0f * u : 0.0f;
                else if (mode == 0) { float sv, dsv; silu_ref(aa, &sv, &dsv); ub[j] = gt[j] * sv; gb[j] = gt[j] * u * dsv; }
                else { ub[j] = gt[j] * (float)sgn(zg[j]); gb[j] = gt[j] * u; }
            }
            for (size_t j = 0; j < H; j++) { CHECK(same(ua[j], ub[j]), "bwd_u8 gu mode=%d j=%zu", mode, j); if (mode != 2) CHECK(same(ga[j], gb[j]), "bwd_u8 ga mode=%d j=%zu", mode, j); }
            free(a); free(b); free(ga); free(gb); free(ua); free(ub);
        }
        free(zg); free(zu); free(tab); free(dtab); free(fa); free(fu); free(gt);
    }
}

static void test_xent_head(void) {
    for (int it = 0; it < 300; it++) {
        const size_t V = 1 + rnd64() % 300;
        float *l = malloc(V * 4), *ga = malloc(V * 4), *gb = malloc(V * 4); int32_t *z = malloc(V * 4);
        for (size_t v = 0; v < V; v++) {
            l[v] = (it % 3 == 0 ? 30.0f : 4.0f) * rndf(); z[v] = (int32_t)(rnd64() % 2001) - 1000;
            if (it % 7 == 0 && v % 5 == 0) l[v] = 1.5f;   /* empates en el maximo */
        }
        const size_t t = rnd64() % V, n = 1 + rnd64() % 50;
        int oka = 0, okb = 0; double la, lb;
        ternary_avx512_xent_row(l, V, t, n, ga, &oka, &la);
        /* referencia escalar (mismo orden canonico) */
        size_t arg = 0; float m = l[0];
        for (size_t v = 1; v < V; v++) if (l[v] > m) { m = l[v]; arg = v; }
        okb = (arg == t);
        double dl[16] = {0};
        for (size_t v = 0; v < V; v++) dl[v & 15] += (double)ternary_expf_nonpos(l[v] - m);
        const double sum = ternary_canon_reduce_d16(dl);
        lb = log(sum) - (double)(l[t] - m);
        const double inv = 1.0 / (sum * (double)n);
        for (size_t v = 0; v < V; v++) gb[v] = (float)((double)ternary_expf_nonpos(l[v] - m) * inv);
        gb[t] -= (float)(1.0 / (double)n);
        CHECK(oka == okb && la == lb, "xent_row V=%zu loss %.17g vs %.17g ok %d %d", V, la, lb, oka, okb);
        for (size_t v = 0; v < V; v++) CHECK(same(ga[v], gb[v]), "xent_row g V=%zu v=%zu", V, v);
        const double sx = 0.0123 * (double)(1 + rnd64() % 20);
        CHECK(ternary_avx512_dot_gz(ga, z, V, 0, 0.0) == ternary_canon_dot_gz(ga, z, V), "dot_gz V=%zu", V);
        CHECK(ternary_avx512_dot_gz(ga, z, V, 1, sx) == ternary_canon_dot_gsz(ga, z, V, sx), "dot_gsz V=%zu", V);
        const float sc = rndf();
        memcpy(gb, ga, V * 4); ternary_avx512_scale_f32(ga, V, sc);
        for (size_t v = 0; v < V; v++) CHECK(same(ga[v], gb[v] * sc), "scale_f32 V=%zu v=%zu", V, v);
        free(l); free(ga); free(gb); free(z);
    }
    /* la exponencial compartida frente a expf: error relativo pequeno */
    double worst = 0;
    for (int i = 0; i < 20000; i++) { float x = -(float)(rnd64() % 600000) / 10000.0f; double r = (double)ternary_expf_nonpos(x), e = exp((double)x); if (e > 1e-9) { double rel = fabs(r - e) / e; if (rel > worst) worst = rel; } }
    CHECK(worst < 1e-6, "expf_nonpos error relativo %.3g", worst);
}

static void test_misc(void) {
    for (int it = 0; it < 300; it++) {
        const size_t n = 1 + rnd64() % 300;
        float *x = malloc(n * 4);
        for (size_t i = 0; i < n; i++) x[i] = rndf() * (it % 5 == 0 ? 500.0f : 1.0f);
        if (it % 11 == 0) for (size_t i = 0; i < n; i++) x[i] = 0.0f;
        /* cuantizacion uint8 */
        TernaryU8 *a = ternary_u8_alloc(1, n), *b = ternary_u8_alloc(1, n);
        ternary_avx512_u8_quantize_row(x, n, a->q, a->scale, a->zp);
        {   /* referencia: la de ternary_u8_quantize, copiada */
            float lo = 0.0f, hi = 0.0f;
            for (size_t i = 0; i < n; i++) { if (x[i] < lo) lo = x[i]; if (x[i] > hi) hi = x[i]; }
            float sc = (hi - lo) / 255.0f;
            if (!(sc > 0.0f) || !isfinite(sc)) { b->scale[0] = 1.0f; b->zp[0] = 0; memset(b->q, 0, n); }
            else {
                int zp = (int)rintf(-lo / sc); if (zp < 0) zp = 0; if (zp > 255) zp = 255;
                b->scale[0] = sc; b->zp[0] = zp; const float inv = 1.0f / sc;
                for (size_t i = 0; i < n; i++) { float qf = rintf(x[i] * inv) + (float)zp; if (qf < 0.0f) qf = 0.0f; if (qf > 255.0f) qf = 255.0f; b->q[i] = (uint8_t)qf; }
            }
        }
        CHECK(a->scale[0] == b->scale[0] && a->zp[0] == b->zp[0] && !memcmp(a->q, b->q, n), "u8_quantize_row n=%zu", n);
        ternary_u8_free(a); ternary_u8_free(b);
        /* residual */
        float *y = malloc(n * 4), *o1 = malloc(n * 4), *o2 = malloc(n * 4), *d1 = malloc(n * 4), *d2 = malloc(n * 4);
        for (size_t i = 0; i < n; i++) { y[i] = rndf(); d1[i] = d2[i] = rndf(); }
        ternary_avx512_add_f32(o1, x, y, n); ternary_avx512_acc_add2(d1, x, y, n);
        for (size_t i = 0; i < n; i++) { o2[i] = x[i] + y[i]; d2[i] += x[i] + y[i]; }
        for (size_t i = 0; i < n; i++) CHECK(same(o1[i], o2[i]) && same(d1[i], d2[i]), "residual n=%zu i=%zu", n, i);
        /* softmax entero */
        const size_t tl = 2 * (1 + rnd64() % 400) + 1;
        float *tab = malloc(tl * 4); for (size_t k = 0; k < tl; k++) tab[k] = (float)exp(-(double)k * 0.05);
        int32_t *sc = malloc(n * 4); for (size_t i = 0; i < n; i++) sc[i] = (int32_t)(rnd64() % 900) - 450;
        ternary_avx512_softmax_fwd(sc, n, tab, tl, o1);
        { int32_t m = sc[0]; for (size_t i = 1; i < n; i++) if (sc[i] > m) m = sc[i];
          float part[16] = {0};
          for (size_t i = 0; i < n; i++) { int64_t k = (int64_t)m - sc[i]; if (k >= (int64_t)tl) k = (int64_t)tl - 1; o2[i] = tab[k]; part[i & 15] += o2[i]; }
          const float inv = 1.0f / ternary_sum16(part); for (size_t i = 0; i < n; i++) o2[i] *= inv; }
        for (size_t i = 0; i < n; i++) CHECK(same(o1[i], o2[i]), "softmax_fwd n=%zu i=%zu", n, i);
        ternary_avx512_softmax_bwd(o1, y, n, 0.177f, d1);
        { float part[16] = {0}; for (size_t j = 0; j < n; j++) part[j & 15] += y[j] * o1[j];
          const float dot = ternary_sum16(part); for (size_t i = 0; i < n; i++) d2[i] = 0.177f * o1[i] * (y[i] - dot); }
        for (size_t i = 0; i < n; i++) CHECK(same(d1[i], d2[i]), "softmax_bwd n=%zu i=%zu", n, i);
        /* utilidades de GQA */
        int32_t *z = malloc(n * 4), *r1 = malloc(n * 4), *r2 = malloc(n * 4);
        for (size_t i = 0; i < n; i++) z[i] = (int32_t)(rnd64() % 200001) - 100000;
        ternary_avx512_shift_round_i32(z, n, 8, r1);
        for (size_t i = 0; i < n; i++) r2[i] = (z[i] + (1 << 7)) >> 8;
        CHECK(!memcmp(r1, r2, n * 4), "shift_round n=%zu", n);
        const size_t stride = 1 + rnd64() % 5;
        float *f1 = calloc(n * stride + 1, 4), *f2 = calloc(n * stride + 1, 4);
        ternary_avx512_sign_f32(z, n, f1, stride);
        for (size_t i = 0; i < n; i++) f2[i * stride] = (float)sgn(z[i]);
        CHECK(!memcmp(f1, f2, (n * stride + 1) * 4), "sign_f32 n=%zu stride=%zu", n, stride);
        const float kk = 0.0137f * (float)(1 + rnd64() % 9);
        ternary_avx512_scale_i32_f32(z, n, kk, f1, stride);
        for (size_t i = 0; i < n; i++) f2[i * stride] = kk * (float)z[i];
        CHECK(!memcmp(f1, f2, (n * stride + 1) * 4), "scale_i32_f32 n=%zu stride=%zu", n, stride);
        int32_t *i1 = calloc(n * stride + 1, 4), *i2 = calloc(n * stride + 1, 4);
        ternary_avx512_scatter_i32(z, n, i1, stride);
        for (size_t i = 0; i < n; i++) i2[i * stride] = z[i];
        CHECK(!memcmp(i1, i2, (n * stride + 1) * 4), "scatter_i32 n=%zu stride=%zu", n, stride);
        free(x); free(y); free(o1); free(o2); free(d1); free(d2); free(tab); free(sc); free(z); free(r1); free(r2); free(f1); free(f2); free(i1); free(i2);
    }
    /* RoPE */
    for (int it = 0; it < 200; it++) {
        const size_t P = 1 + rnd64() % 40, H = 2 * P + 2 * (rnd64() % 4), heads = 1 + rnd64() % 4;
        int16_t *c = malloc(P * 2), *sn = malloc(P * 2);
        for (size_t i = 0; i < P; i++) { c[i] = (int16_t)((int)(rnd64() % 2001) - 1000) * 16; sn[i] = (int16_t)((int)(rnd64() % 2001) - 1000) * 16; }
        int32_t *z = malloc(heads * H * 4), *o1 = malloc(heads * H * 4), *o2 = malloc(heads * H * 4);
        float *g1 = malloc(heads * H * 4), *g2 = malloc(heads * H * 4);
        for (size_t i = 0; i < heads * H; i++) { z[i] = (int32_t)(rnd64() % 20001) - 10000; o1[i] = o2[i] = 7; g1[i] = g2[i] = rndf(); }
        ternary_avx512_rope_rotate_int(z, heads, H, P, c, sn, o1);
        for (size_t h = 0; h < heads; h++) for (size_t i = 0; i < P; i++) {
            int32_t a = z[h * H + 2 * i], b = z[h * H + 2 * i + 1];
            o2[h * H + 2 * i] = a * c[i] - b * sn[i]; o2[h * H + 2 * i + 1] = a * sn[i] + b * c[i]; }
        CHECK(!memcmp(o1, o2, heads * H * 4), "rope_rotate P=%zu H=%zu heads=%zu", P, H, heads);
        const float inv = 1.0f / 32768.0f;
        ternary_avx512_rope_unrotate_grad(g1, heads, H, P, c, sn, inv);
        for (size_t h = 0; h < heads; h++) for (size_t i = 0; i < P; i++) {
            float ga = g2[h * H + 2 * i], gb = g2[h * H + 2 * i + 1], cf = c[i] * inv, sf = sn[i] * inv;
            g2[h * H + 2 * i] = ga * cf + gb * sf; g2[h * H + 2 * i + 1] = -ga * sf + gb * cf; }
        for (size_t i = 0; i < heads * H; i++) CHECK(same(g1[i], g2[i]), "rope_unrotate P=%zu H=%zu i=%zu", P, H, i);
        free(c); free(sn); free(z); free(o1); free(o2); free(g1); free(g2);
    }
}

static void test_bits(void) {
    for (int it = 0; it < 300; it++) {
        const size_t n = 1 + rnd64() % 300, bit0 = rnd64() % 200, tot = bit0 + n + 70, w = (tot + 63) / 64;
        uint64_t *m1 = malloc(w * 8), *s1 = malloc(w * 8), *m2 = malloc(w * 8), *s2 = malloc(w * 8);
        for (size_t k = 0; k < w; k++) { m1[k] = m2[k] = rnd64(); s1[k] = s2[k] = rnd64(); }
        int32_t *z = malloc(n * 4); for (size_t i = 0; i < n; i++) z[i] = (int32_t)(rnd64() % 5) - 2;
        ternary_avx512_pack_sign_i32(z, n, m1, s1, bit0);
        for (size_t i = 0; i < n; i++) {
            const size_t b = bit0 + i; const uint64_t bb = 1ull << (b & 63);
            m2[b >> 6] = (m2[b >> 6] & ~bb) | (z[i] != 0 ? bb : 0); s2[b >> 6] = (s2[b >> 6] & ~bb) | (z[i] < 0 ? bb : 0); }
        CHECK(!memcmp(m1, m2, w * 8) && !memcmp(s1, s2, w * 8), "pack_sign_i32 n=%zu bit0=%zu", n, bit0);
        /* expand_trits desde el bit 0 */
        uint64_t *pm = calloc(w, 8), *ps = calloc(w, 8); float *f1 = malloc(n * 4);
        for (size_t i = 0; i < n; i++) { int v = (int)(rnd64() % 3) - 1; if (v) { pm[i >> 6] |= 1ull << (i & 63); if (v < 0) ps[i >> 6] |= 1ull << (i & 63); } }
        ternary_avx512_expand_trits(pm, ps, n, f1);
        for (size_t i = 0; i < n; i++) { const int m = (int)((pm[i >> 6] >> (i & 63)) & 1u), sg = (int)((ps[i >> 6] >> (i & 63)) & 1u); CHECK(f1[i] == (float)(m * (1 - 2 * sg)), "expand_trits n=%zu i=%zu", n, i); }
        free(m1); free(s1); free(m2); free(s2); free(z); free(pm); free(ps); free(f1);
    }
}

int main(void) {
    if (!ternary_avx512_enabled()) { printf("test_avx512_u8: sin AVX-512, se salta\n"); return 0; }
    test_dense_kernels();
    test_swiglu();
    test_xent_head();
    test_misc();
    test_bits();
    if (fails) { printf("test_avx512_u8: %d FALLOS\n", fails); return 1; }
    printf("test_avx512_u8: OK\n");
    return 0;
}
