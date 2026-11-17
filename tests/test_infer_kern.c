/* Nucleo NEON del motor contra el escalar: matvec, matmul, argmax y puntajes con formas y valores de borde (enteros: igualdad exacta). */
#include "../src/ternary_infer_internal.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

static uint64_t rs = 0x9E3779B97F4A7C15ULL;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 32); }
static int fails = 0, total = 0;
#define CHECK(c, ...) do { total++; if (!(c)) { fails++; printf("FALLO: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

int main(void) {
    const TernaryInferKernels *N = &ternaryai_infer_k_neon, *S = &ternaryai_infer_k_scalar;
    if (!N->matvec || !ternaryai_infer_cpu_has_neon()) { printf("NEON no disponible: test omitido\n"); return 0; }
    const int rows_l[] = { 1, 3, 4, 5, 16, 17, 31, 64, 100 }, cols_l[] = { 1, 15, 16, 17, 48, 63, 64, 65, 80, 128, 130, 192, 300, 1024, 1100, 2050 };
    for (int ri = 0; ri < 9; ri++)
        for (int ci = 0; ci < 16; ci++)
            for (int mode = 0; mode < 3; mode++) {
                const int rows = rows_l[ri], cols = cols_l[ci];
                int8_t *w = malloc((size_t)rows * cols);
                for (int i = 0; i < rows * cols; i++) w[i] = mode == 0 ? (int8_t)((int)(rnd() % 3) - 1) : mode == 1 ? 1 : -1;
                TernaryInferMat ms, mn;
                CHECK(ternaryai_infer_mat_build(&ms, w, rows, cols, 1.0f, TERNARYAI_INFER_LAYOUT_ROW) && ternaryai_infer_mat_build(&mn, w, rows, cols, 1.0f, TERNARYAI_INFER_LAYOUT_NEON), "build");
                const int cp = ms.cp;
                uint8_t *q[5]; int32_t zp[5]; int32_t *zs[5], *zn[5];
                for (int b = 0; b < 5; b++) {
                    q[b] = malloc((size_t)cp + 16); zs[b] = calloc((size_t)ms.rb * 16 + 16, 4); zn[b] = calloc((size_t)ms.rb * 16 + 16, 4);
                    for (int i = 0; i < cp; i++) q[b][i] = (mode == 1 && b == 0) ? 255 : (uint8_t)rnd();
                    zp[b] = (int32_t)(rnd() % 256);
                }
                for (int b = 0; b < 5; b++) {
                    S->matvec(&ms, q[b], zp[b], zs[b]); N->matvec(&mn, q[b], zp[b], zn[b]);
                    CHECK(!memcmp(zs[b], zn[b], (size_t)rows * 4), "matvec rows=%d cols=%d mode=%d b=%d", rows, cols, mode, b);
                    CHECK(S->argmax(zs[b], rows) == N->argmax(zn[b], rows), "argmax rows=%d", rows);
                    memset(zn[b], 0, (size_t)rows * 4);
                }
                for (int B = 1; B <= 5; B++) {
                    N->matmul(&mn, (const uint8_t *const *)q, zp, zn, B);
                    for (int b = 0; b < B; b++) { S->matvec(&ms, q[b], zp[b], zs[b]); CHECK(!memcmp(zs[b], zn[b], (size_t)rows * 4), "matmul rows=%d cols=%d B=%d b=%d", rows, cols, B, b); }
                }
                for (int b = 0; b < 5; b++) { free(q[b]); free(zs[b]); free(zn[b]); }
                ternaryai_infer_mat_free(&ms); ternaryai_infer_mat_free(&mn);
                /* cambio de orden en el sitio: ROW -> NEON -> ROW conserva los pesos */
                if (mode == 0) {
                    TernaryInferMat mr; ternaryai_infer_mat_build(&mr, w, rows, cols, 1.0f, TERNARYAI_INFER_LAYOUT_ROW);
                    TernaryInferMat mc; ternaryai_infer_mat_build(&mc, w, rows, cols, 1.0f, TERNARYAI_INFER_LAYOUT_ROW);
                    CHECK(ternaryai_infer_mat_relayout(&mc, TERNARYAI_INFER_LAYOUT_NEON) && ternaryai_infer_mat_relayout(&mc, TERNARYAI_INFER_LAYOUT_ROW), "relayout");
                    CHECK(!memcmp(mr.w, mc.w, (size_t)mr.rb * 16 * mr.cp / 4), "relayout ida y vuelta rows=%d cols=%d", rows, cols);
                    ternaryai_infer_mat_free(&mr); ternaryai_infer_mat_free(&mc);
                }
                free(w);
            }
    /* puntajes */
    {
        const int hds[] = { 4, 8, 16, 32, 64 }, cnts[] = { 1, 3, 4, 5, 16, 17, 100 }, Gs[] = { 1, 2, 3, 4, 8 };
        const int lims[] = { 3000, 2147483647 };
        for (int hi = 0; hi < 5; hi++) for (int ci = 0; ci < 7; ci++) for (int gi = 0; gi < 5; gi++) for (int li = 0; li < 2; li++) for (int big = 0; big < 2; big++) {
            const int hd = hds[hi], cnt = cnts[ci], G = Gs[gi], tp = (cnt + 16 + 15) & ~15, start = (int)(rnd() % 8), shift = (int)(rnd() % 4);
            int32_t *kT = calloc((size_t)hd * (tp + 8), 4); int32_t *q[8], *a[8], *b[8]; int64_t *scr = malloc((size_t)(cnt + 8) * 8);
            for (int i = 0; i < hd * (tp + 8); i++) kT[i] = big ? (int32_t)rnd() : (int32_t)(rnd() % 20001) - 10000;
            for (int g = 0; g < G; g++) { q[g] = malloc(hd * 4); a[g] = malloc((cnt + 4) * 4); b[g] = malloc((cnt + 4) * 4);
                for (int d = 0; d < hd; d++) q[g][d] = big ? (int32_t)rnd() : (int32_t)(rnd() % 20001) - 10000; }
            const int cnt2 = (start + cnt <= tp + 8) ? cnt : 1;
            S->scores(kT, tp + 8, hd, start, cnt2, (const int32_t *const *)q, G, shift, lims[li], a, scr);
            N->scores(kT, tp + 8, hd, start, cnt2, (const int32_t *const *)q, G, shift, lims[li], b, scr);
            for (int g = 0; g < G; g++) CHECK(!memcmp(a[g], b[g], (size_t)cnt2 * 4), "scores hd=%d cnt=%d G=%d lim=%d big=%d", hd, cnt2, G, lims[li], big);
            for (int g = 0; g < G; g++) { free(q[g]); free(a[g]); free(b[g]); }
            free(kT); free(scr);
        }
    }
    /* partes float del motor: NEON (hibrido) contra formulas escalares, igualdad exacta (datos sin denormales) */
    {
        const TernaryInferKernels *Hy = &ternaryai_infer_k_neon_hybrid;
        enum { KT = 1200 };
        static float tab[KT + 1];
        for (int k = 0; k <= KT; k++) tab[k] = expf(-(float)k / 64.0f);
        const int cnts[] = { 1, 2, 3, 4, 15, 16, 17, 31, 33, 64, 100, 129, 320 };
        for (int ci = 0; ci < 13; ci++) for (int rep = 0; rep < 20; rep++) {
            const int cnt = cnts[ci];
            int32_t sc[330]; float e1[340], e2[340];
            for (int j = 0; j < cnt; j++) sc[j] = (int32_t)(rnd() % 6000) - 3000;
            if (rep == 0) for (int j = 0; j < cnt; j++) sc[j] = 7;          /* todos iguales */
            int32_t mmax = sc[0]; for (int j = 1; j < cnt; j++) if (sc[j] > mmax) mmax = sc[j];
            float part[16] = { 0 };
            for (int j = 0; j < cnt; j++) { const uint32_t k = (uint32_t)(mmax - sc[j]); const float e = tab[k < KT ? k : KT]; e1[j] = e; part[j & 15] += e; }
            float t8[8], t4[4]; for (int l = 0; l < 8; l++) t8[l] = part[l] + part[l + 8]; for (int l = 0; l < 4; l++) t4[l] = t8[l] + t8[l + 4];
            const float tot = (t4[0] + t4[2]) + (t4[1] + t4[3]);
            const float tn = Hy->softmax(sc, cnt, tab, KT, e2);
            CHECK(!memcmp(&tot, &tn, 4) && !memcmp(e1, e2, (size_t)cnt * 4), "softmax cnt=%d", cnt);
        }
        const int hds[] = { 2, 4, 6, 8, 12, 16, 24, 32, 64 };
        for (int hi = 0; hi < 9; hi++) for (int nc = 1; nc <= 8; nc++) for (int ci = 0; ci < 6; ci++) {
            const int hd = hds[hi], cnt = (int[]){ 1, 2, 5, 16, 33, 100 }[ci];
            float *ee[8], *a1[8], *a2[8]; int8_t *v[8]; float vs[128];
            for (int j = 0; j < cnt; j++) vs[j] = (float)(1 + rnd() % 1000) / 4096.0f;
            for (int c = 0; c < nc; c++) {
                ee[c] = malloc(cnt * 4); v[c] = malloc((size_t)cnt * hd); a1[c] = malloc(hd * 4); a2[c] = malloc(hd * 4);
                for (int j = 0; j < cnt; j++) ee[c][j] = (float)(rnd() % 1000) / 1000.0f;
                for (int i = 0; i < cnt * hd; i++) v[c][i] = (int8_t)(rnd() % 256);
                for (int d = 0; d < hd; d++) a1[c][d] = a2[c][d] = (float)(rnd() % 100) / 100.0f;
                for (int j = 0; j < cnt; j++) for (int d = 0; d < hd; d++) { const float vv = (float)v[c][(size_t)j * hd + d] * vs[j]; a1[c][d] += ee[c][j] * vv; }
            }
            Hy->pv_multi(nc, (const float *const *)ee, cnt, (const int8_t *const *)v, vs, hd, a2);
            for (int c = 0; c < nc; c++) { CHECK(!memcmp(a1[c], a2[c], (size_t)hd * 4), "pv hd=%d nc=%d cnt=%d", hd, nc, cnt); free(ee[c]); free(v[c]); free(a1[c]); free(a2[c]); }
        }
        for (int rep = 0; rep < 400; rep++) {
            const int n = 1 + (int)(rnd() % 300);
            int32_t z[300]; float h1[300], h2[300], x[300];
            const float k = (float)(1 + rnd() % 1000) / 100000.0f;
            for (int j = 0; j < n; j++) z[j] = (int32_t)(rnd() % 200001) - 100000;
            for (int j = 0; j < n; j++) { const float a = k * (float)z[j]; const float r = a > 0.0f ? a : 0.0f; h1[j] = r * r; }
            Hy->relu2(z, n, k, h2);
            CHECK(!memcmp(h1, h2, (size_t)n * 4), "relu2 n=%d", n);
            for (int j = 0; j < n; j++) x[j] = ((float)(int32_t)rnd() / 2147483648.0f) * (rep % 3 == 0 ? 50.0f : 1.0f) + (rep % 5 == 0 ? 3.0f : 0.0f);
            if (rep % 7 == 0) for (int j = 0; j < n; j++) x[j] = 0.0f;
            uint8_t q1[304], q2[304]; float s1, s2; int32_t z1, z2;
            ternaryai_infer_set_backend(TERNARYAI_INFER_BACKEND_SCALAR);
            ternaryai_infer_quantize_row(x, n, q1, &s1, &z1);
            Hy->quant(x, n, q2, &s2, &z2);
            CHECK(!memcmp(q1, q2, (size_t)n) && !memcmp(&s1, &s2, 4) && z1 == z2, "quant n=%d", n);
        }
    }
    /* puntajes int16 planos: igual que scores16_planar del motor (copia aqui), con bordes */
    {
        const TernaryInferKernels *Hy = &ternaryai_infer_k_neon_hybrid;
        const int hds[] = { 2, 4, 6, 8, 16, 32, 64, 128, 256 }, cnts[] = { 1, 2, 7, 8, 9, 15, 16, 17, 100, 129, 320 }, Gs[] = { 1, 2, 3, 4, 5, 7, 8 };
        for (int hi = 0; hi < 9; hi++) for (int ci = 0; ci < 11; ci++) for (int gi = 0; gi < 7; gi++) for (int rep = 0; rep < 3; rep++) {
            const int hd = hds[hi], cnt = cnts[ci], G = Gs[gi], tp = (cnt + 40) & ~7, start = (int)(rnd() % 8), shift = (int)(rnd() % 5);
            const int32_t lim = rep == 2 ? 2147483647 : 3000;
            const int amp = rep == 1 ? 32767 : 120;                    /* rep 1: valores grandes (el acumulador sigue sin desbordar: hd*amp*amp < 2^31 cuando hd <= 2) */
            const int usedamp = (rep == 1 && (int64_t)hd * amp * amp >= 2147483647LL) ? 3000 : amp;
            int16_t *k16 = malloc((size_t)hd * tp * 2); int32_t *q[8], *a[8], *b[8];
            for (int i = 0; i < hd * tp; i++) k16[i] = (int16_t)((int)(rnd() % (2 * usedamp + 1)) - usedamp);
            for (int g = 0; g < G; g++) { q[g] = malloc(hd * 4); a[g] = malloc((cnt + 8) * 4); b[g] = malloc((cnt + 8) * 4);
                for (int d = 0; d < hd; d++) q[g][d] = (int32_t)((int)(rnd() % (2 * usedamp + 1)) - usedamp); }
            const int sh = 16 + shift > 31 ? 31 : 16 + shift;
            for (int g = 0; g < G; g++) {
                for (int j = 0; j < cnt; j++) a[g][j] = 0;
                for (int d = 0; d < hd; d++) { const int32_t qd = (int16_t)q[g][d]; const int16_t *row = k16 + (size_t)d * tp + start; for (int j = 0; j < cnt; j++) a[g][j] += qd * (int32_t)row[j]; }
                for (int j = 0; j < cnt; j++) { int32_t v = a[g][j] >> sh; if (v > lim) v = lim; else if (v < -lim) v = -lim; a[g][j] = v; }
            }
            Hy->scores16p(k16, tp, hd, start, cnt, (const int32_t *const *)q, G, shift, lim, b);
            for (int g = 0; g < G; g++) CHECK(!memcmp(a[g], b[g], (size_t)cnt * 4), "scores16p hd=%d cnt=%d G=%d rep=%d", hd, cnt, G, rep);
            for (int g = 0; g < G; g++) { free(q[g]); free(a[g]); free(b[g]); }
            free(k16);
        }
    }
    printf("%d/%d pruebas correctas\n", total - fails, total);
    return fails ? 1 : 0;
}
