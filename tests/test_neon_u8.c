/* Nucleos NEON de la ruta uint8 contra el escalar: deben dar los MISMOS bits (el signo de un cero puede diferir solo
 * donde el AVX-512 tambien lo permite; aqui se exige igualdad de bits salvo -0 == +0). Formas al azar con colas
 * parciales. Sin NEON en la compilacion o en la CPU el test se salta. */
#include "ternary_neon.h"
#include "ternary_u8.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 88172645463325252ull;
static uint64_t rnd64(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static float rndf(void) { int k = (int)(rnd64() % 7); if (k == 0) return 0.0f; if (k == 1) return -0.0f; return (float)((int)(rnd64() % 2001) - 1000) * 0.0137f; }
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { if (fails < 10) { printf("FALLA (linea %d): ", __LINE__); printf(__VA_ARGS__); printf("\n"); } fails++; } } while (0)
static int same(float a, float b) { return a == b || (a != a && b != b); }

int main(void) {
    if (!ternary_neon_enabled()) { printf("NEON no disponible: test omitido\n"); return 0; }
    /* Longitudes que cruzan los limites: palabra de 64, tanda de 32 palabras (2048) y mas. */
    static const size_t fixed[] = {1, 3, 4, 15, 16, 17, 63, 64, 65, 127, 128, 2047, 2048, 2049, 4096, 4097, 70000};
    for (int it = 0; it < 600; it++) {
        const size_t n = (it < (int)(sizeof(fixed) / sizeof(fixed[0]))) ? fixed[it] : 1 + rnd64() % 400;
        const size_t w = (n + 63) / 64;
        uint64_t *wm = calloc(w, 8), *ws = calloc(w, 8);
        int8_t *wv = malloc(n);
        uint8_t *qa = malloc(n + 1);          /* sin relleno: una lectura mas alla de n lo delataria bajo valgrind/ASan */
        float *xf = malloc(n * 4), *xr = malloc(n * 4);
        const int dense = (it % 5 == 0);      /* todos los pesos activos y todos los bytes 255: el peor caso de desbordamiento */
        for (size_t i = 0; i < n; i++) {
            int v = dense ? ((i & 1) ? 1 : -1) : (int)(rnd64() % 3) - 1;
            if (dense && it % 10 == 0) v = 1;
            wv[i] = (int8_t)v;
            if (v) { wm[i >> 6] |= 1ull << (i & 63); if (v < 0) ws[i >> 6] |= 1ull << (i & 63); }
            qa[i] = dense ? 255 : (uint8_t)(rnd64() % 256);
        }
        const int32_t zp = (int32_t)(rnd64() % 256);
        const int32_t a = ternary_neon_dot_u8(wm, ws, n, qa, zp);
        CHECK(a == ternary_dot_u8_scalar(wm, ws, n, qa, zp), "dot_u8 n=%zu", n);
        CHECK(a == ternary_dot_u8_reference(wv, n, qa, zp), "dot_u8 vs referencia n=%zu", n);

        const float sx = 0.003f * (float)(1 + rnd64() % 50);
        ternary_neon_u8_to_f32(qa, n, sx, zp, xf);
        for (size_t i = 0; i < n; i++) xr[i] = sx * (float)((int32_t)qa[i] - zp);
        CHECK(!memcmp(xf, xr, n * 4), "u8_to_f32 n=%zu", n);

        float *x1 = malloc(n * 4), *x2 = malloc(n * 4);
        for (size_t i = 0; i < n; i++) x1[i] = x2[i] = rndf();
        const float g = rndf();
        ternary_neon_outer_acc(x1, n, g, xf);
        for (size_t i = 0; i < n; i++) x2[i] += g * xf[i];
        for (size_t i = 0; i < n; i++) CHECK(same(x1[i], x2[i]) && !memcmp(&x1[i], &x2[i], 4), "outer_acc n=%zu i=%zu (%a vs %a)", n, i, x1[i], x2[i]);

        for (size_t i = 0; i < n; i++) x1[i] = x2[i] = rndf();
        const float gs = rndf();
        ternary_neon_gradx_row(x1, n, wm, ws, gs);
        for (size_t i = 0; i < n; i++) if (wv[i]) x2[i] += wv[i] < 0 ? -gs : gs;
        for (size_t i = 0; i < n; i++) CHECK(!memcmp(&x1[i], &x2[i], 4), "gradx_row n=%zu i=%zu (%a vs %a)", n, i, x1[i], x2[i]);
        free(wm); free(ws); free(wv); free(qa); free(xf); free(xr); free(x1); free(x2);
    }

    /* Sonda de denormales (informativa): NEON de ARMv7 los lleva a cero, el escalar VFP no. */
    {
        volatile float tiny = 1e-39f, one = 1.0f;     /* denormal: < 1.17549435e-38 */
        float g1[4] = {tiny, tiny, tiny, tiny}, g2[4] = {tiny, tiny, tiny, tiny}, x[4] = {one, one, one, one};
        ternary_neon_outer_acc(g1, 4, 0.0f, x);
        for (int i = 0; i < 4; i++) g2[i] += 0.0f * x[i];
        printf("sonda denormales: neon=%a escalar=%a -> %s\n", g1[0], g2[0], memcmp(g1, g2, 16) ? "DIFIEREN (flush-to-zero)" : "iguales");
    }
    if (fails) { printf("test_neon_u8: %d FALLAS\n", fails); return 1; }
    printf("TODAS LAS PRUEBAS NEON U8 PASARON\n");
    return 0;
}
